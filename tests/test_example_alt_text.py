"""Offline run of examples/workflow-alt-text.json: CSV of catalog URLs -> catalog record -> object type
and the first image of the manifest -> describer -> one RESULT row per object. A local HTTP server
stands in for the catalog, the manifest server and the image server, and the model call answers
through LiteLLM's local mock reply, so the whole workflow runs as it is written without a network."""

import json
import os
import pathlib
import tempfile
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from io import BytesIO

from PIL import Image

from chai.result import ItemResult
from chai.workflow import Workflow

EXAMPLE = pathlib.Path(__file__).resolve().parents[1] / "examples" / "workflow-alt-text.json"
REPLY = "A drawing of a man's head. He looks to the right. His hair is gray."

RECORDS = {
    # print: the type is object_name_ss, not genre_ss; its manifest has two canvases
    "tms:1": {"collection_ss": ["Prints and Drawings"], "object_name_ss": ["drawing"],
              "genre_ss": ["drawing", "portrait"], "manifest_ss": ["obj/1"]},
    # archive: the type is genre_ss
    "tms:2": {"collection_ss": ["Archives"], "object_name_ss": ["box"], "genre_ss": ["Letters."],
              "manifest_ss": ["obj/2"]},
    # a collection with no rule: no type, so the model names what it sees
    "tms:3": {"collection_ss": ["Reference Library"], "object_name_ss": ["book"], "manifest_ss": ["obj/3"]},
}
CANVASES = {"1": 2, "2": 1, "3": 1}


def jpeg(shade):
    buf = BytesIO()
    Image.new("RGB", (40, 30), (shade, 90, 40)).save(buf, "JPEG")
    return buf.getvalue()


class Server:
    """The catalog (``/catalog/<id>.json``), manifests (``/manifests/obj/<n>``) and images (``/img/...``)."""

    def __init__(self):
        routes = {}
        self.httpd = ThreadingHTTPServer(("127.0.0.1", 0), self.handler(routes))
        self.base = f"http://127.0.0.1:{self.httpd.server_address[1]}"
        for cid, doc in RECORDS.items():
            routes[f"/catalog/{cid}.json"] = ("application/json", json.dumps({"response": {"document": doc}}).encode())
        for n, count in CANVASES.items():
            items = [
                {"id": f"{self.base}/canvas/{n}/{i}", "label": {"en": [f"view {i}"]},
                 "items": [{"items": [{"motivation": "painting",
                                       "body": {"type": "Image", "id": f"{self.base}/img/{n}-{i}.jpg"}}]}]}
                for i in range(count)
            ]
            routes[f"/manifests/obj/{n}"] = ("application/json", json.dumps({"items": items}).encode())
            for i in range(count):
                routes[f"/img/{n}-{i}.jpg"] = ("image/jpeg", jpeg(40 * i))
        self.requests = []
        threading.Thread(target=self.httpd.serve_forever, daemon=True).start()

    def handler(self, routes):
        server = self

        class Handler(BaseHTTPRequestHandler):
            def do_GET(self):
                server.requests.append((self.path, self.headers.get("User-Agent")))
                if self.path not in routes:
                    self.send_error(404)
                    return
                ctype, body = routes[self.path]
                self.send_response(200)
                self.send_header("Content-Type", ctype)
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def log_message(self, *args):
                pass

        return Handler

    def close(self):
        self.httpd.shutdown()
        self.httpd.server_close()


def find_step(tree, cid):
    if tree.get("id") == cid:
        return tree
    for key in ("steps", "next_steps"):
        for child in tree.get(key, []):
            found = find_step(child, cid)
            if found:
                return found
    return None


class TestAltTextExample(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.server = Server()

    @classmethod
    def tearDownClass(cls):
        cls.server.close()

    def build(self, tmp):
        tree = json.loads(EXAMPLE.read_text())
        urls = pathlib.Path(tmp) / "urls.csv"
        ids = ["tms:1", "tms:2", "tms:3", "tms:404"]
        urls.write_text("url\n" + "".join(f"{self.server.base}/catalog/{cid}\n" for cid in ids))
        find_step(tree, "catalog_urls")["input"] = str(urls)
        find_step(tree, "catalog")["settings"]["retries"] = 0
        iiif = find_step(tree, "iiif")["settings"]
        iiif["base_url"] = f"{self.server.base}/manifests/"
        iiif["directory"] = os.path.join(tmp, "downloads")
        describe = find_step(tree, "describe")["settings"]
        describe["litellm_params"] = {"mock_response": REPLY}
        describe["retries"] = 0
        log = pathlib.Path(tmp) / "run.jsonl"
        find_step(tree, "log_describe")["settings"]["file"] = str(log)
        return Workflow(tree), log

    def test_one_row_per_object(self):
        with tempfile.TemporaryDirectory() as tmp:
            wf, log = self.build(tmp)
            wf.run()
            rows = sorted((json.loads(line) for line in log.read_text().splitlines()), key=lambda r: r["input_sequence"])
            downloads = pathlib.Path(tmp) / "downloads"
            # one directory per manifest, and only the first canvas of the two-canvas one
            self.assertEqual(sorted(p.name for p in downloads.iterdir()), ["manifests_obj_1", "manifests_obj_2", "manifests_obj_3"])
            self.assertEqual(sorted(p.name for p in (downloads / "manifests_obj_1").glob("*.jpg")), ["image_0000.jpg"])
        # tms:404 has no catalog record: an error entry of the iterator, and no row
        self.assertEqual([r["input_sequence"] for r in rows], [0, 1, 2])
        self.assertEqual([r["input"] for r in rows], [f"{self.server.base}/catalog/tms:{n}.json" for n in (1, 2, 3)])
        for row in rows:
            self.assertTrue(row["was_successful"])
            self.assertEqual(row["value"], REPLY)
        catalog_agents = {ua for path, ua in self.server.requests if path.startswith("/catalog/")}
        self.assertEqual(catalog_agents, {"curl/8.7.1 (chai-alt-text)"})

    def test_object_type_per_collection_reaches_the_prompt_with_the_image(self):
        with tempfile.TemporaryDirectory() as tmp:
            wf, _ = self.build(tmp)
            fan = wf.get_component_by_id("type_and_image")
            flatten = wf.get_component_by_id("type_with_image")
            describe = wf.get_component_by_id("describe")
            expected = {"tms:1": "['drawing']", "tms:2": "['Letters.']", "tms:3": "None"}
            for cid, object_type in expected.items():
                record = ItemResult({"response": {"document": RECORDS[cid]}}, metadata={"type": "DATA"})
                prompt, parts = describe.build_contents(flatten._process(fan._process(record)))
                self.assertTrue(prompt.endswith(f"object type(s): {object_type}"), prompt)
                self.assertEqual([p["type"] for p in parts], ["image_url"])  # the image itself, once
        # the fixed rules are the system message, ahead of the image: the part a KV cache can reuse
        self.assertIn("between 4 and 6 sentences", describe.system_instruction)
        self.assertNotIn("{text_input_0}", describe.system_instruction)


class TestIIIFProviderManifestAndSize(unittest.TestCase):
    """The YUAG alt-text workflow hands IIIFDirFileProvider the manifest a UrlJsonProvider fetched,
    and asks the image service for a scaled copy rather than a full-size museum image."""

    @classmethod
    def setUpClass(cls):
        cls.server = Server()

    @classmethod
    def tearDownClass(cls):
        cls.server.close()

    def provider(self, tmp, **settings):
        wf = Workflow({"id": "iiif_wf", "type": "workflow.Workflow"})
        settings = {"directory": tmp, "per_manifest": True, "limit": 1, **settings}
        return wf._make_step({"type": "provider.IIIFDirFileProvider", "settings": settings}, wf)

    def test_a_fetched_manifest_is_not_fetched_again(self):
        base = self.server.base
        manifest = {"id": f"{base}/manifests/obj/2",
                    "items": [{"id": "c", "items": [{"items": [{"motivation": "painting",
                                                                 "body": {"type": "Image", "id": f"{base}/img/2-0.jpg"}}]}]}]}
        with tempfile.TemporaryDirectory() as tmp:
            before = len(self.server.requests)
            out = self.provider(tmp)._process(ItemResult(manifest, metadata={"type": "DATA"}))
            paths = [p for p, _ in self.server.requests[before:]]
            self.assertEqual(paths, ["/img/2-0.jpg"])
            self.assertEqual([os.path.basename(f) for f in out.value], ["image_0000.jpg"])
            self.assertTrue(out.value[0].endswith(os.path.join("manifests_obj_2", "image_0000.jpg")))

    def test_size_asks_the_image_service(self):
        with tempfile.TemporaryDirectory() as tmp:
            sized = self.provider(tmp, size="!1600,1600")
            plain = self.provider(tmp)
        v3 = {"id": "https://img/x/full/full/0/default.jpg", "service": [{"id": "https://img/x", "type": "ImageService3"}]}
        v2 = {"id": "https://img/y/full/full/0/default.jpg", "service": [{"@id": "https://img/y/"}]}
        bare = {"id": "https://img/z.jpg", "service": []}
        self.assertEqual(sized.image_url(v3), "https://img/x/full/!1600,1600/0/default.jpg")
        self.assertEqual(sized.image_url(v2), "https://img/y/full/!1600,1600/0/default.jpg")
        self.assertEqual(sized.image_url(bare), "https://img/z.jpg")
        self.assertEqual(plain.image_url(v3), v3["id"])


if __name__ == "__main__":
    unittest.main()
