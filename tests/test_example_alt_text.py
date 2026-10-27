"""Offline run of examples/alt_text.py: catalog lookup -> IIIF first image -> describer. The catalog
fetch is a stand-in, the manifest and its images are already in the provider's cache directory, and
the model call answers through LiteLLM's local mock reply, so nothing touches the network."""

import importlib.util
import json
import os
import pathlib
import tempfile
import unittest

from PIL import Image

from chai.result import FileItemResult

EXAMPLE = pathlib.Path(__file__).resolve().parents[1] / "examples" / "alt_text.py"
spec = importlib.util.spec_from_file_location("alt_text", EXAMPLE)
example = importlib.util.module_from_spec(spec)
spec.loader.exec_module(example)

URL = "https://collections.britishart.yale.edu/catalog/tms:1000"
REPLY = "A painting of passion flowers climbing a stone pillar. The petals are white."
DOC = {
    "collection_ss": ["Prints and Drawings"],
    "title_ss": ["The Blue Passion Flower"],
    "object_name_ss": ["Engravings -- Hand-colored.", "Maps -- 1886.", "engravings -- hand-colored"],
    "manifest_ss": ["obj/1000"],
}


def manifest(n):
    return {
        "items": [
            {
                "id": f"canvas/{i}",
                "label": {"en": [f"view {i}"]},
                "items": [{"items": [{"motivation": "painting", "body": {"type": "Image", "id": f"http://img/{i}.jpg"}}]}],
            }
            for i in range(n)
        ]
    }


def cache_object(downloads, catalog_id, canvases=2):
    """Put a manifest and its images where the provider would have downloaded them."""
    directory = os.path.join(downloads, catalog_id.replace(":", "_"))
    os.makedirs(directory)
    with open(os.path.join(directory, "manifest.json"), "w") as fh:
        json.dump(manifest(canvases), fh)
    for i in range(canvases):
        Image.new("RGB", (40, 30), (30 * i, 90, 40)).save(os.path.join(directory, f"image_{i:04d}.jpg"))
    with open(os.path.join(directory, "notes.txt"), "w") as fh:
        fh.write("not an image of this manifest")
    return directory


class TestAltTextExample(unittest.TestCase):
    def test_object_types_per_collection(self):
        types, field = example.extract_object_types(DOC)
        self.assertEqual(field, "object_name_ss")
        self.assertEqual(types, ["Engravings (Hand-colored)", "Maps"])  # date dropped, duplicate folded
        archive = {"collection_ss": ["Archives"], "object_name_ss": ["x"], "genre_ss": ["Letters."]}
        self.assertEqual(example.extract_object_types(archive), (["Letters"], "genre_ss"))
        self.assertEqual(example.extract_object_types({}), ([], None))

    def test_catalog_outage_falls_back_to_tms_manifest(self):
        def down(url):
            raise RuntimeError("403")

        info = example.resolve_object(URL, fetch=down)
        self.assertEqual(info["manifest_url"], "https://manifests.collections.yale.edu/ycba/obj/1000")
        self.assertEqual(info["object_types"], [])
        self.assertIn("403", info["catalog_error"])

    def test_first_image_described_with_object_type(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = cache_object(tmp, "tms:1000")
            row = example.describe_object(
                URL, tmp, fetch=lambda url: DOC, retries=0, litellm_params={"mock_response": REPLY}
            )
            self.assertEqual(row["status"], "ok", row["error"])
            self.assertEqual(row["alt_text"], REPLY)
            self.assertEqual(os.path.basename(row["image_file"]), "image_0000.jpg")  # limit 1: first canvas only
            self.assertEqual(row["object_types"], ["Engravings (Hand-colored)", "Maps"])
            self.assertEqual(row["manifest_url"], "https://manifests.collections.yale.edu/ycba/obj/1000")

            # The catalogued type reaches the prompt through {ADDITIONAL_CONTEXT}
            wf = example.alt_text_workflow(directory)
            describe = wf.get_component_by_id("describe")
            describe.substitutions["ADDITIONAL_CONTEXT"] = example.object_type_context(row["object_types"])
            prompt, parts = describe.build_contents(FileItemResult(os.path.join(directory, "image_0000.jpg")))
            self.assertIn("lists the object type(s) of this work as: Engravings (Hand-colored); Maps.", prompt)
            self.assertNotIn("{ADDITIONAL_CONTEXT}", prompt)
            self.assertEqual([p["type"] for p in parts], ["image_url"])

    def test_provider_lists_only_the_manifests_images_in_order(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = cache_object(tmp, "tms:2", canvases=3)
            wf = example.alt_text_workflow(directory)
            provider = wf.get_component_by_id("iiif")
            provider.settings["limit"] = 0
            provider.steps = []  # the provider's own output, not its iterator's
            files = provider._process("http://manifest")
            names = [os.path.basename(str(getattr(f, "file_path", f))) for f in files.value]
            self.assertEqual(names, ["image_0000.jpg", "image_0001.jpg", "image_0002.jpg"])  # no notes.txt


if __name__ == "__main__":
    unittest.main()
