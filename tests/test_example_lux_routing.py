"""examples/workflow-lux-routing.json without a network or model weights: the classifier's reply comes
from LiteLLM's local mock, Chandra OCR 2 is stood in for by a deterministic describer, and each label
is checked to reach its reader -- plus the pieces the workflow rests on: AI classifiers that return
labels, and the hardware every local call records."""

import json
import pathlib
import tempfile
import unittest

from PIL import Image

from chai.classifier import GeminiClassifier
from chai.hardware import hardware_info, hardware_key, hardware_label, is_local_url, local_compute, profile_key
from chai.persistence import read_run_records, run_records_file
from chai.result import FileItemResult, ItemResult, LabelListResult, ListResult
from chai.usage import UsageLedger
from chai.workflow import Workflow

EXAMPLE = pathlib.Path(__file__).resolve().parents[1] / "examples" / "workflow-lux-routing.json"
LABELS = ["photograph", "typed_document", "typed_document_complex_layout", "newspaper", "handwritten_easy", "handwritten_hard"]


def find_step(tree, cid):
    if tree.get("id") == cid:
        return tree
    for key in ("steps", "next_steps", "true_steps", "false_steps"):
        for child in tree.get(key, []):
            found = find_step(child, cid)
            if found:
                return found
    return None


class TestLuxRouting(unittest.TestCase):
    def run_with(self, reply):
        """Run the workflow on two images with the classifier answering *reply*; the RESULT rows."""
        with tempfile.TemporaryDirectory() as tmp:
            images = pathlib.Path(tmp) / "images"
            images.mkdir()
            for i in range(2):
                Image.new("RGB", (60, 40), (200, 190 - 50 * i, 170)).save(images / f"page-{i}.jpg")
            tree = json.loads(EXAMPLE.read_text())
            find_step(tree, "images")["input"] = str(images)
            log = pathlib.Path(tmp) / "run.jsonl"
            for cid in ("log_classify", "log_alt_text", "log_gemini_htr", "log_chandra"):
                find_step(tree, cid)["settings"]["file"] = str(log)
            for cid, mock in (("classify", reply), ("alt_text", "A black-and-white photograph of a street."),
                              ("gemini_htr", "Item the xij day of Maye")):
                settings = find_step(tree, cid)["settings"]
                settings["litellm_params"] = {"mock_response": mock}
                settings["retries"] = 0
            # Chandra's weights are ~10 GB: a deterministic step stands in for it on the same branch
            chandra = find_step(tree, "chandra")
            chandra["type"] = "describer.FileInfoDescriber"
            chandra["settings"] = {}
            Workflow(tree).run()
            return [json.loads(line) for line in log.read_text().splitlines()]

    def routes(self, rows):
        return sorted({r["process_id"] for r in rows} - {"classify"})

    def test_each_label_reaches_its_reader(self):
        expected = {
            "photograph": ["alt_text"],
            "handwritten_hard": ["gemini_htr"],
            "Handwritten (hard)": ["gemini_htr"],
            "newspaper": ["chandra"],
            "typed_document": ["chandra"],
            "typed document complex layout": ["chandra"],
            "handwritten_easy": ["chandra"],
        }
        for reply, route in expected.items():
            with self.subTest(reply=reply):
                rows = self.run_with(reply)
                self.assertEqual(self.routes(rows), route)
                labels = [r["value"] for r in rows if r["process_id"] == "classify"]
                self.assertEqual(len(labels), 2)
                self.assertEqual(len(rows), 4)  # a label and a reading per image
                self.assertTrue(all(len(lbl) == 1 and lbl[0] in LABELS for lbl in labels))

    def test_a_reply_that_is_no_label_reads_nothing(self):
        rows = self.run_with("a painting")
        self.assertEqual(rows, [])


class TestAIClassifier(unittest.TestCase):
    def classifier(self, reply, **settings):
        wf = Workflow({"id": "classify_wf", "type": "workflow.Workflow"})
        settings = {"model": "gemini-3.1-flash-lite", "prompt": "Which label?", "tools": [],
                    "litellm_params": {"mock_response": reply}, **settings}
        return wf._make_step({"type": "classifier.GeminiClassifier", "id": "c", "settings": settings}, wf)

    def test_reply_becomes_labels(self):
        out = self.classifier("Typed document, complex layout.", labels=LABELS).process(ItemResult("x", metadata={"type": "TEXT"}))
        self.assertIsInstance(out, LabelListResult)
        self.assertEqual(out.value, ["typed_document_complex_layout"])
        self.assertEqual(out.metadata["reply"], "Typed document, complex layout.")
        self.assertIn("token_usage", out.metadata)

    def test_json_replies_and_free_labels(self):
        self.assertEqual(self.classifier('{"labels": ["a", "b"]}').labels_from('{"labels": ["a", "b"]}'), ["a", "b"])
        c = self.classifier("x", labels=LABELS)
        self.assertEqual(c.labels_from({"label": "newspaper"}), ["newspaper"])
        self.assertEqual(c.labels_from("This is handwritten_easy."), ["handwritten_easy"])
        with self.assertRaises(ValueError):
            c.labels_from("a painting")

    def test_generated_for_every_backend(self):
        import chai.classifier as module

        for name in ("GeminiClassifier", "LLMClassifier", "OllamaClassifier", "VLLMClassifier"):
            self.assertTrue(hasattr(module, name), name)
        self.assertTrue(issubclass(GeminiClassifier, module.Classifier))


class TestLocalCompute(unittest.TestCase):
    def test_hardware_info(self):
        info = hardware_info()
        for key in ("os", "os_version", "machine", "cpu", "cpu_cores_logical", "ram_gb", "gpus"):
            self.assertIn(key, info)
        self.assertTrue(info["cpu"])
        self.assertGreater(info["ram_gb"], 0)
        self.assertIsInstance(info["gpus"], list)
        self.assertIn(info["cpu"], hardware_label(info))

    def test_local_compute_entry_names_the_machine_by_key(self):
        entry = local_compute(7200, "cuda:0")
        self.assertEqual(entry, {"hardware": hardware_key(), "device": "cuda:0", "seconds": 7200.0, "hours": 2.0})
        self.assertRegex(entry["hardware"], r"^hw_[0-9a-f]{10}$")
        # the key is the profile's: stable for a machine, different for another
        self.assertEqual(hardware_key(), profile_key(dict(reversed(list(hardware_info().items())))))
        self.assertNotEqual(hardware_key(), profile_key({**hardware_info(), "ram_gb": 1.0}))

    def test_local_urls(self):
        for url in ("http://localhost:11434", "127.0.0.1:1234/v1", "http://[::1]:8000"):
            self.assertTrue(is_local_url(url), url)
        for url in ("", None, "https://api.example.org/v1", "http://gpu-box.internal:8000"):
            self.assertFalse(is_local_url(url), url)

    def test_a_local_server_call_records_the_machine_and_a_hosted_one_does_not(self):
        wf = Workflow({"id": "local_wf", "type": "workflow.Workflow"})
        local = wf._make_step({"type": "describer.OllamaDescriber", "id": "local", "settings": {
            "model": "gemma3", "prompt": "Describe {text_input_0}", "expected_output": "text",
            "litellm_params": {"mock_response": "a page"}}}, wf)
        hosted = wf._make_step({"type": "describer.GeminiDescriber", "id": "hosted", "settings": {
            "model": "gemini-3.1-flash-lite", "prompt": "Describe {text_input_0}", "expected_output": "text", "tools": [],
            "litellm_params": {"mock_response": "a page"}}}, wf)
        text = ItemResult("x", metadata={"type": "TEXT"})
        self.assertEqual(local.process(text).metadata["local_compute"]["hardware"], hardware_key())
        self.assertNotIn("local_compute", hosted.process(text).metadata)

    def test_rows_carry_the_key_and_the_run_carries_the_profile_once(self):
        with tempfile.TemporaryDirectory() as tmp:
            log = pathlib.Path(tmp) / "run.jsonl"
            wf = Workflow({"id": "keyed_wf", "type": "workflow.Workflow", "steps": [{
                "id": "each", "type": "iterator.Iterator", "steps": [{
                    "id": "local", "type": "describer.OllamaDescriber",
                    "settings": {"model": "gemma3", "prompt": "Describe {text_input_0}", "expected_output": "text",
                                 "litellm_params": {"mock_response": "a page"}},
                    "next_steps": [{"id": "log", "type": "storage.JsonLinesStorage", "settings": {"file": str(log)}}]}]}]})
            wf.run(ListResult([ItemResult(f"page {i}", metadata={"type": "TEXT"}) for i in range(3)]))
            rows = [json.loads(line) for line in log.read_text().splitlines()]
            records = read_run_records(run_records_file(str(log)))
        self.assertEqual(len(rows), 3)
        for row in rows:
            compute = row["metadata"]["local_compute"]
            self.assertEqual(set(compute), {"hardware", "device", "seconds", "hours"})
            self.assertEqual(compute["hardware"], hardware_key())
        self.assertEqual(wf.hardware, {hardware_key(): hardware_info()})
        (run,) = records["WORKFLOW_RUN"]
        self.assertEqual(run["metadata"]["hardware"], {hardware_key(): hardware_info()})
        self.assertEqual(run["metadata"]["usage"]["local_calls"], 3)
        self.assertNotIn("hardware", run["metadata"]["usage"])

    def test_ledger_totals_hours_per_machine(self):
        class C:
            id, name = "chandra", "chandra"

        ledger = UsageLedger()
        md = {"token_usage": {"total": 10, "prompt": 8, "result": 2}, "duration": 1800.0, "model": "m",
              "local_compute": local_compute(1800.0, "mps")}
        ledger.record(C(), md, key="a")
        ledger.record(C(), dict(md), key="b")
        ledger.record(C(), {"token_usage": {"total": 5}, "cost": 0.01, "duration": 3.0, "model": "hosted"}, key="c")
        summary = ledger.summary()
        self.assertEqual(summary["local_calls"], 2)
        self.assertEqual(summary["local_hours"], 1.0)
        self.assertEqual(summary["calls"], 3)
        (key, row), = summary["by_hardware"].items()
        self.assertEqual(key, hardware_key())
        self.assertEqual(row["label"], hardware_label(hardware_info()))
        self.assertEqual(summary["hardware"], {key: hardware_info()})
        self.assertEqual(row["devices"], ["mps"])
        self.assertEqual(row["local_hours"], 1.0)
        self.assertEqual(summary["by_model"]["hosted"]["local_hours"], 0.0)


if __name__ == "__main__":
    unittest.main()
