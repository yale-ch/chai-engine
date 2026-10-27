"""Offline run of examples/workflow-stockbonds.json: each certificate image goes to the extractor, whose
reply (held to the workflow's schema by the API) is written as one RESULT row. The model call answers
through LiteLLM's local mock reply, so the wiring is exercised end to end without a network."""

import json
import pathlib
import tempfile
import unittest

from PIL import Image

from chai.persistence import RESULT_COLUMNS
from chai.workflow import Workflow

EXAMPLE = pathlib.Path(__file__).resolve().parents[1] / "examples" / "workflow-stockbonds.json"

REPLY = {
    "blank_page": False,
    "transcription": "UNITED PAPERBOARD COMPANY\nThis is to Certify that",
    "company_name": "UNITED PAPERBOARD COMPANY",
    "color": "orange",
    "number_of_shares": "100",
}


def find_step(tree, cid):
    if tree.get("id") == cid:
        return tree
    for key in ("steps", "next_steps"):
        for child in tree.get(key, []):
            found = find_step(child, cid)
            if found:
                return found
    return None


class TestStockbondsExample(unittest.TestCase):
    def build(self, tmp, reply):
        tree = json.loads(EXAMPLE.read_text())
        images = pathlib.Path(tmp) / "images"
        images.mkdir()
        for i in range(3):
            Image.new("RGB", (64, 40), (200, 120 + i, 40)).save(images / f"bond-{i}.jpg")
        find_step(tree, "stockbond_images")["input"] = str(images)
        settings = find_step(tree, "extract")["settings"]
        settings["litellm_params"] = {"mock_response": reply}
        settings["retries"] = 0
        log = pathlib.Path(tmp) / "run.jsonl"
        find_step(tree, "log_extract")["settings"]["file"] = str(log)
        return Workflow(tree), log

    def test_one_row_per_certificate(self):
        with tempfile.TemporaryDirectory() as tmp:
            wf, log = self.build(tmp, json.dumps(REPLY))
            wf.run()
            rows = [json.loads(line) for line in log.read_text().splitlines()]
        self.assertEqual(len(rows), 3)
        self.assertEqual(list(rows[0]), list(RESULT_COLUMNS))
        self.assertEqual(sorted(pathlib.Path(r["input"]).name for r in rows), ["bond-0.jpg", "bond-1.jpg", "bond-2.jpg"])
        self.assertEqual(sorted(r["input_sequence"] for r in rows), [0, 1, 2])
        for row in rows:
            self.assertTrue(row["was_successful"])
            value = row["value"] if isinstance(row["value"], dict) else json.loads(row["value"])
            self.assertEqual(value, REPLY)


if __name__ == "__main__":
    unittest.main()
