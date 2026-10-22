"""A run that stores from several places leaves one Parquet file, handed on once the run has ended.

``ParquetStorage`` steps naming the same file share it, and its ``file_steps`` run on the written
file after the workflow finishes -- here a ``FileSystemUploader``, standing in for ``S3Uploader``, so
what reaches the drop folder is the complete file with the finished run in its metadata.
"""

import os
import shutil
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pyarrow.parquet as pq  # noqa: E402

from chai.persistence import parquet_run_records  # noqa: E402
from chai.workflow import Workflow  # noqa: E402


class TestParquetHandOff(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.mkdtemp()
        self.out = os.path.join(self.dir, "out", "run-{run_id}.parquet")
        self.drop = os.path.join(self.dir, "drop")

    def tearDown(self):
        shutil.rmtree(self.dir, ignore_errors=True)

    def store(self, file_steps=None):
        step = {"type": "storage.ParquetStorage", "settings": {"file": self.out, "flush_on_exit": False}}
        if file_steps:
            step["file_steps"] = file_steps
        return step

    def workflow(self, file_steps):
        upload = [{"id": "to_drop", "type": "uploader.FileSystemUploader", "settings": {"directory": self.drop}}]
        return Workflow(
            {
                "id": "handoff_wf",
                "type": "workflow.Workflow",
                "project": "Hand-off",
                "steps": [
                    {
                        "type": "provider.StaticProvider",
                        "settings": {"values": ["Ada wrote it.", "Babbage built it."]},
                        "steps": [
                            {
                                "type": "iterator.Iterator",
                                "steps": [
                                    {
                                        "id": "words",
                                        "type": "segmenter.WordSegmenter",
                                        "next_steps": [self.store(upload if file_steps == "first" else None)],
                                    },
                                    {
                                        "id": "upper",
                                        "type": "translator.GlossaryTranslator",
                                        "settings": {"glossary": {"wrote": "WROTE"}},
                                        "next_steps": [self.store(upload if file_steps == "second" else None)],
                                    },
                                ],
                            }
                        ],
                    }
                ],
            }
        )

    def test_one_file_for_the_run_uploaded_after_it_ends(self):
        for where in ("first", "second"):  # the upload may hang off either of the sharing steps
            with self.subTest(file_steps=where):
                shutil.rmtree(self.drop, ignore_errors=True)
                wf = self.workflow(where)
                wf.run("go")
                [uploaded] = os.listdir(self.drop)
                self.assertEqual(uploaded, f"run-{wf.run_id}.parquet")
                table = pq.read_table(os.path.join(self.drop, uploaded)).to_pydict()
                # both steps' rows, from both entries, in the one file
                self.assertEqual(sorted(set(table["process_id"])), ["upper", "words"])
                self.assertEqual(len(table["process_id"]), 4)
                # the copy was made after the run ended: its run row says how the run went
                records = parquet_run_records(os.path.join(self.drop, uploaded))
                run = records["WORKFLOW_RUN"]
                self.assertEqual(run["id"], wf.run_id)
                self.assertTrue(run["was_successful"])
                self.assertIsNotNone(run["duration"])
                # the rows a loader creates first are all there: the run, its workflow and project
                self.assertEqual(records["WORKFLOW"]["id"], wf.uuid)
                self.assertEqual(records["PROJECT"]["name"], "Hand-off")
                # and the RESULT columns leave out created/modified, which the central loader refuses
                self.assertNotIn("created", table)
                self.assertNotIn("modified", table)

    def test_a_failing_file_step_leaves_the_file(self):
        wf = self.workflow("first")
        os.makedirs(self.drop)
        with open(os.path.join(self.drop, "blocker"), "w"):
            pass
        storage = next(c for c in wf.registry_ids.values() if c.__class__.__name__ == "ParquetStorage")
        storage.file_steps[0].directory = os.path.join(self.drop, "blocker")  # a file, not a directory
        with self.assertLogs("chai", level="ERROR"):
            wf.run("go")
        self.assertTrue(os.path.exists(self.out.replace("{run_id}", wf.run_id)))
        [failed] = storage.file_results
        self.assertEqual(failed.metadata["type"], "ERROR")

    def test_a_jsonl_run_without_its_run_rows_is_written_with_a_warning(self):
        from chai.persistence import jsonl_to_parquet

        jsonl = os.path.join(self.dir, "run.jsonl")
        Workflow(
            {
                "id": "jsonl_wf",
                "type": "workflow.Workflow",
                "steps": [
                    {
                        "type": "translator.GlossaryTranslator",
                        "settings": {"glossary": {"one": "un"}},
                        "next_steps": [{"type": "storage.JsonLinesStorage", "settings": {"file": jsonl}}],
                    }
                ],
            }
        ).run("one")
        os.remove(os.path.join(self.dir, "run.records.json"))  # as if the run rows were never kept
        with self.assertLogs("chai", level="WARNING") as logged:
            info = jsonl_to_parquet(jsonl, os.path.join(self.dir, "run.parquet"))
        self.assertEqual(info["rows"], 1)
        self.assertIn("no PROJECT/WORKFLOW/WORKFLOW_RUN rows", "\n".join(logged.output))

    def test_steps_that_disagree_about_what_the_file_holds_are_refused(self):
        tree = {
            "id": "mixed_wf",
            "type": "workflow.Workflow",
            "steps": [
                self.store(),
                {"type": "storage.ParquetStorage", "settings": {"file": self.out, "fields": ["value"], "flush_on_exit": False}},
            ],
        }
        with self.assertRaises(ValueError):
            Workflow(tree)


if __name__ == "__main__":
    unittest.main()
