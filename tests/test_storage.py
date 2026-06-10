import json
import os
import shutil
import sqlite3
import tempfile
import unittest

from chai.result import FileItemResult, ItemResult
from chai.storage import get_result, list_processors, list_results, save_correction
from chai.workflow import Workflow


def make_file_result(path="/nonexistent/crop.png", content=b"\x89PNG\r\n\x1a\nfakebytes"):
    """An in-memory FileItemResult whose bytes never touch disk (the path does not exist)."""
    fr = FileItemResult(path)
    fr.file_bytes = content
    return fr


class TestFileSystemStorage(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.mkdtemp()
        self.wf = Workflow({"id": "fs_storage_wf", "type": "workflow.Workflow"})
        self.storage = self.wf._make_step(
            {"type": "storage.FileSystemStorage", "settings": {"directory": self.dir}}, self.wf
        )

    def tearDown(self):
        shutil.rmtree(self.dir, ignore_errors=True)

    def stored_path(self, result, processor_dir="base"):
        return os.path.join(self.dir, processor_dir, result.id[0:2], result.id[2:4], f"{result.id}.json")

    def test_stores_item_result_as_pairtree_json(self):
        res = ItemResult("hello world", metadata={"type": "TEXT"})
        self.storage.process(res)
        fn = self.stored_path(res)
        self.assertTrue(os.path.exists(fn))
        with open(fn) as fh:
            js = json.load(fh)
        self.assertEqual(js["id"], res.id)
        self.assertEqual(js["type"], "ItemResult")
        self.assertEqual(js["value"], "hello world")
        self.assertEqual(js["metadata"]["type"], "TEXT")

    def test_file_item_result_with_bytes_does_not_crash(self):
        res = make_file_result()
        self.storage.process(res)  # must not try to JSON-encode the PNG bytes
        with open(self.stored_path(res)) as fh:
            js = json.load(fh)
        # the file path, not the file content, is persisted
        self.assertEqual(js["value"], "/nonexistent/crop.png")
        self.assertEqual(js["metadata"]["type"], "IMAGE")

    def test_raw_bytes_value_gets_placeholder(self):
        res = ItemResult(b"binary blob")
        self.storage.process(res)
        with open(self.stored_path(res)) as fh:
            js = json.load(fh)
        self.assertEqual(js["value"], {"__bytes__": len(b"binary blob")})


class TestSqliteStorage(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.mkdtemp()
        self.db = os.path.join(self.dir, "results.db")
        self.wf = Workflow({"id": "sqlite_storage_wf", "type": "workflow.Workflow"})
        self.storage = self.wf._make_step(
            {"type": "storage.SqliteStorage", "settings": {"database": self.db}}, self.wf
        )
        self.comp_a = self.wf._make_step({"type": "describer.FileInfoDescriber", "id": "comp_a"}, self.wf)
        self.comp_b = self.wf._make_step({"type": "describer.FileInfoDescriber", "id": "comp_b"}, self.wf)

    def tearDown(self):
        shutil.rmtree(self.dir, ignore_errors=True)

    def store(self, result, processor=None):
        if processor is not None:
            result.processor = processor
        result.workflow = self.wf
        self.storage.process(result)
        return result

    def store_samples(self):
        r1 = self.store(ItemResult("hello world"), self.comp_a)
        r2 = self.store(ItemResult({"name": "Ada", "year": 1815}), self.comp_a)
        r3 = self.store(make_file_result(), self.comp_b)
        return r1, r2, r3

    def test_rows_exist_for_each_stored_result(self):
        r1, r2, r3 = self.store_samples()
        conn = sqlite3.connect(self.db)
        try:
            ids = {row[0] for row in conn.execute("SELECT id FROM results")}
            self.assertEqual(ids, {r1.id, r2.id, r3.id})
            value_json = conn.execute(
                "SELECT value_json FROM results WHERE id = ?", (r3.id,)
            ).fetchone()[0]
        finally:
            conn.close()
        js = json.loads(value_json)  # FileItemResult row holds valid JSON with the path, not bytes
        self.assertEqual(js["value"], "/nonexistent/crop.png")
        self.assertEqual(js["processorId"], "comp_b")

    def test_value_json_is_full_result_json(self):
        r2 = self.store(ItemResult({"name": "Ada", "year": 1815}), self.comp_a)
        row = get_result(self.db, r2.id)
        self.assertEqual(row["value"]["id"], r2.id)
        self.assertEqual(row["value"]["type"], "ItemResult")
        self.assertEqual(row["value"]["value"], {"name": "Ada", "year": 1815})
        self.assertEqual(row["value"]["workflowId"], "sqlite_storage_wf")
        self.assertIn("timestamp", row["metadata"])

    def test_bytes_value_gets_placeholder(self):
        res = self.store(ItemResult(b"binary blob"), self.comp_a)
        row = get_result(self.db, res.id)
        self.assertEqual(row["value"]["value"], {"__bytes__": len(b"binary blob")})

    def test_list_results_and_filters(self):
        r1, r2, r3 = self.store_samples()
        rows = list_results(self.db)
        self.assertEqual({r["id"] for r in rows}, {r1.id, r2.id, r3.id})
        for row in rows:
            self.assertFalse(row["corrected"])
            self.assertIsNotNone(row["created_at"])
        by_proc = list_results(self.db, processor_id="comp_a")
        self.assertEqual({r["id"] for r in by_proc}, {r1.id, r2.id})
        by_wf = list_results(self.db, workflow_id="sqlite_storage_wf")
        self.assertEqual(len(by_wf), 3)
        self.assertEqual(list_results(self.db, workflow_id="no_such_wf"), [])
        self.assertEqual(len(list_results(self.db, limit=2)), 2)
        self.assertEqual(len(list_results(self.db, limit=2, offset=2)), 1)

    def test_get_result_round_trip(self):
        r1, _, _ = self.store_samples()
        row = get_result(self.db, r1.id)
        self.assertEqual(row["processor_id"], "comp_a")
        self.assertEqual(row["workflow_id"], "sqlite_storage_wf")
        self.assertEqual(row["value"]["value"], "hello world")
        self.assertIsNone(get_result(self.db, "no-such-id"))

    def test_save_correction(self):
        r1, _, _ = self.store_samples()
        self.assertTrue(save_correction(self.db, r1.id, {"text": "hello world, corrected"}))
        row = get_result(self.db, r1.id)
        self.assertTrue(row["corrected"])
        self.assertEqual(row["corrected_value"], {"text": "hello world, corrected"})
        self.assertIsNotNone(row["corrected_at"])
        # the original value is untouched
        self.assertEqual(row["value"]["value"], "hello world")
        self.assertFalse(save_correction(self.db, "no-such-id", "x"))

    def test_restoring_keeps_correction(self):
        r1 = self.store(ItemResult("hello world"), self.comp_a)
        save_correction(self.db, r1.id, "fixed")
        self.store(r1)  # same result stored again (e.g. a re-run)
        row = get_result(self.db, r1.id)
        self.assertTrue(row["corrected"])
        self.assertEqual(row["corrected_value"], "fixed")

    def test_list_processors(self):
        self.store_samples()
        procs = {p["processor_id"]: p["count"] for p in list_processors(self.db)}
        self.assertEqual(procs, {"comp_a": 2, "comp_b": 1})

    def test_derivative_storage(self):
        src = ItemResult("source text")
        deriv = ItemResult(["label"], processor=self.comp_a, register_on=src)
        self.store(src, self.comp_b)
        conn = sqlite3.connect(self.db)
        try:
            row = conn.execute(
                "SELECT source_id, component_id, result_json FROM derivatives WHERE id = ?", (deriv.id,)
            ).fetchone()
        finally:
            conn.close()
        self.assertIsNotNone(row)
        self.assertEqual(row[0], src.id)
        self.assertEqual(row[1], "comp_a")
        self.assertEqual(json.loads(row[2])["value"], ["label"])

    def test_upgrades_old_schema_in_place(self):
        old_db = os.path.join(self.dir, "old.db")
        conn = sqlite3.connect(old_db)
        conn.execute(
            """CREATE TABLE results (
                id TEXT PRIMARY KEY, processor_id TEXT, workflow_id TEXT, value_json TEXT,
                metadata_json TEXT, extra_json TEXT, created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )"""
        )
        conn.execute("INSERT INTO results (id, processor_id, value_json) VALUES ('old1', 'p1', '\"hi\"')")
        conn.commit()
        conn.close()
        rows = list_results(old_db)  # must ALTER in the correction columns without complaint
        self.assertEqual(rows[0]["id"], "old1")
        self.assertFalse(rows[0]["corrected"])
        self.assertTrue(save_correction(old_db, "old1", "hi there"))
        self.assertEqual(get_result(old_db, "old1")["corrected_value"], "hi there")


if __name__ == "__main__":
    unittest.main()
