"""Tests for ``chai_tea_import``: a run's stored results loaded into the CHAI-TEA schema.

The pure-Python parts (normalizing a record, reading each storage layer, costing a call) run
anywhere. The tests that insert rows need a PostgreSQL server and skip without one, as the other
PostgreSQL tests in this suite do.
"""

import json
import os
import shutil
import sqlite3
import sys
import tempfile
import unittest
import uuid

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import chai_tea_import
import chai_tea_schema
from chai.storage import _pg_connect, postgres_params

try:
    conn = _pg_connect(postgres_params({"database": "postgres"}), autocommit=True)
    conn.close()
    HAS_POSTGRES = True
except Exception:
    HAS_POSTGRES = False

try:
    import pyarrow  # noqa: F401

    HAS_PYARROW = True
except ImportError:
    HAS_PYARROW = False


def gemini_metadata(prompt=1200, images=800, thinking=150, result=640, duration=2.5):
    """The metadata ``GeminiTranscriber`` records for one call."""
    return {
        "token_usage": {
            "total": prompt + images + thinking + result,
            "prompt": prompt,
            "images": images,
            "thinking": thinking,
            "result": result,
        },
        "duration": duration,
        "type": "TEXT",
        "engine": "gemini",
        "model": "gemini-3.1-flash-lite-preview",
    }


def result_record(**overrides):
    """One line as ``JsonLinesStorage`` writes it."""
    record = {
        "id": str(uuid.uuid4()),
        "type": "ItemResult",
        "workflowId": "wf1",
        "processorId": "transcribe1",
        "metadata": gemini_metadata(),
        "extraInfo": None,
        "input": "page00.jpg",
        "value": "the text",
    }
    record.update(overrides)
    return record


class TestNormalize(unittest.TestCase):
    """Both key spellings the storage layers use have to map onto the same row."""

    def test_result_to_json_keys(self):
        row = chai_tea_import.normalize(result_record())
        self.assertEqual(row["process_id"], "transcribe1")
        self.assertEqual(row["workflow_id"], "wf1")
        self.assertEqual(row["input"], "page00.jpg")
        self.assertTrue(row["was_successful"])
        self.assertEqual(row["md_duration"], 2.5)

    def test_tabular_column_keys(self):
        row = chai_tea_import.normalize(
            {
                "id": str(uuid.uuid4()),
                "processor_id": "transcribe1",
                "workflow_id": "wf1",
                "value_json": json.dumps("the text"),
                "metadata_json": json.dumps(gemini_metadata()),
                "extra_json": json.dumps({"note": "x"}),
                "input_uri": "page00.jpg",
                "input_hash": "abc",
                "input_locator": json.dumps([{"page": 3}, {"start": 10, "end": 40}]),
                "corrects_id": None,
                "created_at": "2026-01-15 10:00:00",
            }
        )
        self.assertEqual(row["process_id"], "transcribe1")
        self.assertEqual(row["input"], "page00.jpg")
        self.assertEqual(row["input_hash"], "abc")
        self.assertEqual(json.loads(row["input_segment"]), [{"page": 3}, {"start": 10, "end": 40}])
        self.assertEqual(row["input_sequence"], 3)
        self.assertEqual(row["extra_data"], {"note": "x"})
        self.assertIsNotNone(row["md_timestamp"])

    def test_the_whole_ai_metadata_dict_survives(self):
        """The point of the importer: nothing a provider recorded may be dropped on the way in."""
        metadata = gemini_metadata()
        row = chai_tea_import.normalize(result_record(metadata=metadata))
        self.assertEqual(row["metadata"], metadata)
        self.assertEqual(row["metadata"]["token_usage"]["thinking"], 150)
        self.assertEqual(row["metadata"]["model"], "gemini-3.1-flash-lite-preview")
        self.assertEqual(row["metadata"]["engine"], "gemini")

    def test_an_errored_step_is_not_successful(self):
        row = chai_tea_import.normalize(
            result_record(metadata={"type": "ERROR", "error": "429", "error_class": "Quota"})
        )
        self.assertFalse(row["was_successful"])

    def test_a_non_uuid_id_becomes_a_stable_uuid(self):
        first = chai_tea_import.normalize(result_record(id="page-1-transcript"))
        second = chai_tea_import.normalize(result_record(id="page-1-transcript"))
        self.assertEqual(first["id"], second["id"])
        uuid.UUID(first["id"])  # raises if it is not one
        self.assertNotEqual(first["id"], chai_tea_import.normalize(result_record(id="page-2"))["id"])

    def test_a_uuid_id_is_kept_as_it_is(self):
        given = str(uuid.uuid4())
        self.assertEqual(chai_tea_import.normalize(result_record(id=given))["id"], given)


class TestCosting(unittest.TestCase):
    def test_prices_parse(self):
        self.assertEqual(chai_tea_import.parse_prices(["m=0.10/0.40"]), {"m": (0.10, 0.40)})
        self.assertEqual(chai_tea_import.parse_prices(["m=0.5"]), {"m": (0.5, 0.5)})
        for bad in (["nope"], ["m=x/y"]):
            with self.assertRaises(ValueError):
                chai_tea_import.parse_prices(bad)

    def test_cost_bills_thinking_as_output(self):
        # (1200 + 800) in at $0.10/M, (640 + 150) out at $0.40/M
        cost = chai_tea_import.cost_of(gemini_metadata(), {"*": (0.10, 0.40)})
        self.assertAlmostEqual(cost, 2000 * 0.10 / 1e6 + 790 * 0.40 / 1e6)

    def test_negative_counts_read_as_zero(self):
        """Providers write -1 for a modality they did not break down; it must not subtract."""
        metadata = {
            "token_usage": {"total": 1400, "prompt": 900, "images": -1, "thinking": -1, "result": 500},
            "model": "llama3.2",
        }
        cost = chai_tea_import.cost_of(metadata, {"*": (1.0, 1.0)})
        self.assertAlmostEqual(cost, 900 / 1e6 + 500 / 1e6)

    def test_an_exact_model_name_beats_a_glob(self):
        prices = {"*": (9.0, 9.0), "gemini-*": (0.10, 0.40)}
        self.assertEqual(chai_tea_import.price_for("gemini-3.1-flash-lite-preview", prices), (0.10, 0.40))
        self.assertEqual(chai_tea_import.price_for("llama3.2", prices), (9.0, 9.0))

    def test_no_price_leaves_the_cost_unknown(self):
        self.assertIsNone(chai_tea_import.cost_of(gemini_metadata(), {}))
        self.assertIsNone(chai_tea_import.cost_of({"type": "TEXT"}, {"*": (1.0, 1.0)}))


class TestReaders(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.mkdtemp()
        self.records = [result_record() for _ in range(3)]

    def tearDown(self):
        shutil.rmtree(self.dir, ignore_errors=True)

    def test_detect_source(self):
        self.assertEqual(chai_tea_import.detect_source("run.jsonl"), "jsonl")
        self.assertEqual(chai_tea_import.detect_source("run.parquet"), "parquet")
        self.assertEqual(chai_tea_import.detect_source("results.db"), "sqlite")
        self.assertEqual(chai_tea_import.detect_source("postgresql://host/db"), "postgres")
        with self.assertRaises(ValueError):
            chai_tea_import.detect_source("results.txt")

    def test_a_truncated_last_line_does_not_lose_the_file(self):
        """A run killed part way through leaves a partial line; the rest still has to import."""
        path = os.path.join(self.dir, "run.jsonl")
        with open(path, "w", encoding="utf-8") as fh:
            for record in self.records:
                fh.write(json.dumps(record) + "\n")
            fh.write('{"id": "half-written')
        self.assertEqual(len(list(chai_tea_import.read_jsonl(path))), 3)

    def test_sqlite_rows_read_back(self):
        path = os.path.join(self.dir, "results.db")
        conn = sqlite3.connect(path)
        conn.execute(
            "CREATE TABLE results (id TEXT PRIMARY KEY, processor_id TEXT, workflow_id TEXT, "
            "value_json TEXT, metadata_json TEXT, extra_json TEXT, input_uri TEXT, input_hash TEXT, "
            "input_locator TEXT, corrects_id TEXT, created_at TIMESTAMP)"
        )
        for record in self.records:
            conn.execute(
                "INSERT INTO results VALUES (?,?,?,?,?,?,?,?,?,?,?)",
                (
                    record["id"], record["processorId"], record["workflowId"],
                    json.dumps(record["value"]), json.dumps(record["metadata"]), None,
                    record["input"], "abc", None, None, "2026-01-15 10:00:00",
                ),
            )
        conn.commit()
        conn.close()
        rows = list(chai_tea_import.read_sqlite(path))
        self.assertEqual(len(rows), 3)
        self.assertEqual(
            chai_tea_import.normalize(rows[0])["metadata"]["model"], "gemini-3.1-flash-lite-preview"
        )

    @unittest.skipUnless(HAS_PYARROW, "pyarrow is not installed")
    def test_parquet_rows_read_back(self):
        from chai.storage import ParquetRecordWriter

        path = os.path.join(self.dir, "run.parquet")
        writer = ParquetRecordWriter(path)
        for record in self.records:
            writer.add(record)
        writer.close()
        rows = list(chai_tea_import.read_parquet(path))
        self.assertEqual(len(rows), 3)
        self.assertEqual(
            chai_tea_import.normalize(rows[0])["metadata"]["token_usage"]["thinking"], 150
        )


@unittest.skipUnless(HAS_POSTGRES, "no PostgreSQL server on localhost:5432")
class TestImportIntoChaiTea(unittest.TestCase):
    """A run imported into a real CHAI-TEA database."""

    def setUp(self):
        self.dir = tempfile.mkdtemp()
        self.database = f"chai_tea_import_{uuid.uuid4().hex[:8]}"
        chai_tea_schema.create_schema({"database": self.database})
        self.records = [result_record() for _ in range(3)]
        self.records.append(
            result_record(
                processorId="translate1",
                metadata={
                    "token_usage": {"total": 1400, "prompt": 900, "images": -1,
                                    "thinking": -1, "result": 500},
                    "duration": 1.2, "type": "TEXT", "engine": "ollama", "model": "llama3.2",
                },
            )
        )
        self.records.append(result_record(metadata={"type": "ERROR", "error": "429 quota"}))
        self.records.append(result_record(processorId="segment1", metadata={"type": "TEXT"}))
        self.path = os.path.join(self.dir, "run.jsonl")
        with open(self.path, "w", encoding="utf-8") as fh:
            for record in self.records:
                fh.write(json.dumps(record) + "\n")

    def tearDown(self):
        shutil.rmtree(self.dir, ignore_errors=True)
        conn = _pg_connect(postgres_params({"database": "postgres"}), autocommit=True)
        try:
            conn.cursor().execute(f'DROP DATABASE IF EXISTS "{self.database}"')
        finally:
            conn.close()

    def run_import(self, **kwargs):
        kwargs.setdefault("project", "M804")
        return chai_tea_import.import_records(
            chai_tea_import.read_records(self.path), {"database": self.database}, **kwargs
        )

    def query(self, sql, args=()):
        conn = _pg_connect(postgres_params({"database": self.database}))
        try:
            cursor = conn.cursor()
            cursor.execute(sql, args)
            return cursor.fetchall()
        finally:
            conn.close()

    def test_a_run_lands_with_its_project_workflow_and_run_rows(self):
        summary = self.run_import()
        self.assertEqual(summary["records"], 6)
        self.assertEqual(summary["inserted"], 6)
        self.assertEqual(summary["errors"], 1)
        self.assertEqual(self.query("SELECT name FROM projects"), [("M804",)])
        self.assertEqual(self.query("SELECT name FROM workflows"), [("wf1",)])
        self.assertEqual(self.query("SELECT count(*) FROM workflow_runs"), [(1,)])
        self.assertEqual(self.query("SELECT count(*) FROM results"), [(6,)])

    def test_every_ai_metadata_field_is_queryable_afterwards(self):
        self.run_import()
        rows = self.query(
            "SELECT metadata->>'engine', metadata->>'model', "
            "       (metadata->'token_usage'->>'thinking')::int, "
            "       (metadata->'token_usage'->>'total')::int, md_duration "
            "  FROM results WHERE process_id = 'transcribe1' AND was_successful "
            "  ORDER BY md_duration LIMIT 1"
        )
        self.assertEqual(rows, [("gemini", "gemini-3.1-flash-lite-preview", 150, 2790, 2.5)])

    def test_token_totals_add_up_across_models(self):
        self.run_import()
        rows = self.query(
            "SELECT metadata->>'model', count(*), sum((metadata->'token_usage'->>'total')::int) "
            "  FROM results WHERE metadata ? 'token_usage' GROUP BY 1 ORDER BY 1"
        )
        self.assertEqual(rows, [("gemini-3.1-flash-lite-preview", 3, 2790 * 3), ("llama3.2", 1, 1400)])

    def test_md_cost_is_filled_in_from_the_price_table(self):
        summary = self.run_import(prices={"gemini-3.1-flash-lite-preview": (0.10, 0.40)})
        self.assertEqual(summary["costed"], 3)
        per_call = 2000 * 0.10 / 1e6 + 790 * 0.40 / 1e6
        self.assertAlmostEqual(summary["cost"], per_call * 3)
        # llama3.2 had no price, so its row is left unknown rather than costed at zero
        self.assertEqual(
            self.query("SELECT count(*) FROM results WHERE md_cost IS NULL"), [(3,)]
        )

    def test_md_cost_stays_null_without_prices(self):
        self.run_import()
        self.assertEqual(self.query("SELECT count(*) FROM results WHERE md_cost IS NOT NULL"), [(0,)])

    def test_md_timestamp_is_never_null(self):
        """chai_tea_parquet.py's --since export filters on it, so a NULL would hide the row."""
        self.run_import()
        self.assertEqual(self.query("SELECT count(*) FROM results WHERE md_timestamp IS NULL"), [(0,)])

    def test_importing_the_same_run_twice_adds_nothing(self):
        self.run_import()
        again = self.run_import()
        self.assertEqual(again["inserted"], 0)
        self.assertEqual(again["duplicates"], 6)
        self.assertEqual(self.query("SELECT count(*) FROM results"), [(6,)])

    def test_the_run_row_records_how_the_run_went(self):
        self.run_import()
        rows = self.query("SELECT was_successful, duration, last_run IS NOT NULL FROM workflow_runs")
        # one step errored, so the run as a whole did not succeed
        self.assertEqual(rows[0][0], False)
        self.assertAlmostEqual(rows[0][1], 2.5 * 3 + 1.2)
        self.assertTrue(rows[0][2])

    def test_a_second_run_of_the_same_workflow_reuses_the_project_and_workflow(self):
        self.run_import()
        with open(self.path, "w", encoding="utf-8") as fh:
            fh.write(json.dumps(result_record()) + "\n")
        self.run_import()
        self.assertEqual(self.query("SELECT count(*) FROM projects"), [(1,)])
        self.assertEqual(self.query("SELECT count(*) FROM workflows"), [(1,)])
        self.assertEqual(self.query("SELECT count(*) FROM workflow_runs"), [(2,)])

    def test_previous_id_links_up_regardless_of_record_order(self):
        """A correction can be written before the row it corrects; the link is made afterwards."""
        target = result_record()
        correction = result_record(value="fixed")
        correction["corrects_id"] = target["id"]
        with open(self.path, "w", encoding="utf-8") as fh:
            fh.write(json.dumps(correction) + "\n")  # the correction comes first
            fh.write(json.dumps(target) + "\n")
        summary = self.run_import()
        self.assertEqual(summary["inserted"], 2)
        self.assertEqual(summary["linked"], 1)
        self.assertEqual(
            self.query("SELECT previous_id FROM results WHERE id = %s", (correction["id"],)),
            [(uuid.UUID(target["id"]),)],
        )

    def test_a_correction_of_a_row_that_was_never_imported_does_not_fail_the_run(self):
        correction = result_record(value="fixed")
        correction["corrects_id"] = str(uuid.uuid4())  # target is in neither the file nor the database
        with open(self.path, "w", encoding="utf-8") as fh:
            fh.write(json.dumps(correction) + "\n")
        summary = self.run_import()
        self.assertEqual(summary["inserted"], 1)
        self.assertEqual(summary["dangling"], 1)
        self.assertEqual(
            self.query("SELECT previous_id FROM results WHERE id = %s", (correction["id"],)), [(None,)]
        )

    def test_deleting_the_project_cascades_to_the_imported_results(self):
        self.run_import()
        conn = _pg_connect(postgres_params({"database": self.database}))
        conn.cursor().execute("DELETE FROM projects")
        conn.commit()
        conn.close()
        self.assertEqual(self.query("SELECT count(*) FROM results"), [(0,)])
        self.assertEqual(self.query("SELECT count(*) FROM workflow_runs"), [(0,)])


if __name__ == "__main__":
    unittest.main()
