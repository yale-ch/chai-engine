"""Tests for ``chai_tea_rows``: a run's stored results reshaped into exactly the CHAI-TEA RESULT columns.

The reshaping and the Parquet file run anywhere. Loading the file into a real CHAI-TEA database needs
a PostgreSQL server and skips without one, as the other PostgreSQL tests in this suite do.
"""

import datetime
import json
import os
import shutil
import sys
import tempfile
import unittest
import uuid

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pyarrow.parquet as pq

import chai_tea_rows
import chai_tea_schema
from chai.storage import _pg_connect, parquet_columns, parquet_to_postgres, postgres_params

try:
    conn = _pg_connect(postgres_params({"database": "postgres"}), autocommit=True)
    conn.close()
    HAS_POSTGRES = True
except Exception:
    HAS_POSTGRES = False

RESULT_COLUMNS = list(chai_tea_schema.column_types("RESULT"))

# Two lines as the m804 workflow's JsonLinesStorage writes them: value stored as "text", the page
# the result was made from as "page", the component as "step".
M804_FIELDS = {"text": "value", "page": "input_uri"}


def m804_record(step="transcribe", text="Ci commencent les croniques", page="inputs/m804/split/f0001r.jpg"):
    return {
        "id": str(uuid.uuid4()),
        "step": step,
        "page": page,
        "metadata": {
            "token_usage": {"total": 6450, "prompt": 364, "images": 1080, "thinking": 4393, "result": 613},
            "duration": 31.85,
            "type": "TEXT",
            "timestamp": 1789336426.17,
            "model": "gemini-3.8-flash",
        },
        "text": text,
    }


def write_jsonl(path, records):
    with open(path, "w", encoding="utf-8") as fh:
        for record in records:
            fh.write(json.dumps(record) + "\n")
    return path


class TestResultRow(unittest.TestCase):
    def test_a_row_has_exactly_the_result_columns_in_order(self):
        row = chai_tea_rows.result_row(m804_record(), uuid.uuid4(), fields=M804_FIELDS)
        self.assertEqual(list(row), RESULT_COLUMNS)

    def test_renamed_fields_land_in_their_columns(self):
        record = m804_record()
        run = uuid.uuid4()
        row = chai_tea_rows.result_row(record, run, fields=M804_FIELDS)
        self.assertEqual(row["id"], record["id"])
        self.assertEqual(row["workflow_run_id"], str(run))
        self.assertEqual(row["process_id"], "transcribe")
        self.assertEqual(row["value"], "Ci commencent les croniques")
        self.assertEqual(row["input"], "inputs/m804/split/f0001r.jpg")
        self.assertEqual(row["metadata"], record["metadata"])
        self.assertEqual(row["md_duration"], 31.85)
        self.assertTrue(row["was_successful"])
        self.assertIsNone(row["editor_user_id"])

    def test_without_the_mapping_the_value_is_lost(self):
        # Why --field exists: normalize does not know the m804 names
        row = chai_tea_rows.result_row(m804_record(), uuid.uuid4())
        self.assertIsNone(row["value"])
        self.assertIsNone(row["input"])

    def test_timestamps_come_from_the_result_and_the_transform(self):
        now = datetime.datetime(2026, 10, 20, tzinfo=datetime.timezone.utc)
        row = chai_tea_rows.result_row(m804_record(), uuid.uuid4(), fields=M804_FIELDS, now=now)
        self.assertEqual(row["md_timestamp"].timestamp(), 1789336426.17)
        self.assertEqual(row["created"], now)
        self.assertEqual(row["modified"], now)
        record = m804_record()
        del record["metadata"]["timestamp"]
        self.assertEqual(chai_tea_rows.result_row(record, uuid.uuid4(), now=now)["md_timestamp"], now)

    def test_cost_is_priced_or_left_null(self):
        record = m804_record()
        self.assertIsNone(chai_tea_rows.result_row(record, uuid.uuid4())["md_cost"])
        prices = {"gemini-3.8-flash": (1.0, 2.0)}
        cost = chai_tea_rows.result_row(record, uuid.uuid4(), prices=prices)["md_cost"]
        self.assertAlmostEqual(cost, (364 + 1080) / 1e6 + (613 + 4393) * 2 / 1e6)

    def test_an_error_result_is_not_successful(self):
        record = m804_record()
        record["metadata"] = {"type": "ERROR", "error": "boom"}
        self.assertFalse(chai_tea_rows.result_row(record, uuid.uuid4())["was_successful"])

    def test_a_record_with_no_id_is_not_a_row(self):
        record = m804_record()
        del record["id"]
        self.assertIsNone(chai_tea_rows.result_row(record, uuid.uuid4()))

    def test_field_specs(self):
        self.assertEqual(chai_tea_rows.parse_fields(["text=value", " page = input_uri"]), M804_FIELDS)
        with self.assertRaises(ValueError):
            chai_tea_rows.parse_fields(["text"])


class TestTransformFile(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.mkdtemp()
        self.records = [m804_record(), m804_record("translate", "Here begin the chronicles")]
        self.source = write_jsonl(os.path.join(self.dir, "run.jsonl"), self.records)
        self.rows = os.path.join(self.dir, "run.result.jsonl")
        self.parquet = os.path.join(self.dir, "run.result.parquet")

    def tearDown(self):
        shutil.rmtree(self.dir, ignore_errors=True)

    def test_every_line_is_a_result_row(self):
        with open(self.source, "a") as fh:
            fh.write(json.dumps({"step": "x"}) + "\n")  # no id
            fh.write("{partial line\n")  # a run killed mid-write
        info = chai_tea_rows.transform_jsonl(self.source, self.rows, fields=M804_FIELDS)
        self.assertEqual((info["rows"], info["skipped"]), (2, 1))
        with open(self.rows) as fh:
            rows = [json.loads(line) for line in fh]
        self.assertEqual([list(r) for r in rows], [RESULT_COLUMNS] * 2)
        self.assertEqual({r["workflow_run_id"] for r in rows}, {info["workflow_run_id"]})

    def test_the_run_id_is_stable_for_the_same_file(self):
        first = chai_tea_rows.transform_jsonl(self.source, self.rows)["workflow_run_id"]
        self.assertEqual(chai_tea_rows.transform_jsonl(self.source, self.rows)["workflow_run_id"], first)
        uuid.UUID(first)
        given = str(uuid.uuid4())
        self.assertEqual(chai_tea_rows.transform_jsonl(self.source, self.rows, given)["workflow_run_id"], given)

    def test_the_parquet_file_has_the_result_columns_and_types(self):
        info = chai_tea_rows.transform_jsonl(self.source, self.rows, fields=M804_FIELDS)
        chai_tea_rows.rows_to_parquet(self.rows, self.parquet, info["workflow_run_id"])
        schema = pq.read_schema(self.parquet)
        self.assertEqual(schema.names, RESULT_COLUMNS)
        self.assertEqual(parquet_columns(self.parquet), chai_tea_rows.RESULT_KINDS)
        self.assertEqual(schema.metadata[b"chai_tea_workflow_run_id"].decode(), info["workflow_run_id"])
        table = pq.read_table(self.parquet).to_pylist()
        # jsonb columns hold JSON text: a string value is a JSON string, so it loads as one
        self.assertEqual(json.loads(table[0]["value"]), "Ci commencent les croniques")
        self.assertEqual(json.loads(table[0]["metadata"])["model"], "gemini-3.8-flash")
        self.assertIsNone(table[0]["editor_user_id"])
        self.assertIsNotNone(table[0]["created"].tzinfo)

    def test_an_empty_run_still_makes_a_file_with_the_columns(self):
        empty = write_jsonl(os.path.join(self.dir, "empty.jsonl"), [])
        info = chai_tea_rows.transform_jsonl(empty, self.rows)
        packed = chai_tea_rows.rows_to_parquet(self.rows, self.parquet, info["workflow_run_id"])
        self.assertEqual(packed["rows"], 0)
        self.assertEqual(pq.read_schema(self.parquet).names, RESULT_COLUMNS)

    def test_the_command_line_writes_both_files(self):
        out = os.path.join(self.dir, "out.parquet")
        chai_tea_rows.main([self.source, "--field", "text=value", "--parquet", out])
        self.assertTrue(os.path.exists(os.path.join(self.dir, "run.result.jsonl")))
        self.assertEqual(pq.read_table(out).num_rows, 2)


class TestStorageOutput(unittest.TestCase):
    """A file the storages wrote with their default fields is already RESULT rows; it passes through."""

    def setUp(self):
        self.dir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.dir, ignore_errors=True)

    def test_a_default_storage_file_keeps_its_own_run_and_values(self):
        from chai.result import ItemResult
        from chai.workflow import Workflow

        source = os.path.join(self.dir, "run.jsonl")
        wf = Workflow({"id": "out_wf", "type": "workflow.Workflow"})
        storage = wf._make_step({"type": "storage.JsonLinesStorage", "settings": {"file": source}}, wf)
        made = ItemResult("Ci commencent", processor=storage, metadata={"duration": 2.0, "cost": 0.01})
        storage.process(made)
        rows = os.path.join(self.dir, "rows.jsonl")
        info = chai_tea_rows.transform_jsonl(source, rows)
        self.assertEqual(info["workflow_run_id"], wf.run_id)
        with open(rows) as fh:
            [row] = [json.loads(line) for line in fh]
        with open(source) as fh:
            [stored] = [json.loads(line) for line in fh]
        for column in ("id", "workflow_run_id", "process_id", "input", "value", "metadata", "md_cost",
                       "md_duration", "md_timestamp", "created", "was_successful"):
            self.assertEqual(row[column], stored[column], column)


@unittest.skipUnless(HAS_POSTGRES, "no PostgreSQL server to load the CHAI-TEA schema into")
class TestLoadIntoChaiTea(unittest.TestCase):
    """The file loads straight into a CHAI-TEA RESULT table once its WORKFLOW_RUN exists."""

    def setUp(self):
        self.dir = tempfile.mkdtemp()
        self.database = f"chai_tea_rows_{uuid.uuid4().hex[:8]}"
        chai_tea_schema.create_schema({"database": self.database})

    def tearDown(self):
        shutil.rmtree(self.dir, ignore_errors=True)
        conn = _pg_connect(postgres_params({"database": "postgres"}), autocommit=True)
        try:
            conn.cursor().execute(f'DROP DATABASE IF EXISTS "{self.database}"')
        finally:
            conn.close()

    def test_parquet_to_postgres_loads_the_rows(self):
        source = write_jsonl(os.path.join(self.dir, "run.jsonl"), [m804_record(), m804_record("translate")])
        rows = os.path.join(self.dir, "rows.jsonl")
        parquet = os.path.join(self.dir, "rows.parquet")
        info = chai_tea_rows.transform_jsonl(source, rows, fields=M804_FIELDS)
        chai_tea_rows.rows_to_parquet(rows, parquet, info["workflow_run_id"])

        names = chai_tea_schema.table_names()
        conn = _pg_connect(postgres_params({"database": self.database}))
        try:
            cursor = conn.cursor()
            cursor.execute(f"INSERT INTO {names['PROJECT']} (name) VALUES ('M804') RETURNING id")
            project = cursor.fetchone()[0]
            cursor.execute(
                f"INSERT INTO {names['WORKFLOW']} (project_id, name) VALUES (%s, 'm804') RETURNING id", [project]
            )
            workflow = cursor.fetchone()[0]
            cursor.execute(
                f"INSERT INTO {names['WORKFLOW_RUN']} (id, workflow_id) VALUES (%s, %s)",
                [info["workflow_run_id"], workflow],
            )
            conn.commit()
        finally:
            conn.close()

        loaded = parquet_to_postgres(parquet, names["RESULT"], {"database": self.database}, create=False)
        self.assertEqual(loaded["rows"], 2)
        conn = _pg_connect(postgres_params({"database": self.database}))
        try:
            cursor = conn.cursor()
            cursor.execute(f"SELECT process_id, value, metadata->>'model' FROM {names['RESULT']} ORDER BY process_id")
            self.assertEqual(
                cursor.fetchall(),
                [
                    ("transcribe", "Ci commencent les croniques", "gemini-3.8-flash"),
                    ("translate", "Ci commencent les croniques", "gemini-3.8-flash"),
                ],
            )
        finally:
            conn.close()


if __name__ == "__main__":
    unittest.main()
