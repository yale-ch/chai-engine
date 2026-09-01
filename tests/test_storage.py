import hashlib
import json
import os
import shutil
import sqlite3
import tempfile
import unittest
import uuid
from datetime import datetime, timezone

from chai.result import FileItemResult, ItemResult, ListResult
import pyarrow.parquet as pq

from chai.storage import (
    COMPUTED_FIELDS,
    build_record,
    content_md5,
    correction_target,
    input_reference,
    result_md5,
    ParquetRecordWriter,
    PostgresStorage,
    _pg_connect,
    _pg_identifier,
    ensure_postgres_database,
    ensure_postgres_schema,
    get_result,
    jsonl_to_parquet,
    list_processors,
    list_results,
    parquet_to_postgres,
    postgres_params,
    save_correction,
    save_postgres_correction,
    source_value,
)
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


class TestJsonLinesStorage(unittest.TestCase):
    """One JSON line per result, written when the result is produced rather than at the end."""

    def setUp(self):
        self.dir = tempfile.mkdtemp()
        self.out = os.path.join(self.dir, "nested", "out.jsonl")
        self.wf = Workflow({"id": "jsonl_wf", "type": "workflow.Workflow"})
        self.comp_a = self.wf._make_step({"type": "describer.FileInfoDescriber", "id": "src_a"}, self.wf)

    def tearDown(self):
        shutil.rmtree(self.dir, ignore_errors=True)

    def make_storage(self, **settings):
        settings.setdefault("file", self.out)
        return self.wf._make_step({"type": "storage.JsonLinesStorage", "settings": settings}, self.wf)

    def lines(self, path=None):
        with open(path or self.out) as fh:
            return [json.loads(line) for line in fh if line.strip()]

    def test_appends_a_line_per_result_as_it_is_processed(self):
        storage = self.make_storage()
        self.assertTrue(os.path.exists(self.out))  # the parent directory is created up front
        first = ItemResult("one", metadata={"type": "TEXT"})
        out = storage.process(first)
        self.assertIs(out, first)  # pass-through, like the other Storage components
        self.assertEqual([js["value"] for js in self.lines()], ["one"])
        storage.process(ItemResult("two"))
        # the second line is there without anything having been flushed at the end of a run
        self.assertEqual([js["value"] for js in self.lines()], ["one", "two"])
        self.assertEqual(self.lines()[0]["id"], first.id)
        self.assertEqual(self.lines()[0]["metadata"]["type"], "TEXT")

    def test_append_mode_keeps_previous_lines(self):
        self.make_storage().process(ItemResult("first run"))
        # a second component (or a second run) adds to the file rather than replacing it
        self.make_storage().process(ItemResult("second run"))
        self.assertEqual([js["value"] for js in self.lines()], ["first run", "second run"])

    def test_truncate_mode_empties_the_file_when_built(self):
        self.make_storage().process(ItemResult("stale"))
        fresh = self.make_storage(mode="truncate")
        self.assertEqual(self.lines(), [])
        fresh.process(ItemResult("current"))
        self.assertEqual([js["value"] for js in self.lines()], ["current"])

    def test_fields_list_selects_keys(self):
        self.make_storage(fields=["value", "processorId"]).process(
            ItemResult("hello", processor=self.comp_a)
        )
        self.assertEqual(self.lines(), [{"value": "hello", "processorId": "src_a"}])

    def test_fields_dict_renames_keys(self):
        self.make_storage(fields={"value": "parsed"}).process(ItemResult({"first_name": "Ada"}))
        self.assertEqual(self.lines(), [{"parsed": {"first_name": "Ada"}}])

    def test_sources_record_the_input_they_came_from(self):
        row = ItemResult({"name": "Ada Lovelace"}, processor=self.comp_a)
        parsed = ItemResult({"first_name": "Ada"}, input=row)
        self.make_storage(fields={"value": "parsed"}, sources={"input": "src_a"}).process(parsed)
        self.assertEqual(self.lines(), [{"parsed": {"first_name": "Ada"}, "input": {"name": "Ada Lovelace"}}])

    def test_unknown_source_records_null(self):
        self.make_storage(fields=["value"], sources={"input": "no_such_component"}).process(
            ItemResult("x")
        )
        self.assertEqual(self.lines(), [{"value": "x", "input": None}])

    def test_source_value_walks_the_chain(self):
        row = ItemResult({"name": "Ada"}, processor=self.comp_a)
        name = ItemResult("Ada", input=row)
        parsed = ItemResult({"first_name": "Ada"}, input=name)
        self.assertEqual(source_value(parsed, "src_a"), {"name": "Ada"})
        self.assertIsNone(source_value(parsed, "src_b"))
        # a file result contributes its path, never the file content
        fr = make_file_result()
        fr.processor = self.comp_a
        self.assertEqual(source_value(ItemResult("text", input=fr), "src_a"), "/nonexistent/crop.png")

    def test_bytes_and_file_results_stay_json(self):
        storage = self.make_storage()
        storage.process(ItemResult(b"binary blob"))
        storage.process(make_file_result())
        values = [js["value"] for js in self.lines()]
        self.assertEqual(values, [{"__bytes__": len(b"binary blob")}, "/nonexistent/crop.png"])

    def test_lines_survive_a_failing_run(self):
        """A run that dies part way through still leaves a valid file of what it had done."""
        tree = {
            "id": "jsonl_fail_wf",
            "type": "workflow.Workflow",
            "steps": [
                {
                    "id": "iter",
                    "type": "iterator.Iterator",
                    "steps": [
                        {
                            "id": "double",
                            "type": "extractor.DoubleExtractor",
                            "next_steps": [
                                {"type": "storage.JsonLinesStorage", "settings": {"file": self.out}}
                            ],
                        }
                    ],
                }
            ],
        }
        wf = Workflow(tree)
        # the None entry makes DoubleExtractor raise, and without continue_on_error the run aborts
        with self.assertRaises(TypeError):
            wf.get_component_by_id("iter").process(ListResult(["a", "b", None, "c"]))
        self.assertEqual([js["value"] for js in self.lines()], ["aa", "bb"])

    def make_csv(self, count):
        path = os.path.join(self.dir, "people.csv")
        with open(path, "w", newline="") as fh:
            fh.write("name,title\n")
            fh.write("".join(f"name{i},title{i}\n" for i in range(count)))
        return path

    def streaming_workflow(self, csv_path, iterator_settings=None):
        return {
            "id": "jsonl_stream_wf",
            "type": "workflow.Workflow",
            "steps": [
                {
                    "id": "csv",
                    "type": "provider.CsvFileProvider",
                    "input": csv_path,
                    "steps": [
                        {
                            "id": "rows",
                            "type": "iterator.Iterator",
                            "settings": iterator_settings or {"retain_results": False},
                            "steps": [
                                {
                                    "id": "xp",
                                    "type": "extractor.JsonXpathExtractor",
                                    "settings": {"xpath": "/name"},
                                    "next_steps": [
                                        {
                                            "id": "jsonl",
                                            "type": "storage.JsonLinesStorage",
                                            "settings": {
                                                "file": self.out,
                                                "mode": "truncate",
                                                "fields": {"value": "name"},
                                                # the iterator stamps itself on each entry it runs
                                                "sources": {"input": "rows"},
                                            },
                                        }
                                    ],
                                }
                            ],
                        }
                    ],
                }
            ],
        }

    def test_streams_every_row_while_retaining_none_of_them(self):
        wf = Workflow(self.streaming_workflow(self.make_csv(5)))
        res = wf.run()
        self.assertEqual(
            self.lines(),
            [{"name": f"name{i}", "input": {"name": f"name{i}", "title": f"title{i}"}} for i in range(5)],
        )
        # every row reached the file, and none of the results were kept to be written at the end
        merged = res.value[0].value[0]
        self.assertIs(merged.processor, wf.get_component_by_id("rows"))
        self.assertEqual(merged.value, [])
        self.assertEqual(merged.metadata["processed"], 5)

    def test_concurrent_workers_write_whole_lines(self):
        settings = {"retain_results": False, "workers": 4}
        wf = Workflow(self.streaming_workflow(self.make_csv(40), settings))
        wf.run()
        got = sorted(js["name"] for js in self.lines())  # every line parses; none are interleaved
        self.assertEqual(got, sorted(f"name{i}" for i in range(40)))

    def test_error_branch_can_share_the_file(self):
        """A value branch and an error branch writing one file give a line per input either way."""
        tree = {
            "id": "jsonl_err_wf",
            "type": "workflow.Workflow",
            "steps": [
                {
                    "id": "iter",
                    "type": "iterator.Iterator",
                    "settings": {"retain_results": False},
                    "steps": [
                        {
                            "id": "double",
                            "type": "extractor.DoubleExtractor",
                            "error_steps": [
                                {
                                    "id": "err_jsonl",
                                    "type": "storage.JsonLinesStorage",
                                    "settings": {"file": self.out, "fields": {"value": "error"}},
                                }
                            ],
                            "next_steps": [
                                {
                                    "id": "ok_jsonl",
                                    "type": "storage.JsonLinesStorage",
                                    "settings": {"file": self.out, "fields": {"value": "doubled"}},
                                }
                            ],
                        }
                    ],
                }
            ],
        }
        wf = Workflow(tree)
        res = wf.get_component_by_id("iter").process(ListResult(["a", None, "b"]))
        keys = [sorted(js) for js in self.lines()]
        self.assertEqual(keys, [["doubled"], ["error"], ["doubled"]])
        self.assertIn("NoneType", self.lines()[1]["error"])
        # the failure was handled by the error branch, so the iterator saw no error of its own
        self.assertEqual(res.metadata["processed"], 3)
        self.assertEqual(res.metadata["errors"], 0)


class TestRecordFields(unittest.TestCase):
    """The computed columns a record can carry beside the result: its input, and what it corrects."""

    def setUp(self):
        self.dir = tempfile.mkdtemp()
        self.wf = Workflow({"id": "fields_wf", "type": "workflow.Workflow"})
        self.provider = self.wf._make_step(
            {"type": "provider.StaticProvider", "id": "prov", "settings": {"values": []}}, self.wf
        )

    def tearDown(self):
        shutil.rmtree(self.dir, ignore_errors=True)

    def file_result(self, name="page.txt", content="some page text"):
        path = os.path.join(self.dir, name)
        with open(path, "w") as fh:
            fh.write(content)
        return path, FileItemResult(path)

    def test_input_uri_and_hash_come_from_the_file_a_result_was_made_from(self):
        path, source = self.file_result()
        transcript = ItemResult("some page text", input=source)
        uri, digest = input_reference(transcript)
        self.assertEqual(uri, path)
        self.assertEqual(digest, hashlib.md5(b"some page text").hexdigest())

    def test_the_uri_is_found_further_up_the_chain(self):
        path, source = self.file_result()
        segment = ItemResult("some page", input=source)
        parsed = ItemResult({"words": 2}, input=segment)
        uri, digest = input_reference(parsed)
        self.assertEqual(uri, path)  # the file two steps back
        self.assertEqual(digest, content_md5("some page"))  # the hash is of the immediate input

    def test_the_raw_input_a_provider_was_given_counts_as_the_uri(self):
        path, _ = self.file_result("people.csv", "name\nAda\n")
        rows = ListResult([{"name": "Ada"}], input=path, processor=self.provider)
        row = ItemResult({"name": "Ada"}, input=rows)
        parsed = ItemResult({"first_name": "Ada"}, input=row)
        uri, digest = input_reference(parsed)
        self.assertEqual(uri, path)
        self.assertEqual(digest, content_md5({"name": "Ada"}))
        # a value that is not a path or a URI is content, not a location
        self.assertIsNone(input_reference(ItemResult("x", input=ItemResult("just some text")))[0])
        self.assertEqual(input_reference(ItemResult("x", input="https://example.org/a.json"))[0],
                         "https://example.org/a.json")

    def test_input_source_names_the_component_whose_result_is_the_input(self):
        row = ItemResult({"name": "Ada"}, processor=self.provider)
        name = ItemResult("Ada", input=row)
        parsed = ItemResult({"first_name": "Ada"}, input=name)
        self.assertEqual(input_reference(parsed)[1], content_md5("Ada"))  # the immediate input
        self.assertEqual(input_reference(parsed, "prov")[1], content_md5({"name": "Ada"}))
        self.assertEqual(input_reference(parsed, "no_such_component"), (None, None))

    def test_no_input_means_nothing_to_record(self):
        self.assertEqual(input_reference(ItemResult("standalone")), (None, None))
        self.assertEqual(input_reference(ItemResult("x", input=ItemResult("y")), with_hash=False)[1], None)

    def test_content_md5_does_not_depend_on_key_order(self):
        self.assertEqual(content_md5({"a": 1, "b": 2}), content_md5({"b": 2, "a": 1}))
        self.assertEqual(content_md5(b"bytes"), hashlib.md5(b"bytes").hexdigest())
        self.assertEqual(content_md5("text"), hashlib.md5(b"text").hexdigest())
        self.assertIsNone(content_md5(None))

    def test_a_file_is_hashed_once_and_remembered(self):
        path, source = self.file_result()
        expected = hashlib.md5(b"some page text").hexdigest()
        self.assertEqual(result_md5(source), expected)
        os.remove(path)  # the digest is cached on the result, so it survives the file going away
        self.assertEqual(result_md5(source), expected)
        # and bytes a component put on the result are hashed instead of re-reading the file
        crop = FileItemResult("/nonexistent/crop.png")
        crop.file_bytes = b"cropped"
        self.assertEqual(result_md5(crop), hashlib.md5(b"cropped").hexdigest())

    def test_correction_target_reads_the_result_it_corrects(self):
        original = ItemResult("first go")
        self.assertIsNone(correction_target(original))
        self.assertEqual(correction_target(ItemResult("fixed", extra={"corrects": original})), original.id)
        self.assertEqual(correction_target(ItemResult("fixed", extra={"corrects": "an-id"})), "an-id")
        self.assertEqual(correction_target(ItemResult("fixed", metadata={"corrects": "an-id"})), "an-id")

    def test_build_record_computed_fields(self):
        path, source = self.file_result()
        original = ItemResult("frst go", input=source)
        fixed = ItemResult("first go", input=source, extra={"corrects": original})
        record = build_record(
            fixed,
            {
                "id": "id",
                "@record": "value_json",
                "@input_uri": "input_uri",
                "@input_hash": "input_hash",
                "@corrects": "corrects_id",
            },
        )
        self.assertEqual(list(record), ["id", "value_json", "input_uri", "input_hash", "corrects_id"])
        self.assertEqual(record["value_json"]["value"], "first go")
        self.assertEqual(record["input_uri"], path)
        self.assertEqual(record["input_hash"], hashlib.md5(b"some page text").hexdigest())
        self.assertEqual(record["corrects_id"], original.id)
        self.assertEqual(record["value_json"]["extraInfo"], {"corrects": original.id})  # a Result, by id

    def test_build_record_rejects_an_unknown_computed_field(self):
        with self.assertRaises(ValueError) as caught:
            build_record(ItemResult("x"), {"@nonsense": "col"})
        self.assertIn("@nonsense", str(caught.exception))
        self.assertIn("@input_uri", str(caught.exception))
        for key in COMPUTED_FIELDS:
            self.assertIn(key, COMPUTED_FIELDS)


class TestParquetStorage(unittest.TestCase):
    """One Parquet file per run, holding a row per result, for bulk loading into a remote database."""

    def setUp(self):
        self.dir = tempfile.mkdtemp()
        self.out = os.path.join(self.dir, "nested", "run.parquet")
        self.wf = Workflow({"id": "parquet_wf", "type": "workflow.Workflow"})
        self.comp_a = self.wf._make_step({"type": "describer.FileInfoDescriber", "id": "src_a"}, self.wf)

    def tearDown(self):
        shutil.rmtree(self.dir, ignore_errors=True)

    def make_storage(self, workflow=None, **settings):
        settings.setdefault("file", self.out)
        # the atexit hook would otherwise write into the deleted temp directory at interpreter exit
        settings.setdefault("flush_on_exit", False)
        wf = workflow or self.wf
        return wf._make_step({"type": "storage.ParquetStorage", "settings": settings}, wf)

    def table(self, path=None):
        return pq.read_table(path or self.out)

    def columns(self, path=None):
        return self.table(path).to_pydict()

    def test_writes_a_row_per_result_when_the_file_is_closed(self):
        storage = self.make_storage()
        first = ItemResult("one", metadata={"type": "TEXT"})
        out = storage.process(first)
        self.assertIs(out, first)  # pass-through, like the other Storage components
        storage.process(ItemResult("two"))
        # nothing on disk yet: a Parquet file only becomes valid once its footer is written
        self.assertFalse(os.path.exists(self.out))
        info = storage.close()
        self.assertEqual(info["path"], self.out)
        self.assertEqual(info["rows"], 2)
        cols = self.columns()
        self.assertEqual(cols["value"], ["one", "two"])
        self.assertEqual(cols["id"][0], first.id)
        self.assertEqual(json.loads(cols["metadata"][0])["type"], "TEXT")

    def test_infers_a_column_type_per_column(self):
        storage = self.make_storage(fields=["value"], run_columns=False)
        for value in ({"name": "Ada"}, {"name": "Grace"}):
            storage.process(ItemResult(value))
        storage.close()
        # dicts (and lists, and mixed columns) become JSON text so the far side can parse them
        self.assertEqual(str(self.table().schema.field("value").type), "string")
        self.assertEqual([json.loads(v)["name"] for v in self.columns()["value"]], ["Ada", "Grace"])

        os.remove(self.out)
        storage = self.make_storage(fields=["value"], run_columns=False)
        for value in (1, 2.5, None):
            storage.process(ItemResult(value))
        storage.close()
        # ints and floats in one column widen to float rather than falling back to text
        self.assertEqual(str(self.table().schema.field("value").type), "double")
        self.assertEqual(self.columns()["value"], [1.0, 2.5, None])

    def test_mixed_types_in_one_column_become_json(self):
        storage = self.make_storage(fields=["value"], run_columns=False)
        storage.process(ItemResult("plain"))
        storage.process(ItemResult({"name": "Ada"}))
        storage.close()
        # every cell of a json column parses the same way, strings included
        self.assertEqual([json.loads(v) for v in self.columns()["value"]], ["plain", {"name": "Ada"}])

    def test_fields_sources_and_constants_shape_the_row(self):
        row = ItemResult({"name": "Ada Lovelace"}, processor=self.comp_a)
        parsed = ItemResult({"first_name": "Ada"}, input=row)
        storage = self.make_storage(
            fields={"value": "parsed"},
            sources={"source_row": "src_a"},
            constants={"batch": "b1"},
            run_columns=False,
        )
        storage.process(parsed)
        storage.close()
        cols = self.columns()
        self.assertEqual(list(cols), ["parsed", "source_row", "batch"])
        self.assertEqual(json.loads(cols["parsed"][0]), {"first_name": "Ada"})
        self.assertEqual(json.loads(cols["source_row"][0]), {"name": "Ada Lovelace"})
        self.assertEqual(cols["batch"], ["b1"])

    def test_run_columns_tag_every_row_with_the_run(self):
        storage = self.make_storage(fields=["value"])
        storage.process(ItemResult("x"))
        run_id = storage.run_id
        storage.close()
        cols = self.columns()
        self.assertEqual(cols["run_id"], [run_id])
        self.assertEqual(str(self.table().schema.field("stored_at").type), "timestamp[us, tz=UTC]")
        metadata = {k.decode(): v.decode() for k, v in self.table().schema.metadata.items()}
        self.assertEqual(metadata["chai_run_id"], run_id)
        self.assertEqual(metadata["chai_workflow_id"], "parquet_wf")

    def test_declared_schema_fixes_the_columns_and_their_types(self):
        storage = self.make_storage(
            fields={"value": "score"},
            run_columns=False,
            schema={"score": "float", "reviewed": "bool", "note": "string"},
        )
        storage.process(ItemResult("1.5"))  # a string in a float column is converted, not dropped
        storage.process(ItemResult(2))
        storage.close()
        schema = self.table().schema
        # declared order is kept, and a column no result produced is still there, all nulls
        self.assertEqual(schema.names, ["score", "reviewed", "note"])
        self.assertEqual(str(schema.field("reviewed").type), "bool")
        self.assertEqual(self.columns()["score"], [1.5, 2.0])
        self.assertEqual(self.columns()["note"], [None, None])

    def test_batch_size_writes_row_groups_during_the_run(self):
        storage = self.make_storage(fields=["value"], run_columns=False, batch_size=2)
        for value in ["a", "b", "c", "d", "e"]:
            storage.process(ItemResult(value))
        storage.close()
        self.assertEqual(pq.ParquetFile(self.out).num_row_groups, 3)  # 2 + 2 + 1
        self.assertEqual(self.columns()["value"], ["a", "b", "c", "d", "e"])

    def test_a_column_that_appears_after_the_schema_is_fixed_is_dropped(self):
        writer = ParquetRecordWriter(self.out, batch_size=1)
        writer.add({"name": "Ada"})  # the first batch fixes the columns
        with self.assertLogs("chai", level="WARNING") as logs:
            writer.add({"name": "Grace", "late": 1})
            info = writer.close()
        self.assertIn("'late'", logs.output[0])
        self.assertEqual(info["rows"], 2)
        self.assertEqual(self.columns(), {"name": ["Ada", "Grace"]})

    def workflow_tree(self, csv_path, out, storage_settings=None):
        settings = {"file": out, "flush_on_exit": False, "fields": {"value": "name"}}
        settings.update(storage_settings or {})
        return {
            "id": "parquet_run_wf",
            "type": "workflow.Workflow",
            "steps": [
                {
                    "id": "csv",
                    "type": "provider.CsvFileProvider",
                    "input": csv_path,
                    "steps": [
                        {
                            "id": "rows",
                            "type": "iterator.Iterator",
                            "settings": {"retain_results": False, "workers": 3},
                            "steps": [
                                {
                                    "id": "xp",
                                    "type": "extractor.JsonXpathExtractor",
                                    "settings": {"xpath": "/name"},
                                    "next_steps": [
                                        {"id": "pq", "type": "storage.ParquetStorage", "settings": settings}
                                    ],
                                }
                            ],
                        }
                    ],
                }
            ],
        }

    def make_csv(self, count):
        path = os.path.join(self.dir, "people.csv")
        with open(path, "w", newline="") as fh:
            fh.write("name,title\n")
            fh.write("".join(f"name{i},title{i}\n" for i in range(count)))
        return path

    def test_the_whole_run_lands_in_one_file_when_the_workflow_finishes(self):
        wf = Workflow(self.workflow_tree(self.make_csv(5), self.out, {"sources": {"row": "rows"}}))
        wf.run()  # nothing is closed by hand: the workflow's own component_end writes the file
        cols = self.columns()
        self.assertEqual(sorted(cols["name"]), [f"name{i}" for i in range(5)])
        self.assertEqual(sorted(json.loads(v)["title"] for v in cols["row"]), [f"title{i}" for i in range(5)])
        self.assertEqual(len(set(cols["run_id"])), 1)

    def test_a_run_that_fails_still_leaves_the_rows_it_produced(self):
        tree = self.workflow_tree(self.make_csv(3), self.out)
        tree["steps"][0]["steps"][0]["settings"] = {"workers": 1}  # one at a time, so 'third' is third
        wf = Workflow(tree)
        extractor = wf.get_component_by_id("xp")
        original, state = extractor._process, {"seen": 0}

        def exploding(input):
            state["seen"] += 1
            if state["seen"] == 3:
                raise ValueError("boom")
            return original(input)

        # the third row aborts the run; the workflow's component_error still writes the first two
        extractor._process = exploding
        with self.assertRaises(ValueError):
            wf.run()
        self.assertEqual(self.columns()["name"], ["name0", "name1"])

    def test_placeholders_give_each_run_its_own_file(self):
        out = os.path.join(self.dir, "runs", "results-{run_id}.parquet")
        wf = Workflow(self.workflow_tree(self.make_csv(2), out))
        wf.run()
        wf.run()  # the same workflow object again: a new run id, so a second file rather than an overwrite
        files = sorted(os.listdir(os.path.join(self.dir, "runs")))
        self.assertEqual(len(files), 2)
        for name in files:
            cols = self.columns(os.path.join(self.dir, "runs", name))
            self.assertEqual(sorted(cols["name"]), ["name0", "name1"])  # the iterator runs workers
            self.assertIn(cols["run_id"][0], name)

    def test_an_empty_run_writes_a_file_only_when_the_schema_says_what_it_holds(self):
        tree = self.workflow_tree(self.make_csv(0), self.out)
        Workflow(tree).run()
        self.assertFalse(os.path.exists(self.out))  # nothing to infer a schema from
        tree["steps"][0]["steps"][0]["steps"][0]["next_steps"][0]["settings"]["schema"] = {"name": "string"}
        Workflow(tree).run()
        # the declared schema is extended with the columns the component adds to every row, so an
        # empty run's file has the same shape as one with rows in it
        self.assertEqual(self.table().schema.names, ["name", "run_id", "stored_at"])
        self.assertEqual(self.table().num_rows, 0)

    def test_jsonl_to_parquet_converts_a_streamed_run(self):
        jsonl = os.path.join(self.dir, "out.jsonl")
        storage = self.wf._make_step(
            {"type": "storage.JsonLinesStorage", "settings": {"file": jsonl, "fields": ["value", "id"]}},
            self.wf,
        )
        first = ItemResult({"name": "Ada"})
        storage.process(first)
        storage.process(ItemResult({"name": "Grace"}))
        info = jsonl_to_parquet(jsonl, self.out, batch_size=1)
        self.assertEqual(info["rows"], 2)
        cols = self.columns()
        self.assertEqual([json.loads(v)["name"] for v in cols["value"]], ["Ada", "Grace"])
        self.assertEqual(cols["id"][0], first.id)


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
            self.assertEqual(row["correction_count"], 0)
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

    def test_a_correction_is_an_entry_pointing_at_what_it_corrects(self):
        r1, _, _ = self.store_samples()
        correction_id = save_correction(
            self.db, r1.id, {"text": "hello world, corrected"}, agent="a.reviewer@example.org",
            metadata={"reason": "typo", "agent_type": "human"},
        )
        self.assertIsNotNone(correction_id)
        row = get_result(self.db, r1.id)
        self.assertEqual(row["value"]["value"], "hello world")  # the original is untouched
        self.assertEqual(row["correction_count"], 1)
        self.assertTrue(row["corrected"])

        # the correction is a row of its own, with its own metadata about who made it
        correction = row["corrections"][0]
        self.assertEqual(correction["id"], correction_id)
        self.assertEqual(correction["corrects_id"], r1.id)
        self.assertEqual(correction["value"]["value"], {"text": "hello world, corrected"})
        self.assertEqual(correction["value"]["type"], "Correction")
        self.assertEqual(correction["value"]["input"], r1.id)  # its provenance is what it corrects
        self.assertEqual(correction["metadata"]["agent"], "a.reviewer@example.org")
        self.assertEqual(correction["metadata"]["reason"], "typo")
        self.assertEqual(correction["correction_count"], 0)
        # and it is reachable as a row in its own right, and by what it corrects
        self.assertEqual(get_result(self.db, correction_id)["corrects_id"], r1.id)
        self.assertEqual([c["id"] for c in list_results(self.db, corrects_id=r1.id)], [correction_id])
        self.assertIsNone(save_correction(self.db, "no-such-id", "x"))

    def test_a_result_can_be_corrected_more_than_once(self):
        r1, _, _ = self.store_samples()
        first = save_correction(self.db, r1.id, "fixed once", agent="reviewer")
        second = save_correction(self.db, r1.id, "fixed again", agent="gpt-brand-x", metadata={"agent_type": "model"})
        row = get_result(self.db, r1.id)
        self.assertEqual(row["correction_count"], 2)
        corrections = {c["id"]: c for c in row["corrections"]}
        self.assertEqual(set(corrections), {first, second})
        self.assertEqual(corrections[first]["metadata"]["agent"], "reviewer")
        self.assertEqual(corrections[second]["metadata"], {"agent": "gpt-brand-x", "agent_type": "model"})
        self.assertEqual(corrections[second]["value"]["value"], "fixed again")

    def test_restoring_a_result_keeps_the_corrections_pointing_at_it(self):
        r1 = self.store(ItemResult("hello world"), self.comp_a)
        correction_id = save_correction(self.db, r1.id, "fixed")
        self.store(r1)  # same result stored again (e.g. a re-run)
        row = get_result(self.db, r1.id)
        self.assertEqual([c["id"] for c in row["corrections"]], [correction_id])
        self.assertEqual(row["value"]["value"], "hello world")

    def test_a_correction_made_by_a_component_records_what_it_corrects(self):
        r1, _, _ = self.store_samples()
        # a component says what its result corrects by putting it in the result's extra
        corrected = ItemResult("hello world, corrected", extra={"corrects": r1})
        self.store(corrected, self.comp_a)
        row = get_result(self.db, r1.id)
        self.assertEqual([c["id"] for c in row["corrections"]], [corrected.id])
        self.assertEqual(get_result(self.db, corrected.id)["corrects_id"], r1.id)

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

    def test_rows_record_the_input_they_were_generated_from(self):
        path = os.path.join(self.dir, "page.txt")
        with open(path, "w") as fh:
            fh.write("some page text")
        source = FileItemResult(path)
        transcript = ItemResult("some page text", input=source)
        self.store(transcript, self.comp_a)
        row = get_result(self.db, transcript.id)
        self.assertEqual(row["input_uri"], path)
        self.assertEqual(row["input_hash"], hashlib.md5(b"some page text").hexdigest())
        # every entry made from the same input can be found by its hash
        self.assertEqual([r["id"] for r in list_results(self.db, input_hash=row["input_hash"])], [transcript.id])

    def test_a_result_with_no_input_has_nothing_to_record(self):
        res = self.store(ItemResult("standalone"), self.comp_a)
        row = get_result(self.db, res.id)
        self.assertIsNone(row["input_uri"])
        self.assertIsNone(row["input_hash"])

    def test_hash_input_false_skips_the_digest(self):
        path = os.path.join(self.dir, "page.txt")
        with open(path, "w") as fh:
            fh.write("some page text")
        storage = self.wf._make_step(
            {"type": "storage.SqliteStorage", "settings": {"database": self.db, "hash_input": False}}, self.wf
        )
        res = ItemResult("some page text", input=FileItemResult(path))
        storage.process(res)
        row = get_result(self.db, res.id)
        self.assertEqual(row["input_uri"], path)  # the path is free; the digest is the part that costs
        self.assertIsNone(row["input_hash"])


def postgres_available():
    """True when there is a server to test against; the whole PostgreSQL class is skipped if not."""
    try:
        ensure_postgres_database(postgres_params({}))
        _pg_connect(postgres_params({})).close()
        return True
    except Exception:
        return False


HAS_POSTGRES = postgres_available()


@unittest.skipUnless(HAS_POSTGRES, "no PostgreSQL server on localhost:5432 with a 'chai' database")
class TestPostgresStorage(unittest.TestCase):
    """The same rows a run writes straight into PostgreSQL, and the same rows loaded from Parquet."""

    def setUp(self):
        self.dir = tempfile.mkdtemp()
        self.out = os.path.join(self.dir, "run.parquet")
        self.params = postgres_params({})
        # a table name of this run's own, so a suite running twice at once does not fight over rows
        self.table = f"test_pg_{uuid.uuid4().hex[:8]}"
        self.loaded_table = f"{self.table}_loaded"
        self.wf = Workflow({"id": "pg_wf", "type": "workflow.Workflow"})
        self.comp_a = self.wf._make_step({"type": "describer.FileInfoDescriber", "id": "src_a"}, self.wf)

    def tearDown(self):
        shutil.rmtree(self.dir, ignore_errors=True)
        for component in list(self.wf.registry_ids.values()):
            if isinstance(component, PostgresStorage):
                component.close()  # nothing ran the workflow, so nothing has given the connection back
        conn = _pg_connect(self.params, autocommit=True)
        try:
            for table in (self.table, self.loaded_table):
                conn.cursor().execute(f"DROP TABLE IF EXISTS {table}_derivatives")
                conn.cursor().execute(f"DROP TABLE IF EXISTS {table}")
        finally:
            conn.close()

    def make_storage(self, **settings):
        settings.setdefault("table", self.table)
        return self.wf._make_step({"type": "storage.PostgresStorage", "settings": settings}, self.wf)

    def query(self, sql, params=None):
        conn = _pg_connect(self.params)
        try:
            cursor = conn.cursor()
            cursor.execute(sql, params)
            return cursor.fetchall()
        finally:
            conn.close()

    def execute(self, sql, params=None):
        conn = _pg_connect(self.params, autocommit=True)
        try:
            conn.cursor().execute(sql, params)
        finally:
            conn.close()

    def test_stores_a_result_row_with_json_columns(self):
        storage = self.make_storage()
        res = ItemResult("hello world", metadata={"type": "TEXT"}, processor=self.comp_a)
        self.assertIs(storage.process(res), res)  # pass-through, like the other Storage components
        rows = self.query(
            f"SELECT id, processor_id, workflow_id, value_json, metadata_json, extra_json FROM {self.table}"
        )
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0][0], res.id)
        self.assertEqual(rows[0][1], "src_a")
        self.assertEqual(rows[0][2], "pg_wf")
        # value_json is the whole result, parsed back out of jsonb rather than returned as text
        self.assertEqual(rows[0][3]["value"], "hello world")
        self.assertEqual(rows[0][4]["type"], "TEXT")  # metadata_json is its own queryable column
        self.assertIsNone(rows[0][5])  # nothing in extra: NULL rather than an empty object
        types = dict(
            self.query(
                "SELECT column_name, data_type FROM information_schema.columns WHERE table_name = %s",
                (self.table,),
            )
        )
        self.assertEqual(types["value_json"], "jsonb")
        self.assertEqual(types["created_at"], "timestamp with time zone")

    def test_storing_a_result_again_is_an_upsert(self):
        storage = self.make_storage()
        res = ItemResult("first", processor=self.comp_a)
        storage.process(res)
        created = self.query(f"SELECT created_at FROM {self.table}")[0][0]
        correction_id = save_postgres_correction(
            res.id, "fixed", agent="a.reviewer@example.org", params=self.params, table=self.table
        )
        res.value = "second"
        storage.process(res)
        rows = self.query(f"SELECT value_json, created_at FROM {self.table} WHERE id = %s", (res.id,))
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0][0]["value"], "second")
        self.assertEqual(rows[0][1], created)
        # the correction is its own row and still points at the result that was re-stored
        corrections = self.query(
            f"SELECT id, corrects_id, metadata_json FROM {self.table} WHERE corrects_id = %s", (res.id,)
        )
        self.assertEqual(corrections[0][0], correction_id)
        self.assertEqual(corrections[0][2]["agent"], "a.reviewer@example.org")

    def test_derivatives_are_stored_against_their_source(self):
        storage = self.make_storage()
        src = ItemResult("source")
        deriv = ItemResult(["label"], processor=self.comp_a)
        src.derivative_results = {self.comp_a: [deriv]}
        storage.process(src)
        rows = self.query(
            f"SELECT source_id, component_id, result_json FROM {self.table}_derivatives WHERE id = %s",
            (deriv.id,),
        )
        self.assertEqual(rows[0][0], src.id)
        self.assertEqual(rows[0][1], "src_a")
        self.assertEqual(rows[0][2]["value"], ["label"])

    def test_a_run_reaches_the_table_and_the_parquet_file_alike(self):
        csv_path = os.path.join(self.dir, "people.csv")
        with open(csv_path, "w", newline="") as fh:
            fh.write("name,title\n")
            fh.write("".join(f"name{i},title{i}\n" for i in range(6)))
        parquet_fields = {
            "id": "id",
            "processorId": "processor_id",
            "workflowId": "workflow_id",
            "@record": "value_json",
            "metadata": "metadata_json",
            "extraInfo": "extra_json",
            "@input_uri": "input_uri",
            "@input_hash": "input_hash",
            "@corrects": "corrects_id",
        }
        parquet_schema = {
            "id": "string",
            "processor_id": "string",
            "workflow_id": "string",
            "value_json": "json",
            "metadata_json": "json",
            "extra_json": "json",
            "input_uri": "string",
            "input_hash": "string",
            "corrects_id": "string",
        }
        tree = {
            "id": "pg_run_wf",
            "type": "workflow.Workflow",
            "steps": [
                {
                    "id": "csv",
                    "type": "provider.CsvFileProvider",
                    "input": csv_path,
                    "steps": [
                        {
                            "id": "rows",
                            "type": "iterator.Iterator",
                            "settings": {"retain_results": False, "workers": 3},
                            "steps": [
                                {
                                    "id": "xp",
                                    "type": "extractor.JsonXpathExtractor",
                                    "settings": {"xpath": "/name"},
                                    "next_steps": [
                                        {
                                            "type": "storage.PostgresStorage",
                                            "settings": {"table": self.table},
                                        },
                                        {
                                            "id": "pq",
                                            "type": "storage.ParquetStorage",
                                            "settings": {
                                                "file": self.out,
                                                "fields": parquet_fields,
                                                "schema": parquet_schema,
                                                "null_if_empty": True,
                                                "run_columns": False,
                                                "flush_on_exit": False,
                                            },
                                        },
                                    ],
                                }
                            ],
                        }
                    ],
                }
            ],
        }
        Workflow(tree).run()
        # the loaded table is made with the same DDL, so the two can be compared column for column
        ensure_postgres_schema(self.params, table=self.loaded_table)
        info = parquet_to_postgres(self.out, self.loaded_table, self.params)
        self.assertEqual(info["rows"], 6)
        # the run's rows say what they were made from: the CSV, and the md5 of the row itself
        uris = self.query(f"SELECT DISTINCT input_uri FROM {self.table}")
        self.assertEqual(uris, [(csv_path,)])
        self.assertEqual(self.query(f"SELECT count(DISTINCT input_hash) FROM {self.table}")[0][0], 6)
        columns = (
            "id, processor_id, workflow_id, value_json, metadata_json, extra_json, "
            "input_uri, input_hash, corrects_id"
        )
        self.assertEqual(self.query(f"SELECT count(*) FROM {self.table}")[0][0], 6)
        self.assertEqual(self.query(f"SELECT count(*) FROM {self.loaded_table}")[0][0], 6)
        # what one route stored and the other loaded is the same set of rows, both ways round
        self.assertEqual(
            self.query(f"SELECT {columns} FROM {self.table} EXCEPT SELECT {columns} FROM {self.loaded_table}"),
            [],
        )
        self.assertEqual(
            self.query(f"SELECT {columns} FROM {self.loaded_table} EXCEPT SELECT {columns} FROM {self.table}"),
            [],
        )

    def test_uploader_creates_a_table_from_the_parquet_columns(self):
        writer = ParquetRecordWriter(
            self.out, schema={"name": "string", "payload": "json", "n": "int", "ratio": "float",
                              "ok": "bool", "seen": "timestamp"}
        )
        moment = datetime(2026, 8, 31, 12, 30, tzinfo=timezone.utc)
        writer.add({"name": "Ada", "payload": {"a": 1}, "n": 3, "ratio": 1.5, "ok": True, "seen": moment})
        writer.add({"name": None, "payload": None, "n": None, "ratio": None, "ok": False, "seen": None})
        writer.close()
        info = parquet_to_postgres(self.out, self.loaded_table, self.params)
        self.assertEqual(info["rows"], 2)
        types = dict(
            self.query(
                "SELECT column_name, data_type FROM information_schema.columns WHERE table_name = %s",
                (self.loaded_table,),
            )
        )
        self.assertEqual(
            types,
            {
                "name": "text",
                "payload": "jsonb",
                "n": "bigint",
                "ratio": "double precision",
                "ok": "boolean",
                "seen": "timestamp with time zone",
            },
        )
        rows = self.query(f"SELECT name, payload, n, ratio, ok, seen FROM {self.loaded_table} ORDER BY n NULLS LAST")
        self.assertEqual(rows[0], ("Ada", {"a": 1}, 3, 1.5, True, moment))
        self.assertEqual(rows[1], (None, None, None, None, False, None))

    def test_uploader_reloads_and_refuses_a_column_the_table_lacks(self):
        writer = ParquetRecordWriter(self.out, schema={"name": "string"})
        writer.add({"name": "Ada"})
        writer.close()
        parquet_to_postgres(self.out, self.loaded_table, self.params)
        parquet_to_postgres(self.out, self.loaded_table, self.params)  # loads again, adding rows
        self.assertEqual(self.query(f"SELECT count(*) FROM {self.loaded_table}")[0][0], 2)
        parquet_to_postgres(self.out, self.loaded_table, self.params, truncate=True)
        self.assertEqual(self.query(f"SELECT count(*) FROM {self.loaded_table}")[0][0], 1)

        wider = os.path.join(self.dir, "wider.parquet")
        writer = ParquetRecordWriter(wider, schema={"name": "string", "extra": "string"})
        writer.add({"name": "Grace", "extra": "unexpected"})
        writer.close()
        with self.assertRaises(ValueError) as caught:
            parquet_to_postgres(wider, self.loaded_table, self.params, create=False)
        self.assertIn("extra", str(caught.exception))


    def test_works_through_the_psycopg2_fallback(self):
        try:
            import psycopg2
        except ImportError:
            self.skipTest("psycopg2 is not installed")
        import chai.storage

        driver = chai.storage._pg_driver
        chai.storage._pg_driver = lambda: psycopg2  # the older driver, used when psycopg 3 is absent
        try:
            storage = self.make_storage()
            res = ItemResult("through psycopg2", metadata={"type": "TEXT"})
            storage.process(res)
            storage.close()
            writer = ParquetRecordWriter(self.out, schema={"name": "string", "payload": "json"})
            writer.add({"name": "Ada", "payload": {"a": 1}})
            writer.close()
            # the uploader has no COPY on psycopg2 and inserts the batch instead
            self.assertEqual(parquet_to_postgres(self.out, self.loaded_table, self.params)["rows"], 1)
        finally:
            chai.storage._pg_driver = driver
        self.assertEqual(self.query(f"SELECT value_json FROM {self.table}")[0][0]["value"], "through psycopg2")
        self.assertEqual(self.query(f"SELECT payload FROM {self.loaded_table}")[0][0], {"a": 1})


class TestPostgresIdentifiers(unittest.TestCase):
    """Table names come from configuration, so they are checked before they reach any SQL."""

    def test_rejects_anything_that_is_not_a_plain_identifier(self):
        self.assertEqual(_pg_identifier("results_2"), "results_2")
        for bad in ('results"; DROP TABLE x --', "results table", "2results", ""):
            with self.assertRaises(ValueError):
                _pg_identifier(bad)


if __name__ == "__main__":
    unittest.main()
