import hashlib
import json
import os
import shutil
import sqlite3
import sys
import tempfile
import unittest
import uuid
from datetime import datetime, timezone

# the CHAI-TEA schema modules sit at the repository root, outside the package
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from chai.result import FileItemResult, ItemResult, ListResult
import pyarrow.parquet as pq

from chai.storage import (
    COMPUTED_FIELDS,
    RESULT_COLUMNS,
    result_row,
    build_record,
    content_md5,
    correction_target,
    input_provenance,
    recorded_source,
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
    store_postgres_rows,
    store_rows,
    source_value,
    token_usage_summary,
)
from chai.workflow import Workflow
import chai.schema as chai_schema


class TestResultRows(unittest.TestCase):
    """What the record-writing storages produce by default: rows of the CHAI-TEA RESULT table."""

    def setUp(self):
        self.dir = tempfile.mkdtemp()
        self.out = os.path.join(self.dir, "run.jsonl")
        self.wf = Workflow({"id": "rows_wf", "type": "workflow.Workflow"})
        self.comp_a = self.wf._make_step({"type": "describer.FileInfoDescriber", "id": "src_a"}, self.wf)

    def tearDown(self):
        shutil.rmtree(self.dir, ignore_errors=True)

    def lines(self):
        with open(self.out) as fh:
            return [json.loads(line) for line in fh if line.strip()]

    def storage(self, **settings):
        settings.setdefault("file", self.out)
        return self.wf._make_step({"type": "storage.JsonLinesStorage", "settings": settings}, self.wf)

    def test_the_columns_are_the_chai_tea_result_table(self):
        import chai_tea_parquet  # top level, outside the package -- the schema's own definition

        self.assertEqual(RESULT_COLUMNS, chai_tea_parquet.column_kinds("RESULT"))

    def test_a_line_is_exactly_a_result_row(self):
        res = ItemResult(
            "hello",
            processor=self.comp_a,
            metadata={"type": "TEXT", "duration": 1.5, "cost": 0.002, "timestamp": 1789336426.0},
        )
        self.storage().process(res)
        [row] = self.lines()
        self.assertEqual(list(row), list(RESULT_COLUMNS))
        self.assertEqual(row["id"], res.id)
        self.assertEqual(row["workflow_run_id"], self.wf.run_id)
        self.assertEqual(row["process_id"], "src_a")
        self.assertEqual(row["value"], "hello")
        self.assertTrue(row["was_successful"])
        self.assertEqual((row["md_cost"], row["md_duration"]), (0.002, 1.5))
        self.assertEqual(datetime.fromisoformat(row["md_timestamp"]).timestamp(), 1789336426.0)
        self.assertIsNone(row["editor_user_id"])
        self.assertIsNone(row["extra_data"])

    def test_input_is_the_file_the_result_came_from_and_where_in_it(self):
        page = make_file_result("/pages/f0001r.jpg")
        # the page provider is the run's recorded source, so the locator runs from the page down
        page.processor = self.wf._make_step(
            {"type": "describer.FileInfoDescriber", "id": "pages", "source": True}, self.wf
        )
        region = ItemResult("text", input=page, extra={"locator": {"bbox": [1, 2, 3, 4]}})
        line = ItemResult("a line", input=region, extra={"locator": {"index": 7}})
        row = result_row(line, self.wf.run_id)
        self.assertEqual(row["input"], "/pages/f0001r.jpg")
        self.assertEqual(json.loads(row["input_segment"]), [{"bbox": [1, 2, 3, 4]}, {"index": 7}])
        self.assertEqual(row["input_sequence"], 7)
        # with no file up the chain, the input is the result it was made from
        made_from = ItemResult("y")
        self.assertEqual(result_row(ItemResult("x", input=made_from))["input"], made_from.id)

    def test_errors_and_corrections(self):
        original = ItemResult("teh")
        self.assertFalse(result_row(ItemResult(None, metadata={"type": "ERROR", "error": "boom"}))["was_successful"])
        fixed = result_row(ItemResult("the", extra={"corrects": original, "note": "typo"}))
        self.assertEqual(fixed["previous_id"], original.id)
        self.assertEqual(fixed["extra_data"], {"note": "typo"})  # 'corrects' became previous_id

    def test_sources_and_constants_are_kept_in_extra_data(self):
        row = ItemResult({"name": "Ada"}, processor=self.comp_a)
        self.storage(sources={"row": "src_a"}).process(ItemResult("Ada", input=row))
        [line] = self.lines()
        self.assertEqual(list(line), list(RESULT_COLUMNS))
        self.assertEqual(line["extra_data"], {"row": {"name": "Ada"}})

    def test_every_storage_in_a_run_records_the_same_run(self):
        tree = {
            "id": "two_files_wf",
            "type": "workflow.Workflow",
            "steps": [
                {
                    "type": "provider.StaticProvider",
                    "settings": {"values": ["a", "b"]},
                    "input": "x",
                    "steps": [
                        {
                            "type": "iterator.Iterator",
                            "steps": [
                                {
                                    "type": "extractor.DoubleExtractor",
                                    "next_steps": [
                                        {"type": "storage.JsonLinesStorage", "settings": {"file": self.out}},
                                        {
                                            "type": "storage.ParquetStorage",
                                            "settings": {"file": os.path.join(self.dir, "run.parquet"), "flush_on_exit": False},
                                        },
                                    ],
                                }
                            ],
                        }
                    ],
                }
            ],
        }
        wf = Workflow(tree)
        wf.run()
        first = wf.run_id
        parquet = pq.read_table(os.path.join(self.dir, "run.parquet")).to_pydict()
        self.assertEqual({r["workflow_run_id"] for r in self.lines()}, {first})
        self.assertEqual(set(parquet["workflow_run_id"]), {first})
        self.assertEqual(list(parquet), list(RESULT_COLUMNS))
        wf.run()
        self.assertNotEqual(wf.run_id, first)
        self.assertEqual({r["workflow_run_id"] for r in self.lines()[2:]}, {wf.run_id})

    def test_a_configured_run_id_that_is_not_a_uuid_is_hashed_into_one(self):
        self.storage(run_id="slice-3").process(ItemResult("x"))
        run = self.lines()[0]["workflow_run_id"]
        self.assertEqual(str(uuid.UUID(run)), run)
        self.storage(run_id="slice-3").process(ItemResult("y"))
        self.assertEqual(self.lines()[1]["workflow_run_id"], run)

    def test_jsonl_to_parquet_types_result_rows_by_the_schema(self):
        storage = self.storage()
        storage.process(ItemResult("plain text"))
        storage.process(ItemResult({"a": 1}))
        out = os.path.join(self.dir, "run.parquet")
        info = jsonl_to_parquet(self.out, out)
        self.assertEqual(info["columns"], RESULT_COLUMNS)
        values = [json.loads(v) for v in pq.read_table(out).column("value").to_pylist()]
        self.assertEqual(values, ["plain text", {"a": 1}])

    def test_an_empty_parquet_run_still_has_the_columns(self):
        out = os.path.join(self.dir, "empty.parquet")
        tree = {
            "id": "empty_wf",
            "type": "workflow.Workflow",
            "steps": [
                {
                    "type": "provider.StaticProvider",
                    "settings": {"values": []},
                    "input": "x",
                    "steps": [
                        {
                            "type": "iterator.Iterator",
                            "steps": [{"type": "storage.ParquetStorage", "settings": {"file": out, "flush_on_exit": False}}],
                        }
                    ],
                }
            ],
        }
        Workflow(tree).run()
        self.assertEqual(pq.read_schema(out).names, list(RESULT_COLUMNS))

    def test_token_usage_is_totalled_per_process(self):
        usage = {"total": 30, "prompt": 10, "images": 0, "thinking": 5, "result": 15}
        self.storage().process(ItemResult("x", processor=self.comp_a, metadata={"token_usage": usage, "duration": 2}))
        summary = token_usage_summary(self.out)
        self.assertEqual(summary["components"]["src_a"]["calls"], 1)
        self.assertEqual(summary["total"]["output"], 15)


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

    def test_truncate_mode_leaves_the_file_alone_when_built(self):
        # building a workflow -- to inspect it, say -- must never cost the last run's output
        self.make_storage().process(ItemResult("stale"))
        fresh = self.make_storage(mode="truncate")
        self.assertEqual([js["value"] for js in self.lines()], ["stale"])
        # outside a run, the file is emptied before the first write, and only then
        fresh.process(ItemResult("current"))
        fresh.process(ItemResult("next"))
        self.assertEqual([js["value"] for js in self.lines()], ["current", "next"])

    def truncating_workflow(self, values, steps=1):
        store = {"type": "storage.JsonLinesStorage", "settings": {"file": self.out, "mode": "truncate"}}
        return {
            "id": "truncate_wf",
            "type": "workflow.Workflow",
            "steps": [
                {
                    "type": "provider.StaticProvider",
                    "settings": {"values": values},
                    "input": "x",
                    "steps": [
                        {
                            "type": "iterator.Iterator",
                            "steps": [{"type": "extractor.DoubleExtractor", "next_steps": [dict(store) for _ in range(steps)]}],
                        }
                    ],
                }
            ],
        }

    def test_truncate_mode_empties_the_file_when_a_run_starts(self):
        self.make_storage().process(ItemResult("stale"))
        wf = Workflow(self.truncating_workflow(["a", "b"]))
        self.assertEqual([js["value"] for js in self.lines()], ["stale"])  # built: nothing lost yet
        wf.run()
        self.assertEqual([js["value"] for js in self.lines()], ["aa", "bb"])
        wf.run()  # each run replaces the last run's lines
        self.assertEqual([js["value"] for js in self.lines()], ["aa", "bb"])

    def test_a_run_that_writes_nothing_still_empties_the_file(self):
        self.make_storage().process(ItemResult("stale"))
        Workflow(self.truncating_workflow([])).run()
        self.assertEqual(self.lines(), [])

    def test_steps_sharing_a_file_empty_it_once_per_run(self):
        # two truncating steps writing one file (as workflow-m804.json has) keep each other's lines
        wf = Workflow(self.truncating_workflow(["a", "b"], steps=2))
        wf.run()
        self.assertEqual([js["value"] for js in self.lines()], ["aa", "aa", "bb", "bb"])
        wf.run()
        self.assertEqual(len(self.lines()), 4)

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
        provenance = input_provenance(transcript)
        self.assertEqual(provenance["uri"], path)
        self.assertEqual(provenance["hash"], hashlib.md5(b"some page text").hexdigest())
        self.assertIsNone(provenance["locator"])  # nothing on the way said where it sits

    def test_the_uri_is_found_further_up_the_chain(self):
        path, source = self.file_result()
        segment = ItemResult("some page", input=source)
        parsed = ItemResult({"words": 2}, input=segment)
        provenance = input_provenance(parsed)
        self.assertEqual(provenance["uri"], path)  # the file two steps back
        self.assertEqual(provenance["hash"], content_md5("some page"))  # the hash is of the input

    def test_the_raw_input_a_provider_was_given_counts_as_the_uri(self):
        path, _ = self.file_result("people.csv", "name\nAda\n")
        rows = ListResult([{"name": "Ada"}], input=path, processor=self.provider)
        row = ItemResult({"name": "Ada"}, input=rows)
        parsed = ItemResult({"first_name": "Ada"}, input=row)
        provenance = input_provenance(parsed)
        self.assertEqual(provenance["uri"], path)
        self.assertEqual(provenance["hash"], content_md5({"name": "Ada"}))
        # a value that is not a path or a URI is content, not a location
        self.assertIsNone(input_provenance(ItemResult("x", input=ItemResult("just some text")))["uri"])
        self.assertEqual(
            input_provenance(ItemResult("x", input="https://example.org/a.json"))["uri"],
            "https://example.org/a.json",
        )

    def test_input_source_names_the_component_whose_result_is_the_input(self):
        row = ItemResult({"name": "Ada"}, processor=self.provider)
        name = ItemResult("Ada", input=row)
        parsed = ItemResult({"first_name": "Ada"}, input=name)
        self.assertEqual(input_provenance(parsed)["hash"], content_md5("Ada"))  # the immediate input
        self.assertEqual(input_provenance(parsed, "prov")["hash"], content_md5({"name": "Ada"}))
        self.assertEqual(input_provenance(parsed, "no_such_component")["hash"], None)

    def test_no_input_means_nothing_to_record(self):
        self.assertEqual(input_provenance(ItemResult("standalone")), {"uri": None, "hash": None, "locator": None})
        self.assertIsNone(input_provenance(ItemResult("x", input=ItemResult("y")), with_hash=False)["hash"])

    def test_a_marked_component_is_the_recorded_source(self):
        path, page = self.file_result("page.txt", "One. Two! Three?")
        # a workflow marks the component whose results are the source of everything downstream
        pages = self.wf._make_step(
            {"type": "iterator.Iterator", "id": "pages", "source": True, "steps": []}, self.wf
        )
        page.processor = pages
        page.input = ListResult([path], input=os.path.dirname(path))
        transcript = ItemResult("One. Two! Three?", input=page)
        sentence = ItemResult("Two!", input=transcript)
        parsed = ItemResult({"words": 1}, input=sentence)
        # four steps later the row still records the page, not the sentence it came from
        self.assertIs(recorded_source(parsed), page)
        provenance = input_provenance(parsed)
        self.assertEqual(provenance["uri"], path)
        self.assertEqual(provenance["hash"], content_md5("One. Two! Three?"))
        # ... and without the mark it would record the sentence it happens to sit behind
        pages.is_source = False
        self.assertIsNone(recorded_source(parsed))
        self.assertEqual(input_provenance(parsed)["hash"], content_md5("Two!"))

    def test_locators_stack_from_the_source_down_to_the_result(self):
        path, page = self.file_result("scan.png", "not really an image")
        page.processor = self.wf._make_step(
            {"type": "iterator.Iterator", "id": "scans", "source": True, "steps": []}, self.wf
        )
        region = ItemResult("region bytes", input=page)
        region.locator = {"bbox": [10, 20, 300, 80]}  # where the region is on the page
        transcript = ItemResult("A sentence. And another.", input=region)
        sentence = ItemResult("And another.", input=transcript)
        sentence.locator = {"start": 12, "end": 24}  # where the sentence is in that region's text
        provenance = input_provenance(sentence)
        self.assertEqual(provenance["uri"], path)  # the page, which is all that was materialized
        self.assertEqual(
            provenance["locator"], [{"bbox": [10, 20, 300, 80]}, {"start": 12, "end": 24}]
        )
        # the region's own row carries only its own frame
        self.assertEqual(input_provenance(region)["locator"], [{"bbox": [10, 20, 300, 80]}])

    def test_a_segmenter_records_where_each_segment_was(self):
        text = "One. Two!  Three?"
        segmenter = self.wf._make_step(
            {"type": "segmenter.TextSegmenter", "id": "sentences", "settings": {"mode": "sentence", "locate": True}},
            self.wf,
        )
        segments = segmenter.process(ItemResult(text))
        self.assertEqual([s.value for s in segments.value], ["One.", "Two!", "Three?"])
        for segment in segments.value:
            start, end = segment.locator["start"], segment.locator["end"]
            # the offsets address the segment's own characters in the text it was cut from
            self.assertEqual(text[start:end], segment.value)
        # off by default, so the segments stay plain strings
        plain = self.wf._make_step(
            {"type": "segmenter.TextSegmenter", "id": "plain", "settings": {"mode": "sentence"}}, self.wf
        )
        self.assertEqual(plain.process(ItemResult(text)).value, ["One.", "Two!", "Three?"])

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
                "@input_locator": "input_locator",
                "@corrects": "corrects_id",
            },
        )
        self.assertEqual(
            list(record),
            ["id", "value_json", "input_uri", "input_hash", "input_locator", "corrects_id"],
        )
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
        # the default row is a CHAI-TEA RESULT row: value is a jsonb column, so it holds JSON text
        self.assertEqual(list(cols), list(RESULT_COLUMNS))
        self.assertEqual([json.loads(v) for v in cols["value"]], ["one", "two"])
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
    """A run stored in SQLite, in the CHAI-TEA tables: RESULT rows under their run, workflow and project."""

    def setUp(self):
        self.dir = tempfile.mkdtemp()
        self.db = os.path.join(self.dir, "results.db")
        self.wf = Workflow(
            {
                "id": "sqlite_storage_wf",
                "type": "workflow.Workflow",
                "name": "SQLite storage",
                "description": "results stored by the tests",
                "project": {"name": "Tests", "description": "the test suite's project"},
            }
        )
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

    def query(self, sql, params=()):
        conn = sqlite3.connect(self.db)
        try:
            return conn.execute(sql, params).fetchall()
        finally:
            conn.close()

    def test_the_database_has_the_chai_tea_tables(self):
        self.store_samples()
        tables = {row[0] for row in self.query("SELECT name FROM sqlite_master WHERE type = 'table'")}
        self.assertEqual(tables, set(chai_schema.TABLES.values()))
        for entity, table in chai_schema.TABLES.items():
            columns = [row[1] for row in self.query(f"PRAGMA table_info({table})")]
            self.assertEqual(columns, list(chai_schema.column_types(entity)), entity)

    def test_rows_exist_for_each_stored_result_under_its_run(self):
        r1, r2, r3 = self.store_samples()
        rows = {r["id"]: r for r in list_results(self.db)}
        self.assertEqual(set(rows), {r1.id, r2.id, r3.id})
        self.assertEqual(rows[r3.id]["value"], "/nonexistent/crop.png")  # a file's path, never its bytes
        self.assertEqual(rows[r3.id]["process_id"], "comp_b")
        self.assertEqual({r["workflow_run_id"] for r in rows.values()}, {self.wf.run_id})
        # the rows they hang under come from the objects: the run, the workflow, its project
        self.assertEqual(self.query("SELECT id, workflow_id FROM workflow_runs"), [(self.wf.run_id, self.wf.uuid)])
        self.assertEqual(
            self.query("SELECT id, project_id, name, description, is_active FROM workflows"),
            [(self.wf.uuid, self.wf.project.id, "SQLite storage", "results stored by the tests", 1)],
        )
        self.assertEqual(self.query("SELECT name, description FROM projects"), [("Tests", "the test suite's project")])

    def test_a_row_carries_the_result_and_its_own_timestamps(self):
        r2 = self.store(ItemResult({"name": "Ada", "year": 1815}, metadata={"duration": 1.25}), self.comp_a)
        row = get_result(self.db, r2.id)
        self.assertEqual(row["value"], {"name": "Ada", "year": 1815})
        self.assertEqual(row["md_duration"], 1.25)
        self.assertIn("timestamp", row["metadata"])
        self.assertTrue(row["was_successful"])
        self.assertEqual(datetime.fromisoformat(row["created"]), r2.created)
        self.assertEqual(datetime.fromisoformat(row["modified"]), r2.modified)

    def test_bytes_value_gets_placeholder(self):
        res = self.store(ItemResult(b"binary blob"), self.comp_a)
        self.assertEqual(get_result(self.db, res.id)["value"], {"__bytes__": len(b"binary blob")})

    def test_an_error_result_is_not_successful(self):
        res = self.store(ItemResult(None, metadata={"type": "ERROR", "error": "boom"}), self.comp_a)
        self.assertFalse(get_result(self.db, res.id)["was_successful"])

    def test_list_results_and_filters(self):
        r1, r2, r3 = self.store_samples()
        rows = list_results(self.db)
        self.assertEqual({r["id"] for r in rows}, {r1.id, r2.id, r3.id})
        for row in rows:
            self.assertEqual(row["correction_count"], 0)
            self.assertIsNotNone(row["created"])
        self.assertEqual({r["id"] for r in list_results(self.db, process_id="comp_a")}, {r1.id, r2.id})
        self.assertEqual(len(list_results(self.db, workflow_run_id=self.wf.run_id)), 3)
        self.assertEqual(list_results(self.db, workflow_run_id=str(uuid.uuid4())), [])
        self.assertEqual(len(list_results(self.db, limit=2)), 2)
        self.assertEqual(len(list_results(self.db, limit=2, offset=2)), 1)

    def test_get_result_round_trip(self):
        r1, _, _ = self.store_samples()
        row = get_result(self.db, r1.id)
        self.assertEqual(list(row)[: len(RESULT_COLUMNS)], list(RESULT_COLUMNS))
        self.assertEqual(row["process_id"], "comp_a")
        self.assertEqual(row["value"], "hello world")
        self.assertIsNone(get_result(self.db, "no-such-id"))

    def test_a_correction_is_an_entry_pointing_at_what_it_corrects(self):
        r1, _, _ = self.store_samples()
        correction_id = save_correction(
            self.db, r1.id, {"text": "hello world, corrected"}, agent="a.reviewer@example.org",
            metadata={"reason": "typo", "agent_type": "human"},
        )
        self.assertIsNotNone(correction_id)
        row = get_result(self.db, r1.id)
        self.assertEqual(row["value"], "hello world")  # the original is untouched
        self.assertEqual(row["correction_count"], 1)
        self.assertTrue(row["corrected"])

        # the correction is a row of its own, in the same run, made from the same input
        correction = row["corrections"][0]
        self.assertEqual(correction["id"], correction_id)
        self.assertEqual(correction["previous_id"], r1.id)
        self.assertEqual(correction["workflow_run_id"], self.wf.run_id)
        self.assertEqual(correction["value"], {"text": "hello world, corrected"})
        self.assertEqual(correction["metadata"]["agent"], "a.reviewer@example.org")
        self.assertEqual(correction["metadata"]["reason"], "typo")
        self.assertEqual(correction["correction_count"], 0)
        # and it is reachable as a row in its own right, and by what it corrects
        self.assertEqual(get_result(self.db, correction_id)["previous_id"], r1.id)
        self.assertEqual([c["id"] for c in list_results(self.db, previous_id=r1.id)], [correction_id])
        self.assertIsNone(save_correction(self.db, "no-such-id", "x"))

    def test_a_correction_names_its_editor_when_that_user_exists(self):
        r1, _, _ = self.store_samples()
        now = datetime.now(timezone.utc)
        user = {"id": str(uuid.uuid4()), "name": "A. Reviewer", "netid": "ar123", "email_address": None,
                "is_active": True, "auth_key": None, "created": now, "modified": now}
        store_rows(self.db, {"USER": user})
        known = save_correction(self.db, r1.id, "fixed", editor_user_id=user["id"])
        unknown = save_correction(self.db, r1.id, "fixed", editor_user_id=str(uuid.uuid4()))
        self.assertEqual(get_result(self.db, known)["editor_user_id"], user["id"])
        self.assertIsNone(get_result(self.db, unknown)["editor_user_id"])  # no USER row to point at

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
        self.assertEqual(corrections[second]["value"], "fixed again")

    def test_restoring_a_result_keeps_its_created_and_the_corrections_pointing_at_it(self):
        r1 = self.store(ItemResult("hello world"), self.comp_a)
        created = get_result(self.db, r1.id)["created"]
        correction_id = save_correction(self.db, r1.id, "fixed")
        r1.value = "hello again"  # changing the value marks the result modified
        self.store(r1)
        row = get_result(self.db, r1.id)
        self.assertEqual([c["id"] for c in row["corrections"]], [correction_id])
        self.assertEqual(row["value"], "hello again")
        self.assertEqual(row["created"], created)
        self.assertGreaterEqual(row["modified"], created)

    def test_a_correction_made_by_a_component_records_what_it_corrects(self):
        r1, _, _ = self.store_samples()
        # a component says what its result corrects by putting it in the result's extra
        corrected = ItemResult("hello world, corrected", extra={"corrects": r1})
        self.store(corrected, self.comp_a)
        row = get_result(self.db, r1.id)
        self.assertEqual([c["id"] for c in row["corrections"]], [corrected.id])
        self.assertEqual(get_result(self.db, corrected.id)["previous_id"], r1.id)

    def test_list_processors(self):
        self.store_samples()
        procs = {p["process_id"]: p["count"] for p in list_processors(self.db)}
        self.assertEqual(procs, {"comp_a": 2, "comp_b": 1})

    def test_a_derivative_is_a_result_row_of_its_own(self):
        src = ItemResult("source text")
        deriv = ItemResult(["label"], processor=self.comp_a, register_on=src)
        self.store(src, self.comp_b)
        row = get_result(self.db, deriv.id)
        self.assertEqual(row["value"], ["label"])
        self.assertEqual(row["process_id"], "comp_a")
        self.assertEqual(row["input"], src.id)  # made from the result it is a derivative of
        self.assertEqual(row["workflow_run_id"], self.wf.run_id)

    def test_rows_record_the_input_they_were_generated_from(self):
        path = os.path.join(self.dir, "page.txt")
        with open(path, "w") as fh:
            fh.write("some page text")
        source = FileItemResult(path)
        transcript = ItemResult("some page text", input=source)
        self.store(transcript, self.comp_a)
        row = get_result(self.db, transcript.id)
        self.assertEqual(row["input"], path)
        self.assertEqual(row["input_hash"], hashlib.md5(b"some page text").hexdigest())
        # every entry made from the same input can be found by its hash
        self.assertEqual([r["id"] for r in list_results(self.db, input_hash=row["input_hash"])], [transcript.id])

    def test_a_run_records_the_document_and_where_in_it_each_row_is(self):
        docs = os.path.join(self.dir, "docs")
        os.makedirs(docs)
        text = "Ada wrote it. Babbage built it."
        with open(os.path.join(docs, "note.txt"), "w") as fh:
            fh.write(text)
        tree = {
            "id": "locator_wf",
            "type": "workflow.Workflow",
            "steps": [
                {
                    "id": "dir",
                    "type": "provider.DirFileProvider",
                    "input": docs,
                    "steps": [
                        {
                            "id": "each_doc",
                            "type": "iterator.Iterator",
                            "source": True,  # every row downstream records the document
                            "steps": [
                                {
                                    "id": "read",
                                    "type": "transcriber.TextFileTranscriber",
                                    "next_steps": [
                                        {
                                            "id": "sentences",
                                            "type": "segmenter.TextSegmenter",
                                            "settings": {"mode": "sentence", "locate": True},
                                            "next_steps": [
                                                {
                                                    "id": "each_sentence",
                                                    "type": "iterator.Iterator",
                                                    "steps": [
                                                        {
                                                            "id": "words",
                                                            "type": "classifier.KeywordClassifier",
                                                            "settings": {"labels": {"who": ["ada", "babbage"]}},
                                                            "next_steps": [
                                                                {
                                                                    "type": "storage.SqliteStorage",
                                                                    "settings": {"database": self.db},
                                                                }
                                                            ],
                                                        }
                                                    ],
                                                }
                                            ],
                                        }
                                    ],
                                }
                            ],
                        }
                    ],
                }
            ],
        }
        wf = Workflow(tree)
        wf.run()
        rows = [r for r in list_results(self.db, process_id="words") if r["input_segment"]]
        self.assertEqual(len(rows), 2)  # one per sentence, and no sentence was written anywhere
        located = {}
        for row in rows:
            # every row names the document, not the sentence or the transcript it sat behind
            self.assertEqual(row["input"], os.path.join(docs, "note.txt"))
            self.assertEqual(row["input_hash"], hashlib.md5(text.encode()).hexdigest())
            frame = json.loads(row["input_segment"])[0]
            located[text[frame["start"] : frame["end"]]] = row["value"]
        # the offsets read back out of the document as the sentences the labels were assigned to
        self.assertEqual(located, {"Ada wrote it.": ["who"], "Babbage built it.": ["who"]})
        # the finished run was written back: it succeeded, and says how long it took
        [(successful, duration, metadata)] = self.query(
            "SELECT was_successful, duration, metadata FROM workflow_runs WHERE id = ?", (wf.run_id,)
        )
        self.assertEqual(successful, 1)
        self.assertGreater(duration, 0)
        self.assertIn("usage", json.loads(metadata))

    def test_each_run_is_a_workflow_run_row(self):
        tree = {
            "id": "two_runs_wf",
            "type": "workflow.Workflow",
            "steps": [
                {
                    "type": "provider.StaticProvider",
                    "settings": {"values": ["a"]},
                    "input": "x",
                    "steps": [
                        {
                            "type": "iterator.Iterator",
                            "steps": [
                                {
                                    "type": "extractor.DoubleExtractor",
                                    "next_steps": [{"type": "storage.SqliteStorage", "settings": {"database": self.db}}],
                                }
                            ],
                        }
                    ],
                }
            ],
        }
        wf = Workflow(tree)
        wf.run()
        first = wf.run_id
        wf.run()
        runs = dict(self.query("SELECT id, workflow_id FROM workflow_runs"))
        self.assertEqual(set(runs), {first, wf.run_id})
        self.assertEqual(set(runs.values()), {wf.uuid})
        self.assertEqual(len(self.query("SELECT id FROM workflows")), 1)  # the same workflow, run twice

    def test_a_result_with_no_input_has_nothing_to_record(self):
        res = self.store(ItemResult("standalone"), self.comp_a)
        row = get_result(self.db, res.id)
        self.assertIsNone(row["input"])
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
        self.assertEqual(row["input"], path)  # the path is free; the digest is the part that costs
        self.assertIsNone(row["input_hash"])

    def test_foreign_keys_are_enforced(self):
        stray = {column: None for column in RESULT_COLUMNS}
        now = datetime.now(timezone.utc)
        stray.update(id=str(uuid.uuid4()), workflow_run_id=str(uuid.uuid4()), created=now, modified=now)
        with self.assertRaises(sqlite3.IntegrityError):  # a RESULT row must belong to a run that exists
            store_rows(self.db, {"RESULT": stray})

    def test_a_database_in_the_old_layout_is_refused(self):
        old = os.path.join(self.dir, "old.db")
        conn = sqlite3.connect(old)
        conn.execute("CREATE TABLE results (id TEXT PRIMARY KEY, processor_id TEXT, value_json TEXT)")
        conn.commit()
        conn.close()
        with self.assertRaises(RuntimeError):
            list_results(old)


def postgres_available():
    """True when there is a server to test against; the PostgreSQL classes are skipped if not."""
    try:
        _pg_connect(postgres_params({"database": "postgres"})).close()
        return True
    except Exception:
        return False


HAS_POSTGRES = postgres_available()


class _ThrowawayDatabase(unittest.TestCase):
    """A PostgreSQL database of the class's own, created for it and dropped afterwards."""

    @classmethod
    def setUpClass(cls):
        cls.params = postgres_params({"database": f"chai_test_{uuid.uuid4().hex[:8]}"})
        ensure_postgres_database(cls.params)

    @classmethod
    def tearDownClass(cls):
        conn = _pg_connect(postgres_params({"database": "postgres"}), autocommit=True)
        try:
            conn.cursor().execute(f'DROP DATABASE IF EXISTS "{cls.params["dbname"]}" WITH (FORCE)')
        finally:
            conn.close()

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


@unittest.skipUnless(HAS_POSTGRES, "no PostgreSQL server to test against (PGHOST/PGPORT, default localhost:5432)")
class TestPostgresStorage(_ThrowawayDatabase):
    """The same rows a run writes straight into PostgreSQL, and the same rows loaded from Parquet."""

    def setUp(self):
        self.dir = tempfile.mkdtemp()
        self.out = os.path.join(self.dir, "run.parquet")
        # tables of this test's own, so the tests do not see each other's rows
        suffix = uuid.uuid4().hex[:6]
        self.tables = {entity: f"{name}_{suffix}" for entity, name in chai_schema.TABLES.items()}
        self.names = self.tables
        self.loaded_table = f"loaded_{suffix}"
        self.wf = Workflow(
            {"id": "pg_wf", "type": "workflow.Workflow", "name": "PostgreSQL storage", "project": "Tests"}
        )
        self.comp_a = self.wf._make_step({"type": "describer.FileInfoDescriber", "id": "src_a"}, self.wf)

    def tearDown(self):
        shutil.rmtree(self.dir, ignore_errors=True)
        for component in list(self.wf.registry_ids.values()):
            if isinstance(component, PostgresStorage):
                component.close()  # nothing ran the workflow, so nothing has given the connection back

    def make_storage(self, workflow=None, **settings):
        settings.setdefault("tables", self.tables)
        settings.setdefault("database", self.params["dbname"])
        wf = workflow or self.wf
        return wf._make_step({"type": "storage.PostgresStorage", "settings": settings}, wf)

    def test_stores_a_result_row_under_its_run(self):
        storage = self.make_storage()
        res = ItemResult("hello world", metadata={"type": "TEXT"}, processor=self.comp_a)
        self.assertIs(storage.process(res), res)  # pass-through, like the other Storage components
        results = self.names["RESULT"]
        rows = self.query(
            f"SELECT id::text, workflow_run_id::text, process_id, value, metadata, extra_data, created FROM {results}"
        )
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0][:3], (res.id, self.wf.run_id, "src_a"))
        self.assertEqual(rows[0][3], "hello world")  # jsonb, parsed back out rather than returned as text
        self.assertEqual(rows[0][4]["type"], "TEXT")
        self.assertIsNone(rows[0][5])  # nothing in extra: NULL rather than an empty object
        self.assertEqual(rows[0][6], res.created)
        columns = [r[0] for r in self.query(
            "SELECT column_name FROM information_schema.columns WHERE table_name = %s ORDER BY ordinal_position",
            (results,),
        )]
        self.assertEqual(columns, list(RESULT_COLUMNS))
        self.assertEqual(
            self.query(f"SELECT w.name, p.name FROM {self.names['WORKFLOW']} w JOIN {self.names['PROJECT']} p ON p.id = w.project_id"),
            [("PostgreSQL storage", "Tests")],
        )

    def test_storing_a_result_again_is_an_upsert(self):
        storage = self.make_storage()
        res = ItemResult("first", processor=self.comp_a)
        storage.process(res)
        results = self.names["RESULT"]
        created = self.query(f"SELECT created FROM {results}")[0][0]
        correction_id = save_postgres_correction(
            res.id, "fixed", agent="a.reviewer@example.org", params=self.params, tables=self.tables
        )
        res.value = "second"
        storage.process(res)
        rows = self.query(f"SELECT value, created FROM {results} WHERE id = %s::uuid", (res.id,))
        self.assertEqual(rows, [("second", created)])
        # the correction is its own row and still points at the result that was re-stored
        corrections = self.query(
            f"SELECT id::text, metadata, workflow_run_id::text FROM {results} WHERE previous_id = %s::uuid", (res.id,)
        )
        self.assertEqual(corrections[0][0], correction_id)
        self.assertEqual(corrections[0][1]["agent"], "a.reviewer@example.org")
        self.assertEqual(corrections[0][2], self.wf.run_id)
        self.assertIsNone(save_postgres_correction(str(uuid.uuid4()), "x", params=self.params, tables=self.tables))

    def test_a_derivative_is_a_result_row_of_its_own(self):
        storage = self.make_storage()
        src = ItemResult("source")
        deriv = ItemResult(["label"], processor=self.comp_a, input=src)
        src.derivative_results = {self.comp_a: [deriv]}
        storage.process(src)
        rows = self.query(
            f"SELECT process_id, input, value FROM {self.names['RESULT']} WHERE id = %s::uuid", (deriv.id,)
        )
        self.assertEqual(rows, [("src_a", src.id, ["label"])])

    def test_a_run_reaches_the_tables_and_the_parquet_file_alike(self):
        csv_path = os.path.join(self.dir, "people.csv")
        with open(csv_path, "w", newline="") as fh:
            fh.write("name,title\n")
            fh.write("".join(f"name{i},title{i}\n" for i in range(6)))
        tree = {
            "id": "pg_run_wf",
            "type": "workflow.Workflow",
            "name": "PostgreSQL and Parquet",
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
                                            "settings": {"tables": self.tables, "database": self.params["dbname"]},
                                        },
                                        # no fields: the file's rows are RESULT rows too
                                        {"id": "pq", "type": "storage.ParquetStorage",
                                         "settings": {"file": self.out, "flush_on_exit": False}},
                                    ],
                                }
                            ],
                        }
                    ],
                }
            ],
        }
        wf = Workflow(tree)
        wf.run()
        # the Parquet route into a second set of tables: the run's own rows (carried in the file's
        # metadata) first, then the RESULT rows, bulk loaded
        loaded = {entity: f"{name}_l" for entity, name in self.tables.items()}
        records = json.loads(pq.read_schema(self.out).metadata[b"chai_tea_records"])
        store_postgres_rows(records, self.params, tables=loaded)
        info = parquet_to_postgres(self.out, loaded["RESULT"], self.params, create=False)
        self.assertEqual(info["rows"], 6)
        direct = self.tables["RESULT"]
        # the run's rows say what they were made from: the CSV, and the md5 of the row itself
        self.assertEqual(self.query(f"SELECT DISTINCT input FROM {direct}"), [(csv_path,)])
        self.assertEqual(self.query(f"SELECT count(DISTINCT input_hash) FROM {direct}")[0][0], 6)
        columns = ", ".join(RESULT_COLUMNS)
        self.assertEqual(self.query(f"SELECT count(*) FROM {direct}")[0][0], 6)
        # what one route stored and the other loaded is the same set of rows -- every column, both ways
        self.assertEqual(self.query(f"SELECT {columns} FROM {direct} EXCEPT SELECT {columns} FROM {loaded['RESULT']}"), [])
        self.assertEqual(self.query(f"SELECT {columns} FROM {loaded['RESULT']} EXCEPT SELECT {columns} FROM {direct}"), [])
        # and the run finished in both: successful, with its duration
        for names in (self.tables, loaded):
            self.assertEqual(
                self.query(f"SELECT id::text, was_successful, duration > 0 FROM {names['WORKFLOW_RUN']}"),
                [(wf.run_id, True, True)],
            )

    def test_a_table_in_the_old_layout_is_refused(self):
        self.execute(f"CREATE TABLE {self.tables['RESULT']} (id TEXT PRIMARY KEY, value_json JSONB)")
        with self.assertRaises(RuntimeError):
            self.make_storage().process(ItemResult("x"))

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
        self.assertEqual(self.query(f"SELECT value FROM {self.tables['RESULT']}")[0][0], "through psycopg2")
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


class TestTokenUsageSummary(unittest.TestCase):
    """Totalling the token usage an AI run recorded in its JSON-Lines log."""

    def setUp(self):
        self.dir = tempfile.mkdtemp()
        self.log = os.path.join(self.dir, "run.jsonl")

    def tearDown(self):
        shutil.rmtree(self.dir, ignore_errors=True)

    def write(self, records):
        with open(self.log, "w") as fh:
            for record in records:
                fh.write(json.dumps(record) + "\n")

    def usage(self, total=1000, prompt=200, images=700, thinking=50, result=50):
        return {
            "type": "TEXT",
            "duration": 1.5,
            "token_usage": {
                "total": total,
                "prompt": prompt,
                "images": images,
                "thinking": thinking,
                "result": result,
            },
        }

    def test_totals_per_component_and_overall(self):
        self.write(
            [
                {"processorId": "transcribe", "metadata": self.usage()},
                {"processorId": "transcribe", "metadata": self.usage()},
                {"processorId": "translate", "metadata": self.usage(500, 400, 0, 0, 100)},
            ]
        )
        summary = token_usage_summary(self.log)
        self.assertEqual(summary["records"], 3)
        self.assertEqual(summary["components"]["transcribe"]["calls"], 2)
        self.assertEqual(summary["components"]["transcribe"]["input"], 1800)
        self.assertEqual(summary["components"]["transcribe"]["images"], 1400)
        self.assertEqual(summary["components"]["translate"]["input"], 400)
        self.assertEqual(summary["total"]["total"], 2500)
        self.assertEqual(summary["total"]["duration"], 4.5)

    def test_results_without_usage_are_counted_not_ignored(self):
        self.write(
            [
                {"processorId": "transcribe", "metadata": self.usage()},
                {"processorId": "save_text", "metadata": {"type": "TEXT"}},
            ]
        )
        summary = token_usage_summary(self.log)
        self.assertEqual(summary["without_usage"], 1)
        self.assertEqual(summary["total"]["calls"], 1)
        self.assertNotIn("save_text", summary["components"])

    def test_missing_modality_breakdown_falls_back_to_the_total(self):
        # The API did not break the prompt down by modality; what is left of the total went in
        self.write([{"processorId": "t", "metadata": self.usage(1000, -1, -1, 100, 200)}])
        summary = token_usage_summary(self.log)
        self.assertEqual(summary["total"]["input"], 700)
        self.assertEqual(summary["total"]["total"], 1000)

    def test_prices_give_a_cost_with_thinking_billed_as_output(self):
        self.write([{"processorId": "t", "metadata": self.usage(1_000_000, 1_000_000, 0, 500_000, 500_000)}])
        summary = token_usage_summary(self.log, input_price=1.0, output_price=10.0)
        # 1M in at $1, and 0.5M thinking + 0.5M output at $10
        self.assertAlmostEqual(summary["total"]["cost"], 11.0)

    def test_reads_usage_from_a_whole_result_column(self):
        self.write([{"step": "t", "@record": {"id": "x", "metadata": self.usage()}}])
        summary = token_usage_summary(self.log)
        self.assertEqual(summary["components"]["t"]["calls"], 1)

    def test_renamed_step_column(self):
        self.write([{"step": "transcribe", "metadata": self.usage()}])
        self.assertIn("transcribe", token_usage_summary(self.log)["components"])
