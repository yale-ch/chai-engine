import json
import os
import tempfile
import unittest

from chai.provider import CsvFileProvider
from chai.result import ItemResult, ListResult
from chai.workflow import Workflow

PEOPLE_CSV = os.path.join(os.path.dirname(os.path.dirname(__file__)), "inputs", "people.csv")

CSV_TEXT = """name,title,publisher,year
"Ford, Brinsley [Sir, 1908-1999] (introduction by)","Eliot Hodgkin, 1905-1987 : painter and collector","Hazlitt, Gooden & Fox",1990
"Booth, Sally (author)","Edges & extremes : Shetland","Arch Ventures Press",2013
"Brett, Simon (author)","The life and art of Clifford Webb","Little Toller Books",2019
"""


def write_csv(tmpdir, text=CSV_TEXT, name="people.csv"):
    path = os.path.join(tmpdir, name)
    with open(path, "w", newline="") as fh:
        fh.write(text)
    return path


def make_provider(path, settings=None, steps=None):
    """Build a one-provider workflow and return (workflow, provider)."""
    tree = {
        "id": "csv_wf",
        "type": "workflow.Workflow",
        "steps": [
            {
                "id": "csv_provider",
                "type": "provider.CsvFileProvider",
                "input": path,
                "settings": settings or {},
                "steps": steps or [],
            }
        ],
    }
    wf = Workflow(tree)
    return wf, wf.get_component_by_id("csv_provider")


class TestCsvFileProvider(unittest.TestCase):
    def test_rows_become_dicts(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = write_csv(tmpdir)
            wf, provider = make_provider(path)
            res = provider.run()

            self.assertIsInstance(res, ListResult)
            self.assertEqual(len(res.value), 3)
            self.assertEqual(res.metadata["columns"], ["name", "title", "publisher", "year"])

            first = res.value[0]
            self.assertIsInstance(first, ItemResult)
            self.assertEqual(
                first.value,
                {
                    "name": "Ford, Brinsley [Sir, 1908-1999] (introduction by)",
                    "title": "Eliot Hodgkin, 1905-1987 : painter and collector",
                    "publisher": "Hazlitt, Gooden & Fox",
                    "year": "1990",
                },
            )
            # DATA-typed, so JSON-shaped components downstream will accept it
            self.assertEqual(first.metadata["type"], "DATA")
            self.assertEqual(first.metadata["row"], 0)
            self.assertIs(first.processor, provider)

    def test_result_is_iterable(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = write_csv(tmpdir)
            wf, provider = make_provider(path)
            res = provider.run()

            rows = list(res)
            self.assertEqual(len(rows), 3)
            # Iteration passes the row Results through untouched, keeping their metadata
            for i, row in enumerate(rows):
                self.assertIsInstance(row, ItemResult)
                self.assertEqual(row.metadata["row"], i)
                self.assertIsInstance(row.value, dict)
            self.assertEqual(
                [r.value["name"] for r in rows],
                [
                    "Ford, Brinsley [Sir, 1908-1999] (introduction by)",
                    "Booth, Sally (author)",
                    "Brett, Simon (author)",
                ],
            )
            # Indexing matches iteration
            self.assertEqual(res[1].value, rows[1].value)

    def test_limit_setting(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = write_csv(tmpdir)
            wf, provider = make_provider(path, {"limit": 2})
            res = provider.run()
            self.assertEqual(len(res.value), 2)
            self.assertEqual(res.metadata["columns"], ["name", "title", "publisher", "year"])

    def test_explicit_columns_no_header(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = write_csv(tmpdir, '"Booth, Sally (author)",Shetland\n')
            wf, provider = make_provider(path, {"columns": ["name", "title"]})
            res = provider.run()
            self.assertEqual(len(res.value), 1)
            self.assertEqual(res.value[0].value, {"name": "Booth, Sally (author)", "title": "Shetland"})

    def test_delimiter_setting(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = write_csv(tmpdir, "name\ttitle\nBooth, Sally\tShetland\n")
            wf, provider = make_provider(path, {"delimiter": "\t"})
            res = provider.run()
            self.assertEqual(res.value[0].value, {"name": "Booth, Sally", "title": "Shetland"})

    def test_ragged_rows(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = write_csv(tmpdir, "name,title\nshort\nlong,title,surplus\n")
            wf, provider = make_provider(path)
            res = provider.run()
            self.assertEqual(res.value[0].value, {"name": "short", "title": ""})
            self.assertEqual(res.value[1].value, {"name": "long", "title": "title", "_extra": ["surplus"]})

    def test_missing_file_raises(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            wf, provider = make_provider(os.path.join(tmpdir, "nope.csv"))
            with self.assertRaises(ValueError):
                provider.run()

    def test_empty_file_raises(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = write_csv(tmpdir, "")
            wf, provider = make_provider(path)
            with self.assertRaises(ValueError):
                provider.run()

    @unittest.skipUnless(os.path.exists(PEOPLE_CSV), f"{PEOPLE_CSV} not present")
    def test_reads_people_csv(self):
        wf, provider = make_provider(PEOPLE_CSV)
        res = provider.run()
        self.assertIn("name", res.metadata["columns"])
        self.assertTrue(len(res.value) > 1)
        for row in res:
            self.assertIsInstance(row.value, dict)
            self.assertTrue(row.value["name"])


class TestCsvXpathChain(unittest.TestCase):
    """The workflow-ppl.json shape: CSV rows -> Iterator -> JsonXpathExtractor('/name') -> parser.

    The real workflow's parser is a local transformers model, so here the downstream step is the
    deterministic ``WordCountExtractor``: it proves the extracted name is what gets passed on.
    """

    def build(self, path, downstream=None):
        xpath_step = {
            "id": "row_name",
            "type": "extractor.JsonXpathExtractor",
            "settings": {"xpath": "/name"},
        }
        if downstream:
            xpath_step["next_steps"] = [downstream]
        return make_provider(
            path,
            steps=[{"id": "row_iter", "type": "iterator.Iterator", "steps": [xpath_step]}],
        )

    def test_xpath_extracts_name_column(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = write_csv(tmpdir)
            wf, provider = self.build(path)
            # The provider merges its steps' output, so unwrap to the Iterator's result
            res = provider.run().value[0]

            # Iterator output: one ListResult of step outputs per row
            self.assertEqual(len(res.value), 3)
            names = [row.value[0].value for row in res.value]
            self.assertEqual(
                names,
                [
                    "Ford, Brinsley [Sir, 1908-1999] (introduction by)",
                    "Booth, Sally (author)",
                    "Brett, Simon (author)",
                ],
            )
            # TEXT-typed, which is what an AI component (the name parser) requires of its input
            self.assertEqual(res.value[0].value[0].metadata["type"], "TEXT")

    def test_name_is_passed_to_downstream_step(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = write_csv(tmpdir)
            wf, provider = self.build(
                path, {"id": "counter", "type": "extractor.WordCountExtractor"}
            )
            res = provider.run().value[0]

            # Each row's chain now ends in the downstream extractor's output
            counted = json.loads(res.value[1].value[0].value)
            self.assertEqual(counted, {"booth,": 1, "sally": 1, "(author)": 1})

    def test_workflow_ppl_json_is_valid(self):
        """workflow-ppl.json must be loadable and wired the way the chain above expects."""
        wf_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "workflow-ppl.json")
        with open(wf_path) as fh:
            tree = json.load(fh)

        provider = tree["steps"][0]
        self.assertEqual(provider["type"], "provider.CsvFileProvider")
        iterator = provider["steps"][0]
        self.assertEqual(iterator["type"], "iterator.Iterator")
        xpath = iterator["steps"][0]
        self.assertEqual(xpath["type"], "extractor.JsonXpathExtractor")
        self.assertEqual(xpath["settings"]["xpath"], "/name")
        parser = xpath["next_steps"][0]
        self.assertEqual(parser["type"], "extractor.TransformersExtractor")
        self.assertIn("{text_input_0}", parser["settings"]["prompt"])


if __name__ == "__main__":
    unittest.main()
