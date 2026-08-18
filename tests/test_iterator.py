import os
import tempfile
import unittest

from chai.result import ListResult
from chai.workflow import Workflow

LETTERS = list("abcdefghij")


def make_iterator(itype="iterator.Iterator", settings=None):
    """Build a workflow of one iterator over a DoubleExtractor and return (workflow, iterator)."""
    tree = {
        "id": "iter_wf",
        "type": "workflow.Workflow",
        "steps": [
            {
                "id": "iter",
                "type": itype,
                "settings": settings or {},
                "steps": [{"id": "double", "type": "extractor.DoubleExtractor"}],
            }
        ],
    }
    wf = Workflow(tree)
    return wf, wf.get_component_by_id("iter")


def selected(res):
    """The input entries the iterator actually processed, in order."""
    # Each entry of the output is a ListResult of step outputs, built from the entry it ran on
    return [entry.input.value for entry in res.value]


def step_outputs(res):
    """The output of the child step for each processed entry."""
    return [entry.value[0].value for entry in res.value]


class TestIterator(unittest.TestCase):
    """The base Iterator still processes every entry after the select_items refactor."""

    def test_processes_every_entry(self):
        wf, it = make_iterator()
        res = it.process(ListResult(LETTERS))
        self.assertEqual(selected(res), LETTERS)
        self.assertEqual(step_outputs(res), [x * 2 for x in LETTERS])

    def test_select_items_returns_all_entries(self):
        wf, it = make_iterator()
        items = it.select_items(ListResult(LETTERS))
        self.assertEqual([i.value for i in items], LETTERS)


class TestSliceIterator(unittest.TestCase):
    def slice_of(self, offset, max_slices, values=LETTERS):
        wf, it = make_iterator("iterator.SliceIterator", {"slice": offset, "max_slices": max_slices})
        return selected(it.process(ListResult(list(values))))

    def test_every_nth_entry(self):
        # 10 letters, 3 slices: 0 -> a,d,g,j  1 -> b,e,h  2 -> c,f,i
        self.assertEqual(self.slice_of(0, 3), ["a", "d", "g", "j"])
        self.assertEqual(self.slice_of(1, 3), ["b", "e", "h"])
        self.assertEqual(self.slice_of(2, 3), ["c", "f", "i"])

    def test_slice_zero_starts_with_first_record(self):
        # The documented case: slice 0 of 24 is every 24th record, starting at the first
        values = list(range(100))
        self.assertEqual(self.slice_of(0, 24, values), [0, 24, 48, 72, 96])
        self.assertEqual(self.slice_of(1, 24, values), [1, 25, 49, 73, 97])
        self.assertEqual(self.slice_of(23, 24, values), [23, 47, 71, 95])

    def test_slices_are_disjoint_and_complete(self):
        max_slices = 4
        seen = []
        for offset in range(max_slices):
            seen.extend(self.slice_of(offset, max_slices))
        self.assertEqual(len(seen), len(LETTERS))
        self.assertEqual(sorted(seen), sorted(LETTERS))

    def test_child_steps_still_run(self):
        wf, it = make_iterator("iterator.SliceIterator", {"slice": 1, "max_slices": 5})
        res = it.process(ListResult(LETTERS))
        self.assertEqual(step_outputs(res), ["bb", "gg"])

    def test_more_slices_than_entries(self):
        # A slice past the end of the input is simply empty, not an error
        self.assertEqual(self.slice_of(2, 20), ["c"])
        self.assertEqual(self.slice_of(15, 20), [])

    def test_defaults_to_every_entry(self):
        wf, it = make_iterator("iterator.SliceIterator")
        self.assertEqual(selected(it.process(ListResult(LETTERS))), LETTERS)

    def test_invalid_slice_raises(self):
        for settings in ({"slice": 3, "max_slices": 3}, {"slice": -1, "max_slices": 3}):
            wf, it = make_iterator("iterator.SliceIterator", settings)
            with self.assertRaises(ValueError):
                it.process(ListResult(LETTERS))

    def test_invalid_max_slices_raises(self):
        wf, it = make_iterator("iterator.SliceIterator", {"slice": 0, "max_slices": -2})
        with self.assertRaises(ValueError):
            it.process(ListResult(LETTERS))


class TestSliceIteratorOverCsv(unittest.TestCase):
    """Sliced CSV rows keep their original row number, so slices stay identifiable."""

    def test_row_metadata_survives_slicing(self):
        rows = "\n".join(f"name{i},t{i}" for i in range(10))
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "p.csv")
            with open(path, "w", newline="") as fh:
                fh.write(f"name,title\n{rows}\n")

            tree = {
                "id": "csv_slice_wf",
                "type": "workflow.Workflow",
                "steps": [
                    {
                        "id": "csv",
                        "type": "provider.CsvFileProvider",
                        "input": path,
                        "steps": [
                            {
                                "id": "iter",
                                "type": "iterator.SliceIterator",
                                "settings": {"slice": 1, "max_slices": 4},
                                "steps": [
                                    {
                                        "id": "xp",
                                        "type": "extractor.JsonXpathExtractor",
                                        "settings": {"xpath": "/name"},
                                    }
                                ],
                            }
                        ],
                    }
                ],
            }
            wf = Workflow(tree)
            res = wf.get_component_by_id("csv").run().value[0]

            self.assertEqual(step_outputs(res), ["name1", "name5", "name9"])
            # The rows carry their position in the file, not their position in the slice
            self.assertEqual([e.input.metadata["row"] for e in res.value], [1, 5, 9])


if __name__ == "__main__":
    unittest.main()
