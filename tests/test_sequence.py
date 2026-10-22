"""``input_sequence``: every stored row says where it comes in order, so a front end can ORDER BY it.

The pages of a run are numbered by their position in the input, a page's transcription and
translation share its number, and the lines of one page are numbered within it -- in reading order
when they come from a detector.
"""

import os
import shutil
import sqlite3
import tempfile
import unittest

from chai.result import ItemResult, ListResult
from chai.segmenter import reading_order
from chai.persistence import result_row
from chai.workflow import Workflow

PAGES = {
    # written out of order: the directory listing sorts them, and that is the order they are read in
    "f0003r.txt": "third page, line one\nthird page, line two",
    "f0001r.txt": "first page, line one\nfirst page, line two\nfirst page, line three",
    "f0002r.txt": "second page, line one",
}


def pages_and_lines_workflow(pages_dir, db, workers=1):
    """Each page is read (its 'transcription') and copied (its 'translation'), then split into lines."""
    store = {"type": "storage.SqliteStorage", "settings": {"database": db}}
    return {
        "id": "sequence_wf",
        "type": "workflow.Workflow",
        "steps": [
            {
                "type": "provider.DirFileProvider",
                "input": pages_dir,
                "steps": [
                    {
                        "id": "pages",
                        "type": "iterator.Iterator",
                        "source": True,
                        "settings": {"workers": workers},
                        "steps": [
                            {
                                "id": "transcribe",
                                "type": "transcriber.TextFileTranscriber",
                                "next_steps": [
                                    dict(store),
                                    {
                                        "id": "translate",
                                        "type": "translator.GlossaryTranslator",
                                        "settings": {"glossary": {"page": "folio"}},
                                        "next_steps": [dict(store)],
                                    },
                                    {
                                        "id": "lines",
                                        "type": "segmenter.TextSegmenter",
                                        "settings": {"mode": "line"},
                                        "next_steps": [
                                            {
                                                "id": "each_line",
                                                "type": "iterator.Iterator",
                                                "steps": [
                                                    {
                                                        "id": "line",
                                                        "type": "translator.GlossaryTranslator",
                                                        "settings": {"glossary": {"zzz": "zzz"}},
                                                        "next_steps": [dict(store)],
                                                    }
                                                ],
                                            }
                                        ],
                                    },
                                ],
                            }
                        ],
                    }
                ],
            }
        ],
    }


class TestIteratorPositions(unittest.TestCase):
    def iterate(self, itype="iterator.Iterator", settings=None, values=None):
        tree = {
            "id": "positions_wf",
            "type": "workflow.Workflow",
            "steps": [
                {"id": "iter", "type": itype, "settings": settings or {},
                 "steps": [{"id": "double", "type": "extractor.DoubleExtractor"}]}
            ],
        }
        wf = Workflow(tree)
        res = wf.get_component_by_id("iter").process(ListResult(values or list("abcdef")))
        return [(entry.input.value, entry.input.sequence) for entry in res.value]

    def test_each_entry_is_stamped_with_its_position(self):
        self.assertEqual(self.iterate(), [(c, i) for i, c in enumerate("abcdef")])

    def test_workers_do_not_change_the_positions(self):
        self.assertEqual(self.iterate(settings={"workers": 4}), [(c, i) for i, c in enumerate("abcdef")])

    def test_a_slice_keeps_the_positions_of_the_whole_input(self):
        got = self.iterate("iterator.SliceIterator", {"max_slices": 3, "slice": 1})
        self.assertEqual(got, [("b", 1), ("e", 4)])

    def test_a_row_takes_the_nearest_position_up_its_chain(self):
        page = ItemResult("page")
        page.sequence = 4
        line = ItemResult("line", input=ItemResult("lines", input=page))
        line.sequence = 2
        self.assertEqual(result_row(ItemResult("translated line", input=line))["input_sequence"], 2)
        self.assertEqual(result_row(ItemResult("translated page", input=page))["input_sequence"], 4)
        self.assertIsNone(result_row(ItemResult("on its own"))["input_sequence"])
        # the position is a column of its own, not also a key of extra_data
        self.assertIsNone(result_row(page)["extra_data"])


class TestStoredOrder(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.mkdtemp()
        self.pages = os.path.join(self.dir, "pages")
        os.makedirs(self.pages)
        for name, text in PAGES.items():
            with open(os.path.join(self.pages, name), "w") as fh:
                fh.write(text)
        self.db = os.path.join(self.dir, "run.db")

    def tearDown(self):
        shutil.rmtree(self.dir, ignore_errors=True)

    def query(self, sql):
        conn = sqlite3.connect(self.db)
        try:
            return conn.execute(sql).fetchall()
        finally:
            conn.close()

    def check(self, workers):
        Workflow(pages_and_lines_workflow(self.pages, self.db, workers)).run()
        # the pages: the transcription and translation of the nth page are both n
        pages = self.query(
            "SELECT process_id, input_sequence, input FROM results "
            "WHERE process_id IN ('transcribe', 'translate') ORDER BY process_id, input_sequence"
        )
        names = [os.path.basename(path) for _, _, path in pages]
        self.assertEqual(names, ["f0001r.txt", "f0002r.txt", "f0003r.txt"] * 2)
        self.assertEqual([seq for _, seq, _ in pages], [0, 1, 2] * 2)
        # the lines: numbered within their page, so ORDER BY input, input_sequence reads the run
        lines = self.query(
            "SELECT input_sequence, json_extract(value, '$') FROM results "
            "WHERE process_id = 'line' ORDER BY input, input_sequence"
        )
        self.assertEqual(
            [text for _, text in lines],
            [line for name in sorted(PAGES) for line in PAGES[name].split("\n")],
        )
        self.assertEqual([seq for seq, _ in lines], [0, 1, 2, 0, 0, 1])

    def test_pages_and_their_lines_sort_into_reading_order(self):
        self.check(workers=1)

    def test_the_order_survives_parallel_workers(self):
        self.check(workers=3)


class TestReadingOrder(unittest.TestCase):
    """The order ``YoloSegmenter`` emits its regions in, before they are numbered."""

    # A heading across the top of a page, then two columns of three lines, scrambled as a detector
    # (roughly by confidence) would return them
    BOXES = {
        "R2": [520, 300, 900, 340],
        "L1": [100, 200, 480, 240],
        "H": [100, 80, 900, 140],
        "R1": [520, 200, 900, 240],
        "L3": [100, 400, 480, 440],
        "L2": [100, 300, 480, 340],
        "R3": [520, 400, 900, 440],
    }

    def ordered(self, order):
        names = list(self.BOXES)
        return [names[i] for i in reading_order(list(self.BOXES.values()), order)]

    def test_columns_are_read_left_then_right_after_a_heading_over_both(self):
        self.assertEqual(self.ordered("reading"), ["H", "L1", "L2", "L3", "R1", "R2", "R3"])

    def test_top_to_bottom_reads_across_the_columns(self):
        self.assertEqual(self.ordered("top_to_bottom"), ["H", "L1", "R1", "L2", "R2", "L3", "R3"])

    def test_detected_keeps_the_detector_order(self):
        self.assertEqual(self.ordered("detected"), list(self.BOXES))

    def test_a_single_column_is_read_top_to_bottom(self):
        lines = [[100, 300, 480, 340], [100, 200, 480, 240], [100, 250, 470, 290]]
        self.assertEqual(reading_order(lines), [1, 2, 0])

    def test_an_unknown_order_is_refused(self):
        with self.assertRaises(ValueError):
            reading_order([], "sideways")


if __name__ == "__main__":
    unittest.main()
