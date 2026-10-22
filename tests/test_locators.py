"""Locator frames and the components that write them: what was found and where, for a UI to draw.

``chai.persistence.locators`` defines the frames; ``TextSegmenter`` and ``TextSpanLocator`` write
character ranges, ``KeywordExtractor`` gives the locator something deterministic to find, and a run
stored in SQLite shows the rows a front end reads: each span names the text row its offsets index.
"""

import json
import os
import shutil
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from chai.persistence import bbox_frame, frame_kind, list_results, segment_frames, text_frame  # noqa: E402
from chai.result import FileItemResult, ItemResult  # noqa: E402
from chai.workflow import Workflow  # noqa: E402


class TestFrames(unittest.TestCase):
    def test_a_bbox_frame(self):
        self.assertEqual(bbox_frame(1, 2, 3, 4), {"bbox": [1.0, 2.0, 3.0, 4.0]})
        self.assertEqual(frame_kind(bbox_frame(1, 2, 3, 4)), "bbox")

    def test_a_text_frame_names_the_result_it_indexes(self):
        transcript = ItemResult("Ada wrote it.")
        frame = text_frame(0, 3, result=transcript, exact="Ada")
        self.assertEqual(frame, {"start": 0, "end": 3, "result": transcript.id, "exact": "Ada"})
        self.assertEqual(frame_kind(frame), "text")

    def test_a_file_or_raw_input_is_left_to_the_rows_input(self):
        path = os.path.join(tempfile.mkdtemp(), "note.txt")
        self.assertEqual(text_frame(0, 3, result=FileItemResult(path)), {"start": 0, "end": 3})
        self.assertEqual(text_frame(0, 3), {"start": 0, "end": 3})

    def test_segment_frames_reads_a_stored_input_segment(self):
        self.assertEqual(segment_frames('[{"bbox": [1, 2, 3, 4]}, {"index": 7}]'), [{"bbox": [1, 2, 3, 4]}, {"index": 7}])
        self.assertEqual(segment_frames({"page": 3}), [{"page": 3}])
        self.assertEqual(segment_frames(None), [])
        self.assertEqual(segment_frames("not json"), [])
        self.assertEqual(frame_kind({"page": 3}), "index")
        self.assertIsNone(frame_kind({"something": "else"}))


class TestLocating(unittest.TestCase):
    def setUp(self):
        self.wf = Workflow({"id": "locating_wf", "type": "workflow.Workflow", "steps": []})
        self.made = 0

    def step(self, type_, **settings):
        self.made += 1
        return self.wf._make_step({"type": type_, "id": f"step{self.made}", "settings": settings}, self.wf)

    def locate(self, text, labels, **settings):
        transcript = ItemResult(text)
        extracted = self.step("extractor.KeywordExtractor", labels=labels).process(transcript)
        return transcript, self.step("segmenter.TextSpanLocator", **settings).process(extracted).value

    def test_the_keyword_extractor_reports_what_it_found(self):
        out = self.step("extractor.KeywordExtractor", labels={"PERSON": ["ada", "babbage"]}).process(
            ItemResult("Ada met Babbage. Ada wrote.")
        )
        self.assertEqual(out.value, [{"label": "PERSON", "text": "Ada"}, {"label": "PERSON", "text": "Babbage"}])

    def test_each_occurrence_becomes_a_located_span(self):
        text = "Ada met Babbage. Ada wrote the notes."
        transcript, spans = self.locate(text, {"PERSON": ["Ada", "Babbage"]})
        self.assertEqual([(s.value["text"], s.locator["start"]) for s in spans], [("Ada", 0), ("Babbage", 8), ("Ada", 17)])
        for span in spans:
            frame = span.locator
            self.assertEqual(text[frame["start"] : frame["end"]], frame["exact"])
            self.assertEqual(frame["result"], transcript.id)  # the text the offsets index
            self.assertEqual(span.value["label"], "PERSON")
            self.assertTrue(span.metadata["located"])

    def test_only_the_first_occurrence(self):
        _, spans = self.locate("Ada met Babbage. Ada wrote.", {"PERSON": ["Ada"]}, occurrences="first")
        self.assertEqual([s.locator["start"] for s in spans], [0])

    def test_longer_values_win(self):
        text = "Ada Lovelace and Ada"
        transcript = ItemResult(text)
        extracted = ItemResult(
            [{"type": "PERSON", "text": "Ada"}, {"type": "PERSON", "text": "Ada Lovelace"}], input=transcript
        )
        spans = self.step("segmenter.TextSpanLocator").process(extracted).value
        self.assertEqual([(s.locator["start"], s.locator["end"]) for s in spans], [(0, 12), (17, 20)])

    def test_a_value_not_in_the_text_is_kept_unlocated(self):
        transcript = ItemResult("Ada wrote it.")
        extracted = ItemResult(json.dumps({"author": "Ada", "place": "London"}), input=transcript)
        spans = self.step("segmenter.TextSpanLocator").process(extracted).value
        self.assertEqual([(s.value, s.metadata["located"], s.locator) for s in spans][1:], [({"label": "place", "text": "London"}, False, None)])
        self.assertEqual(spans[0].value, {"label": "author", "text": "Ada"})

    def test_fields_and_case(self):
        transcript = ItemResult("ADA in LONDON")
        extracted = ItemResult({"who": "Ada", "where": "London"}, input=transcript)
        spans = self.step("segmenter.TextSpanLocator", fields="where", case_sensitive=False).process(extracted).value
        self.assertEqual([(s.value["label"], s.locator["exact"]) for s in spans], [("where", "LONDON")])

    def test_the_text_can_be_named(self):
        page = ItemResult("Ada wrote it.")
        summary = self.step("translator.GlossaryTranslator", glossary={"wrote": "penned"}).process(page)
        extracted = ItemResult({"who": "Ada"}, input=summary)
        named = self.step("segmenter.TextSpanLocator", text_from=summary.processor.id).process(extracted)
        self.assertEqual(named.value[0].locator["result"], summary.id)

    def test_a_text_files_spans_leave_the_file_to_the_rows_input(self):
        path = os.path.join(tempfile.mkdtemp(), "note.txt")
        with open(path, "w") as fh:
            fh.write("Ada wrote it.")
        extracted = ItemResult({"who": "Ada"}, input=FileItemResult(path))
        [span] = self.step("segmenter.TextSpanLocator").process(extracted).value
        self.assertEqual(span.locator, {"start": 0, "end": 3, "exact": "Ada"})

    def test_a_segmenter_names_the_text_it_cut(self):
        transcript = ItemResult("One. Two!")
        segments = self.step("segmenter.TextSegmenter", mode="sentence", locate=True).process(transcript)
        self.assertEqual([s.locator for s in segments.value], [
            {"start": 0, "end": 4, "result": transcript.id},
            {"start": 5, "end": 9, "result": transcript.id},
        ])


class TestStoredSpans(unittest.TestCase):
    """Documents transcribed, keywords extracted and located, everything stored: the rows a UI reads."""

    def setUp(self):
        self.dir = tempfile.mkdtemp()
        self.docs = os.path.join(self.dir, "docs")
        os.makedirs(self.docs)
        self.text = "Ada Lovelace met Babbage in London."
        with open(os.path.join(self.docs, "note.txt"), "w") as fh:
            fh.write(self.text)
        self.db = os.path.join(self.dir, "run.db")

    def tearDown(self):
        shutil.rmtree(self.dir, ignore_errors=True)

    def test_each_span_points_into_a_stored_text(self):
        store = {"type": "storage.SqliteStorage", "settings": {"database": self.db}}
        wf = Workflow(
            {
                "id": "spans_wf",
                "type": "workflow.Workflow",
                "steps": [
                    {
                        "type": "provider.DirFileProvider",
                        "input": self.docs,
                        "steps": [
                            {
                                "type": "iterator.Iterator",
                                "source": True,
                                "steps": [
                                    {
                                        "id": "transcribe",
                                        "type": "translator.GlossaryTranslator",
                                        "settings": {"glossary": {"met": "met with"}},
                                        "next_steps": [
                                            dict(store),
                                            {
                                                "id": "entities",
                                                "type": "extractor.KeywordExtractor",
                                                "settings": {
                                                    "labels": {"PERSON": ["Ada Lovelace", "Babbage"], "PLACE": ["London"]}
                                                },
                                                "next_steps": [
                                                    {
                                                        "id": "locate",
                                                        "type": "segmenter.TextSpanLocator",
                                                        "next_steps": [
                                                            {"type": "iterator.Iterator", "steps": [dict(store)]}
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
        )
        wf.run()
        [transcript] = list_results(self.db, process_id="transcribe")
        spans = [r for r in list_results(self.db) if r["input_segment"] and r["process_id"] != "transcribe"]
        self.assertEqual(len(spans), 3)
        found = {}
        for row in spans:
            [frame] = segment_frames(row["input_segment"])
            # each span names the document it came from, and the stored text its offsets index
            self.assertEqual(row["input"], os.path.join(self.docs, "note.txt"))
            self.assertEqual(frame["result"], transcript["id"])
            self.assertEqual(transcript["value"][frame["start"] : frame["end"]], frame["exact"])
            found[frame["exact"]] = row["value"]["label"]
        self.assertEqual(found, {"Ada Lovelace": "PERSON", "Babbage": "PERSON", "London": "PLACE"})


if __name__ == "__main__":
    unittest.main()
