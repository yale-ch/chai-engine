"""Tests for chai.context.ContextWindow and storage.TextFileStorage.

The window is exercised with TextJoinReducer as its step: the reducer joins every text it is given,
so the text it produces shows exactly what the window injected alongside the entry, and the window
then captures from that text -- injection and capture in one deterministic run, no model involved.
"""

import logging
import os
import shutil
import tempfile
import unittest

from chai.context import ContextWindow
from chai.result import ItemResult
from chai.workflow import Workflow

PAGES = ["alpha one\nalpha two", "beta one\nbeta two", "gamma one\ngamma two"]


def make_workflow(window_settings=None, iterator_settings=None, pages=None, steps=None):
    """A workflow of StaticProvider -> Iterator -> ContextWindow(TextJoinReducer)."""
    tree = {
        "id": "ctx_wf",
        "type": "workflow.Workflow",
        "steps": [
            {
                "id": "pages",
                "type": "provider.StaticProvider",
                # StaticProvider ignores its input, but Workflow._run only starts input-less
                # steps when handed a runtime input -- so configure one.
                "input": "",
                "settings": {"values": list(pages if pages is not None else PAGES)},
                "steps": [
                    {
                        "id": "each",
                        "type": "iterator.Iterator",
                        "settings": iterator_settings or {},
                        "steps": [
                            {
                                "id": "window",
                                "type": "context.ContextWindow",
                                "settings": window_settings or {},
                                "steps": steps or [{"id": "join", "type": "reducer.TextJoinReducer"}],
                            }
                        ],
                    }
                ],
            }
        ],
    }
    return Workflow(tree)


def joined(res):
    """The text each entry's window produced, in order."""
    # workflow -> provider list -> iterator list -> per-entry list -> the window's single output
    iterator = res.value[0].value[0]
    return [entry.value[0].value for entry in iterator.value]


class TestContextInjection(unittest.TestCase):
    def test_first_entry_gets_the_empty_text(self):
        res = make_workflow().run()
        self.assertTrue(joined(res)[0].startswith("(none)"))

    def test_following_entries_get_the_previous_last_line(self):
        texts = joined(make_workflow().run())
        self.assertEqual(texts[1], "alpha two\n\nbeta one\nbeta two")
        self.assertEqual(texts[2], "beta two\n\ngamma one\ngamma two")

    def test_template_wraps_the_carried_text(self):
        texts = joined(make_workflow({"template": "PREVIOUS LINE: {context}"}).run())
        self.assertTrue(texts[1].startswith("PREVIOUS LINE: alpha two\n\nbeta one"))
        # The template is only used when there is something to carry
        self.assertTrue(texts[0].startswith("(none)"))

    def test_empty_text_is_configurable(self):
        texts = joined(make_workflow({"empty": "this is the first page"}).run())
        self.assertTrue(texts[0].startswith("this is the first page"))

    def test_initial_context_is_carried_into_the_first_entry(self):
        texts = joined(make_workflow({"initial": "the line before the run"}).run())
        self.assertTrue(texts[0].startswith("the line before the run\n\nalpha one"))

    def test_position_after_puts_the_context_last(self):
        # The reducer echoes what it was given, so the injected context is now at the end
        texts = joined(make_workflow({"position": "after", "capture_from": "each"}).run())
        self.assertEqual(texts[0], "alpha one\nalpha two\n\n(none)")
        self.assertEqual(texts[1], "beta one\nbeta two\n\nalpha two")

    def test_context_used_metadata_records_whether_anything_was_carried(self):
        res = make_workflow().run()
        entries = res.value[0].value[0].value
        used = [entry.value[0].metadata["context_used"] for entry in entries]
        self.assertEqual(used, [False, True, True])

    def test_single_step_result_passes_through_unwrapped(self):
        res = make_workflow().run()
        out = res.value[0].value[0].value[0].value[0]
        self.assertIsInstance(out, ItemResult)
        self.assertEqual(out.processor.id, "join")


class TestContextCapture(unittest.TestCase):
    def test_all_keeps_the_whole_text(self):
        texts = joined(make_workflow({"capture": "all"}).run())
        # The second entry sees everything the first produced, injected text and all
        self.assertTrue(texts[1].startswith("(none)\n\nalpha one\nalpha two\n\nbeta one"))

    def test_last_lines_keeps_n_lines(self):
        texts = joined(make_workflow({"capture": "last_lines", "lines": 2}).run())
        self.assertTrue(texts[1].startswith("alpha one\nalpha two\n\nbeta one"))

    def test_first_line_keeps_the_first_line(self):
        texts = joined(make_workflow({"capture": "first_line"}).run())
        self.assertTrue(texts[1].startswith("(none)\n\nbeta one"))

    def test_tail_chars_keeps_the_end_of_the_text(self):
        texts = joined(make_workflow({"capture": "tail_chars", "chars": 5}).run())
        self.assertTrue(texts[1].startswith("a two\n\nbeta one"))

    def test_max_chars_caps_the_carried_text(self):
        # The whole first text is captured, then capped to its last 9 characters: 'alpha two'
        texts = joined(make_workflow({"capture": "all", "max_chars": 9}).run())
        self.assertTrue(texts[1].startswith("alpha two\n\nbeta one"))

    def test_capture_from_reaches_back_up_the_input_chain(self):
        # 'each' is the iterator, which stamps itself on every entry: this captures the entry the
        # window was given rather than what its steps made of it
        texts = joined(make_workflow({"capture": "first_line", "capture_from": "each"}).run())
        self.assertTrue(texts[1].startswith("alpha one\n\nbeta one"))

    def test_capture_from_finds_a_step_inside_the_window(self):
        texts = joined(make_workflow({"capture_from": "join"}).run())
        self.assertEqual(texts[1], "alpha two\n\nbeta one\nbeta two")

    def test_unknown_capture_from_warns_and_carries_nothing(self):
        with self.assertLogs("chai", level="WARNING") as logs:
            texts = joined(make_workflow({"capture_from": "nosuchstep"}).run())
        self.assertTrue(any("nosuchstep" in line for line in logs.output))
        self.assertTrue(all(text.startswith("(none)") for text in texts))

    def test_ignored_values_are_not_carried(self):
        pages = ["VLM-NO-TEXT", "beta one\nbeta two"]
        texts = joined(make_workflow({"ignore": ["VLM-NO-TEXT"]}, pages=pages).run())
        # The first entry's text is the injected '(none)' plus the marker, whose last line is ignored
        self.assertTrue(texts[1].startswith("(none)\n\nbeta one"))

    def test_on_empty_keep_holds_the_previous_context(self):
        pages = ["alpha one\nalpha two", "VLM-NO-TEXT", "gamma one"]
        settings = {"ignore": ["VLM-NO-TEXT"], "on_empty": "keep"}
        texts = joined(make_workflow(settings, pages=pages).run())
        self.assertTrue(texts[1].startswith("alpha two\n\nVLM-NO-TEXT"))
        self.assertTrue(texts[2].startswith("alpha two\n\ngamma one"))

    def test_reset_starts_a_new_sequence(self):
        wf = make_workflow()
        wf.run()
        window = wf.get_component_by_id("window")
        self.assertEqual(window.context, "gamma two")
        window.reset()
        self.assertEqual(window.context, "")
        self.assertTrue(joined(wf.run())[0].startswith("(none)"))


class TestContextConcurrencyWarning(unittest.TestCase):
    def test_warns_inside_a_concurrent_iterator(self):
        with self.assertLogs("chai", level="WARNING") as logs:
            make_workflow(iterator_settings={"workers": 4})
        self.assertTrue(any("workers" in line for line in logs.output))

    def test_no_warning_when_sequential(self):
        with self.assertLogs("chai", level="WARNING") as logs:
            make_workflow(iterator_settings={"workers": 1})
            logging.getLogger("chai").warning("probe")  # assertLogs needs at least one record
        self.assertEqual([line for line in logs.output if "workers" in line], [])


class TestTextFileStorage(unittest.TestCase):
    """One text file per page, named after the page it came from."""

    def setUp(self):
        self.dir = tempfile.mkdtemp()
        self.pages = os.path.join(self.dir, "pages")
        self.out = os.path.join(self.dir, "text")
        os.makedirs(self.pages)
        for name, text in (("f0001r.txt", "first page"), ("f0001v.txt", "second page")):
            with open(os.path.join(self.pages, name), "w") as fh:
                fh.write(text)

    def tearDown(self):
        shutil.rmtree(self.dir, ignore_errors=True)

    def run_workflow(self, storage_settings=None):
        tree = {
            "id": "text_storage_wf",
            "type": "workflow.Workflow",
            "steps": [
                {
                    "id": "files",
                    "type": "provider.DirFileProvider",
                    "input": self.pages,
                    "steps": [
                        {
                            "id": "each",
                            "type": "iterator.Iterator",
                            "steps": [
                                {
                                    "id": "read",
                                    "type": "transcriber.TextFileTranscriber",
                                    "next_steps": [
                                        {
                                            "id": "save",
                                            "type": "storage.TextFileStorage",
                                            "settings": {
                                                "directory": self.out,
                                                **(storage_settings or {}),
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
        return Workflow(tree).run()

    def test_one_file_per_page_named_after_the_source(self):
        self.run_workflow()
        self.assertEqual(sorted(os.listdir(self.out)), ["f0001r.txt", "f0001v.txt"])
        with open(os.path.join(self.out, "f0001v.txt")) as fh:
            self.assertEqual(fh.read(), "second page")

    def test_suffix_and_extension_are_configurable(self):
        self.run_workflow({"suffix": "-en", "extension": ".md"})
        self.assertEqual(sorted(os.listdir(self.out)), ["f0001r-en.md", "f0001v-en.md"])

    def test_no_overwrite_versions_the_file(self):
        self.run_workflow({"overwrite": False})
        self.run_workflow({"overwrite": False})
        self.assertIn("f0001r.txt.1", os.listdir(self.out))

    def test_name_from_names_a_component_in_the_chain(self):
        self.run_workflow({"name_from": "each"})  # the iterator stamps itself on each page
        self.assertEqual(sorted(os.listdir(self.out)), ["f0001r.txt", "f0001v.txt"])

    def test_file_results_are_not_copied_back_out(self):
        wf = Workflow({"id": "direct_wf", "type": "workflow.Workflow"})
        storage = wf._make_step(
            {"type": "storage.TextFileStorage", "settings": {"directory": self.out}}, wf
        )
        from chai.result import FileItemResult

        with self.assertLogs("chai", level="WARNING"):
            storage.process(FileItemResult(os.path.join(self.pages, "f0001r.txt")))
        self.assertEqual(os.listdir(self.out), [])


class TestContextWindowUnit(unittest.TestCase):
    """The window's pieces on their own, without a run around them."""

    def setUp(self):
        self.wf = Workflow({"id": "unit_wf", "type": "workflow.Workflow"})

    def window(self, settings=None):
        return self.wf._make_step(
            {"type": "context.ContextWindow", "settings": settings or {}}, self.wf
        )

    def test_combine_keeps_the_input_in_the_chain(self):
        window = self.window()
        self.assertIsInstance(window, ContextWindow)
        page = ItemResult("a page", metadata={"type": "TEXT"})
        combined = window.combine(window.context_result(page), page)
        self.assertEqual([r.value for r in combined], ["(none)", "a page"])
        self.assertIs(combined.input, page)

    def test_extract_skips_binary_results(self):
        window = self.window()
        with self.assertLogs("chai", level="WARNING"):
            self.assertEqual(window.extract(ItemResult(b"\x89PNG", metadata={"type": "IMAGE"})), "")


if __name__ == "__main__":
    unittest.main()
