import tempfile
import unittest

from chai.result import ItemResult, ListResult
from chai.workflow import Workflow


class Flaky(object):
    """Mixin-style helper: a counter the test components share."""

    calls = 0


def make_flaky(fail_times, exc=RuntimeError):
    """A Component subclass whose _process fails the first *fail_times* calls."""
    from chai.core import Component

    class FlakyComponent(Component):
        calls = 0

        def _process(self, input):
            FlakyComponent.calls += 1
            if FlakyComponent.calls <= fail_times:
                raise exc(f"boom {FlakyComponent.calls}")
            return ItemResult(f"ok after {FlakyComponent.calls}", input=input, processor=self)

    return FlakyComponent


class TestErrorPolicy(unittest.TestCase):
    def setUp(self):
        self.wf = Workflow({"id": "resilience_wf", "type": "workflow.Workflow"})

    def test_retries_recover(self):
        cls = make_flaky(2)
        comp = cls({"id": "flaky1", "settings": {"retries": 2, "retry_delay": 0}}, self.wf)
        out = comp.process(ItemResult("x"))
        self.assertEqual(out.value, "ok after 3")
        self.assertEqual(cls.calls, 3)

    def test_retries_exhausted_raises(self):
        cls = make_flaky(5)
        comp = cls({"id": "flaky2", "settings": {"retries": 1, "retry_delay": 0}}, self.wf)
        with self.assertRaises(RuntimeError):
            comp.process(ItemResult("x"))

    def test_on_error_skip(self):
        cls = make_flaky(5)
        comp = cls({"id": "flaky3", "settings": {"on_error": "skip", "retry_delay": 0}}, self.wf)
        self.assertIsNone(comp.process(ItemResult("x")))

    def test_error_steps_branch(self):
        cls = make_flaky(5)
        comp = cls(
            {
                "id": "flaky4",
                "settings": {"retry_delay": 0},
                "error_steps": [{"type": "describer.FileInfoDescriber", "id": "err_describe"}],
            },
            self.wf,
        )
        out = comp.process(ItemResult("x"))
        # the error branch ran with an ERROR result describing the failure
        self.assertIsNotNone(out)
        self.assertEqual(out.input.metadata["type"], "ERROR")
        self.assertIn("boom", out.input.metadata["error"])
        self.assertEqual(out.input.metadata["component"], "flaky4")

    def test_retry_events_emitted(self):
        events = []
        self.wf.add_listener(lambda ev: events.append(ev["event"]))
        cls = make_flaky(1)
        comp = cls({"id": "flaky5", "settings": {"retries": 1, "retry_delay": 0}}, self.wf)
        comp.process(ItemResult("x"))
        self.assertIn("component_retry", events)
        self.assertIn("component_end", events)


class TestIterator(unittest.TestCase):
    def setUp(self):
        self.wf = Workflow({"id": "iter_resilience_wf", "type": "workflow.Workflow"})

    def _gloss_iterator(self, extra_settings, iter_id):
        return self.wf._make_step(
            {
                "type": "iterator.Iterator",
                "id": iter_id,
                "settings": extra_settings,
                "steps": [
                    {
                        "type": "translator.GlossaryTranslator",
                        "settings": {"glossary": {"cat": "Katze"}},
                    }
                ],
            },
            self.wf,
        )

    def test_workers_preserve_order(self):
        it = self._gloss_iterator({"workers": 4}, "iter_w")
        out = it.process(ListResult([ItemResult(f"cat {i}") for i in range(8)]))
        texts = [entry.value[0].value for entry in out.value]
        self.assertEqual(texts, [f"Katze {i}" for i in range(8)])

    def test_continue_on_error_records_failures(self):
        # the record evaluator raises on non-record input ("not json" fails json.loads)
        it = self.wf._make_step(
            {
                "type": "iterator.Iterator",
                "id": "iter_e",
                "settings": {"continue_on_error": True},
                "steps": [
                    {"type": "evaluator.RecordFieldEvaluator", "settings": {"expected": {"a": "1"}}}
                ],
            },
            self.wf,
        )
        out = it.process(ListResult([ItemResult({"a": "1"}), ItemResult("not json")]))
        self.assertEqual(len(out.value), 2)
        self.assertEqual(out.value[0].value[0].value["fields"]["a"], "correct")
        self.assertEqual(out.value[1].metadata.get("type"), "ERROR")

    def test_cache_replays_results(self):
        db = tempfile.mktemp(suffix=".db")
        events = []
        self.wf.add_listener(lambda ev: events.append(ev["event"]))
        it1 = self._gloss_iterator({"cache": db}, "iter_c1")
        out1 = it1.process(ListResult([ItemResult("the cat")]))
        self.assertEqual(out1.value[0].value[0].value, "the Katze")
        self.assertNotIn("iterator_cache_hit", events)

        # same config, new run: replayed from cache
        it2 = self._gloss_iterator({"cache": db}, "iter_c2")
        out2 = it2.process(ListResult([ItemResult("the cat")]))
        self.assertEqual(out2.value[0].value[0].value, "the Katze")
        self.assertIn("iterator_cache_hit", events)


if __name__ == "__main__":
    unittest.main()
