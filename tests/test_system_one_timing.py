"""What a System One call is timed for: the model answering, not the model loading.

A local model loads its weights the first time it is needed. That used to happen inside the first
call's clock, so the first result of a run recorded seconds of ``duration`` where every later one
recorded a fraction of a second. These tests use a provider with the real readout code and a
``load`` that takes a while, as weights do -- no model, no network.
"""

import threading
import time
import unittest

from chai.core import importClass
from chai.result import ItemResult
from chai.system_one import FAMILIES, LogitProvider, System1Model, make_question
from chai.usage import UsageLedger
from chai.workflow import Workflow

NOUL = make_question("noul", "The text mentions a painter.")
STATE = "The Night Café was painted by Van Gogh in 1888."
LOAD = 0.25  # seconds the stand-in for loading weights takes
QUICK = LOAD / 2  # anything under this did not wait for a load


class Slow(LogitProvider):
    """Fixed logits read out by the real code, behind a ``load`` that takes ``LOAD`` seconds once."""

    name = "slow"
    loads = 0  # how many times the weights were loaded, by any instance

    def __init__(self, model, settings):
        super().__init__(model, settings)
        self.loaded = False

    def load(self):
        if not self.loaded:
            time.sleep(LOAD)
            type(self).loads += 1
            self.loaded = True

    def logits(self, state, question):
        if not self.loaded:
            raise RuntimeError("asked before the weights were loaded")
        return [0.0, 2.0]


class Broken(Slow):
    """Fails to load the first time, as a download cut short does."""

    name = "broken"

    def load(self):
        if not getattr(self, "tried", False):
            self.tried = True
            raise OSError("the weights could not be read")
        super().load()


class Unloaded(LogitProvider):
    """A provider with nothing to load."""

    name = "unloaded"

    def logits(self, state, question):
        return [0.0, 2.0]


class TimingCase(unittest.TestCase):
    FAMILY = {"slow-test": Slow, "broken-test": Broken, "unloaded-test": Unloaded}

    def setUp(self):
        FAMILIES.update(self.FAMILY)
        Slow.loads = 0

    def tearDown(self):
        for prefix in self.FAMILY:
            FAMILIES.pop(prefix, None)


class TestCallTiming(TimingCase):
    def test_the_first_call_is_not_charged_for_loading(self):
        model = System1Model({"model": "slow-test/weights"})
        answers, first = model.decide(STATE, {"q": NOUL})
        self.assertTrue(answers["q"]["truth"])
        self.assertLess(first["duration"], QUICK)
        self.assertGreaterEqual(first["load_seconds"], LOAD - 0.01)
        self.assertLess(first["local_compute"]["seconds"], QUICK)  # the machine-time follows the duration
        _, second = model.decide(STATE, {"q": NOUL})
        self.assertLess(second["duration"], QUICK)
        self.assertNotIn("load_seconds", second)  # loading is reported once, by the call that did it
        self.assertEqual(Slow.loads, 1)

    def test_a_model_can_be_loaded_ahead_of_its_first_call(self):
        model = System1Model({"model": "slow-test/weights"})
        self.assertIsNone(model.load_seconds)  # building the model loads nothing
        self.assertEqual(Slow.loads, 0)
        self.assertGreaterEqual(model.load(), LOAD - 0.01)
        self.assertEqual(model.load(), 0.0)  # already in memory
        self.assertGreaterEqual(model.load_seconds, LOAD - 0.01)
        _, meta = model.decide(STATE, {"q": NOUL})
        self.assertNotIn("load_seconds", meta)
        self.assertLess(meta["duration"], QUICK)
        self.assertEqual(Slow.loads, 1)

    def test_nothing_to_load_reports_no_load(self):
        model = System1Model({"model": "unloaded-test/x"})
        _, meta = model.decide(STATE, {"q": NOUL})
        self.assertNotIn("load_seconds", meta)
        self.assertEqual(model.load_seconds, 0.0)

    def test_calls_that_arrive_while_the_model_loads_wait_outside_their_clocks(self):
        model = System1Model({"model": "slow-test/weights"})
        metas = []

        def call():
            metas.append(model.decide(STATE, {"q": NOUL})[1])

        threads = [threading.Thread(target=call) for _ in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        self.assertEqual(len(metas), 4)
        self.assertEqual(Slow.loads, 1)
        self.assertEqual(sum("load_seconds" in m for m in metas), 1)  # one call loaded it
        for meta in metas:
            self.assertLess(meta["duration"], QUICK)

    def test_a_load_that_fails_is_tried_again(self):
        model = System1Model({"model": "broken-test/weights"})
        with self.assertRaises(OSError):
            model.decide(STATE, {"q": NOUL})
        self.assertIsNone(model.load_seconds)
        _, meta = model.decide(STATE, {"q": NOUL})
        self.assertGreaterEqual(meta["load_seconds"], LOAD - 0.01)


class TestLoadTimeInARun(TimingCase):
    def classifier(self, wf):
        tree = {"type": "decision.DecisionClassifier", "id": "decide",
                "settings": {"model": "slow-test/weights", "question_type": "noul", "question": "The text mentions a painter."}}
        return importClass(tree["type"])(tree, wf)

    def test_a_result_records_the_load_beside_its_duration(self):
        wf = Workflow({"type": "Workflow", "id": "w"})
        step = self.classifier(wf)
        first = step.process(ItemResult(STATE, metadata={"type": "TEXT"}))
        self.assertEqual(first.value, ["true"])
        self.assertLess(first.metadata["duration"], QUICK)
        self.assertGreaterEqual(first.metadata["load_seconds"], LOAD - 0.01)
        second = step.process(ItemResult(STATE, metadata={"type": "TEXT"}))
        self.assertNotIn("load_seconds", second.metadata)
        self.assertLess(second.metadata["duration"], QUICK)

    def test_a_run_totals_loading_apart_from_answering(self):
        wf = Workflow({"type": "Workflow", "id": "w", "steps": [
            {"type": "iterator.Iterator", "id": "each", "steps": [
                {"type": "decision.DecisionClassifier", "id": "decide",
                 "settings": {"model": "slow-test/weights", "question_type": "noul", "question": "The text mentions a painter."}}]}]})
        from chai.result import ListResult

        texts = ListResult([ItemResult(STATE, metadata={"type": "TEXT"}) for _ in range(3)])
        wf.run(texts)
        usage = wf.usage_summary()
        self.assertEqual(usage["calls"], 3)
        self.assertGreaterEqual(usage["load_seconds"], LOAD - 0.01)
        self.assertLess(usage["seconds"], QUICK)  # three answers, none of them the load
        self.assertGreaterEqual(usage["by_component"]["decide"]["load_seconds"], LOAD - 0.01)
        self.assertGreaterEqual(wf.current_run.to_row()["metadata"]["usage"]["load_seconds"], LOAD - 0.01)
        # the model stays loaded: the next run of the same workflow loads nothing
        wf.run(texts)
        self.assertEqual(wf.usage_summary()["load_seconds"], 0.0)
        self.assertEqual(Slow.loads, 1)

    def test_a_gate_reports_the_load_too(self):
        wf = Workflow({"type": "Workflow", "id": "w"})
        tree = {"type": "decision.DecisionGate", "id": "gate",
                "settings": {"model": "slow-test/weights", "question": "The text mentions a painter."}}
        gate = importClass(tree["type"])(tree, wf)
        text = ItemResult(STATE, metadata={"type": "TEXT"})
        gate.process(text)
        record = text.extra["decisions"][0]
        self.assertGreaterEqual(record["load_seconds"], LOAD - 0.01)
        self.assertLess(record["duration"], QUICK)
        self.assertGreaterEqual(wf.usage_summary()["load_seconds"], LOAD - 0.01)


class TestLedger(unittest.TestCase):
    def test_load_seconds_are_totalled_apart_from_seconds(self):
        entry = UsageLedger.entry_from_metadata({"token_usage": None, "cost": 0.0, "duration": 0.2, "load_seconds": 6.5})
        self.assertEqual(entry["seconds"], 0.2)
        self.assertEqual(entry["load_seconds"], 6.5)
        self.assertEqual(UsageLedger.entry_from_metadata({"cost": 0.0, "duration": 0.2})["load_seconds"], 0.0)
        self.assertEqual(UsageLedger.entry_from_metadata({"cost": 0.0, "load_seconds": "soon"})["load_seconds"], 0.0)

        class C:
            id = name = "decide"

        ledger = UsageLedger()
        ledger.record(C(), {"cost": 0.0, "duration": 0.2, "load_seconds": 6.5, "model": "kev"})
        ledger.record(C(), {"cost": 0.0, "duration": 0.2, "model": "kev"})
        summary = ledger.summary()
        self.assertAlmostEqual(summary["seconds"], 0.4)
        self.assertEqual(summary["load_seconds"], 6.5)
        self.assertEqual(summary["by_model"]["kev"]["load_seconds"], 6.5)


if __name__ == "__main__":
    unittest.main()
