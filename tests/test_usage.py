"""Tests for run-wide usage and cost accounting (chai.usage.UsageLedger via the Workflow)."""

import unittest

from chai.result import ItemResult
from chai.usage import UsageLedger
from chai.workflow import Workflow


def llm_step(cid, reply='{"ok": true}', model="openai/gpt-4o-mini", **settings):
    return {
        "type": "extractor.LLMExtractor",
        "id": cid,
        "settings": {"model": model, "prompt": "Do {text_input_0}", "litellm_params": {"mock_response": reply}, **settings},
    }


class TestLedgerEntries(unittest.TestCase):
    def test_no_usage_no_entry(self):
        self.assertIsNone(UsageLedger.entry_from_metadata({"type": "TEXT"}))
        self.assertIsNone(UsageLedger.entry_from_metadata(None))

    def test_entry_fields_and_unknowns(self):
        entry = UsageLedger.entry_from_metadata(
            {"token_usage": {"prompt": 10, "result": 5, "thinking": -1, "total": None}, "cost": None, "duration": "0.5"}
        )
        self.assertEqual(entry["prompt_tokens"], 10)
        self.assertEqual(entry["completion_tokens"], 5)
        self.assertEqual(entry["thinking_tokens"], 0)  # -1 means unknown
        self.assertEqual(entry["total_tokens"], 15)  # summed when the backend gave no total
        self.assertEqual(entry["cost"], 0.0)
        self.assertEqual(entry["unpriced_calls"], 1)
        self.assertEqual(entry["seconds"], 0.5)

    def test_totals_by_component_and_model(self):
        ledger = UsageLedger()

        class C:
            def __init__(self, cid):
                self.id = cid
                self.name = cid

        ledger.record(C("a"), {"token_usage": {"prompt": 10, "result": 2, "total": 12}, "cost": 0.001, "model": "m1"})
        ledger.record(C("a"), {"token_usage": {"prompt": 10, "result": 2, "total": 12}, "cost": 0.001, "model": "m1"})
        ledger.record(C("b"), {"token_usage": {"prompt": 1, "result": 1, "total": 2}, "cost": None, "model": "m2"})
        s = ledger.summary()
        self.assertEqual(s["calls"], 3)
        self.assertEqual(s["total_tokens"], 26)
        self.assertAlmostEqual(s["cost"], 0.002)
        self.assertEqual(s["unpriced_calls"], 1)
        self.assertEqual(s["by_component"]["a"]["calls"], 2)
        self.assertEqual(s["by_component"]["b"]["unpriced_calls"], 1)
        self.assertEqual(set(s["by_model"]), {"m1", "m2"})
        ledger.reset()
        self.assertEqual(ledger.summary()["calls"], 0)


class TestWorkflowAccounting(unittest.TestCase):
    def test_run_records_every_call_and_reports_on_the_end_event(self):
        events = []
        wf = Workflow({"type": "Workflow", "id": "wf", "steps": [llm_step("first"), llm_step("second", model="gemini/gemini-3.1-flash-lite")]})
        wf.add_listener(lambda ev: events.append(ev))
        wf.run("text")
        summary = wf.usage_summary()
        self.assertEqual(summary["calls"], 2)
        self.assertEqual(set(summary["by_component"]), {"first", "second"})
        self.assertGreater(summary["cost"], 0)  # the Gemini call is priced
        self.assertEqual(summary["unpriced_calls"], 0)  # gpt-4o-mini is priced too
        end = [e for e in events if e["event"] == "component_end" and e["component_id"] == "wf"][-1]
        self.assertEqual(end["usage"]["calls"], 2)
        # a second run starts from zero
        wf.run("again")
        self.assertEqual(wf.usage_summary()["calls"], 2)

    def test_per_result_cost_is_recorded(self):
        wf = Workflow({"type": "Workflow", "id": "wf", "steps": [llm_step("only", model="gemini/gemini-3.1-flash-lite")]})
        out = wf.run("x").value[0]
        self.assertIn("cost", out.metadata)
        self.assertAlmostEqual(out.metadata["cost"], wf.usage_summary()["cost"])

    def test_iterator_workers_are_counted_once_each(self):
        wf = Workflow(
            {
                "type": "Workflow",
                "id": "wf",
                "steps": [
                    {
                        "type": "provider.StaticProvider",
                        "id": "items",
                        "input": "",
                        "settings": {"values": [f"item {i}" for i in range(8)]},
                        "steps": [{"type": "iterator.Iterator", "id": "it", "settings": {"workers": 4}, "steps": [llm_step("llm")]}],
                    }
                ],
            }
        )
        wf.run()
        summary = wf.usage_summary()
        self.assertEqual(summary["calls"], 8)
        self.assertEqual(summary["by_component"]["llm"]["calls"], 8)
        self.assertEqual(len(summary["by_model"]), 1)


if __name__ == "__main__":
    unittest.main()
