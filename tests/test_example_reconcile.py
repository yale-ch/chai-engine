"""Offline run of examples/reconcile_wikidata.py: NER (LLM) -> Wikidata candidates (plain Python)
-> LLM selection. The two model calls answer through LiteLLM's local mock replies and the Wikidata
lookup is a stand-in, so the pipeline's wiring is exercised end to end without a network."""

import importlib.util
import json
import pathlib
import unittest

EXAMPLE = pathlib.Path(__file__).resolve().parents[1] / "examples" / "reconcile_wikidata.py"
spec = importlib.util.spec_from_file_location("reconcile_wikidata", EXAMPLE)
example = importlib.util.module_from_spec(spec)
spec.loader.exec_module(example)

MODEL = "openai/gpt-4o-mini"  # any LiteLLM id; replies come from mock_response, never the network
TEXT = "The Night Café was painted by Vincent van Gogh in Arles in 1888."

NER_REPLY = json.dumps(
    {
        "entities": [
            {"name": "Vincent van Gogh", "type": "Person", "mentions": ["Vincent van Gogh"], "gloss": "painter of The Night Café"},
            {"name": "Arles", "type": "place", "mentions": ["Arles"]},
            {"name": "", "type": "work"},  # dropped: no name
        ]
    }
)
SELECT_REPLY = json.dumps(
    {"answers": {"q": {"choice": "wikidata:Q5582", "probabilities": {"wikidata:Q5582": 0.95, "wikidata:Q17029510": 0.03, "none": 0.02}, "confidence": 0.95}}}
)
VAN_GOGH = [
    {"key": "wikidata:Q5582", "label": "Vincent van Gogh", "summary": "Vincent van Gogh — Dutch painter (1853–1890)"},
    {"key": "wikidata:Q17029510", "label": "Van Gogh", "summary": "Van Gogh — music album"},
]


def fake_lookup(name, limit=8):
    return VAN_GOGH[:limit] if name == "Vincent van Gogh" else []


class TestReconcileExample(unittest.TestCase):
    def test_pipeline_offline(self):
        out = example.reconcile(
            TEXT,
            model=MODEL,
            lookup=fake_lookup,
            ner_settings={"litellm_params": {"mock_response": NER_REPLY}},
            select_settings={"litellm_params": {"mock_response": SELECT_REPLY}},
        )
        ents = out["entities"]
        self.assertEqual([e["name"] for e in ents], ["Vincent van Gogh", "Arles"])
        self.assertEqual(ents[0]["type"], "person")  # normalized
        self.assertEqual(ents[0]["choice"], "wikidata:Q5582")
        self.assertEqual(ents[0]["record"]["label"], "Vincent van Gogh")
        self.assertAlmostEqual(ents[0]["confidence"], 0.95)
        self.assertEqual(set(ents[0]["probabilities"]), {"wikidata:Q5582", "wikidata:Q17029510", "none"})
        # Wikidata found nothing for Arles: a null decision, and no model call was spent on it
        self.assertEqual(ents[1]["choice"], "none")
        self.assertIsNone(ents[1]["record"])
        self.assertEqual(out["usage"]["ner"]["calls"], 1)
        self.assertEqual(out["usage"]["selection"]["calls"], 1)
        self.assertGreaterEqual(out["usage"]["cost"], 0.0)

    def test_ner_workflow_settings_per_model(self):
        wf = example.ner_workflow("openai/gpt-4o-mini")
        comp = wf.get_component_by_id("ner")
        self.assertEqual(comp.__class__.__name__, "LLMExtractor")
        self.assertEqual(comp.base_kwargs["response_format"], {"type": "json_object"})
        self.assertIn("{text_input_0}", comp.prompt_text)

    def test_selection_state_and_question(self):
        wf = example.select_workflow(MODEL, litellm_params={"mock_response": SELECT_REPLY})
        from chai.result import ItemResult

        comp = wf.get_component_by_id("select")
        rec = {"name": "Vincent van Gogh", "type": "person", "gloss": "painter", "context": TEXT, "candidates": VAN_GOGH}
        out = comp.process(ItemResult(rec, metadata={"type": "DATA"}))
        decision = out.metadata["decision"]
        self.assertEqual(decision["question"]["instructions"], "Which Wikidata record is the person 'Vincent van Gogh' (painter) referring to in the text?")
        self.assertTrue(decision["state"].startswith(f"Text: {TEXT}"))
        self.assertIn("Candidate Wikidata records for 'Vincent van Gogh':", decision["state"])
        self.assertIn("- wikidata:Q5582: Vincent van Gogh — Dutch painter (1853–1890)", decision["state"])
        self.assertEqual(decision["question"]["options"][-1]["key"], "none")

    def test_wikidata_response_parsing(self):
        calls = []

        class FakeResponse:
            def raise_for_status(self):
                pass

            def json(self):
                return {
                    "search": [
                        {"id": "Q5582", "label": "Vincent van Gogh", "description": "Dutch painter (1853–1890)", "aliases": ["Van Gogh", "Vincent van Gogh"], "concepturi": "http://www.wikidata.org/entity/Q5582"},
                        {"id": "Q17029510", "label": "Van Gogh"},
                    ]
                }

        class FakeSession:
            def get(self, url, params=None, headers=None, timeout=None):
                calls.append({"url": url, "params": params, "headers": headers})
                return FakeResponse()

        cands = example.wikidata_candidates("Van Gogh", limit=3, session=FakeSession())
        self.assertEqual(calls[0]["url"], example.WIKIDATA_API)
        self.assertEqual(calls[0]["params"]["search"], "Van Gogh")
        self.assertEqual(calls[0]["params"]["limit"], 3)
        self.assertIn("User-Agent", calls[0]["headers"])
        self.assertEqual([c["key"] for c in cands], ["wikidata:Q5582", "wikidata:Q17029510"])
        self.assertEqual(cands[0]["summary"], "Vincent van Gogh — Dutch painter (1853–1890) (aka Van Gogh)")
        self.assertEqual(cands[1]["summary"], "Van Gogh")
        self.assertEqual(cands[1]["url"], "https://www.wikidata.org/wiki/Q17029510")


if __name__ == "__main__":
    unittest.main()
