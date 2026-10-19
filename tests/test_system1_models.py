"""Live tests of the System One providers against the real models, and of the comparison example.

Each class runs only where its model can, and skips (with the reason) everywhere else, so the
suite stays fast and offline by default:

* Laya and GLiNER2.5-Decide -- when ``laya`` / ``gliner2`` are installed and the weights are in
  the Hugging Face cache. They load from the cache only: the Hub is switched to offline mode for
  this module, so nothing is downloaded.
* Jev-Omni (Gemma 4 12B) -- additionally only with ``CHAI_TEST_JEV_OMNI=1``: it needs ~25 GB of
  accelerator memory and ~20 s to load.
* The hosted Jev API -- only with ``TYPESAFE_API_KEY`` (or ``JEV_API_KEY``) set; it is the one
  class here that uses the network.

The assertions are about behaviour any sound decision model shows -- probabilities over exactly
the asked keys that sum to 1, a painter record preferred to a music album for "Van Gogh", a true
statement rated above a false one, an art text scored as more about art than a football report --
plus the exact pick where the model is known to get it right.
"""

import importlib.util
import os
import pathlib
import unittest

from chai.ai.system1 import System1Model, make_question
from chai.core import importClass
from chai.result import ItemResult, ListResult
from chai.workflow import Workflow

ART_TEXT = (
    "The Night Café was painted by Vincent van Gogh in Arles in 1888. Van Gogh wrote to his brother "
    "Theo about it."
)
SPORT_TEXT = "Liverpool beat Everton 2-1 at Anfield on Saturday; Salah scored the winner in the 89th minute."
PAINTER, ALBUM = "wikidata:Q5582", "wikidata:Q17029510"
RECORDS = {
    PAINTER: "Vincent van Gogh, Dutch painter (1853-1890)",
    ALBUM: "Van Gogh, music album",
    "none": "None of the listed records is the entity referred to in the text",
}
WHICH = "Which candidate record is the mention 'Van Gogh' referring to in the text? Choose none if no listed record fits."
ABOUT_ART = make_question("score", "How much is the text about art?", {"1": "not at all", "2": "partly", "3": "entirely"})

_saved_offline = None


def setUpModule():
    """Local weights come from the Hugging Face cache only; never download during tests."""
    global _saved_offline
    try:
        import huggingface_hub.constants as hf
    except ImportError:
        return
    _saved_offline = hf.HF_HUB_OFFLINE
    hf.HF_HUB_OFFLINE = True


def tearDownModule():
    if _saved_offline is not None:
        import huggingface_hub.constants as hf

        hf.HF_HUB_OFFLINE = _saved_offline


def installed(package: str) -> bool:
    return importlib.util.find_spec(package) is not None


def cached(repo: str, filename: str) -> bool:
    try:
        from huggingface_hub import try_to_load_from_cache
    except ImportError:
        return False
    path = try_to_load_from_cache(repo, filename)
    return isinstance(path, str) and os.path.exists(path)


LAYA_READY = installed("laya") and cached("convaiinnovations/laya", "model.safetensors")
GLINER_READY = installed("gliner2") and cached("fastino/GLiNER2.5-Decide", "model.safetensors")
JEV_OMNI_READY = os.getenv("CHAI_TEST_JEV_OMNI") == "1" and cached("akhilaaa3/Jev-Omni", "head.pt")
JEV_API_READY = bool(os.getenv("TYPESAFE_API_KEY") or os.getenv("JEV_API_KEY"))


class LiveProviderChecks:
    """The checks every provider runs; subclasses set the model and its best input layout."""

    MODEL = ""
    SETTINGS: dict = {}
    LAYOUT = "described"
    STATE = "Text: {text}"  # GLiNER needs the mention in the state (it is not sent the question)
    KNOWS_VAN_GOGH = True  # whether this model picks the painter outright on the example below
    FREE = True  # local models cost 0; the Jev API is unpriced

    @classmethod
    def setUpClass(cls):
        cls.model = System1Model({"model": cls.MODEL, **cls.SETTINGS})

    def state(self, text):
        return self.STATE.replace("{text}", text)

    def assert_distribution(self, answer, keys):
        self.assertEqual(set(answer["probabilities"]), set(keys))
        self.assertAlmostEqual(sum(answer["probabilities"].values()), 1.0, places=3)
        self.assertTrue(all(0.0 <= p <= 1.0 for p in answer["probabilities"].values()))
        self.assertGreaterEqual(answer["confidence"], 0.0)
        self.assertLessEqual(answer["confidence"], 1.0)

    def test_choice(self):
        answers, meta = self.model.decide(self.state(ART_TEXT), {"q": make_question("choice", WHICH, RECORDS)}, self.LAYOUT)
        a = answers["q"]
        self.assert_distribution(a, RECORDS)
        self.assertIn(a["choice"], RECORDS)
        self.assertGreater(a["probabilities"][PAINTER], a["probabilities"][ALBUM])
        if self.KNOWS_VAN_GOGH:
            self.assertEqual(a["choice"], PAINTER)
        self.assertEqual(meta["cost"], 0.0 if self.FREE else meta["cost"])
        self.assertGreater(meta["duration"], 0.0)

    def test_noul_rates_a_true_statement_above_a_false_one(self):
        questions = {"painter": make_question("noul", "The text mentions a painter."), "football": make_question("noul", "The text is about a football match.")}
        answers, _ = self.model.decide(self.state(ART_TEXT), questions)
        for a in answers.values():
            self.assert_distribution(a, ["true", "false"])
            self.assertEqual(a["truth"], a["probability"] >= 0.5)
        self.assertGreater(answers["painter"]["probability"], answers["football"]["probability"])

    def test_score_follows_the_text(self):
        art, _ = self.model.decide(self.state(ART_TEXT), {"q": ABOUT_ART})
        sport, _ = self.model.decide(self.state(SPORT_TEXT), {"q": ABOUT_ART})
        for a in (art["q"], sport["q"]):
            self.assert_distribution(a, ["1", "2", "3"])
            self.assertIn(a["score"], ["1", "2", "3"])
            self.assertGreaterEqual(a["expected"], 1.0)
            self.assertLessEqual(a["expected"], 3.0)
        self.assertGreater(art["q"]["expected"], sport["q"]["expected"])

    def test_decision_classifier_in_a_workflow(self):
        wf = Workflow(
            {
                "type": "Workflow",
                "id": "live_wf",
                "steps": [
                    {
                        "type": "iterator.Iterator",
                        "id": "each",
                        "steps": [
                            {
                                "type": "decision.DecisionClassifier",
                                "id": "select",
                                "register_on": ["each"],
                                "settings": {
                                    **self.SETTINGS,
                                    "model": self.MODEL,
                                    "question": "Which candidate record is the mention '{name}' referring to in the text?",
                                    "state": self.STATE.replace("{text}", "{context}"),
                                    "options_from": "candidates",
                                    "option_text": "{summary}",
                                    "none_option": RECORDS["none"],
                                    "layout": self.LAYOUT,
                                    "options_heading": "Candidate records for the mention '{name}':",
                                },
                            }
                        ],
                    }
                ],
            }
        )
        record = {
            "name": "Van Gogh",
            "context": ART_TEXT,
            "candidates": [{"key": PAINTER, "summary": RECORDS[PAINTER]}, {"key": ALBUM, "summary": RECORDS[ALBUM]}],
        }
        res = wf.run(ListResult([ItemResult(record, metadata={"type": "DATA"})]))
        label = res.value[0].value[0].value[0]
        self.assertIn(label.value[0], [PAINTER, ALBUM, "none"])
        if self.KNOWS_VAN_GOGH:
            self.assertEqual(label.value, [PAINTER])
        self.assertTrue(label.metadata["model"])
        usage = wf.usage_summary()
        self.assertEqual(usage["calls"], 1)
        self.assertEqual(usage["unpriced_calls"], 0 if self.FREE else 1)


@unittest.skipUnless(LAYA_READY, "Laya not installed (pip install laya) or its weights are not cached")
class TestLaya(LiveProviderChecks, unittest.TestCase):
    MODEL = "laya/convaiinnovations/laya"
    LAYOUT = "in_state"  # Laya's best layout: candidates listed in the state, bare keys as options

    def test_all_questions_in_one_call(self):
        questions = {"q": make_question("choice", WHICH, RECORDS), "painter": make_question("noul", "The text mentions a painter."), "art": ABOUT_ART}
        answers, meta = self.model.decide(self.state(ART_TEXT), questions, "in_state")
        self.assertEqual(set(answers), {"q", "painter", "art"})
        self.assertEqual(answers["q"]["choice"], PAINTER)
        self.assertTrue(answers["painter"]["truth"])
        self.assertGreater(meta["token_usage"]["prompt"], 0)  # Laya reports the tokens it read


@unittest.skipUnless(GLINER_READY, "GLiNER2 not installed (pip install 'gliner2[local]') or GLiNER2.5-Decide is not cached")
class TestGliner(LiveProviderChecks, unittest.TestCase):
    MODEL = "gliner/fastino/GLiNER2.5-Decide"
    SETTINGS = {"device": "cpu"} if os.getenv("CHAI_TEST_DEVICE") == "cpu" else {}
    STATE = "Mention: Van Gogh\nEntity type: person\nText: {text}"
    KNOWS_VAN_GOGH = False  # the base model answers "none" here; the reconciliation fine-tune does not


@unittest.skipUnless(JEV_OMNI_READY, "set CHAI_TEST_JEV_OMNI=1 with Jev-Omni cached (~48 GB of weights, ~25 GB of accelerator memory)")
class TestJevOmni(LiveProviderChecks, unittest.TestCase):
    MODEL = "jev-omni/akhilaaa3/Jev-Omni"

    def test_confident_on_the_easy_cases(self):
        answers, _ = self.model.decide(self.state(ART_TEXT), {"q": make_question("choice", WHICH, RECORDS), "painter": make_question("noul", "The text mentions a painter.")})
        self.assertGreater(answers["q"]["probabilities"][PAINTER], 0.9)
        self.assertGreater(answers["painter"]["probability"], 0.9)


@unittest.skipUnless(JEV_API_READY, "set TYPESAFE_API_KEY (or JEV_API_KEY) to test the hosted Jev API")
class TestJevApi(LiveProviderChecks, unittest.TestCase):
    MODEL = "jev/jev-latest"
    LAYOUT = "in_state"
    FREE = False


@unittest.skipUnless(LAYA_READY, "the example's offline check runs Laya")
class TestComparisonExample(unittest.TestCase):
    """examples/system1_models.py end to end with one local model."""

    @classmethod
    def setUpClass(cls):
        path = pathlib.Path(__file__).resolve().parents[1] / "examples" / "system1_models.py"
        spec = importlib.util.spec_from_file_location("system1_models", path)
        cls.example = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cls.example)

    def test_runs_and_reports(self):
        rows, triage_rows, usage = self.example.run(["laya"])
        self.assertEqual([r["name"] for r in rows], [m["name"] for m in self.example.MENTIONS])
        for row in rows:
            keys = {c["key"] for c in next(m for m in self.example.MENTIONS if m["name"] == row["name"])["candidates"]} | {"none"}
            self.assertIn(row["picks"]["laya"]["choice"], keys)
        self.assertTrue(rows[0]["picks"]["laya"]["correct"])  # Van Gogh
        art, sport = (t["answers"]["laya"] for t in triage_rows)
        self.assertGreater(art["painter"]["probability"], sport["painter"]["probability"])
        self.assertGreater(art["about_art"]["expected"], sport["about_art"]["expected"])
        self.assertEqual(usage["calls"], len(rows) + len(triage_rows))
        self.assertEqual(usage["cost"], 0.0)
        self.assertEqual(usage["unpriced_calls"], 0)

    def test_availability_reasons(self):
        ok, why = self.example.availability("nope")
        self.assertFalse(ok)
        if not JEV_API_READY:
            ok, why = self.example.availability("jev")
            self.assertFalse(ok)
            self.assertIn("TYPESAFE_API_KEY", why)


if __name__ == "__main__":
    unittest.main()
