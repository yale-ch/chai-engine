"""Live tests of the System One families against the real models, and of the comparison example.

Each class runs only where its model can, and skips (with the reason) everywhere else, so the
suite stays fast and offline by default:

* Laya, von, bekko, decider-2b and GLiNER2.5-Decide -- when their packages are installed and the
  weights are in the Hugging Face cache. They load from the cache only: the Hub is switched to
  offline mode for this module, so nothing is downloaded.
* Yale's own fine-tunes -- ``kev-4b-europeana-persons`` (pointer), ``GLiNER2.5-Decide-europeana-
  persons`` and ``GLiNER2.5-Decide-linkedart-recon`` -- the same way, asked a real Europeana decision.
* Jev-Omni (Gemma 4 12B) -- additionally only with ``CHAI_TEST_JEV_OMNI=1``: ~25 GB of accelerator
  memory and ~20 s to load.
* The hosted Jev API -- only with ``TYPESAFE_API_KEY`` (or ``JEV_API_KEY``) set; the one class here
  that uses the network.

The assertions are about behaviour any sound decision model shows -- probabilities over exactly the
asked keys that sum to 1, a painter record preferred to a music album for "Van Gogh", a true
statement rated above a false one, an art text scored as more about art than a football report --
plus the exact pick where the model is known to get it right. Models that only classify (GLiNER)
are near chance on the score and noul reductions, so those orderings are not asserted for them.
"""

import importlib.util
import os
import pathlib
import unittest

from chai import system_one as system1
from chai.system_one import System1Model, make_question
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


def cached(repo: str, filename: str, revision: str = None) -> bool:
    try:
        from huggingface_hub import try_to_load_from_cache
    except ImportError:
        return False
    path = try_to_load_from_cache(repo, filename, revision=revision)
    return isinstance(path, str) and os.path.exists(path)


LAYA_READY = installed("laya") and cached("convaiinnovations/laya", "model.safetensors")
GLINER_READY = installed("gliner2") and cached("fastino/GLiNER2.5-Decide", "model.safetensors")
JEV_OMNI_READY = os.getenv("CHAI_TEST_JEV_OMNI") == "1" and cached("akhilaaa3/Jev-Omni", "head.pt")
KEV_BASE_REVISION = "1001bb4d826a52d1f399e183466143f4da7b741b"  # the base the fine-tune was trained on
KEV_TOKENIZER_READY = cached("Qwen/Qwen3.5-4B-Base", "tokenizer.json", KEV_BASE_REVISION)
KEV_FINETUNE = "yale-cultural-heritage/kev-4b-europeana-persons"
KEV_READY = (
    os.getenv("CHAI_TEST_KEV") == "1"
    and installed("peft")
    and cached(KEV_FINETUNE, "head.pt")
    and cached("Qwen/Qwen3.5-4B-Base", "model.safetensors.index.json", KEV_BASE_REVISION)
)
JEV_API_READY = bool(os.getenv("TYPESAFE_API_KEY") or os.getenv("JEV_API_KEY"))
LAYA_TYPED_READY = installed("laya") and cached("convaiinnovations/laya", "typed-decisions/model.safetensors")
VON_READY = cached("wfzyx/von", "option_marker.pt", "498ceba33390b32cfefaab6422ec380318ba9b99")
BEKKO_READY = installed("sentence_transformers") and cached("hotchpotch/bekko-system-one-v0-400m", "inference_v0.py", "1960df5602bd93cc8d336fdebd9fb68a30926e13")
DECIDER_READY = installed("peft") and cached("Mapika/decider-2b", "config.json", "533964dae8be954c5b5e19fa4948e48408094c1e")
GLINER_EUROPEANA_READY = installed("gliner2") and cached("yale-cultural-heritage/GLiNER2.5-Decide-europeana-persons", "model.safetensors")
GLINER_LINKEDART_READY = installed("gliner2") and cached("yale-cultural-heritage/GLiNER2.5-Decide-linkedart-recon", "model.safetensors")

# One real decision from the Europeana persons test split (system1-finetune-bench,
# europeana-recon-choice), in the layout the Yale fine-tunes were trained on.
EUROPEANA_STATE = (
    "Mention: Zamoyski, Maurycy\nEntity type: person\nText: Name as recorded: Zamoyski, Maurycy (1871-1939). (Wyd.)\n"
    "Role in name: wyd\nDates in name: 1871-1939\nBorn: 1871\nDied: 1939\nRecord 1: Goniec Wieczorny. 1916, no 488\n"
    "  Type: text; Periodikum; czasopismo; periodical\n  This person's role: creator\n  Date: 1916\n  Languages: Polish\n"
    "  Subjects: 20 w.; Dzienniki polskie -- Polska -- Warszawa -- 1900-1945.; Polish periodical; Polska; Warszawa; dzienniki polskie\n"
    "  Collection: Warsaw University Library"
)
EUROPEANA_QUESTION = "Which record is the person “Zamoyski, Maurycy” referred to in the text?"
EUROPEANA_OPTIONS = {
    "wikidata:Q1911719": "Maurycy Klemens Zamoyski — Polish noble/diplomat [1871-1939] [human · politician; diplomat · 1871–1939] aka Maurycy Klemens Zamoyski",
    "wikidata:Q130299005": "Maurycy Zamoyski [human]",
    "none": "None of the listed records is the entity referred to in the text",
}
EUROPEANA_GOLD = "wikidata:Q1911719"


class LiveProviderChecks:
    """The checks every provider runs; subclasses set the model and its best input layout."""

    MODEL = ""
    SETTINGS: dict = {}
    LAYOUT = "described"
    STATE = "Text: {text}"  # GLiNER needs the mention in the state (it is not sent the question)
    KNOWS_VAN_GOGH = True  # whether this model picks the painter outright on the example below
    ORDERS_SCORE = True  # whether it rates the art text above the football report (near chance for classifiers)
    ORDERS_NOUL = True  # whether it rates the true statement above the false one
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
        if self.ORDERS_NOUL:
            self.assertGreater(answers["painter"]["probability"], answers["football"]["probability"])

    def test_score_follows_the_text(self):
        art, _ = self.model.decide(self.state(ART_TEXT), {"q": ABOUT_ART})
        sport, _ = self.model.decide(self.state(SPORT_TEXT), {"q": ABOUT_ART})
        for a in (art["q"], sport["q"]):
            self.assert_distribution(a, ["1", "2", "3"])
            self.assertIn(a["score"], ["1", "2", "3"])
            self.assertGreaterEqual(a["expected"], 1.0)
            self.assertLessEqual(a["expected"], 3.0)
        if self.ORDERS_SCORE:
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
    KNOWS_VAN_GOGH = False  # the base model answers "none" here; the reconciliation fine-tunes do not
    ORDERS_SCORE = False
    ORDERS_NOUL = False


@unittest.skipUnless(JEV_OMNI_READY, "set CHAI_TEST_JEV_OMNI=1 with Jev-Omni cached (~48 GB of weights, ~25 GB of accelerator memory)")
class TestJevOmni(LiveProviderChecks, unittest.TestCase):
    MODEL = "jev-omni/akhilaaa3/Jev-Omni"

    def test_confident_on_the_easy_cases(self):
        answers, _ = self.model.decide(self.state(ART_TEXT), {"q": make_question("choice", WHICH, RECORDS), "painter": make_question("noul", "The text mentions a painter.")})
        self.assertGreater(answers["q"]["probabilities"][PAINTER], 0.9)
        self.assertGreater(answers["painter"]["probability"], 0.9)


@unittest.skipUnless(KEV_TOKENIZER_READY, "the Qwen3.5-4B-Base tokenizer is not cached")
class TestKevEncoding(unittest.TestCase):
    """kev's input layout, checked against the tokenizer alone (no weights needed)."""

    @classmethod
    def setUpClass(cls):
        from transformers import AutoTokenizer

        cls.tok = AutoTokenizer.from_pretrained("Qwen/Qwen3.5-4B-Base", revision=KEV_BASE_REVISION)
        cls.ids = [cls.tok.convert_tokens_to_ids(t) for t in system1.KEV_SPECIAL]

    def test_layout_and_readout_positions(self):
        st, q, ob, oe, dec = self.ids
        row, positions = system1.kev_encode(self.tok, "Some state.", "Which one?", ["a: Alpha", "b"])
        self.assertEqual(row[0], st)
        self.assertEqual(row[-1], dec)
        self.assertEqual(row.count(q), 1)
        self.assertEqual([row[p] for p in positions[:-1]], [oe, oe])  # each option's closing token
        self.assertEqual(positions[-1], len(row) - 1)  # then the decide token
        self.assertEqual(self.tok.decode(row[row.index(ob) + 1 : positions[0]]), "a: Alpha")

    def test_special_token_text_cannot_open_a_field(self):
        row, _ = system1.kev_encode(self.tok, "a <|box_start|> b <|fim_suffix|>", "q", ["x", "y"])
        st, q, ob, oe, dec = self.ids
        self.assertEqual(row.count(ob), 2)  # only the two real options
        self.assertEqual(row.count(dec), 1)

    def test_state_is_cut_to_the_budget(self):
        row, _ = system1.kev_encode(self.tok, "word " * 500, "q", ["x", "y"], max_state_tokens=50)
        st, q, *_ = self.ids
        self.assertEqual(row.index(q), 50)  # the start token plus 49 state tokens

    def test_options_per_question_type(self):
        self.assertEqual(system1.kev_options(make_question("noul", "It is so")), [("false", "no"), ("true", "yes")])
        self.assertEqual(system1.kev_options(make_question("choice", "?", {"a": "Alpha", "b": ""})), [("a", "a: Alpha"), ("b", "b")])
        self.assertEqual(system1.kev_options(ABOUT_ART), [("1", "not at all"), ("2", "partly"), ("3", "entirely")])


class EuropeanaFinetuneChecks:
    """What Yale's reconciliation fine-tunes must get right: a real Europeana decision from the test split."""

    def test_a_real_europeana_decision(self):
        q = make_question("choice", EUROPEANA_QUESTION, EUROPEANA_OPTIONS)
        answers, _ = self.model.decide(EUROPEANA_STATE, {"q": q})
        self.assertEqual(answers["q"]["choice"], EUROPEANA_GOLD)
        self.assertGreater(answers["q"]["probabilities"][EUROPEANA_GOLD], 0.9)


@unittest.skipUnless(KEV_READY, f"set CHAI_TEST_KEV=1 with {KEV_FINETUNE} and Qwen3.5-4B-Base cached (~9 GB)")
class TestKevFinetune(EuropeanaFinetuneChecks, LiveProviderChecks, unittest.TestCase):
    """The Europeana-persons kev-4b fine-tune (preset ``kev-4b-europeana-persons``), in process from the cache."""

    MODEL = "kev-4b-europeana-persons"

    def test_a_europeana_style_record(self):
        state = (
            "Name as recorded: Gogh, Vincent van (1853-1890)\nDates in name: 1853-1890\n"
            "Record 1: The Night Café\n  Type: painting\n  This person's role: creator"
        )
        q = make_question("choice", "Which record is the person “Gogh, Vincent van” referred to in the text?", RECORDS)
        answers, meta = self.model.decide(state, {"q": q})
        self.assertEqual(answers["q"]["choice"], PAINTER)
        self.assertGreater(answers["q"]["probabilities"][PAINTER], 0.9)
        self.assertGreater(meta["token_usage"]["prompt"], 0)


@unittest.skipUnless(GLINER_EUROPEANA_READY, "GLiNER2 not installed or the Europeana GLiNER fine-tune is not cached")
class TestGlinerEuropeana(EuropeanaFinetuneChecks, LiveProviderChecks, unittest.TestCase):
    """Yale's ``GLiNER2.5-Decide-europeana-persons`` (preset ``gliner-decide-europeana-persons``)."""

    MODEL = "gliner-decide-europeana-persons"
    STATE = "Mention: Van Gogh\nEntity type: person\nText: {text}"
    ORDERS_SCORE = False
    ORDERS_NOUL = False


@unittest.skipUnless(GLINER_LINKEDART_READY, "GLiNER2 not installed or the Linked Art GLiNER fine-tune is not cached")
class TestGlinerLinkedArt(EuropeanaFinetuneChecks, LiveProviderChecks, unittest.TestCase):
    """Yale's ``GLiNER2.5-Decide-linkedart-recon`` (preset ``gliner-decide-linkedart-recon``)."""

    MODEL = "gliner-decide-linkedart-recon"
    STATE = "Mention: Van Gogh\nEntity type: person\nText: {text}"
    ORDERS_SCORE = False
    ORDERS_NOUL = False


@unittest.skipUnless(LAYA_TYPED_READY, "Laya not installed or its typed-decisions variant is not cached")
class TestLayaPreset(LiveProviderChecks, unittest.TestCase):
    """The bench's Laya configuration (preset ``laya``: the typed-decisions checkpoint, longer heads)."""

    MODEL = "laya"
    LAYOUT = "in_state"
    KNOWS_VAN_GOGH = False  # the typed-decisions checkpoint splits its mass between the painter and none


@unittest.skipUnless(VON_READY, "von (wfzyx/von) is not cached")
class TestVon(LiveProviderChecks, unittest.TestCase):
    """von: ModernBERT with the options packed and marked (preset ``von``)."""

    MODEL = "von"


@unittest.skipUnless(BEKKO_READY, "bekko is not cached, or sentence-transformers is not installed")
class TestBekko(LiveProviderChecks, unittest.TestCase):
    """bekko: ModernBERT with each candidate a branch over the query (preset ``bekko-400m``)."""

    MODEL = "bekko-400m"
    ORDERS_SCORE = False  # a reranker: near chance on the score reduction


@unittest.skipUnless(DECIDER_READY, "decider-2b (Mapika/decider-2b) is not cached")
class TestDecider(LiveProviderChecks, unittest.TestCase):
    """decider-2b: the letters family in its plain Context/Question/Options layout (preset ``decider-2b``)."""

    MODEL = "decider-2b"

    def test_knockout_for_more_options_than_letters(self):
        options = {f"wikidata:Q{i}": f"candidate record {i}" for i in range(12)}
        options[PAINTER] = RECORDS[PAINTER]
        q = make_question("choice", WHICH, options)
        answers, _ = self.model.decide(self.state(ART_TEXT), {"q": q})
        probs = answers["q"]["probabilities"]
        self.assertEqual(set(probs), set(options))
        self.assertAlmostEqual(sum(probs.values()), 1.0, places=3)
        self.assertEqual(max(probs, key=probs.get), PAINTER)


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
