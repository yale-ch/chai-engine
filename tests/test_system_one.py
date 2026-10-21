"""Offline tests of ``chai.system_one``: the registry and presets, the logit readout every decoder and
encoder family shares, and the exact prompt each family builds -- checked against the layouts the
bench copied from the models' own code -- without loading any weights.

The encoding tests need only the Qwen3.5-4B-Base tokenizer, which the kev fine-tune's tests cache;
they skip when it is absent.
"""

import json
import os
import unittest
from types import SimpleNamespace

from chai import system_one
from chai.system_one import (
    FAMILIES,
    PRESETS,
    CandidateProvider,
    LetterProvider,
    LogitProvider,
    NLIProvider,
    PointerProvider,
    SlotHeadProvider,
    System1Model,
    make_provider,
    make_question,
    temperatures,
)
from chai.system_one.encoders import BekkoProvider, VonProvider
from chai.system_one.gliner import GlinerProvider

CHOICE = make_question("choice", "Which record?", {"wikidata:Q5582": "Vincent van Gogh, Dutch painter", "none": "None of these"})
SCORE = make_question("score", "How much is the text about art?", {"1": "not at all", "2": "partly", "3": "entirely"})
NOUL = make_question("noul", "The text mentions a painter.")
STATE = "The Night Café was painted by Van Gogh in 1888."


def cached(repo, filename, revision=None):
    try:
        from huggingface_hub import try_to_load_from_cache
    except ImportError:
        return False
    path = try_to_load_from_cache(repo, filename, revision=revision)
    return isinstance(path, str) and os.path.exists(path)


QWEN_REVISION = "1001bb4d826a52d1f399e183466143f4da7b741b"
TOKENIZER_READY = cached("Qwen/Qwen3.5-4B-Base", "tokenizer.json", QWEN_REVISION)


class TestRegistry(unittest.TestCase):
    def test_every_preset_builds_without_loading(self):
        for name, preset in PRESETS.items():
            self.assertIn(preset["family"], FAMILIES, name)
            self.assertTrue(preset.get("checkpoint"), name)
            provider = make_provider(name, {})
            self.assertIsInstance(provider, FAMILIES[preset["family"]], name)
            self.assertFalse(getattr(provider, "rt", None), f"{name} loaded weights at construction")

    def test_presets_carry_the_bench_values(self):
        kev = make_provider("kev-4b", {})
        self.assertEqual((kev.checkpoint, kev.cfg["revision"][:8], kev.cfg["head"]["file"]), ("Qwen/Qwen3.5-4B-Base", "1001bb4d", "head.pt"))
        plumb = make_provider("plumb-4b", {})
        self.assertEqual((plumb.style, plumb.max_options, plumb.temperature["choice"]), ("jevk5", 16, 2.07))
        decider = make_provider("decider-2b", {})
        self.assertEqual((decider.style, decider.max_options, decider.noul_slots), ("decider", 10, ("false", "true")))
        self.assertEqual(decider.temperature, {"choice": 1.164, "noul": 1.624, "score": 1.124})
        jev = make_provider("jev-9b", {})
        self.assertEqual(jev.cfg["adapter"]["subfolder"], "adapter")
        laya = make_provider("laya", {})
        self.assertEqual((laya.subfolder, laya.lengths), ("typed-decisions", {"max_len": 1536, "head_max_len": 512}))
        self.assertEqual(make_provider("gliner-decide-europeana-persons", {}).model, "yale-cultural-heritage/GLiNER2.5-Decide-europeana-persons")

    def test_settings_override_a_preset(self):
        p = make_provider("plumb-4b", {"temperature": 1.0, "device": "cpu", "max_options": 4})
        self.assertEqual((p.temperature["choice"], p.device, p.max_options), (1.0, "cpu", 4))

    def test_prefix_ids_and_fallback(self):
        self.assertIsInstance(make_provider("letters/crh225/plumb-4b", {"style": "jevk5"}), LetterProvider)
        self.assertIsInstance(make_provider("candidate/Qwen/Qwen3.5-9B", {}), CandidateProvider)
        self.assertIsInstance(make_provider("slots/autotrust/JEV-9B", {}), SlotHeadProvider)
        self.assertIsInstance(make_provider("nli/AlexWortega/openjev", {"subfolder": "qwen3.5-4b-nli-v5"}), NLIProvider)
        self.assertIsInstance(make_provider("von/wfzyx/von", {}), VonProvider)
        self.assertIsInstance(make_provider("bekko/hotchpotch/bekko-system-one-v0-400m", {}), BekkoProvider)
        self.assertEqual(make_provider("openai/gpt-4o-mini", {}).name, "generative")
        with self.assertRaises(ValueError):
            make_provider("letters/x", {"style": "roman"})
        with self.assertRaises(ValueError):
            make_provider("", {})

    def test_temperatures(self):
        self.assertEqual(temperatures(None), {"choice": 1.0, "score": 1.0, "noul": 1.0})
        self.assertEqual(temperatures(2.0), {"choice": 2.0, "score": 2.0, "noul": 2.0})
        self.assertEqual(temperatures({"choice": 1.5}), {"choice": 1.5, "score": 1.0, "noul": 1.0})


class Fixed(LogitProvider):
    """A logit provider with logits given in the test: real readout code, no model."""

    name = "fixed"
    max_options = 3

    def __init__(self, table, **settings):
        super().__init__("fixed", settings)
        self.table = table
        self.calls = []

    def logits(self, state, question):
        keys = tuple(self.slots(question))
        self.calls.append(keys)
        return [self.table[k] for k in keys]


class TestLogitReadout(unittest.TestCase):
    def test_softmax_and_temperature(self):
        p = Fixed({"a": 2.0, "b": 0.0}, temperature=1.0)
        q = make_question("choice", "?", {"a": "", "b": ""})
        probs = p.probabilities(STATE, q)
        self.assertAlmostEqual(probs["a"], 0.8808, places=3)
        cooler = Fixed({"a": 2.0, "b": 0.0}, temperature=4.0).probabilities(STATE, q)
        self.assertLess(cooler["a"], probs["a"])
        self.assertAlmostEqual(sum(cooler.values()), 1.0)

    def test_noul_reads_the_true_slot(self):
        p = Fixed({"false": 0.0, "true": 1.0})
        answers, meta = p.decide(STATE, {"q": NOUL})
        probs = answers["q"]["probabilities"]
        self.assertGreater(probs["true"], probs["false"])
        self.assertEqual(meta["cost"], 0.0)
        from chai.system_one import normalize_answer

        self.assertTrue(normalize_answer(NOUL, answers["q"])["truth"])

    def test_knockout_sums_to_one_and_prefers_the_strongest(self):
        table = {k: float(v) for k, v in zip("abcdefg", [0, 1, 5, 2, 4, 3, 6])}
        p = Fixed(table)  # reads three options per pass
        q = make_question("choice", "?", {k: "" for k in "abcdefg"})
        probs = p.probabilities(STATE, q)
        self.assertEqual(set(probs), set("abcdefg"))
        self.assertAlmostEqual(sum(probs.values()), 1.0)
        self.assertEqual(max(probs, key=probs.get), "g")
        # groups of three, then a final among the group winners
        self.assertEqual([len(c) for c in p.calls], [3, 3, 3])
        self.assertEqual(p.calls[-1], ("c", "e", "g"))
        with self.assertRaises(ValueError):
            p.probabilities(STATE, make_question("choice", "?", {str(i): "" for i in range(10)}))


def tokenizer():
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained("Qwen/Qwen3.5-4B-Base", revision=QWEN_REVISION)


def with_tokenizer(provider, tok, **extra):
    """A decoder provider with just a tokenizer (no weights) so its prompt can be built."""
    provider.rt = SimpleNamespace(tok=tok, device="cpu", **extra)
    return provider


@unittest.skipUnless(TOKENIZER_READY, "the Qwen3.5-4B-Base tokenizer is not cached")
class TestDecoderPrompts(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tok = tokenizer()

    def decode(self, ids):
        return self.tok.decode(ids)

    def test_letters_jevk5_payload(self):
        p = with_tokenizer(LetterProvider("crh225/plumb-4b", {"style": "jevk5"}), self.tok)
        text = self.decode(p.encode(STATE, CHOICE))
        self.assertIn("Apply the supplied criterion to the supplied evidence.", text)
        payload = json.loads(text[text.index("{") : text.rindex("}") + 1])
        self.assertEqual(payload["evidence"], STATE)
        self.assertEqual(payload["criterion"], "Which record?")
        self.assertEqual(payload["options"], [{"letter": "A", "description": "wikidata:Q5582: Vincent van Gogh, Dutch painter"}, {"letter": "B", "description": "none: None of these"}])
        self.assertEqual(p.slots(NOUL), ["true", "false"])
        self.assertEqual([t for _, t in p.options(SCORE)], ["0: not at all", "1: partly", "2: entirely"])

    def test_letters_decider_text(self):
        p = with_tokenizer(LetterProvider("Mapika/decider-2b", {"style": "decider"}), self.tok)
        text = self.decode(p.encode(STATE, CHOICE))
        self.assertEqual(
            text,
            f"Context:\n{STATE}\n\nQuestion: Which record?\nOptions:\n(A) wikidata:Q5582: Vincent van Gogh, Dutch painter\n(B) none: None of these\nAnswer: (",
        )
        self.assertEqual(p.options(NOUL), [("false", "no"), ("true", "yes")])
        self.assertEqual(p.max_options, 10)

    def test_letters_tev1_payload(self):
        p = with_tokenizer(LetterProvider("togethercomputer/Tev1-4B-experimental", {"style": "tev1"}), self.tok)
        text = self.decode(p.encode(STATE, NOUL))
        payload = json.loads(text[text.index("{") : text.rindex("}") + 1])
        self.assertEqual(payload["options"], [{"label": "A", "key": "yes", "description": "Yes."}, {"label": "B", "key": "no", "description": "No."}])
        self.assertEqual(p.max_options, 24)

    def test_candidate_prompts(self):
        p = with_tokenizer(CandidateProvider("Qwen/Qwen3.5-9B", {}), self.tok)
        prompts = p.prompts(STATE, CHOICE)
        self.assertEqual(len(prompts), 2)
        self.assertTrue(prompts[0].startswith(f"Context:\n{STATE}\n\nQuestion: Which record?\n"))
        self.assertTrue(prompts[0].endswith("Proposed answer: wikidata:Q5582: Vincent van Gogh, Dutch painter\nIs this proposed answer correct? Answer Yes or No."))
        self.assertEqual(p.prompts(STATE, NOUL), [f"Context:\n{STATE}\n\nQuestion: The text mentions a painter.\nIs the answer to this question yes? Answer Yes or No."])
        self.assertEqual(p.prompts(STATE, SCORE)[1].split("Proposed answer: ")[1].split("\n")[0], "partly")

    def test_slots_text(self):
        p = with_tokenizer(SlotHeadProvider("autotrust/JEV-9B", {}), self.tok, prefix=[], softcap=None, ranges={"noul": [0, 2], "score": [2, 8], "choice": [8, 24]})
        self.assertEqual(
            p.text(STATE, CHOICE),
            f"[kind] choice\n[state] {STATE}\n[question] Which record?\n[options]\nA) wikidata:Q5582: Vincent van Gogh, Dutch painter\nB) none: None of these\n[decision]:",
        )
        self.assertEqual(p.text(STATE, NOUL).split("[options]\n")[1], "false\ntrue\n[decision]:")
        score = p.text(STATE, SCORE)
        self.assertIn("Scale: 0 = not at all; 1 = partly; 2 = entirely", score)
        self.assertTrue(score.endswith("[options]\n0\n1\n2\n[decision]:"))
        with self.assertRaises(ValueError):
            p.text(STATE, make_question("score", "?", [str(i) for i in range(7)]))

    def test_nli_pairs(self):
        p = NLIProvider("AlexWortega/openjev", {"subfolder": "qwen3.5-4b-nli-v5"})
        p.rt = SimpleNamespace(tok=self.tok, device="cpu", entailment=0)
        pairs = p.pairs(STATE, CHOICE)
        self.assertEqual(pairs[0], f'Premise: {STATE}\nHypothesis: The answer to "Which record?" is wikidata:Q5582: Vincent van Gogh, Dutch painter')
        self.assertEqual(p.pairs(STATE, NOUL)[1].split("Hypothesis: ")[1], 'The answer to "The text mentions a painter." is yes: yes')
        self.assertEqual(p.pairs(STATE, SCORE)[2].split(" is ")[-1], "2: entirely")

    def test_pointer_row_reads_each_option_close_then_decide(self):
        p = with_tokenizer(PointerProvider("jaredpalmer/kev-4b", {}), self.tok)
        options = system_one.kev_options(CHOICE)
        row, positions = system_one.kev_encode(self.tok, STATE, CHOICE["instructions"], [t for _, t in options], 3072)
        st, q, ob, oe, dec = [self.tok.convert_tokens_to_ids(t) for t in system_one.KEV_SPECIAL]
        self.assertEqual([row[i] for i in positions], [oe, oe, dec])
        self.assertEqual(p.slots(NOUL), ["false", "true"])


class TestEncoderInputs(unittest.TestCase):
    def test_von_option_texts(self):
        self.assertEqual(VonProvider.option_texts(NOUL), ["Yes, condition holds true.", "No, condition is false."])
        self.assertEqual(VonProvider.option_texts(CHOICE), ["Vincent van Gogh, Dutch painter", "None of these"])
        self.assertEqual(VonProvider("wfzyx/von", {}).noul_slots, ("true", "false"))

    def test_bekko_native_record(self):
        native = BekkoProvider.native(SCORE, STATE)
        decision = native["decisions"][0]
        self.assertEqual(decision["type"], "score")
        self.assertEqual([c["value"] for c in decision["criteria"]], [0.0, 1.0, 2.0])
        self.assertEqual(json.loads(native["state_json"]), STATE)
        noul = BekkoProvider.native(NOUL, STATE)["decisions"][0]
        self.assertEqual([c["id"] for c in noul["criteria"]], ["true", "false"])

    def test_gliner_labels_and_text(self):
        p = GlinerProvider("fastino/GLiNER2.5-Decide", {})
        self.assertEqual(p.labels_for(CHOICE), {"wikidata:Q5582": "Vincent van Gogh, Dutch painter", "none": "None of these"})
        self.assertEqual(p.text_for(CHOICE, STATE), STATE)
        self.assertEqual(p.text_for(NOUL, STATE), f"Statement: The text mentions a painter.\n{STATE}")
        self.assertEqual(p.text_for(SCORE, STATE), f"How much is the text about art?\n{STATE}")
        custom = GlinerProvider("fastino/GLiNER2.5-Decide", {"text_template": "Mention: x\n{state}"})
        self.assertEqual(custom.text_for(CHOICE, STATE), f"Mention: x\n{STATE}")
        self.assertEqual(system_one.gliner_sanitize("Van Gogh (1853-1890) [L]"), "Van Gogh [1853-1890]")


class TestSystem1Model(unittest.TestCase):
    def test_meta_fields(self):
        model = System1Model({"model": "openai/gpt-4o-mini"})
        self.assertEqual(model.provider.name, "generative")
        self.assertEqual(repr(model), "<System1Model openai/gpt-4o-mini via generative>")


if __name__ == "__main__":
    unittest.main()
