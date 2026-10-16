"""Tests for System One support: the decision data model (chai.ai.system1) and the decision roles
(chai.decision). No models and no network: the generative provider is exercised through LiteLLM's
local mock replies, and the Jev API through a throwaway HTTP server in this process."""

import json
import os
import threading
import unittest
from http.server import BaseHTTPRequestHandler, HTTPServer
from unittest import mock

from chai.ai import system1
from chai.ai.system1 import (
    GenerativeProvider,
    GlinerProvider,
    JevProvider,
    LayaProvider,
    System1Model,
    apply_layout,
    level_value,
    make_question,
    normalize_answer,
    normalize_options,
)
from chai.core import importClass
from chai.result import ItemResult, LabelListResult
from chai.workflow import Workflow

MODEL = "openai/gpt-4o-mini"  # any LiteLLM id; replies come from mock_response, never the network

CANDIDATES = [
    {"key": "wikidata:Q5582", "summary": "Vincent van Gogh — Dutch painter (1853–1890)"},
    {"key": "wikidata:Q17029510", "summary": "Van Gogh — music album"},
]
RECORD = {"name": "Van Gogh", "type": "person", "context_text": "The Night Café was painted by Van Gogh in 1888.", "candidates": CANDIDATES}


def answers(**by_name):
    return json.dumps({"answers": by_name})


def classifier(settings, cid="sel", wf=None, **tree_extra):
    wf = wf or Workflow({"type": "Workflow", "id": "w"})
    tree = {"type": "decision.DecisionClassifier", "id": cid, "settings": {"model": MODEL, **settings}, **tree_extra}
    return importClass(tree["type"])(tree, wf), wf


# ------------------------------------------------------------------ the data model


class TestQuestions(unittest.TestCase):
    def test_option_shapes(self):
        want = [{"key": "a", "text": "Alpha"}, {"key": "b", "text": ""}]
        self.assertEqual(normalize_options({"a": "Alpha", "b": None}), want)
        self.assertEqual(normalize_options([{"key": "a", "text": "Alpha"}, "b"]), want)
        self.assertEqual(normalize_options([["a", "Alpha"], ["b", ""]]), want)
        self.assertEqual(normalize_options([{"id": "a", "description": "Alpha"}]), want[:1])
        with self.assertRaises(ValueError):
            normalize_options(["a", "a"])

    def test_make_question_validates(self):
        q = make_question("choice", "Which?", {"a": "A"})
        self.assertEqual(q["options"], [{"key": "a", "text": "A"}])
        with self.assertRaises(ValueError):
            make_question("choice", "Which?", [])
        with self.assertRaises(ValueError):
            make_question("noul", "It is so", {"a": "A"})
        with self.assertRaises(ValueError):
            make_question("guess", "?", {"a": "A"})
        with self.assertRaises(ValueError):
            make_question("choice", "", {"a": "A"})

    def test_layouts(self):
        q = {"q": make_question("choice", "Which?", {"a": "Alpha", "b": "Beta"})}
        state, laid = apply_layout("Text.", q, "in_state", heading="Candidates:")
        self.assertEqual(state, "Text.\n\nCandidates:\n- a: Alpha\n- b: Beta")
        self.assertEqual([o["text"] for o in laid["q"]["options"]], ["", ""])
        state, laid = apply_layout("Text.", q, "bare")
        self.assertEqual(state, "Text.")
        self.assertEqual([o["text"] for o in laid["q"]["options"]], ["", ""])
        self.assertEqual(q["q"]["options"][0]["text"], "Alpha")  # inputs untouched
        with self.assertRaises(ValueError):
            apply_layout("Text.", q, "sideways")


class TestAnswers(unittest.TestCase):
    def test_choice_renormalizes_and_fills_missing(self):
        q = make_question("choice", "?", {"a": "", "b": "", "none": ""})
        a = normalize_answer(q, {"choice": "a", "probabilities": {"a": 0.6, "b": 0.2}, "confidence": 0.55})
        self.assertEqual(a["choice"], "a")
        self.assertAlmostEqual(a["probabilities"]["a"], 0.75)
        self.assertEqual(a["probabilities"]["none"], 0.0)
        self.assertEqual(a["confidence"], 0.55)

    def test_choice_without_mass_falls_back(self):
        q = make_question("choice", "?", {"a": "", "b": "", "none": ""})
        self.assertEqual(normalize_answer(q, {"choice": "b"})["probabilities"]["b"], 1.0)
        self.assertEqual(normalize_answer(q, {})["probabilities"]["none"], 1.0)
        self.assertEqual(normalize_answer(q, {"choice": "zzz"})["choice"], "none")
        q2 = make_question("choice", "?", {"a": "", "b": ""})
        self.assertEqual(normalize_answer(q2, {})["probabilities"], {"a": 0.5, "b": 0.5})

    def test_score_from_index_keyed_and_list_probabilities(self):
        q = make_question("score", "How well?", {"1": "poor", "2": "fair", "3": "good"})
        # Laya's shape: probabilities keyed by 0-based level index, score = expected index
        a = normalize_answer(q, {"type": "score", "score": 1.5, "probabilities": {"0": 0.1, "1": 0.3, "2": 0.6}})
        self.assertEqual(a["score"], "3")
        self.assertAlmostEqual(a["expected"], 0.1 * 1 + 0.3 * 2 + 0.6 * 3)
        a = normalize_answer(q, {"probabilities": [0.5, 0.5, 0.0]})
        self.assertEqual(a["score"], "1")
        self.assertAlmostEqual(a["expected"], 1.5)
        q2 = make_question("score", "?", ["low", "mid", "high"])
        self.assertEqual(level_value("mid", 1), 2.0)
        self.assertAlmostEqual(normalize_answer(q2, {"probabilities": {"high": 1.0}})["expected"], 3.0)

    def test_noul_shapes(self):
        q = make_question("noul", "The text mentions a painter")
        self.assertAlmostEqual(normalize_answer(q, {"noul": 0.8})["probability"], 0.8)
        a = normalize_answer(q, {"probabilities": {"true": 0.3, "false": 0.1}})
        self.assertAlmostEqual(a["probability"], 0.75)
        self.assertTrue(a["truth"])
        self.assertAlmostEqual(a["confidence"], 0.75)
        a = normalize_answer(q, {"truth": False})
        self.assertEqual((a["probability"], a["truth"]), (0.0, False))
        self.assertEqual(normalize_answer(q, {"probability": 1.7})["probability"], 1.0)
        with self.assertRaises(ValueError):
            normalize_answer(q, {})


class TestProviders(unittest.TestCase):
    def test_prefix_resolution(self):
        self.assertIsInstance(System1Model({"model": "laya/convaiinnovations/laya"}).provider, LayaProvider)
        self.assertIsInstance(System1Model({"model": "gliner/fastino/GLiNER2.5-Decide", "device": "cpu"}).provider, GlinerProvider)
        self.assertIsInstance(System1Model({"model": "typesafe/jev-latest", "api_key": "k"}).provider, JevProvider)
        self.assertIsInstance(System1Model({"model": MODEL}).provider, GenerativeProvider)
        with self.assertRaises(ValueError):
            System1Model({})
        with mock.patch.dict(os.environ, {"TYPESAFE_API_KEY": "", "JEV_API_KEY": ""}):
            with self.assertRaises(ValueError):
                System1Model({"model": "jev/jev-latest"})

    def test_gliner_labels(self):
        p = GlinerProvider("fastino/GLiNER2.5-Decide", {"device": "cpu"})
        self.assertEqual(p.labels_for(make_question("choice", "?", {"a": "Alpha (1)", "b": ""})), {"a": "Alpha [1]", "b": "b"})
        self.assertEqual(p.labels_for(make_question("choice", "?", ["a", "b"])), ["a", "b"])
        self.assertEqual(set(p.labels_for(make_question("noul", "so"))), {"true", "false"})
        self.assertEqual(system1.gliner_sanitize("x [L] (y)"), "x [y]")

    def test_generative_prompt_lists_every_question(self):
        p = GenerativeProvider(MODEL, {})
        qs = {"who": make_question("choice", "Who?", {"a": "Alpha"}), "ok": make_question("noul", "It is fine")}
        prompt = p.prompt("The state.", qs)
        self.assertIn("STATE:\nThe state.", prompt)
        self.assertIn("[who] choice: Who?\n  - a: Alpha", prompt)
        self.assertIn("[ok] noul", prompt)
        self.assertNotIn("temperature", p.llm.base_kwargs)  # the provider's default, unless set
        self.assertEqual(p.llm.base_kwargs["response_format"], {"type": "json_object"})
        self.assertEqual(GenerativeProvider(MODEL, {"temperature": 0.0}).llm.base_kwargs["temperature"], 0.0)

    def test_generative_gemini_id_uses_gemini_defaults(self):
        with mock.patch.dict(os.environ, {"GEMINI_API_KEY": "k", "GOOGLE_CLOUD_PROJECT": "", "VERTEXAI_PROJECT": ""}):
            p = GenerativeProvider("gemini/gemini-3.1-flash-lite", {})
        self.assertNotIn("tools", p.llm.base_kwargs)  # grounding off for decisions
        self.assertEqual(p.llm.base_kwargs["api_key"], "k")


class _JevHandler(BaseHTTPRequestHandler):
    requests = []
    throttle_first = False

    def log_message(self, *args):
        pass

    def do_POST(self):
        length = int(self.headers.get("content-length", 0))
        body = json.loads(self.rfile.read(length) or b"{}")
        _JevHandler.requests.append({"path": self.path, "headers": dict(self.headers), "body": body})
        if _JevHandler.throttle_first and len(_JevHandler.requests) == 1:
            self.send_response(429)
            self.send_header("Retry-After", "0")
            self.end_headers()
            return
        reply = {
            "model": "jev-2",
            "usage": {"input_tokens": 50, "output_tokens": 1},
            "answers": {
                name: (
                    {"type": "choice", "choice": list(q["criteria"])[0], "probabilities": {k: (0.9 if i == 0 else 0.1 / max(1, len(q["criteria"]) - 1)) for i, k in enumerate(q["criteria"])}, "confidence": 0.9}
                    if q["type"] == "choice"
                    else {"type": "noul", "noul": 0.2}
                    if q["type"] == "noul"
                    else {"type": "score", "score": 2.0, "probabilities": {"0": 0.0, "1": 0.0, "2": 1.0}}
                )
                for name, q in body["questions"].items()
            },
        }
        data = json.dumps(reply).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)


class TestJevProvider(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.server = HTTPServer(("127.0.0.1", 0), _JevHandler)
        cls.thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.thread.start()
        cls.base = f"http://127.0.0.1:{cls.server.server_address[1]}"

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.server.server_close()

    def setUp(self):
        _JevHandler.requests = []
        _JevHandler.throttle_first = False

    def test_request_shape_and_answers(self):
        model = System1Model({"model": "jev/jev-latest", "api_key": "secret", "api_base": self.base})
        questions = {
            "who": make_question("choice", "Which record?", {"wikidata:Q5582": "Vincent van Gogh", "none": "None of these"}),
            "painter": make_question("noul", "The text mentions a painter"),
            "fit": make_question("score", "How well?", {"1": "poor", "2": "fair", "3": "good"}),
        }
        answers, meta = model.decide("Van Gogh painted it.", questions)
        req = _JevHandler.requests[0]
        self.assertEqual(req["path"], "/v1/systemone")
        self.assertEqual(req["headers"]["Authorization"], "Bearer secret")
        self.assertEqual(req["body"]["model"], "jev-latest")
        self.assertEqual(req["body"]["state"], "Van Gogh painted it.")
        self.assertEqual(req["body"]["questions"]["who"], {"type": "choice", "instructions": "Which record?", "criteria": {"wikidata:Q5582": "Vincent van Gogh", "none": "None of these"}})
        self.assertEqual(req["body"]["questions"]["fit"]["criteria"], ["poor", "fair", "good"])
        self.assertEqual(req["body"]["questions"]["painter"], {"type": "noul", "instructions": "The text mentions a painter"})
        self.assertEqual(answers["who"]["choice"], "wikidata:Q5582")
        self.assertAlmostEqual(answers["who"]["probabilities"]["none"], 0.1)
        self.assertEqual(answers["painter"]["truth"], False)
        self.assertEqual(answers["fit"]["score"], "3")
        self.assertEqual(meta["model"], "jev-2")
        self.assertEqual(meta["token_usage"]["prompt"], 50)
        self.assertIsNone(meta["cost"])  # TypeSafe pricing is not known to chai

    def test_rate_limit_is_retried(self):
        _JevHandler.throttle_first = True
        model = System1Model({"model": "jev/jev-latest", "api_key": "secret", "api_base": self.base})
        answers, _ = model.decide("state", {"q": make_question("noul", "statement")})
        self.assertEqual(len(_JevHandler.requests), 2)
        self.assertIn("probability", answers["q"])

    def test_in_state_layout_moves_descriptions(self):
        model = System1Model({"model": "jev/jev-latest", "api_key": "secret", "api_base": self.base})
        model.decide("Text.", {"q": make_question("choice", "?", {"a": "Alpha"})}, layout="in_state")
        body = _JevHandler.requests[0]["body"]
        self.assertEqual(body["state"], "Text.\n\nOptions:\n- a: Alpha")
        self.assertEqual(body["questions"]["q"]["criteria"], {"a": "a"})


# ------------------------------------------------------------------ the roles


class TestDecisionClassifier(unittest.TestCase):
    def test_choice_over_candidates_in_state(self):
        reply = answers(q={"choice": "wikidata:Q5582", "probabilities": {"wikidata:Q5582": 0.7, "wikidata:Q17029510": 0.2, "none": 0.1}, "confidence": 0.7})
        comp, wf = classifier(
            {
                "question": "Which record is the {type} '{name}' referring to?",
                "options_from": "candidates",
                "option_text": "{summary}",
                "none_option": "None of the listed records",
                "layout": "in_state",
                "options_heading": "Candidate records for '{name}':",
                "state": "Text: {context_text}",
                "litellm_params": {"mock_response": reply},
            }
        )
        out = comp.process(ItemResult(RECORD, metadata={"type": "DATA"}))
        self.assertIsInstance(out, LabelListResult)
        self.assertEqual(out.value, ["wikidata:Q5582"])
        self.assertEqual(out.metadata["choice"], "wikidata:Q5582")
        self.assertAlmostEqual(out.metadata["confidence"], 0.7)
        self.assertEqual(set(out.metadata["probabilities"]), {"wikidata:Q5582", "wikidata:Q17029510", "none"})
        decision = out.metadata["decision"]
        self.assertEqual(decision["question"]["instructions"], "Which record is the person 'Van Gogh' referring to?")
        self.assertTrue(decision["state"].startswith("Text: The Night Café"))
        self.assertIn("- wikidata:Q5582: Vincent van Gogh", decision["state"])
        self.assertEqual([o["text"] for o in decision["question"]["options"]], ["Vincent van Gogh — Dutch painter (1853–1890)", "Van Gogh — music album", "None of the listed records"])
        self.assertEqual(out.metadata["model"], MODEL)
        self.assertEqual(wf.usage_summary()["calls"], 1)  # the generative call is on the ledger
        self.assertEqual(list(wf.usage_summary()["by_model"]), [MODEL])

    def test_none_choice_and_threshold(self):
        comp, _ = classifier({"question": "?", "options": {"a": "A", "b": "B"}, "none_option": True, "litellm_params": {"mock_response": answers(q={"choice": "none", "probabilities": {"none": 0.8, "a": 0.1, "b": 0.1}})}})
        self.assertEqual(comp.process(ItemResult("text")).value, ["none"])
        comp, _ = classifier({"question": "?", "options": {"a": "A", "b": "B"}, "threshold": 0.6, "litellm_params": {"mock_response": answers(q={"choice": "a", "probabilities": {"a": 0.55, "b": 0.45}})}})
        out = comp.process(ItemResult("text"))
        self.assertEqual(out.value, ["none"])
        self.assertEqual(out.metadata["choice"], "a")  # the model's pick is still recorded

    def test_no_candidates_is_a_null_decision_without_a_call(self):
        comp, wf = classifier({"question": "?", "options_from": "candidates", "none_option": True, "litellm_params": {"mock_response": "not valid json at all"}})
        out = comp.process(ItemResult({"name": "Nobody", "candidates": []}))
        self.assertEqual(out.value, ["none"])
        self.assertEqual(out.metadata["decision"]["skipped"], "no options")
        self.assertEqual(wf.usage_summary()["calls"], 0)

    def test_score_and_noul(self):
        comp, _ = classifier({"question_type": "score", "question": "How relevant?", "options": {"1": "not", "2": "somewhat", "3": "very"}, "litellm_params": {"mock_response": answers(q={"score": "2", "probabilities": {"1": 0.2, "2": 0.5, "3": 0.3}})}})
        out = comp.process(ItemResult("text"))
        self.assertEqual(out.value, ["2"])
        self.assertAlmostEqual(out.metadata["expected"], 2.1)
        comp, _ = classifier({"question_type": "noul", "question": "The text is about art.", "litellm_params": {"mock_response": answers(q={"truth": True, "probability": 0.65})}})
        out = comp.process(ItemResult("text"))
        self.assertEqual(out.value, ["true"])
        self.assertAlmostEqual(out.metadata["probability"], 0.65)
        comp, _ = classifier({"question_type": "noul", "question": "The text is about art.", "threshold": 0.7, "litellm_params": {"mock_response": answers(q={"truth": True, "probability": 0.65})}})
        self.assertEqual(comp.process(ItemResult("text")).value, ["false"])

    def test_missing_settings_fail_at_build(self):
        with self.assertRaises(ValueError):
            classifier({"options": {"a": "A"}})  # no question
        with self.assertRaises(ValueError):
            classifier({"question": "?", "layout": "upside_down", "options": {"a": "A"}})
        comp, _ = classifier({"question": "?"})  # choice without options fails when asked
        with self.assertRaises(ValueError):
            comp.process(ItemResult("text"))

    def test_labels_drive_a_gate_in_a_workflow(self):
        reply = answers(q={"choice": "wikidata:Q5582", "probabilities": {"wikidata:Q5582": 0.9, "wikidata:Q17029510": 0.1}})
        wf = Workflow(
            {
                "type": "Workflow",
                "id": "wf",
                "steps": [
                    {
                        "type": "provider.StaticProvider",
                        "id": "records",
                        "input": "",
                        "settings": {"values": [RECORD, {**RECORD, "candidates": []}]},
                        "steps": [
                            {
                                "type": "iterator.Iterator",
                                "id": "entities",
                                "steps": [
                                    {
                                        "type": "decision.DecisionClassifier",
                                        "id": "select",
                                        "register_on": ["entities"],
                                        "settings": {"model": MODEL, "question": "Which record is '{name}'?", "options_from": "candidates", "option_text": "{summary}", "none_option": True, "litellm_params": {"mock_response": reply}},
                                    },
                                    {
                                        "type": "gate.LabelTestGate",
                                        "id": "linked",
                                        "settings": {"component": "select", "label": ["none"]},
                                        "true_steps": [{"type": "describer.FileInfoDescriber", "id": "unlinked"}],
                                        "false_steps": [{"type": "describer.FileInfoDescriber", "id": "linked_desc"}],
                                    },
                                ],
                            }
                        ],
                    }
                ],
            }
        )
        res = wf.run()
        entries = res.value[0].value[0].value  # provider -> its steps (the iterator) -> per-entry lists
        self.assertEqual(entries[0].value[0].value, ["wikidata:Q5582"])
        self.assertEqual(entries[0].value[1].value[0].processor.id, "linked_desc")
        self.assertEqual(entries[1].value[0].value, ["none"])
        self.assertEqual(entries[1].value[1].value[0].processor.id, "unlinked")
        self.assertEqual(wf.usage_summary()["calls"], 1)  # the empty-candidate record made no call


class TestDecisionGate(unittest.TestCase):
    def build(self, reply, threshold=None):
        wf = Workflow({"type": "Workflow", "id": "w"})
        settings = {"model": MODEL, "question": "The text mentions a painter named {name}.", "litellm_params": {"mock_response": reply}}
        if threshold is not None:
            settings["threshold"] = threshold
        tree = {
            "type": "decision.DecisionGate",
            "id": "g",
            "settings": settings,
            "true_steps": [{"type": "describer.FileInfoDescriber", "id": "yes"}],
            "false_steps": [{"type": "describer.FileInfoDescriber", "id": "no"}],
        }
        return importClass(tree["type"])(tree, wf), wf

    def test_routes_and_records(self):
        gate, wf = self.build(answers(q={"truth": True, "probability": 0.9}))
        events = []
        wf.add_listener(events.append)
        inp = ItemResult({"name": "Van Gogh", "text": "Van Gogh painted it."})
        res = gate.process(inp)
        self.assertEqual(res.value[0].processor.id, "yes")
        decision = inp.extra["decisions"][0]
        self.assertEqual(decision["question"]["instructions"], "The text mentions a painter named Van Gogh.")
        self.assertAlmostEqual(decision["answer"]["probability"], 0.9)
        end = [e for e in events if e["event"] == "component_end"][-1]
        self.assertEqual(end["branch"], "true")
        self.assertAlmostEqual(end["probability"], 0.9)
        self.assertEqual(wf.usage_summary()["calls"], 1)

    def test_false_branch_and_threshold(self):
        gate, _ = self.build(answers(q={"truth": False, "probability": 0.3}))
        self.assertEqual(gate.process(ItemResult("text")).value[0].processor.id, "no")
        gate, _ = self.build(answers(q={"truth": False, "probability": 0.3}), threshold=0.25)
        self.assertEqual(gate.process(ItemResult("text")).value[0].processor.id, "yes")


class TestDecisionExtractor(unittest.TestCase):
    def test_several_questions_one_record(self):
        reply = answers(who={"choice": "a", "probabilities": {"a": 0.6, "b": 0.4}}, art={"truth": True, "probability": 0.95}, fit={"score": "3", "probabilities": {"1": 0.0, "2": 0.2, "3": 0.8}})
        wf = Workflow({"type": "Workflow", "id": "w"})
        tree = {
            "type": "decision.DecisionExtractor",
            "id": "e",
            "settings": {
                "model": MODEL,
                "questions": {
                    "who": {"type": "choice", "question": "Who made {title}?", "options": {"a": "Artist A", "b": "Artist B"}},
                    "art": {"type": "noul", "question": "{title} is a painting."},
                    "fit": {"type": "score", "question": "How confident?", "options": ["1", "2", "3"]},
                },
                "litellm_params": {"mock_response": reply},
            },
        }
        comp = importClass(tree["type"])(tree, wf)
        out = comp.process(ItemResult({"title": "The Night Café", "text": "..."}))
        self.assertEqual(out.value["who"]["choice"], "a")
        self.assertTrue(out.value["art"]["truth"])
        self.assertEqual(out.value["fit"]["score"], "3")
        self.assertAlmostEqual(out.value["fit"]["expected"], 2.8)
        self.assertEqual(out.metadata["decision"]["questions"]["who"]["instructions"], "Who made The Night Café?")
        self.assertEqual(out.metadata["type"], "DATA")
        self.assertEqual(wf.usage_summary()["by_component"]["e"]["calls"], 1)

    def test_needs_questions(self):
        with self.assertRaises(ValueError):
            importClass("decision.DecisionExtractor")({"type": "decision.DecisionExtractor", "id": "e", "settings": {"model": MODEL}}, Workflow({"type": "Workflow", "id": "w"}))


if __name__ == "__main__":
    unittest.main()
