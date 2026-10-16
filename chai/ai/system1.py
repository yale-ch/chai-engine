"""System One decision models for chai: typed questions answered with probabilities, not text.

A System One model reads a *state* (text) and answers one or more typed questions about it with a
calibrated probability per option, in a single forward pass -- no generation, no parsing. Three
question types exist, and they are the same three that the ``system1-finetune-bench`` training
format uses, so what a chai workflow asks can be exported as training data as-is:

* ``choice`` -- pick one of the ``options`` (``[{"key", "text"}, ...]``);
* ``score``  -- rate the state on an ordinal scale; ``options`` are the levels, low to high;
* ``noul``   -- is the statement (``instructions``) true of the state? No options.

Every answer comes back in one canonical shape (see ``normalize_answer``)::

    choice: {"type": "choice", "choice": key, "probabilities": {key: p}, "confidence": p}
    score:  {"type": "score", "score": level_key, "expected": float, "probabilities": {level_key: p}, "confidence": p}
    noul:   {"type": "noul", "truth": bool, "probability": P(true), "probabilities": {"true": p, "false": 1-p}, "confidence": p}

``System1Model`` picks its provider from the model id prefix, the way LiteLLM does for chat models:

    jev/jev-latest                      TypeSafe's Jev via POST /v1/systemone (TYPESAFE_API_KEY)
    laya/convaiinnovations/laya         Laya, open weights, in process (pip install laya)
    gliner/fastino/GLiNER2.5-Decide     GLiNER2.5-Decide, open weights, in process (pip install gliner2)
    <any LiteLLM chat model id>         a generative model asked for the same JSON -- a baseline for
                                        comparison; its probabilities are self-reported, not calibrated

Models that only answer choice questions natively (GLiNER) get score and noul as a choice over the
levels / over true-false, exactly as the bench's ``as_choice`` approaches do. ``apply_layout``
implements the bench's inference-time prompt approaches (``described`` / ``in_state`` / ``bare``), so
the layout a model scored best with is one setting away.

The chai roles that use this module are in ``chai.decision``: ``DecisionClassifier`` (one question,
the answer as a label), ``DecisionGate`` (a noul question as a branch) and ``DecisionExtractor``
(several questions, one structured record).
"""

from __future__ import annotations

import json
import logging
import os
import re
import threading
import time
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger("chai")

QUESTION_TYPES = ("choice", "score", "noul")
NONE_KEY = "none"
NULL_KEYS = {"none", "null", "None"}
LAYOUTS = ("described", "in_state", "bare")
TRUE_KEY, FALSE_KEY = "true", "false"
NOUL_OPTIONS = [{"key": TRUE_KEY, "text": "The statement is true"}, {"key": FALSE_KEY, "text": "The statement is false"}]

#: One process-wide lock for every in-process model forward pass. Metal (Apple MPS) aborts the
#: process when two threads submit work at once, and chai's Iterator runs steps on several threads;
#: per-model locks are not enough once two local models share the GPU, so they all take this one.
DEVICE_LOCK = threading.RLock()


# --------------------------------------------------------------------------- questions


def normalize_options(options) -> List[dict]:
    """``[{"key", "text"}, ...]`` from any of the shapes a workflow may spell options in:
    ``{key: text}``, ``["key", ...]``, ``[{"key", "text"}, ...]`` or ``[["key", "text"], ...]``."""
    if options is None:
        return []
    out: List[dict] = []
    if isinstance(options, dict):
        for key, text in options.items():
            out.append({"key": str(key), "text": "" if text is None else str(text)})
        return out
    for entry in options:
        if isinstance(entry, dict):
            key = entry.get("key", entry.get("id", entry.get("label")))
            if key is None:
                raise ValueError(f"option without a key: {entry!r}")
            text = entry.get("text", entry.get("description", entry.get("summary", "")))
            out.append({"key": str(key), "text": "" if text is None else str(text)})
        elif isinstance(entry, (list, tuple)) and len(entry) == 2:
            out.append({"key": str(entry[0]), "text": "" if entry[1] is None else str(entry[1])})
        else:
            out.append({"key": str(entry), "text": ""})
    keys = [o["key"] for o in out]
    if len(set(keys)) != len(keys):
        raise ValueError(f"duplicate option keys: {keys}")
    return out


def make_question(qtype: str, instructions: str, options=None) -> dict:
    """A canonical chai question dict; validates the type and the options it needs."""
    if qtype not in QUESTION_TYPES:
        raise ValueError(f"question type must be one of {QUESTION_TYPES}, got {qtype!r}")
    if not instructions or not str(instructions).strip():
        raise ValueError("a question needs instructions (the question or statement text)")
    opts = normalize_options(options)
    if qtype == "noul":
        if opts:
            raise ValueError("a noul question takes no options")
    elif not opts:
        raise ValueError(f"a {qtype} question needs options")
    return {"type": qtype, "instructions": str(instructions), "options": opts}


def question_keys(question: dict) -> List[str]:
    """The answer space of a question: option keys, or true/false for noul."""
    if question["type"] == "noul":
        return [TRUE_KEY, FALSE_KEY]
    return [o["key"] for o in question["options"]]


def apply_layout(state: str, questions: Dict[str, dict], layout: str = "described", heading: str = "Options:") -> Tuple[str, Dict[str, dict]]:
    """Rewrite *state* and *questions* for an inference-time prompt approach.

    ``described`` (default): options carry their descriptions. ``in_state``: each question's
    options are listed in the state under *heading* and the options become bare keys -- the layout
    Laya scores best with on reconciliation, and GLiNER worst. ``bare``: descriptions dropped.
    Returns new objects; the inputs are not modified.
    """
    if layout not in LAYOUTS:
        raise ValueError(f"layout must be one of {LAYOUTS}, got {layout!r}")
    if layout == "described":
        return state, {n: dict(q) for n, q in questions.items()}
    out = {}
    extra: List[str] = []
    for name, q in questions.items():
        q = dict(q)
        if q["type"] != "noul":
            if layout == "in_state":
                lines = [f"- {o['key']}: {o['text']}" if o["text"] else f"- {o['key']}" for o in q["options"]]
                extra.append(heading + "\n" + "\n".join(lines))
            q["options"] = [{"key": o["key"], "text": ""} for o in q["options"]]
        out[name] = q
    if extra:
        state = state.rstrip() + "\n\n" + "\n\n".join(extra)
    return state, out


def _wire_question(question: dict) -> dict:
    """The ``{type, instructions, criteria}`` shape Jev and Laya take."""
    q = {"type": question["type"], "instructions": question["instructions"]}
    if question["type"] == "choice":
        q["criteria"] = {o["key"]: (o["text"] or o["key"]) for o in question["options"]}
    elif question["type"] == "score":
        q["criteria"] = [o["text"] or o["key"] for o in question["options"]]
    return q


# --------------------------------------------------------------------------- answers


def _as_float(value, default=0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _probabilities(raw, keys: List[str]) -> Dict[str, float]:
    """A probability per key from whatever the provider returned: a dict keyed by option key,
    a dict keyed by 0-based level index (Laya's score answers), or a list aligned with the keys."""
    probs = {k: 0.0 for k in keys}
    if isinstance(raw, dict):
        raw = {str(k): _as_float(v) for k, v in raw.items()}
        # Level keys are often "1".."n" while Laya keys its score probabilities "0".."n-1": a
        # dict whose keys are all option keys is read by key; one that is not, but is all
        # in-range indices, is read by position.
        if raw and set(raw) <= set(keys):
            for k in keys:
                probs[k] = raw.get(k, 0.0)
        elif raw and all(k.isdigit() and int(k) < len(keys) for k in raw):
            for k, v in raw.items():
                probs[keys[int(k)]] = v
        else:
            for k in keys:
                probs[k] = raw.get(k, 0.0)
    elif isinstance(raw, (list, tuple)):
        for k, v in zip(keys, raw):
            probs[k] = _as_float(v)
    return probs


def _settle(probs: Dict[str, float], preferred: Optional[str]) -> Dict[str, float]:
    """Renormalize; when the provider gave no mass at all, fall back to a one-hot on the preferred
    key, else on the none option, else a uniform distribution."""
    total = sum(max(p, 0.0) for p in probs.values())
    if total > 0:
        return {k: max(p, 0.0) / total for k, p in probs.items()}
    keys = list(probs)
    if preferred in probs:
        target = preferred
    else:
        target = next((k for k in keys if k in NULL_KEYS), None)
    if target is None:
        return {k: 1.0 / len(keys) for k in keys}
    return {k: (1.0 if k == target else 0.0) for k in keys}


def level_value(key: str, index: int) -> float:
    """The numeric value of a score level: its key when numeric, else its 1-based position."""
    try:
        return float(key)
    except (TypeError, ValueError):
        return float(index + 1)


def normalize_answer(question: dict, raw) -> dict:
    """Canonicalize a provider's raw answer to *question* (see the module docstring for shapes).

    Accepts the Jev/Laya shapes (``choice``/``probabilities``/``confidence``; ``score`` with
    index-keyed probabilities; ``noul`` as P(true)), a classifier's choice over true/false or over
    the levels, and the generative baseline's JSON (``truth``/``probability``).
    """
    raw = raw if isinstance(raw, dict) else {}
    qtype = question["type"]
    keys = question_keys(question)

    if qtype == "noul":
        p = None
        for field in ("noul", "probability", "p_true"):
            if raw.get(field) is not None:
                p = _as_float(raw[field], None)
                break
        if p is None and isinstance(raw.get("probabilities"), (dict, list)):
            probs = _settle(_probabilities(raw["probabilities"], keys), None)
            p = probs[TRUE_KEY]
        if p is None:
            for field in ("truth", "answer", "choice"):
                if raw.get(field) is not None:
                    v = raw[field]
                    p = 1.0 if (v is True or str(v).lower() in ("true", "yes", "1")) else 0.0
                    break
        if p is None:
            raise ValueError(f"noul answer carries no probability: {raw!r}")
        p = min(max(p, 0.0), 1.0)
        return {
            "type": "noul",
            "truth": p >= 0.5,
            "probability": p,
            "probabilities": {TRUE_KEY: p, FALSE_KEY: 1.0 - p},
            "confidence": _as_float(raw.get("confidence"), max(p, 1.0 - p)),
        }

    picked = raw.get("choice", raw.get("score", raw.get("answer")))
    picked = str(picked) if picked is not None else None
    if qtype == "score" and picked is not None and picked not in keys and picked.isdigit() and int(picked) < len(keys):
        picked = keys[int(picked)]  # a 0-based level index
    probs = _settle(_probabilities(raw.get("probabilities"), keys), picked if picked in keys else None)
    top = max(probs, key=probs.get)
    chosen = picked if picked in keys else top
    answer = {
        "type": qtype,
        "probabilities": probs,
        "confidence": _as_float(raw.get("confidence"), probs[chosen]),
    }
    if qtype == "choice":
        answer["choice"] = chosen
    else:
        answer["score"] = chosen
        answer["expected"] = sum(probs[k] * level_value(k, i) for i, k in enumerate(keys))
    return answer


# --------------------------------------------------------------------------- providers


def _usage_from(raw) -> Optional[dict]:
    """Token counts in chai's ``token_usage`` shape from a provider's usage block."""
    if not isinstance(raw, dict):
        return None
    prompt = raw.get("input_tokens", raw.get("prompt_tokens", raw.get("prompt")))
    result = raw.get("output_tokens", raw.get("completion_tokens", raw.get("result")))
    total = raw.get("total_tokens", raw.get("total"))
    if total is None and (prompt is not None or result is not None):
        total = (prompt or 0) + (result or 0)
    return {"total": total, "prompt": prompt, "result": result, "thinking": None, "images": None}


class DecisionProvider:
    """One way of answering System One questions. ``decide`` returns the provider's raw answers by
    question name plus a meta dict (``model``, ``token_usage``, ``cost``)."""

    name = ""
    model = ""

    def decide(self, state: str, questions: Dict[str, dict]) -> Tuple[Dict[str, dict], dict]:
        raise NotImplementedError


class JevProvider(DecisionProvider):
    """TypeSafe's Jev through ``POST {base}/v1/systemone``: all questions in one call.

    Authenticates with the ``api_key`` setting, else ``TYPESAFE_API_KEY`` / ``JEV_API_KEY``. A 429 or
    529 is retried a few times with backoff, honouring ``Retry-After``; other errors raise.
    """

    name = "jev"

    def __init__(self, model: str, settings: dict):
        self.model = model or "jev-latest"
        self.api_key = settings.get("api_key") or os.getenv("TYPESAFE_API_KEY") or os.getenv("JEV_API_KEY", "")
        if not self.api_key:
            raise ValueError("the Jev System One model needs TYPESAFE_API_KEY or JEV_API_KEY (env) or an `api_key` setting")
        base = settings.get("api_base") or settings.get("base_url") or os.getenv("TYPESAFE_BASE_URL") or "https://api.typesafe.ai"
        self.base_url = str(base).rstrip("/")
        self.timeout = float(settings.get("timeout", 30))
        self.attempts = int(settings.get("rate_limit_retries", 3)) + 1

    def decide(self, state, questions):
        import requests

        body = {"state": state, "model": self.model, "questions": {n: _wire_question(q) for n, q in questions.items()}}
        headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}
        delay = 1.0
        for attempt in range(self.attempts):
            resp = requests.post(f"{self.base_url}/v1/systemone", json=body, headers=headers, timeout=self.timeout)
            if resp.status_code in (429, 529) and attempt < self.attempts - 1:
                retry_after = resp.headers.get("retry-after", "")
                wait = float(retry_after) if re.fullmatch(r"\d+(\.\d+)?", retry_after) else delay
                time.sleep(wait)
                delay *= 2
                continue
            if resp.status_code >= 400:
                raise RuntimeError(f"Jev API {resp.status_code}: {resp.text[:300]}")
            data = resp.json()
            answers = data.get("answers")
            if not isinstance(answers, dict):
                raise RuntimeError(f"Jev API returned no answers: {str(data)[:300]}")
            return answers, {"model": data.get("model") or self.model, "token_usage": _usage_from(data.get("usage")), "cost": None}
        raise RuntimeError("Jev API: retries exhausted")


_LAYA_AGENTS: dict = {}
_LAYA_LOCK = threading.Lock()


class LayaProvider(DecisionProvider):
    """Laya (``convaiinnovations/laya``), in process. Loaded once per process (about 850 MB on
    first use); answers every question about a state in one forward pass. Settings: ``subfolder``
    (checkpoint variant such as 'multilingual' or 'typed-decisions') and ``device`` (torch device)."""

    name = "laya"

    def __init__(self, model: str, settings: dict):
        self.model = model or "convaiinnovations/laya"
        self.subfolder = settings.get("subfolder") or None
        self.device = settings.get("device") or None

    def agent(self):
        key = (self.model, self.subfolder or "", self.device or "")
        with _LAYA_LOCK, DEVICE_LOCK:
            if key not in _LAYA_AGENTS:
                import laya  # heavy import, deferred until first use

                logger.info(f"Loading Laya {self.model} subfolder={self.subfolder!r} device={self.device!r}")
                _LAYA_AGENTS[key] = laya.load(self.model, subfolder=self.subfolder, device=self.device)
            return _LAYA_AGENTS[key]

    def decide(self, state, questions):
        agent = self.agent()
        wire = {n: _wire_question(q) for n, q in questions.items()}
        with DEVICE_LOCK:
            res = agent.predict(state, wire)
        answers = res.get("answers") if isinstance(res, dict) else None
        if not isinstance(answers, dict):
            raise RuntimeError(f"Laya returned no answers: {str(res)[:300]}")
        return answers, {"model": (res.get("model") if isinstance(res, dict) else None) or self.model, "token_usage": _usage_from(res.get("usage")), "cost": 0.0}


_GLINER_RUNTIMES: dict = {}
_GLINER_LOCK = threading.Lock()
_GLINER_MARKERS = re.compile(r"\[(?:P|L|C|E|R|DESCRIPTION|EXAMPLE|OUTPUT)\]")


def gliner_sanitize(value) -> str:
    """Make a string legal as a gliner2 label description or instruction: the library injects them
    into its prompt verbatim and rejects its own marker tokens there, including bare parentheses."""
    value = _GLINER_MARKERS.sub(" ", str(value))
    value = value.replace("(", "[").replace(")", "]")
    return " ".join(value.split()) or "-"


class GlinerProvider(DecisionProvider):
    """GLiNER2.5-Decide (``fastino/GLiNER2.5-Decide``), a 340M encoder classifier, in process.

    Each question becomes one single-label classification over the state: choice options are the
    labels (with their descriptions, this model's best layout), a score is a choice over the levels,
    a noul is a choice over true/false with the statement as the instruction. Settings: ``device``
    ('cuda', 'mps' or 'cpu'; default: first available) and ``use_instruction`` (pass choice/score
    instructions to the model too; default false -- measured worse on reconciliation).
    """

    name = "gliner"
    TASK = "q"

    def __init__(self, model: str, settings: dict):
        self.model = model or "fastino/GLiNER2.5-Decide"
        self.device = settings.get("device") or None
        self.use_instruction = bool(settings.get("use_instruction", False))

    @staticmethod
    def pick_device(requested=None) -> str:
        if requested:
            return requested
        import torch

        if torch.cuda.is_available():
            return "cuda"
        if torch.backends.mps.is_available():
            return "mps"
        return "cpu"

    def classifier(self):
        key = (self.model, self.device or "")
        with _GLINER_LOCK, DEVICE_LOCK:
            if key not in _GLINER_RUNTIMES:
                from gliner2.classification import Classifier  # heavy import, deferred

                device = self.pick_device(self.device)
                logger.info(f"Loading GLiNER2.5-Decide {self.model} on {device} (~1.9 GB)")
                _GLINER_RUNTIMES[key] = (Classifier.from_pretrained(self.model).to(device).eval(), device)
            return _GLINER_RUNTIMES[key]

    def labels_for(self, question: dict):
        options = NOUL_OPTIONS if question["type"] == "noul" else question["options"]
        bad = [o["key"] for o in options if gliner_sanitize(o["key"]) != o["key"]]
        if bad:
            raise ValueError(f"option keys must not contain gliner2 marker characters: {bad}")
        if all(not o["text"] for o in options):
            return [o["key"] for o in options]
        return {o["key"]: gliner_sanitize(o["text"] or o["key"]) for o in options}

    def decide(self, state, questions):
        from gliner2.classification import ClassificationSchema

        classifier, device = self.classifier()
        answers = {}
        for name, q in questions.items():
            instruction = q["instructions"] if (self.use_instruction or q["type"] == "noul") else None
            kw = {"instruction": gliner_sanitize(instruction)} if instruction else {}
            schema = ClassificationSchema().single(self.TASK, self.labels_for(q), **kw)
            with DEVICE_LOCK:
                result = classifier.classify(state, schema)
            probs = {k: float(p) for k, p in result.probabilities(self.TASK).items()}
            choice = result.value(self.TASK)
            answers[name] = {"type": q["type"], "choice": choice, "probabilities": probs, "confidence": probs.get(choice, 0.0)}
        return answers, {"model": self.model, "token_usage": None, "cost": 0.0, "device": device}


SYSTEM1_PROMPT = """You are a System One decision model: you answer typed questions about a state with calibrated probabilities, not prose.

STATE:
<<state>>

QUESTIONS:
<<questions>>

Reply with JSON only, of the form {"answers": {"<question name>": <answer>, ...}} where
- a choice answer is {"choice": "<option key>", "probabilities": {"<option key>": <0-1>, ...}, "confidence": <0-1>}, covering every option key, probabilities summing to 1;
- a score answer is {"score": "<level key>", "probabilities": {"<level key>": <0-1>, ...}, "confidence": <0-1>}, covering every level;
- a noul answer is {"truth": true or false, "probability": <0-1 that the statement is true>}.
Judge only from the state. Pick "none" only when it is listed and no other option fits."""


class GenerativeProvider(DecisionProvider):
    """Any chat model, asked for the same JSON through ``chai.ai.llm`` -- a generative baseline so a
    System One model can be compared like for like. Its probabilities are self-reported, not
    calibrated. JSON mode is on; sampling follows the backend's rules unless ``temperature`` is set
    (Gemini 3+ recommends its own default). The LLM settings (api_key, api_base, timeout,
    reasoning_effort, pricing, litellm_params, ...) pass through. Gemini ids get
    ``GeminiComponent``'s credentials and generation handling, with grounding off."""

    name = "generative"
    PASSTHROUGH = (
        "temperature", "top_p", "max_output_tokens", "reasoning_effort", "thinking_budget", "thinking_level",
        "api_key", "api_base", "api_host", "timeout", "pricing", "litellm_params", "extra_body",
        "system_instruction", "location", "safety",
    )

    def __init__(self, model: str, settings: dict):
        from .gemini import GeminiComponent
        from .llm import LLMComponent

        self.model = model
        llm_settings = {k: settings[k] for k in self.PASSTHROUGH if k in settings}
        llm_settings["model"] = model
        llm_settings.setdefault("max_output_tokens", 2048)
        llm_settings["json_mode"] = True
        llm_settings["expected_output"] = "json"
        head = model.split("/", 1)[0]
        if head in ("gemini", "vertex_ai") or model.startswith("gemini-"):
            llm_settings.setdefault("tools", [])
            cls = GeminiComponent
        else:
            cls = LLMComponent
        self.llm = cls({"type": "system1", "id": f"system1:{model}", "name": f"System One ({model})", "settings": llm_settings}, None)

    @staticmethod
    def render_questions(questions: Dict[str, dict]) -> str:
        lines = []
        for name, q in questions.items():
            if q["type"] == "noul":
                lines.append(f"[{name}] noul (is this statement true of the state?): {q['instructions']}")
                continue
            kind = "choice" if q["type"] == "choice" else "score (levels low to high)"
            lines.append(f"[{name}] {kind}: {q['instructions']}")
            for o in q["options"]:
                lines.append(f"  - {o['key']}: {o['text']}" if o["text"] else f"  - {o['key']}")
        return "\n".join(lines)

    def prompt(self, state: str, questions: Dict[str, dict]) -> str:
        return SYSTEM1_PROMPT.replace("<<state>>", state).replace("<<questions>>", self.render_questions(questions))

    def decide(self, state, questions):
        from .ai_utils import extract_json

        messages = [{"role": "user", "content": [{"type": "text", "text": self.prompt(state, questions)}]}]
        response = self.llm.generate_content(messages)
        data = extract_json(self.llm.extract_text(response)) or {}
        answers = data.get("answers") if isinstance(data.get("answers"), dict) else data
        if not isinstance(answers, dict):
            answers = {}
        usage = self.llm.get_usage(response)
        return {n: answers.get(n, {}) for n in questions}, {
            "model": self.llm.model,  # the resolved LiteLLM id, so the ledger's by_model rows line up
            "token_usage": usage,
            "cost": self.llm.get_cost(response, usage),
        }


PROVIDERS = {"jev": JevProvider, "typesafe": JevProvider, "laya": LayaProvider, "gliner": GlinerProvider}


def make_provider(model: str, settings: dict) -> DecisionProvider:
    """The provider for a System One model id: a ``PROVIDERS`` prefix, else the generative baseline."""
    if not model:
        raise ValueError("a System One model id is required, e.g. 'jev/jev-latest', 'laya/convaiinnovations/laya' or a LiteLLM chat model id")
    head, _, rest = str(model).partition("/")
    if head in PROVIDERS:
        return PROVIDERS[head](rest, settings)
    return GenerativeProvider(str(model), settings)


class System1Model:
    """A System One model behind one ``decide`` call, chosen by its id (see the module docstring).

    ``decide(state, questions, layout)`` applies the layout, asks the provider, canonicalizes every
    answer with ``normalize_answer`` and returns ``(answers, meta)`` where *meta* carries
    ``model``, ``provider``, ``token_usage``, ``cost`` and ``duration`` -- the fields a chai result's
    metadata records so the workflow's usage ledger counts the call.
    """

    def __init__(self, settings: dict):
        self.settings = dict(settings or {})
        self.model = str(self.settings.get("model", "") or "")
        self.provider = make_provider(self.model, self.settings)

    def decide(self, state: str, questions: Dict[str, dict], layout: str = "described", heading: str = "Options:") -> Tuple[Dict[str, dict], dict]:
        for name, q in questions.items():
            if q.get("type") not in QUESTION_TYPES:
                raise ValueError(f"question {name!r}: type must be one of {QUESTION_TYPES}")
        laid_state, laid = apply_layout(state, questions, layout, heading)
        start = time.time()
        raw, meta = self.provider.decide(laid_state, laid)
        duration = time.time() - start
        answers = {name: normalize_answer(questions[name], raw.get(name)) for name in questions}
        meta = dict(meta)
        meta.setdefault("model", self.model)
        meta.update({"provider": self.provider.name, "duration": duration, "layout": layout, "state": laid_state})
        return answers, meta

    def __repr__(self):
        return f"<System1Model {self.model} via {self.provider.name}>"
