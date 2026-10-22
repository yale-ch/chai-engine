"""The decision data model: typed questions, their options and layouts, and canonical answers.

A System One question is one of three types, in the same ``{key, text}`` option shape as the
``system1-finetune-bench`` training format, so what a chai workflow asks can be exported as training
data as-is:

* ``choice`` -- pick one of the ``options``;
* ``score``  -- rate the state on an ordinal scale; ``options`` are the levels, low to high;
* ``noul``   -- is the statement (``instructions``) true of the state? No options.

Every provider's raw answer is canonicalized by ``normalize_answer``::

    choice: {"type": "choice", "choice": key, "probabilities": {key: p}, "confidence": p}
    score:  {"type": "score", "score": level_key, "expected": float, "probabilities": {level_key: p}, "confidence": p}
    noul:   {"type": "noul", "truth": bool, "probability": P(true), "probabilities": {"true": p, "false": 1-p}, "confidence": p}

``apply_layout`` implements the bench's inference-time prompt approaches (``described`` /
``in_state`` / ``bare``); ``wire_question`` is the ``{type, instructions, criteria}`` shape the Jev
API and Laya take.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

QUESTION_TYPES = ("choice", "score", "noul")


NONE_KEY = "none"


NULL_KEYS = {"none", "null", "None"}


LAYOUTS = ("described", "in_state", "bare")


TRUE_KEY, FALSE_KEY = "true", "false"


NOUL_OPTIONS = [{"key": TRUE_KEY, "text": "The statement is true"}, {"key": FALSE_KEY, "text": "The statement is false"}]


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


def wire_question(question: dict) -> dict:
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
