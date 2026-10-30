"""What every System One provider shares: the provider contract, logit readout, the model registry.

``DecisionProvider`` is the contract -- ``decide(state, questions)`` returns raw answers by question
name plus a meta dict (``model``, ``token_usage``, ``cost``, ``device``). ``LogitProvider`` is the shape
most models take: one logit per option, read out with a temperature per question type (the bench's
``LogitAdapter``), with a two-round knockout for models that read at most ``max_options`` options in
one pass. In-process models load once per process under ``DEVICE_LOCK`` (Apple's MPS backend aborts
the process when two threads submit work at once), so a workflow's threads share one copy.

``FAMILIES`` maps a model id prefix to a provider class and ``PRESETS`` names the models the bench
configures, so ``model: "kev-4b"`` and ``model: "pointer/jaredpalmer/kev-4b"`` both work; an id with
no known prefix is a chat model asked for the same JSON (``generative``). ``System1Model`` wraps a
provider behind one ``decide`` that applies the layout and canonicalizes the answers.
"""

from __future__ import annotations

import logging
import math
import threading
from typing import Dict, List, Optional, Tuple

from .questions import QUESTION_TYPES, apply_layout, normalize_answer, question_keys

logger = logging.getLogger("chai")

#: One process-wide lock for every in-process model load and forward pass. Metal (Apple MPS) aborts
#: the process when two threads submit work at once, and chai's Iterator runs steps on several
#: threads; per-model locks are not enough once two local models share the GPU, so they all take it.
DEVICE_LOCK = threading.RLock()

_LOADED: dict = {}
_LOAD_LOCK = threading.Lock()


def load_once(key, loader):
    """Build something expensive (a loaded model) once per process; later callers get the same object."""
    with _LOAD_LOCK, DEVICE_LOCK:
        if key not in _LOADED:
            _LOADED[key] = loader()
        return _LOADED[key]


def pick_device(requested=None) -> str:
    """The torch device to load a local model on: *requested*, else CUDA, else Apple MPS, else CPU."""
    if requested:
        return requested
    import torch

    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def temperatures(value, default=1.0) -> Dict[str, float]:
    """One softmax temperature per question type from a scalar, a dict, or nothing."""
    if isinstance(value, dict):
        return {t: float(value.get(t, default)) for t in QUESTION_TYPES}
    if value is None or value == "":
        return {t: float(default) for t in QUESTION_TYPES}
    return {t: float(value) for t in QUESTION_TYPES}


def usage_from(raw) -> Optional[dict]:
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
    local = True  # runs on this machine; a provider that calls a service says otherwise

    def decide(self, state: str, questions: Dict[str, dict]) -> Tuple[Dict[str, dict], dict]:
        raise NotImplementedError



class LogitProvider(DecisionProvider):
    """A model that produces one logit per option; the shared readout the bench's adapters use.

    Subclasses implement ``logits(state, question)`` -- one float per slot, in ``slots(question)``
    order (the option keys; for a noul the two ``noul_slots``) -- and may implement ``load()``.
    ``probabilities`` is the softmax of the logits over a per-type ``temperature`` (a scalar or a
    ``{choice, score, noul}`` dict in the settings, else the one the checkpoint ships). A model that
    reads at most ``max_options`` options in one pass scores a longer choice by a two-round knockout:
    P(o) = P(o | its group) * P(its group's winner wins the final), which sums to 1.
    """

    name = ""
    noul_slots: Tuple[str, str] = ("false", "true")  # the order the model reads a noul's two options in
    max_options: Optional[int] = None

    def __init__(self, model: str, settings: dict):
        self.model = model
        self.settings = dict(settings or {})
        self.temperature = temperatures(self.settings.get("temperature"))
        self.device = self.settings.get("device") or None
        self.last_tokens: Optional[int] = None

    def load(self):
        """Load the model (once per process); called before the first decision."""

    def slots(self, question: dict) -> List[str]:
        return list(self.noul_slots) if question["type"] == "noul" else question_keys(question)

    def logits(self, state: str, question: dict) -> List[float]:
        raise NotImplementedError

    def softmax(self, z: List[float], qtype: str, keys: List[str]) -> Dict[str, float]:
        t = self.temperature.get(qtype, 1.0) or 1.0
        top = max(z)
        e = [math.exp((v - top) / t) for v in z]
        total = sum(e) or 1.0
        return {k: v / total for k, v in zip(keys, e)}

    def probabilities(self, state: str, question: dict) -> Dict[str, float]:
        cap = self.max_options
        if cap and question["type"] != "noul" and len(question["options"]) > cap:
            return self.knockout(state, question)
        return self.softmax(self.logits(state, question), question["type"], self.slots(question))

    def knockout(self, state: str, question: dict) -> Dict[str, float]:
        """More options than one pass reads: score groups, then the group winners against each other."""
        cap = self.max_options
        options = question["options"]
        groups = [options[i : i + cap] for i in range(0, len(options), cap)]
        if len(groups) > cap:
            raise ValueError(f"{len(options)} options exceed a two-round knockout of {cap}")
        within: Dict[str, Tuple[float, Optional[list]]] = {}
        winners: List[str] = []
        for group in groups:
            if len(group) == 1:
                within[group[0]["key"]] = (1.0, None)
                winners.append(group[0]["key"])
                continue
            sub = {**question, "options": group}
            p = self.softmax(self.logits(state, sub), question["type"], self.slots(sub))
            for key, value in p.items():
                within[key] = (value, group)
            winners.append(max(p, key=p.get))
        final = {**question, "options": [o for o in options if o["key"] in winners]}
        pf = self.softmax(self.logits(state, final), question["type"], self.slots(final))

        def winner_of(key):
            group = within[key][1]
            return key if group is None else next(w for w in winners if any(o["key"] == w for o in group))

        return {o["key"]: within[o["key"]][0] * pf[winner_of(o["key"])] for o in options}

    def decide(self, state, questions):
        self.load()
        answers = {}
        tokens = 0
        for name, q in questions.items():
            self.last_tokens = None
            answers[name] = {"type": q["type"], "probabilities": self.probabilities(state, q)}
            tokens += self.last_tokens or 0
        usage = {"total": tokens, "prompt": tokens, "result": 0, "thinking": None, "images": None} if tokens else None
        return answers, {"model": self.model, "token_usage": usage, "cost": 0.0, "device": str(self.device or "") or None}


# --------------------------------------------------------------------------- registry

#: Model id prefix -> provider class. The provider modules fill it in with ``register``.
FAMILIES: Dict[str, type] = {}


def register(*prefixes):
    """Class decorator: make a provider the one for these model id prefixes."""

    def decorate(cls):
        for prefix in prefixes:
            FAMILIES[prefix] = cls
        return cls

    return decorate


#: The models the bench configures (``configs/models/*.yaml`` in system1-finetune-bench), by name,
#: with the keys inference needs: family, checkpoint and revision, released adapter and head, the
#: prompt style and the shipped temperature. A workflow names one (``model: "kev-4b"``) and may
#: override any key in its settings. Revisions are the Hub commits the bench pinned on 2026-10-07.
PRESETS: Dict[str, dict] = {
    # kev: Qwen3.5 base + LoRA + pointer head over the options' closing tokens (apache-2.0)
    "kev-4b": {"family": "pointer", "checkpoint": "Qwen/Qwen3.5-4B-Base", "revision": "1001bb4d826a52d1f399e183466143f4da7b741b",
               "adapter": {"repo": "jaredpalmer/kev-4b", "revision": "6cfce5c2fa4b4bd64026336ab649c5ca78857d52"},
               "head": {"repo": "jaredpalmer/kev-4b", "file": "head.pt", "revision": "6cfce5c2fa4b4bd64026336ab649c5ca78857d52"},
               "head_dim": 256, "max_state_tokens": 3072},
    "kev-9b": {"family": "pointer", "checkpoint": "Qwen/Qwen3.5-9B-Base", "revision": "68c46c4b3498877f3ef123c856ecfde50c39f404",
               "adapter": {"repo": "jaredpalmer/kev-9b", "revision": "db029f08b290afd9fee4aa4bbcd9ae48602d1eb0"},
               "head": {"repo": "jaredpalmer/kev-9b", "file": "head.pt", "revision": "db029f08b290afd9fee4aa4bbcd9ae48602d1eb0"},
               "head_dim": 256, "max_state_tokens": 3072},
    "kev-27b": {"family": "pointer", "checkpoint": "jaredpalmer/kev-27b", "revision": "af0e6d551bdc2cc724f3e9d7a8bee1cd4fb8f7bf",
                "head": {"repo": "jaredpalmer/kev-27b", "file": "head.pt", "revision": "af0e6d551bdc2cc724f3e9d7a8bee1cd4fb8f7bf"},
                "head_dim": 256, "max_state_tokens": 3072},
    # letters: one prompt listing lettered options; logits from the LM head's letter rows
    "plumb-4b": {"family": "letters", "style": "jevk5", "checkpoint": "crh225/plumb-4b", "revision": "24f7bf77e7ee258a2d158c61ea2dce2b60321010", "temperature": 2.07},
    "jevk5-4b": {"family": "letters", "style": "jevk5", "checkpoint": "alibiserikbay/JevK5", "revision": "c4f7fdb3aeab5582336406e78d3bef11bf98833d", "temperature": 1.22},
    "jevk5-9b": {"family": "letters", "style": "jevk5", "checkpoint": "alibiserikbay/JevK5-9B", "revision": "d6521a18a86999190e9d775c915af3d6d6772fc4", "temperature": 1.316},
    "decider-2b": {"family": "letters", "style": "decider", "checkpoint": "Mapika/decider-2b", "revision": "533964dae8be954c5b5e19fa4948e48408094c1e",
                   "temperature": {"choice": 1.164, "noul": 1.624, "score": 1.124}},
    "decider-4b": {"family": "letters", "style": "decider", "checkpoint": "Mapika/decider-4b", "revision": "eb5fbdfc9448473ec25e399882912863afbdb70e",
                   "temperature": {"choice": 1.11, "noul": 1.56, "score": 1.287}},
    "decider-35b-a3b": {"family": "letters", "style": "decider", "checkpoint": "Mapika/decider-35b-a3b", "revision": "91470e65478656eb68fde8c0ccdfe575d69d0309", "temperature": 1.08},
    "tev1-4b": {"family": "letters", "style": "tev1", "checkpoint": "togethercomputer/Tev1-4B-experimental", "revision": "0b7becf017daa0e5eb222f8ce7483c8c8259c52f"},
    # candidate: Open-Jev, one chat prompt per option and a scalar head (apache-2.0)
    "open-jev-9b": {"family": "candidate", "checkpoint": "Qwen/Qwen3.5-9B", "revision": "c202236235762e1c871ad0ccb60c8ee5ba337b9a",
                    "adapter": {"repo": "ZefanCai/Open-Jev-9B", "subfolder": "package/checkpoint/adapter", "revision": "47e966881e489511c0c7f5633a9e1960a676a551"},
                    "head": {"repo": "ZefanCai/Open-Jev-9B", "subfolder": "package/checkpoint", "file": "head.pt", "temperature_file": "temperature.json", "revision": "47e966881e489511c0c7f5633a9e1960a676a551"},
                    "max_state_tokens": 3072},
    "open-jev-27b": {"family": "candidate", "checkpoint": "Qwen/Qwen3.8-27B", "revision": "1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0",
                     "adapter": {"repo": "ZefanCai/Open-Jev-27B-v1.1", "subfolder": "package/checkpoint/adapter", "revision": "28cf73067d5b337860bbef3c85b8b82ba8730956"},
                     "head": {"repo": "ZefanCai/Open-Jev-27B-v1.1", "subfolder": "package/checkpoint", "file": "head.pt", "temperature_file": "temperature.json", "revision": "28cf73067d5b337860bbef3c85b8b82ba8730956"},
                     "max_state_tokens": 3072},
    # nli: openjev, a 3-way NLI cross-encoder read as log P(entailment) per option (mit)
    "openjev-nli-4b": {"family": "nli", "checkpoint": "AlexWortega/openjev", "subfolder": "qwen3.5-4b-nli-v5", "revision": "a20448012c213128955ca0c693e7c943865cab77", "max_length": 4096},
    # slots: autotrust JEV, a bare template and a 24-slot head on the last token (apache-2.0)
    "jev-9b": {"family": "slots", "checkpoint": "autotrust/JEV-9B", "revision": "b63f651ce8ed64481d3f5e73ecdb05f740042f01",
               "adapter": {"repo": "autotrust/JEV-9B", "subfolder": "adapter", "revision": "b63f651ce8ed64481d3f5e73ecdb05f740042f01"},
               "head": {"repo": "autotrust/JEV-9B", "revision": "b63f651ce8ed64481d3f5e73ecdb05f740042f01"}, "max_state_tokens": 3072},
    "jev-27b": {"family": "slots", "checkpoint": "autotrust/JEV-27B", "revision": "51740a8891c2a8baefd969237fd44187b3e3a115",
                "adapter": {"repo": "autotrust/JEV-27B", "subfolder": "adapter", "revision": "51740a8891c2a8baefd969237fd44187b3e3a115"},
                "head": {"repo": "autotrust/JEV-27B", "revision": "51740a8891c2a8baefd969237fd44187b3e3a115"}, "max_state_tokens": 3072},
    "gev-26b": {"family": "slots", "checkpoint": "autotrust/GEV-26B-Decide", "revision": "7c89590ead085bf77630b4bf68264ea30b6ddc78",
                "adapter": {"repo": "autotrust/GEV-26B-Decide", "subfolder": "adapter", "revision": "7c89590ead085bf77630b4bf68264ea30b6ddc78", "attach": "full"},
                "head": {"repo": "autotrust/GEV-26B-Decide", "revision": "7c89590ead085bf77630b4bf68264ea30b6ddc78"}, "max_state_tokens": 3072},
    # encoders (ModernBERT-based)
    "laya": {"family": "laya", "checkpoint": "convaiinnovations/laya", "subfolder": "typed-decisions", "revision": "7b928d828b7b0e022f929d9bd2e44165aa270148",
             "max_len": 1536, "head_max_len": 512},
    "von": {"family": "von", "checkpoint": "wfzyx/von", "revision": "498ceba33390b32cfefaab6422ec380318ba9b99", "independent_options": True, "digit_split": False, "max_len": 4096},
    "bekko-400m": {"family": "bekko", "checkpoint": "hotchpotch/bekko-system-one-v0-400m", "revision": "1960df5602bd93cc8d336fdebd9fb68a30926e13",
                   "query_length": 4096, "document_length": 512},
    # classifiers and the rest
    "gliner-decide": {"family": "gliner", "checkpoint": "fastino/GLiNER2.5-Decide"},
    "jev-omni": {"family": "jev-omni", "checkpoint": "akhilaaa3/Jev-Omni"},
    # Yale Cultural Heritage fine-tunes (cc-by-nc-4.0 for the GLiNER ones, apache-2.0 for kev)
    "kev-4b-europeana-persons": {"family": "pointer", "checkpoint": "yale-cultural-heritage/kev-4b-europeana-persons"},
    "gliner-decide-europeana-persons": {"family": "gliner", "checkpoint": "yale-cultural-heritage/GLiNER2.5-Decide-europeana-persons"},
    "gliner-decide-linkedart-recon": {"family": "gliner", "checkpoint": "yale-cultural-heritage/GLiNER2.5-Decide-linkedart-recon"},
}

#: Prefixes that mean "a chat model asked for the same JSON" rather than a decision model.
GENERATIVE = "generative"


def make_provider(model: str, settings: dict) -> DecisionProvider:
    """The provider for a System One model id.

    A ``PRESETS`` name expands to its family and checkpoint (the settings override its keys); a
    ``<family>/<checkpoint>`` id picks the family by prefix; anything else is a LiteLLM chat model id
    for the generative baseline.
    """
    if not model:
        raise ValueError(
            "a System One model id is required: a preset name such as 'kev-4b' or 'laya', a '<family>/<checkpoint>' "
            f"id with a family in {sorted(FAMILIES)}, or a LiteLLM chat model id"
        )
    settings = dict(settings or {})
    if model in PRESETS:
        cfg = {**PRESETS[model], **{k: v for k, v in settings.items() if k != "model"}}
        family = cfg.pop("family")
        checkpoint = cfg.pop("checkpoint")
        return FAMILIES[family](checkpoint, cfg)
    head, _, rest = str(model).partition("/")
    if head in FAMILIES:
        return FAMILIES[head](rest, settings)
    return FAMILIES[GENERATIVE](str(model), settings)


class System1Model:
    """A System One model behind one ``decide`` call, chosen by its id (see ``make_provider``).

    ``decide(state, questions, layout)`` applies the layout, asks the provider, canonicalizes every
    answer with ``normalize_answer`` and returns ``(answers, meta)`` where *meta* carries ``model``,
    ``provider``, ``token_usage``, ``cost`` and ``duration`` -- the fields a chai result's metadata
    records so the workflow's usage ledger counts the call.
    """

    def __init__(self, settings: dict):
        self.settings = dict(settings or {})
        self.model = str(self.settings.get("model", "") or "")
        self.provider = make_provider(self.model, self.settings)

    def decide(self, state: str, questions: Dict[str, dict], layout: str = "described", heading: str = "Options:") -> Tuple[Dict[str, dict], dict]:
        import time

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
        if getattr(self.provider, "local", True):
            from ..hardware import local_compute

            meta["local_compute"] = local_compute(duration, meta.get("device") or getattr(self.provider, "device", None))
        return answers, meta

    def __repr__(self):
        return f"<System1Model {self.model} via {self.provider.name}>"
