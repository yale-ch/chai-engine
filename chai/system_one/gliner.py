"""GLiNER2.5-Decide: a DeBERTa-v3 classifier that scores every described label against the text in one pass.

Every question is one single-label classification (``fastino/GLiNER2.5-Decide`` and its fine-tunes,
including Yale's ``GLiNER2.5-Decide-europeana-persons`` and ``-linkedart-recon``). For a choice the
labels are the option keys and their descriptions -- the layout those fine-tunes were trained on,
with the mention and its type named in the state (``Mention: …``, ``Entity type: …``, ``Text: …``)
and no task instruction. A score is a choice over the levels and a noul a choice over true/false,
each with its question put into the text (``text_template``), since level names and ``true`` /
``false`` mean nothing on their own. gliner2 reserves its marker tokens and bare parentheses, so
descriptions are sanitized the way training sanitized them.
"""

from __future__ import annotations

import logging
import re
from types import SimpleNamespace
from typing import Dict

from .base import DEVICE_LOCK, DecisionProvider, load_once, pick_device, register
from .questions import NOUL_OPTIONS

logger = logging.getLogger("chai")

_MARKERS = re.compile(r"\[(?:P|L|C|E|R|DESCRIPTION|EXAMPLE|OUTPUT)\]")

#: Where each question type's text comes from; ``{question}`` and ``{state}`` are filled in.
TEXT_TEMPLATES = {"choice": "{state}", "score": "{question}\n{state}", "noul": "Statement: {question}\n{state}"}


def gliner_sanitize(value) -> str:
    """Make a string legal as a gliner2 label description or instruction.

    The library injects them into its prompt verbatim and rejects its own marker tokens there,
    including bare parentheses; training replaced ``(`` ``)`` with ``[`` ``]``, and a marker that the
    bracket swap produces ("[C]" from "(C)") is stripped again.
    """
    value = _MARKERS.sub(" ", str(value))
    value = value.replace("(", "[").replace(")", "]")
    value = _MARKERS.sub(" ", value)
    return " ".join(value.split()) or "-"


@register("gliner", "gliner2")
class GlinerProvider(DecisionProvider):
    """GLiNER2.5-Decide, in process on CUDA, Apple MPS or CPU; ``gliner/<hub repo or folder>``.

    Settings: ``device`` (default: first available of CUDA, MPS, CPU), ``text_template`` (a string,
    or ``{type: template}`` overriding ``TEXT_TEMPLATES``), ``use_instruction`` (also pass the
    question as gliner2's task instruction; default false -- measured worse on reconciliation), and
    ``temperature`` is not applied: the model's own softmax is the answer.
    """

    name = "gliner"
    TASK = "decision"

    def __init__(self, model: str, settings: dict):
        self.model = model or "fastino/GLiNER2.5-Decide"
        self.settings = dict(settings or {})
        self.device = self.settings.get("device") or None
        self.use_instruction = bool(self.settings.get("use_instruction", False))
        template = self.settings.get("text_template")
        self.templates = dict(TEXT_TEMPLATES)
        if isinstance(template, str):
            self.templates = {t: template for t in self.templates}
        elif isinstance(template, dict):
            self.templates.update(template)
        self.rt = None

    def load(self):
        if self.rt is None:
            self.rt = load_once(("gliner", self.model, self.device or ""), self._load_runtime)
            self.device = self.rt.device

    def _load_runtime(self):
        from gliner2.classification import Classifier  # heavy import, deferred

        device = pick_device(self.device)
        logger.info(f"Loading GLiNER2.5-Decide {self.model} on {device} (~1.9 GB)")
        return SimpleNamespace(classifier=Classifier.from_pretrained(self.model).to(device).eval(), device=device)

    def labels_for(self, question: dict):
        """gliner2 labels for a question: ``{name: description}``, or a list when nothing is described."""
        options = NOUL_OPTIONS if question["type"] == "noul" else question["options"]
        bad = [o["key"] for o in options if gliner_sanitize(o["key"]) != o["key"]]
        if bad:
            raise ValueError(f"option keys must not contain gliner2 marker characters: {bad}")
        if all(not o["text"] for o in options):
            return [o["key"] for o in options]
        return {o["key"]: gliner_sanitize(o["text"] or o["key"]) for o in options}

    def text_for(self, question: dict, state: str) -> str:
        return self.templates[question["type"]].format(question=question["instructions"], state=state).strip()

    def decide(self, state, questions):
        from gliner2.classification import ClassificationSchema

        self.load()
        answers = {}
        for name, q in questions.items():
            kw = {"instruction": gliner_sanitize(q["instructions"])} if self.use_instruction else {}
            schema = ClassificationSchema().single(self.TASK, self.labels_for(q), **kw)
            with DEVICE_LOCK:
                result = self.rt.classifier.classify(self.text_for(q, state), schema)
            probs = {k: float(p) for k, p in result.probabilities(self.TASK).items()}
            choice = result.value(self.TASK)
            answers[name] = {"type": q["type"], "choice": choice, "probabilities": probs, "confidence": probs.get(choice, 0.0)}
        return answers, {"model": self.model, "token_usage": None, "cost": 0.0, "device": self.rt.device}
