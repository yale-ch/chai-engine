"""Decision components: System One questions as chai roles.

A System One model (see ``chai.ai.system1``) answers a typed question about a state with a
probability per option instead of generating text. In chai that is three roles:

* ``DecisionClassifier`` -- asks ONE question (choice, score or noul) about the input and labels it
  with the answer: the chosen option key, the chosen level, or ``true``/``false``. The full answer
  (every option's probability, the confidence, the exact state and question sent) rides in the
  result's metadata, so gates can threshold on ``metadata.confidence`` and storage keeps what the
  model saw.
* ``DecisionGate`` -- a noul question as a branch: ``true_steps`` when P(statement is true) clears
  the threshold. Gates are guarantees; a decision model as a gate is the chai idiom for "only do X
  when the model is sure of Y".
* ``DecisionExtractor`` -- asks SEVERAL questions about one state in one call and returns the
  answers as one structured record.

All three take the same inputs -- a text Result, or a dict-valued Result whose fields can be
templated into the state and the question (``{name}``, ``{type}``...) and whose lists can supply the
options (``options_from``), the way an entity record with retrieved ``candidates`` does in
reconciliation -- and the same ``model`` setting (``jev/jev-latest``, ``laya/convaiinnovations/laya``,
``gliner/fastino/GLiNER2.5-Decide``, or any LiteLLM chat model id as a generative baseline).
"""

from __future__ import annotations

import json
import logging
from typing import Dict, List, Optional

from .ai.system1 import LAYOUTS, NONE_KEY, NULL_KEYS, QUESTION_TYPES, System1Model, make_question, normalize_options
from .classifier import Classifier
from .extractor import Extractor
from .gate import Gate
from .result import ItemResult, LabelListResult, Result
from .utils import text_from_input

logger = logging.getLogger("chai")

STATE_FIELDS = ("state", "context_text", "text", "content", "body")


def render_template(template: str, fields: dict) -> str:
    """Fill ``{field}`` placeholders from *fields*; braces that name no field are left alone."""
    out = str(template)
    for key, value in fields.items():
        if isinstance(value, (str, int, float)) or value is None:
            out = out.replace("{" + str(key) + "}", "" if value is None else str(value))
    return out


def input_fields(input) -> dict:
    """The dict a Result carries (for templating), or an empty dict for plain text."""
    value = input.value if isinstance(input, Result) else input
    return dict(value) if isinstance(value, dict) else {}


def state_from_input(input, template: Optional[str] = None) -> str:
    """The state text a decision is about: *template* rendered over the input's fields (``{text}``
    is the input's text), else a dict input's first text-like field, else the input as text."""
    fields = input_fields(input)
    if template:
        return render_template(template, {**fields, "text": text_from_input(input) if not fields else fields.get("text", "")})
    if fields:
        for name in STATE_FIELDS:
            if isinstance(fields.get(name), str) and fields[name].strip():
                return fields[name]
        return json.dumps(fields, ensure_ascii=False, default=str)
    return text_from_input(input)


def dig(record: dict, path: str):
    cur = record
    for part in str(path).split("."):
        if isinstance(cur, dict) and part in cur:
            cur = cur[part]
        else:
            return None
    return cur


def options_from_records(records, key_template: str = "{key}", text_template: str = "{text}") -> List[dict]:
    """Options from a list of records, keyed and described by templates over each record's fields."""
    options = []
    for rec in records or []:
        if not isinstance(rec, dict):
            options.append({"key": str(rec), "text": ""})
            continue
        key = render_template(key_template, rec).strip()
        if not key:
            raise ValueError(f"option key template {key_template!r} is empty for {rec!r}")
        options.append({"key": key, "text": render_template(text_template, rec).strip()})
    return options


class DecisionBase:
    """Settings and helpers the three decision roles share (mixed into a chai role).

    Settings:
        - model: System One model id (required): 'jev/<model>', 'laya/<hf id>', 'gliner/<hf id>', or a
          LiteLLM chat model id for a generative baseline
        - state: template for the state text sent to the model, over a dict input's fields with
          {text} as the input text (default: the input text, or a dict's state/context_text/text field)
        - layout: described (default) | in_state | bare -- where option descriptions go: on the
          options, listed in the state (Laya's best layout), or nowhere
        - options_heading: heading over the options listed in the state for layout in_state; {field}
          placeholders are filled from a dict input (default 'Options:')
        - max_options: cap on options sent to the model, before any none option (default 0 = all)
        - the model's own settings pass through: api_key, api_base, timeout, device, subfolder,
          use_instruction, temperature, pricing, litellm_params, ... (see chai.ai.system1)
    """

    def _init_decision(self):
        if "model" not in self.settings:
            raise ValueError(f"{self} needs the `model` setting (a System One model id)")
        self.layout = str(self.settings.get("layout", "described"))
        if self.layout not in LAYOUTS:
            raise ValueError(f"{self}: layout must be one of {LAYOUTS}")
        self.options_heading = str(self.settings.get("options_heading", "Options:"))
        self.model = System1Model(self.settings)

    def state_for(self, input) -> str:
        return state_from_input(input, self.settings.get("state"))

    def fields_for(self, input) -> dict:
        return {**input_fields(input), "step_name": self.name}

    def heading_for(self, input) -> str:
        return render_template(self.options_heading, self.fields_for(input))

    def build_options(self, spec: dict, input) -> List[dict]:
        """Options for one question spec: static ``options``, or ``options_from`` a list in the input."""
        options = normalize_options(spec.get("options"))
        path = spec.get("options_from")
        if path:
            options += options_from_records(
                dig(input_fields(input), path), spec.get("option_key", "{key}"), spec.get("option_text", "{text}")
            )
        limit = int(spec.get("max_options", self.settings.get("max_options", 0)) or 0)
        if limit:
            options = options[:limit]
        none = spec.get("none_option")
        if none:
            text = "None of the options" if none is True else str(none)
            options = [o for o in options if o["key"] != NONE_KEY] + [{"key": NONE_KEY, "text": text}]
        return options

    def build_question(self, spec: dict, input) -> dict:
        qtype = str(spec.get("type", spec.get("question_type", "choice")))
        instructions = spec.get("instructions", spec.get("question", ""))
        instructions = render_template(instructions, self.fields_for(input))
        options = None if qtype == "noul" else self.build_options(spec, input)
        if qtype != "noul" and not options:
            raise ValueError(f"{self}: a {qtype} question needs options (`options` or `options_from`)")
        return make_question(qtype, instructions, options)

    def usage_metadata(self, meta: dict) -> dict:
        return {
            "model": meta.get("model"),
            "provider": meta.get("provider"),
            "token_usage": meta.get("token_usage"),
            "cost": meta.get("cost"),
            "duration": meta.get("duration"),
        }


class DecisionClassifier(DecisionBase, Classifier):
    """Ask a System One model one question about the input and label the input with its answer.

    Input is a text Result (the state) or a dict-valued Result (an entity record, a row); a raw
    string works too. Output is a ``LabelListResult`` with one label -- the chosen option key for a
    choice, the chosen level key for a score, ``true``/``false`` for a noul, or ``none`` when the
    none option wins or the top probability is below ``threshold`` -- and metadata: ``probabilities``
    (every option), ``confidence``, ``choice``/``score``+``expected``/``probability``+``truth``,
    ``decision`` (the exact state and question sent and the canonical answer), plus
    ``model``/``token_usage``/``cost`` for the usage ledger. With ``register_on`` a later
    ``LabelTestGate`` or ``SwitchGate`` routes on the label; ``ThresholdGate`` on the confidence.

    Settings:
        - question_type: choice (default) | score | noul
        - question: the question or statement; {field} placeholders are filled from a dict input,
          {step_name} is this component's name (required)
        - options: for choice/score -- {key: description}, [key, ...] or [{key, text}, ...]; for a
          score, the levels low to high (required unless options_from)
        - options_from: dotted path to a list of records in a dict input whose entries become the
          options (e.g. 'candidates')
        - option_key: template over each record for its option key (default '{key}')
        - option_text: template over each record for its description (default '{text}')
        - none_option: add a "none of these" option the model may pick: its description, or true
          for a default text (default: not added)
        - threshold: for choice/score, the top probability below which the label is 'none'
          (default 0 = always the top option); for noul, the P(true) at or above which the label
          is 'true' (default 0.5)
        - model, state, layout, options_heading, max_options: see DecisionBase
    """

    def __init__(self, tree, workflow, parent=None):
        Classifier.__init__(self, tree, workflow, parent)
        self._init_decision()
        self.question_type = str(self.settings.get("question_type", self.settings.get("type", "choice")))
        if self.question_type not in QUESTION_TYPES:
            raise ValueError(f"{self}: question_type must be one of {QUESTION_TYPES}")
        if not self.settings.get("question"):
            raise ValueError(f"{self} needs the `question` setting")
        default_threshold = 0.5 if self.question_type == "noul" else 0.0
        self.threshold = float(self.settings.get("threshold", default_threshold))

    def label_for(self, answer: dict) -> str:
        if answer["type"] == "noul":
            return "true" if answer["probability"] >= self.threshold else "false"
        chosen = answer["choice"] if answer["type"] == "choice" else answer["score"]
        if chosen in NULL_KEYS:
            return NONE_KEY
        if self.threshold and answer["probabilities"].get(chosen, 0.0) < self.threshold:
            return NONE_KEY
        return chosen

    def _process(self, input):
        spec = {**self.settings, "type": self.question_type}
        state = self.state_for(input)
        question = self.build_question(spec, input)
        real_options = [o for o in question.get("options", []) if o["key"] not in NULL_KEYS]
        if self.question_type != "noul" and not real_options:
            # nothing to choose from (e.g. no candidates were retrieved): a null decision, no model call
            answer = {"type": self.question_type, "choice" if self.question_type == "choice" else "score": NONE_KEY,
                      "probabilities": {NONE_KEY: 1.0}, "confidence": 1.0}
            if self.question_type == "score":
                answer["expected"] = 0.0
            meta = {"model": self.model.model, "provider": self.model.provider.name, "skipped": "no options"}
        else:
            answers, meta = self.model.decide(state, {"q": question}, self.layout, self.heading_for(input))
            answer = answers["q"]
        label = self.label_for(answer)
        metadata = {
            "type": "DATA",
            **self.usage_metadata(meta),
            "probabilities": answer["probabilities"],
            "confidence": answer["confidence"],
            "decision": {
                "state": meta.get("state", state),
                "question": question,
                "layout": self.layout,
                "answer": answer,
                "threshold": self.threshold,
                "skipped": meta.get("skipped"),
            },
        }
        for field in ("choice", "score", "expected", "probability", "truth"):
            if field in answer:
                metadata[field] = answer[field]
        self._emit("decision", label=label, confidence=answer["confidence"], question_type=self.question_type)
        return LabelListResult([label], input=input, processor=self, metadata=metadata)


class DecisionGate(DecisionBase, Gate):
    """Branch on a noul question answered by a System One model.

    The statement in ``question`` is put to the model about the input's state; the input goes to
    ``true_steps`` when P(true) is at or above ``threshold`` and to ``false_steps`` otherwise. The
    answer is appended to the input Result's ``extra["decisions"]`` (gates produce no result of
    their own) and reported on the ``component_end`` event as ``probability``; the call is counted
    in the workflow's usage ledger.

    Settings:
        - question: the statement to test, with {field} placeholders from a dict input (required)
        - threshold: P(true) at or above which the true branch runs (default 0.5)
        - model, state: see DecisionBase
    """

    def __init__(self, tree, workflow, parent=None):
        Gate.__init__(self, tree, workflow, parent)
        self._init_decision()
        if not self.settings.get("question"):
            raise ValueError(f"{self} needs the `question` setting (a statement about the input)")
        self.threshold = float(self.settings.get("threshold", 0.5))

    def decide(self, input) -> dict:
        state = self.state_for(input)
        question = self.build_question({"type": "noul", "instructions": self.settings["question"]}, input)
        answers, meta = self.model.decide(state, {"q": question}, "described")
        answer = answers["q"]
        record = {"component": self.id, "state": meta.get("state", state), "question": question, "answer": answer,
                  **self.usage_metadata(meta)}
        if isinstance(input, Result):
            input.extra.setdefault("decisions", []).append(record)
        if self.workflow is not None:
            self.workflow.record_usage(self, record)
        return answer

    def _test(self, input):
        answer = self.decide(input)
        self._last_probability = answer["probability"]
        return answer["probability"] >= self.threshold

    def process(self, input):
        # Gates bypass Component.process; add the probability to the end event.
        self._emit("component_start")
        try:
            case = self._test(input)
            result = self._process(input, case)
        except Exception as e:
            self._emit("component_error", error=str(e))
            raise
        from .core import result_preview

        self._emit(
            "component_end",
            preview=result_preview(result),
            branch="true" if case else "false",
            probability=getattr(self, "_last_probability", None),
            result=result,
        )
        return result


class DecisionExtractor(DecisionBase, Extractor):
    """Ask a System One model several questions about one state and return the answers as a record.

    Input as for ``DecisionClassifier``. Output is an ``ItemResult`` whose value is
    ``{question name: canonical answer}`` (see chai.ai.system1 for the answer shapes), with
    ``decision`` metadata holding the exact state and questions sent, plus
    ``model``/``token_usage``/``cost`` for the usage ledger. Jev answers all questions in one API
    call; the other providers loop.

    Settings:
        - questions: {name: {type, question|instructions, options|options_from, option_key,
          option_text, none_option, max_options}} -- one entry per question (required)
        - model, state, layout, options_heading, max_options: see DecisionBase
    """

    def __init__(self, tree, workflow, parent=None):
        Extractor.__init__(self, tree, workflow, parent)
        self._init_decision()
        specs = self.settings.get("questions")
        if not isinstance(specs, dict) or not specs:
            raise ValueError(f"{self} needs the `questions` setting: {{name: {{type, question, options}}}}")
        self.question_specs: Dict[str, dict] = {str(k): dict(v) for k, v in specs.items()}

    def _process(self, input):
        state = self.state_for(input)
        questions = {name: self.build_question(spec, input) for name, spec in self.question_specs.items()}
        answers, meta = self.model.decide(state, questions, self.layout, self.heading_for(input))
        metadata = {
            "type": "DATA",
            **self.usage_metadata(meta),
            "decision": {"state": meta.get("state", state), "questions": questions, "layout": self.layout},
        }
        return ItemResult(answers, input=input, processor=self, metadata=metadata)
