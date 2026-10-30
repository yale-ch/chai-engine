"""Any chat model asked for the same JSON: a generative baseline, not a calibrated decision model."""

from __future__ import annotations

from typing import Dict, List

from .base import GENERATIVE, DecisionProvider, register

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


@register(GENERATIVE)
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
        from ..ai.gemini import GeminiComponent
        from ..ai.llm import LLMComponent

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
        # a hosted model is a service, not this machine; a server at localhost is this machine
        self.local = self.llm.is_local

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
        from ..ai.ai_utils import extract_json

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
