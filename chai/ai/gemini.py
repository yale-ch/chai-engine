"""Google Gemini backend for chai.

Talks to Gemini through LiteLLM: Vertex AI when ``GOOGLE_CLOUD_PROJECT`` is set (application-default
credentials), otherwise the Gemini API with ``GEMINI_API_KEY`` / ``GOOGLE_API_KEY``. Role mixins such
as ``GeminiTranscriber`` are generated from ``GeminiComponent`` by ``chai.ai.create_all_components``.
"""

import logging
import os
import re

from .llm import LLMComponent

logger = logging.getLogger("chai")


class GeminiComponent(LLMComponent):
    """Gemini with the defaults chai pipelines have relied on, on top of ``LLMComponent``.

    The model id may be bare (``gemini-3.1-flash-lite``) -- it gets the ``vertex_ai/`` prefix when a
    GCP project is configured and ``gemini/`` otherwise -- or already prefixed. Google Search
    grounding is on by default (``tools: []`` turns it off) and every safety filter is off.

    Reasoning is configured per model generation. Gemini 2.5 models take a token budget
    (``thinking_budget``, default 0 = thinking off). Gemini 3+ models take ``thinking_level``
    ('low' | 'medium' | 'high'; also 'minimal' on flash models), default 'low' here to keep latency
    and cost close to the old thinking-off default -- the API's own default is 'high'. For 3+ models
    no sampling defaults are sent (Google recommends the API default), but an explicit
    ``temperature``/``top_p`` is always honored -- verified live on Vertex, which accepts them.

    Settings:
        - model: Gemini model id (default 'gemini-3.1-flash-lite-preview')
        - location: Vertex AI location when using a GCP project (default 'global')
        - thinking_level: reasoning effort for Gemini 3+ models (default 'low')
        - thinking_budget: thinking-token budget for 2.5 models (default 0 = thinking off)
        - temperature: sampling temperature (default 0.4 for pre-3 models, not sent for 3+ unless set)
        - top_p: nucleus sampling threshold (default 0.9 for pre-3 models, not sent for 3+ unless set)
        - max_output_tokens: response token cap (default 8192)
        - tools: server-side tools: 'search', 'url', 'code' (default ['search']; [] disables grounding)
        - safety: threshold for the four harm categories (default 'OFF')
        - response_mime_type: 'application/json' to make the API guarantee JSON (default: none)
        - plus every LLMComponent setting (prompt, system_instruction, schema, api_key, timeout, pricing, ...)
    """

    DEFAULT_MODEL = "gemini-3.1-flash-lite-preview"
    DEFAULT_MAX_OUTPUT_TOKENS = 8192
    DEFAULT_TOOLS = ["search"]
    DEFAULT_SAFETY = "OFF"
    ENGINE_NAME = "gemini"

    def __init__(self, tree, workflow, parent=None):
        # project name is treated as sensitive; it comes from the environment only
        self.project = os.getenv("GOOGLE_CLOUD_PROJECT", "") or os.getenv("VERTEXAI_PROJECT", "")
        self.env_api_key = os.getenv("GEMINI_API_KEY", os.getenv("GOOGLE_API_KEY", ""))
        settings = tree.get("settings", {}) or {}
        if not self.project and not self.env_api_key and not settings.get("api_key"):
            raise ValueError("Either GOOGLE_CLOUD_PROJECT (Vertex AI) or GEMINI_API_KEY / GOOGLE_API_KEY must be set")
        super().__init__(tree, workflow, parent)
        self.model_version = self._model_version(self.model)

    @staticmethod
    def _model_version(model: str) -> tuple:
        """Parse the generation out of a Gemini model id: 'gemini-3.7-flash' -> (3, 7).

        Ids that don't carry a version (custom endpoints, tuned models) come back as (0, 0), which
        gets the legacy pre-Gemini-3 request shape.
        """
        m = re.search(r"gemini-(\d+)(?:\.(\d+))?", model)
        if not m:
            return (0, 0)
        return (int(m.group(1)), int(m.group(2) or 0))

    def resolve_model(self, model: str) -> str:
        if not model:
            return model
        head = model.split("/", 1)[0]
        if head in ("gemini", "vertex_ai"):
            return model
        return f"vertex_ai/{model}" if self.project else f"gemini/{model}"

    def build_base_kwargs(self) -> dict:
        version = self._model_version(self.model)
        if version < (3, 0):
            # Legacy defaults for 2.x and unversioned ids
            if "temperature" not in self.settings:
                self.temperature = 0.4
            if "top_p" not in self.settings:
                self.top_p = 0.9
        kw = super().build_base_kwargs()

        if version >= (3, 0):
            if "thinking_budget" in self.settings and "thinking_level" not in self.settings:
                logger.warning(
                    "%s: %s takes `thinking_level` (low/medium/high), not `thinking_budget`; "
                    "using the default level 'low'",
                    self,
                    self.model,
                )
            kw.pop("thinking", None)
            kw["reasoning_effort"] = str(self.settings.get("thinking_level", self.settings.get("reasoning_effort", "low")))
        elif version == (2, 5) and "thinking" not in kw and "reasoning_effort" not in kw:
            # Default to turning thinking off: it can run away and ignore the budget
            kw["thinking"] = {"type": "disabled", "budget_tokens": 0}

        if self.project:
            kw["vertex_project"] = self.project
            kw["vertex_location"] = self.settings.get("location", "global")
        elif self.env_api_key and "api_key" not in kw:
            kw["api_key"] = self.env_api_key
        return kw
