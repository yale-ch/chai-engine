"""OpenAI and OpenAI-compatible backend for chai.

A huge swath of inference engines speak the OpenAI Chat Completions wire protocol -- the OpenAI API
itself, vLLM, sglang, llama.cpp's server, Together, Groq, OpenRouter, Azure, ... ``OpenAIComponent``
reaches them through LiteLLM's ``openai`` provider with a configurable ``api_host``; the subclasses
``vllm.VLLMComponent`` and ``sglang.SGLangComponent`` preset the conventional local host so workflow
JSON reads more cleanly.
"""

from __future__ import annotations

from .llm import LLMComponent


class OpenAIComponent(LLMComponent):
    """Any OpenAI-compatible chat completions endpoint, on top of ``LLMComponent``.

    With no ``api_host`` this is the OpenAI API (``OPENAI_API_KEY``). With one it is whatever server
    answers there -- most local servers ignore the key, so 'EMPTY' is sent when none is configured.

    Settings:
        - api_host: host:port or URL of the server; '/v1' is appended if missing (default: api.openai.com)
        - api_key: bearer token (default: OPENAI_API_KEY for OpenAI, 'EMPTY' for a custom host)
        - model: model id, as the server advertises it in /v1/models (required)
        - temperature: sampling temperature (default 0.4)
        - top_p: nucleus sampling threshold (default 0.9)
        - max_output_tokens: response token cap (default 4096)
        - image_detail: OpenAI image detail hint 'low' | 'high' | 'auto' (default: not sent)
        - extra_body: dict passed verbatim to the server, e.g. vLLM/sglang extras (default: none)
        - plus every LLMComponent setting (prompt, expected_output, system_instruction, timeout, pricing, ...)
    """

    PROVIDER = "openai"
    API_BASE_SUFFIX = "/v1"
    DEFAULT_TEMPERATURE = 0.4
    DEFAULT_TOP_P = 0.9
    DEFAULT_MAX_OUTPUT_TOKENS = 4096
    ENGINE_NAME = "openai"

    def __init__(self, tree, workflow, parent=None):
        super().__init__(tree, workflow, parent)
        if self.api_base and not self.api_key:
            # a self-hosted server: send a placeholder so the OpenAI client doesn't refuse to start
            self.api_key = "EMPTY"
            self.base_kwargs["api_key"] = "EMPTY"
