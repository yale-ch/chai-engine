"""LM Studio backend for chai.

Talks to a local `LM Studio <https://lmstudio.ai>`_ server (default ``localhost:1234``) through
LiteLLM's ``lm_studio`` provider, which speaks the server's OpenAI-compatible ``/v1`` API. Role mixins
such as ``LMStudioTranscriber`` are generated from ``LMStudioComponent`` by
``chai.ai.create_all_components``.
"""

from .llm import LLMComponent


class LMStudioComponent(LLMComponent):
    """A model served by LM Studio, with the defaults chai has used for it, on top of ``LLMComponent``.

    Settings:
        - api_host: host:port (or URL) of the LM Studio server; '/v1' is appended (default 'localhost:1234')
        - model: model identifier as LM Studio advertises it (default 'qwen/qwen3.5-9b')
        - temperature: sampling temperature (default 0.4)
        - max_output_tokens: response token cap (default 32000)
        - max_image_size: long-edge pixel limit; images are downscaled before upload if set (default 0 = off)
        - plus every LLMComponent setting (prompt, expected_output, system_instruction, timeout, pricing, ...)
    """

    PROVIDER = "lm_studio"
    DEFAULT_MODEL = "qwen/qwen3.5-9b"
    DEFAULT_API_BASE = "http://localhost:1234"
    API_BASE_SUFFIX = "/v1"
    DEFAULT_TEMPERATURE = 0.4
    DEFAULT_MAX_OUTPUT_TOKENS = 32000
    DEFAULT_API_KEY = "EMPTY"
    ENGINE_NAME = "lm_studio"
