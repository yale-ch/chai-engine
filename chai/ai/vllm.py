"""vLLM backend for chai.

Talks to a `vLLM <https://github.com/vllm-project/vllm>`_ server through its OpenAI-compatible chat
completions endpoint (LiteLLM's ``hosted_vllm`` provider). The server can be a Linux/CUDA
``vllm serve`` instance, an Apple Silicon ``vllm-metal`` instance (text-only models), or any other
vLLM deployment exposing ``/v1/chat/completions``. See ``docs/vllm.md`` for setup instructions.
"""

from __future__ import annotations

from .openai import OpenAIComponent


class VLLMComponent(OpenAIComponent):
    """Component speaking to a vLLM server via its OpenAI-compatible API.

    Differs from :class:`~chai.ai.openai.OpenAIComponent` only in defaults: ``localhost:8000`` is
    the vLLM convention, and the LiteLLM provider is ``hosted_vllm``.
    """

    PROVIDER = "hosted_vllm"
    DEFAULT_API_BASE = "http://localhost:8000"
    DEFAULT_API_KEY = "EMPTY"
    ENGINE_NAME = "vllm"
