"""vLLM backend for chai.

Talks to a `vLLM <https://github.com/vllm-project/vllm>`_ server through its
OpenAI-compatible chat completions endpoint. The server can be a Linux/CUDA
``vllm serve`` instance, an Apple Silicon ``vllm-metal`` instance (text-only
models), or any other vLLM deployment exposing ``/v1/chat/completions``.

See ``docs/vllm.md`` for setup instructions on macOS and NVIDIA hardware.
"""

from __future__ import annotations

from .openai import OpenAIComponent


class VLLMComponent(OpenAIComponent):
    """Component speaking to a vLLM server via its OpenAI-compatible API.

    Differs from :class:`~chai.ai.openai.OpenAIComponent` only in defaults:
    ``localhost:8000`` is the vLLM convention.
    """

    DEFAULT_API_HOST = "localhost:8000"
    ENGINE_NAME = "vllm"
