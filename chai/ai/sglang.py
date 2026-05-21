"""sglang backend for chai.

Talks to an `sglang <https://github.com/sgl-project/sglang>`_ server through
its OpenAI-compatible chat completions endpoint. The server can be a
Linux/CUDA ``python -m sglang.launch_server`` instance, an Apple Silicon
MLX-backed instance (``SGLANG_USE_MLX=1``, text-only models), or any other
sglang deployment exposing ``/v1/chat/completions``.

See ``docs/sglang.md`` for setup instructions on macOS and NVIDIA hardware.
"""

from __future__ import annotations

from .openai import OpenAIComponent


class SGLangComponent(OpenAIComponent):
    """Component speaking to an sglang server via its OpenAI-compatible API.

    Differs from :class:`~chai.ai.openai.OpenAIComponent` only in defaults:
    ``localhost:30000`` is the sglang convention.
    """

    DEFAULT_API_HOST = "localhost:30000"
    ENGINE_NAME = "sglang"
