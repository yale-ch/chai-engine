"""Shared Component for OpenAI-compatible inference servers.

A huge swath of inference engines speak the OpenAI Chat Completions wire
protocol -- vLLM, sglang, LM Studio's ``/v1`` endpoint, Ollama's ``/v1``
endpoint, llama.cpp's server, Together, Groq, OpenRouter, Azure, the actual
OpenAI API, etc. ``OpenAIComponent`` handles them all; concrete subclasses
(``vllm.VLLMComponent``, ``sglang.SGLangComponent``) just preset a default
host/port so workflow JSON reads more cleanly.

The ``openai`` Python client is imported lazily so users who only need
``LMStudioComponent`` or ``GeminiComponent`` don't have to install it.
"""

from __future__ import annotations

import base64
import io
import logging
import time
from pathlib import Path
from typing import Any, List, Tuple

from PIL import Image

from ..core import Component
from ..image_operations import bytes_from_image, image_from_bytes, scale
from ..result import FileItemResult, ItemResult, Result
from .ai_utils import extract_json

logger = logging.getLogger("chai")


class OpenAIComponent(Component):
    """Talks to any OpenAI-compatible chat completions endpoint.

    Input is an ``ItemResult`` or list-shaped Result whose entries carry ``type`` metadata (with a
    one-level unwrap for ``Iterator``-re-wrapped items, see ``_unwrap_typed``): TEXT/DATA values fill
    ``{text_input_<i>}`` prompt slots, IMAGE entries become base64 ``image_url`` content parts. Output
    is an ``ItemResult`` whose value is parsed JSON (when ``expected_output`` is 'json') or raw text,
    with ``token_usage``/``duration``/``type``/``engine``/``model`` metadata. A leading ``</think>``
    block (reasoning models) is stripped. Subclasses (``VLLMComponent``, ``SGLangComponent``) only
    override the class defaults ``DEFAULT_API_HOST``/``DEFAULT_MODEL``/``ENGINE_NAME``.

    Settings:
        - api_host: host:port or full base URL; '/v1' is appended automatically if not already
          present (default 'api.openai.com')
        - api_key: bearer token; most local servers ignore it (default 'EMPTY')
        - model: model id, must match what the server advertises in /v1/models (default '')
        - prompt: prompt template; supports {step_name}, {text_input_<i>}, {input_length},
          {first_input} and {last_input} substitutions (default '')
        - expected_output: 'json' to parse the reply as JSON, anything else for raw text (default 'json')
        - temperature: sampling temperature (default 0.4)
        - top_p: nucleus sampling threshold (default 0.9)
        - max_output_tokens: response token cap, mapped to OpenAI max_tokens (default 4096)
        - max_image_size: long-edge pixels; if set, images are downscaled before being base64-encoded
          (default 0 = off)
        - image_mime_type: mime type declared for attached images (default 'image/jpeg')
        - image_detail: OpenAI image detail hint: 'low', 'high' or 'auto' (default 'auto')
        - timeout: request timeout in seconds (default 600)
        - extra_body: dict passed verbatim to the OpenAI client, e.g. vLLM/sglang extras (default unset)
    """

    DEFAULT_API_HOST: str = "api.openai.com"
    DEFAULT_MODEL: str = ""
    ENGINE_NAME: str = "openai"

    def __init__(self, tree, workflow, parent=None):
        super().__init__(tree, workflow, parent)

        self.api_host = self.settings.get("api_host", self.DEFAULT_API_HOST)
        self.api_key = self.settings.get("api_key", "EMPTY")
        self.model = self.settings.get("model", self.DEFAULT_MODEL)
        self.temperature = self.settings.get("temperature", 0.4)
        self.top_p = self.settings.get("top_p", 0.9)
        self.max_output_tokens = self.settings.get("max_output_tokens", 4096)
        self.image_mime_type = self.settings.get("image_mime_type", "image/jpeg")
        self.image_detail = self.settings.get("image_detail", "auto")
        self.timeout = self.settings.get("timeout", 600)

        self.prompt_text = self.settings.get("prompt", "")
        self.expects = self.settings.get("expected_output", "json")
        self.substitutions = {"ADDITIONAL_CONTEXT": ""}

        self.client = None
        self.connect_to_client()

    def _base_url(self) -> str:
        host = self.api_host
        if not host.startswith("http://") and not host.startswith("https://"):
            host = f"http://{host}"
        host = host.rstrip("/")
        if not host.endswith("/v1"):
            host = f"{host}/v1"
        return host

    def connect_to_client(self):
        try:
            from openai import OpenAI
        except ImportError as e:
            raise ImportError(
                f"The 'openai' package is required for the {self.ENGINE_NAME} backend. "
                "Install it with: pip install openai"
            ) from e

        self.client = OpenAI(
            base_url=self._base_url(),
            api_key=self.api_key,
            timeout=self.timeout,
        )

    def _image_bytes(self, image_source) -> bytes:
        if isinstance(image_source, FileItemResult):
            return image_source.value
        if isinstance(image_source, Image.Image):
            buf = io.BytesIO()
            image_source.save(buf, format="PNG")
            return buf.getvalue()
        if isinstance(image_source, (str, Path)):
            with open(image_source, "rb") as fh:
                return fh.read()
        if isinstance(image_source, bytes):
            return image_source
        raise ValueError(f"Unsupported image source type: {type(image_source)}")

    @staticmethod
    def _unwrap_typed(item: Result):
        """Resolve ``(type, value)`` for a result, looking through one level
        of wrapping that chai's ``Iterator`` introduces.

        ``ListResult`` iteration re-wraps each element via ``valueClass``
        (defaults to ``ItemResult``), which produces ``ItemResult(value=<orig>)``
        with empty metadata. We peek one level deeper so steps still see the
        original ``type`` metadata. Bare strings/bytes are inferred as
        ``TEXT``/``IMAGE`` respectively as a fallback.
        """

        typ = item.metadata.get("type", "")
        if typ:
            return typ, item.value

        inner = item.value
        if isinstance(inner, Result):
            inner_type = inner.metadata.get("type", "")
            if inner_type:
                return inner_type, inner.value
            inner = inner.value

        if isinstance(inner, (bytes, bytearray)):
            return "IMAGE", inner
        if isinstance(inner, str):
            return "TEXT", inner
        return typ, item.value

    def image_to_part(self, image_source) -> dict:
        """Convert an image into an OpenAI chat ``image_url`` content part."""

        img_bytes = self._image_bytes(image_source)

        max_size = self.settings.get("max_image_size", 0)
        if max_size:
            img = image_from_bytes(img_bytes)
            img = scale(img, max_size)
            img_bytes = bytes_from_image(img)
            mime = "image/png"
        else:
            mime = self.image_mime_type

        b64 = base64.b64encode(img_bytes).decode("ascii")
        return {
            "type": "image_url",
            "image_url": {
                "url": f"data:{mime};base64,{b64}",
                "detail": self.image_detail,
            },
        }

    def build_contents(self, input) -> Tuple[str, List[dict]]:
        """Render the prompt and collect image content parts from chai input."""

        inputs: List[dict] = []
        format_vars: dict[str, Any] = {"step_name": self.name}
        format_vars.update(self.substitutions)
        prompt_text = self.prompt_text

        if isinstance(input, ItemResult):
            input = [input]

        for i, item in enumerate(input):
            if isinstance(item, Result):
                typ, value = self._unwrap_typed(item)
                if typ in ("DATA", "TEXT"):
                    slot = f"{{text_input_{i}}}"
                    if slot in prompt_text:
                        format_vars[f"text_input_{i}"] = value
                elif typ == "IMAGE":
                    inputs.append(self.image_to_part(item))
                else:
                    raise NotImplementedError(
                        f"Unsupported result type {typ!r} for {self.ENGINE_NAME}: {item}"
                    )

        format_vars["input_length"] = len(inputs)
        first = input[0] if input else None
        last = input[-1] if input else None
        format_vars["first_input"] = first.file_path.name if isinstance(first, FileItemResult) else ""
        format_vars["last_input"] = last.file_path.name if isinstance(last, FileItemResult) else ""

        try:
            p_text = prompt_text.format(**format_vars)
        except KeyError as e:
            logger.warning(f"Missing substitution in prompt for {self}: {e}")
            p_text = prompt_text

        if not p_text:
            raise ValueError(f"Prompt text in {self} is empty")

        return p_text, inputs

    def generate_content(self, contents: Tuple[str, List[dict]]):
        prompt_text, image_parts = contents

        content_parts: List[dict] = [{"type": "text", "text": prompt_text}]
        content_parts.extend(image_parts)

        kwargs = {
            "model": self.model,
            "messages": [{"role": "user", "content": content_parts}],
            "temperature": self.temperature,
            "top_p": self.top_p,
            "max_tokens": self.max_output_tokens,
        }
        extra_body = self.settings.get("extra_body")
        if extra_body:
            kwargs["extra_body"] = extra_body

        return self.client.chat.completions.create(**kwargs)

    @staticmethod
    def get_usage(response) -> dict:
        usage = getattr(response, "usage", None)
        tokens_in = getattr(usage, "prompt_tokens", -1) if usage else -1
        tokens_out = getattr(usage, "completion_tokens", -1) if usage else -1
        total = getattr(usage, "total_tokens", -1) if usage else -1
        return {
            "total": total,
            "prompt": tokens_in,
            "images": -1,
            "thinking": -1,
            "result": tokens_out,
        }

    def extract_text(self, resp) -> str:
        text = resp.choices[0].message.content or ""
        if "\n</think>\n" in text:
            _, text = text.split("\n</think>\n", 1)
        return text.strip()

    def _process(self, input):
        if self.client is None:
            self.connect_to_client()

        contents = self.build_contents(input)

        start = time.time()
        resp = self.generate_content(contents)
        duration = time.time() - start

        data_type = "DATA"
        txt = self.extract_text(resp)
        if self.expects == "json":
            result = extract_json(txt)
        else:
            result = txt
            data_type = "TEXT"

        metadata = {
            "token_usage": self.get_usage(resp),
            "duration": duration,
            "type": data_type,
            "engine": self.ENGINE_NAME,
            "model": self.model,
        }
        return ItemResult(result, metadata=metadata)
