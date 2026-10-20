"""LiteLLM backend for chai: one component for every hosted or served chat model.

`LiteLLM <https://docs.litellm.ai>`_ speaks to Google (the Gemini API and Vertex AI), OpenAI,
Anthropic, Ollama, LM Studio, vLLM, sglang and a hundred other providers through one ``completion``
call, normalizes their replies and usage, and prices every call it knows the model of.
``LLMComponent`` turns a chai input into that call and the reply into an ``ItemResult`` carrying
``token_usage`` and ``cost`` metadata. The provider is part of the model id --
``gemini/gemini-3.1-flash-lite``, ``vertex_ai/gemini-3.7-flash``, ``ollama_chat/llama3.2``,
``lm_studio/qwen/qwen3.5-9b``, ``hosted_vllm/Qwen/Qwen3-8B``, ``openai/gpt-4o-mini``,
``anthropic/claude-sonnet-5-5`` -- so a workflow switches models by editing one string.

The named backends (``GeminiComponent``, ``OllamaComponent``, ``LMStudioComponent``,
``OpenAIComponent``, ``VLLMComponent``, ``SGLangComponent``) are thin subclasses that fill in the
provider prefix, the default host and each provider's defaults, so existing workflow JSON keeps
working. Role composites (``LLMExtractor``, ``GeminiTranscriber``, ...) are generated from all of
them by ``chai.ai.create_all_components``.

``litellm`` itself is imported lazily, when a component is built, so a deterministic pipeline never
pays for it.
"""

from __future__ import annotations

import base64
import io
import logging
import time
from pathlib import Path
from typing import Any, List, Optional, Tuple

from PIL import Image

from ..core import Component
from ..image_operations import bytes_from_image, image_from_bytes, scale
from ..result import FileItemResult, ItemResult, Result
from .ai_utils import extract_json, extract_yaml

logger = logging.getLogger("chai")

IMAGE_MIME_TYPES = {
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".png": "image/png",
    ".webp": "image/webp",
    ".gif": "image/gif",
    ".tif": "image/tiff",
    ".tiff": "image/tiff",
}

AUDIO_MIME_TYPES = {
    ".mp3": "audio/mpeg",
    ".wav": "audio/wav",
    ".m4a": "audio/mp4",
    ".ogg": "audio/ogg",
    ".flac": "audio/flac",
}

# Leading bytes of the image formats providers accept, for image bytes that carry no file name.
IMAGE_SIGNATURES = (
    (b"\x89PNG", "image/png"),
    (b"\xff\xd8\xff", "image/jpeg"),
    (b"GIF87a", "image/gif"),
    (b"GIF89a", "image/gif"),
    (b"II*\x00", "image/tiff"),
    (b"MM\x00*", "image/tiff"),
)


def sniff_image_mime(data) -> Optional[str]:
    """The image mime type *data* starts with the signature of, or ``None``."""
    head = bytes(data[:12]) if isinstance(data, (bytes, bytearray)) else b""
    if head[:4] == b"RIFF" and head[8:12] == b"WEBP":
        return "image/webp"
    for signature, mime in IMAGE_SIGNATURES:
        if head.startswith(signature):
            return mime
    return None


# Gemini server-side tools by their chai names; LiteLLM forwards these to the
# Gemini/Vertex request as-is and rejects them for other providers.
GEMINI_TOOLS = {
    "search": {"googleSearch": {}},
    "url": {"urlContext": {}},
    "code": {"codeExecution": {}},
}

HARM_CATEGORIES = {
    "hate_speech_safety": "HARM_CATEGORY_HATE_SPEECH",
    "dangerous_content_safety": "HARM_CATEGORY_DANGEROUS_CONTENT",
    "sexually_explicit_safety": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
    "harassment_safety": "HARM_CATEGORY_HARASSMENT",
}


def litellm_module():
    """The ``litellm`` package, imported on first use (it is a heavy import)."""
    import litellm

    litellm.suppress_debug_info = True
    return litellm


def known_provider(name: str) -> bool:
    """Whether *name* is a provider prefix LiteLLM recognizes (``gemini``, ``ollama_chat``, ...)."""
    if not name:
        return False
    try:
        providers = litellm_module().provider_list
    except Exception:  # pragma: no cover -- very old litellm
        return False
    return name in {str(getattr(p, "value", p)) for p in providers}


class LLMComponent(Component):
    """Send the input to a chat model through LiteLLM and return its reply.

    Input is an ``ItemResult`` or list-shaped Result whose entries carry ``type`` metadata (with a
    one-level unwrap for ``Iterator``-re-wrapped items, see ``unwrap_typed``): TEXT/DATA values fill
    ``{text_input_<i>}`` prompt slots or are attached as text parts when the prompt has no slot for
    them; IMAGE entries become image parts (base64 data URLs, which LiteLLM converts for each
    provider); AUDIO entries become file parts where the provider accepts them. Output is an
    ``ItemResult`` whose value is parsed JSON (``expected_output`` 'json'), parsed YAML ('yaml') or raw
    text, with ``token_usage`` (``prompt``/``result``/``thinking``/``images``/``total``, ``None`` when
    the provider does not report a figure), ``cost`` (USD, ``None`` when unpriceable), ``duration``,
    ``type``, ``engine`` (the provider) and ``model`` metadata. A leading ``</think>`` block from a
    reasoning model is stripped.

    Settings that are not set are not sent, so each provider's own defaults apply; a setting that is
    set is always sent, and a model that rejects it fails loudly rather than being quietly ignored.

    Settings:
        - model: LiteLLM model id '<provider>/<model>' (required unless the backend has a default)
        - prompt: prompt template; {step_name}, {text_input_<i>}, {input_length}, {first_input},
          {last_input} and {ADDITIONAL_CONTEXT} are substituted, other braces (JSON examples) are
          left alone (default '', or the role's default prompt)
        - system_instruction: standing instructions sent as the system message (default: none)
        - expected_output: 'json' to parse the reply as JSON, 'yaml' for YAML, anything else for raw
          text (default 'json')
        - json_mode: ask the API itself to guarantee JSON (response_format json_object) (default false)
        - response_mime_type: Gemini-style spelling of json_mode: 'application/json' (default: none)
        - schema: JSON Schema (dict or JSON string) the reply must match; sent as a json_schema
          response_format so the API enforces it (default: none)
        - temperature: sampling temperature (default: the provider's)
        - top_p: nucleus sampling threshold (default: the provider's)
        - max_output_tokens: reply token cap, sent as max_tokens (default 4096)
        - reasoning_effort: 'minimal' | 'low' | 'medium' | 'high' | 'disable'; LiteLLM maps it to the
          provider's knob -- Gemini 3 thinking_level, OpenAI reasoning_effort, Anthropic thinking
          (default: not sent)
        - thinking_budget: thinking-token budget for models that take one (Gemini 2.5, Anthropic);
          0 turns thinking off (default: not sent)
        - tools: Gemini server-side tools by name: 'search' (Google Search grounding), 'url', 'code';
          a dict is passed through as a tool definition (default [])
        - safety: Gemini safety threshold for all four harm categories, e.g. 'OFF' or
          'BLOCK_ONLY_HIGH' (default: not sent); hate_speech_safety, dangerous_content_safety,
          sexually_explicit_safety and harassment_safety override it per category
        - api_base: base URL of the server for self-hosted/local providers (default: the provider's);
          api_host (host:port) is accepted as a synonym
        - api_key: credential for the provider (default: the provider's environment variable)
        - timeout: request timeout in seconds (default 600)
        - max_image_size: long-edge pixels; images are downscaled before upload when set (default 0 = off)
        - image_mime_type: mime type declared for image bytes that have no file name (default: detected
          from the bytes' signature, else 'image/png'); a file's own extension always wins
        - image_detail: OpenAI-style detail hint 'low' | 'high' | 'auto' (default: not sent)
        - pricing: {"input_per_million": USD, "output_per_million": USD} to price calls LiteLLM
          cannot, e.g. a self-hosted model (default: LiteLLM's price table)
        - extra_body: dict passed to the provider verbatim, e.g. vLLM/sglang extras (default: none)
        - litellm_params: extra keyword arguments passed verbatim to litellm.completion, e.g.
          drop_params or mock_response (default {})
    """

    DEFAULT_MODEL: str = ""
    PROVIDER: str = ""  # prefix prepended to bare model ids ('gemini', 'ollama_chat', ...)
    DEFAULT_API_BASE: str = ""  # local servers; '' means the provider's own endpoint
    API_BASE_SUFFIX: str = ""  # '/v1' for OpenAI-compatible servers
    ENGINE_NAME: str = "litellm"
    DEFAULT_MAX_OUTPUT_TOKENS: int = 4096
    DEFAULT_TEMPERATURE: Optional[float] = None
    DEFAULT_TOP_P: Optional[float] = None
    DEFAULT_TOOLS: list = []
    DEFAULT_SAFETY: Optional[str] = None
    DEFAULT_API_KEY: Optional[str] = None  # e.g. 'EMPTY' for local servers that ignore it

    def __init__(self, tree, workflow, parent=None):
        super().__init__(tree, workflow, parent)

        self.prompt_text = self.settings.get("prompt", "")
        self.expects = self.settings.get("expected_output", "json")
        self.substitutions = {"ADDITIONAL_CONTEXT": ""}
        self.system_instruction = self.settings.get("system_instruction", "") or None

        self.model = self.resolve_model(str(self.settings.get("model", self.DEFAULT_MODEL) or ""))
        if not self.model:
            raise ValueError(f"{self} needs the `model` setting (a LiteLLM model id such as 'gemini/gemini-3.1-flash-lite')")
        self.provider = self.model.split("/", 1)[0] if "/" in self.model else ""
        self.api_base = self.resolve_api_base()
        self.api_key = self.settings.get("api_key") or self.DEFAULT_API_KEY
        self.timeout = float(self.settings.get("timeout", 600))
        self.temperature = self.settings.get("temperature", self.DEFAULT_TEMPERATURE)
        self.top_p = self.settings.get("top_p", self.DEFAULT_TOP_P)
        self.max_output_tokens = self.settings.get("max_output_tokens", self.DEFAULT_MAX_OUTPUT_TOKENS)
        self.image_mime_type = self.settings.get("image_mime_type") or None

        # Everything about the request except the messages, computed once so a
        # misconfiguration fails when the workflow is built, not on the first file.
        self.base_kwargs = self.build_base_kwargs()
        self.validate_model()

    # ------------------------------------------------------------------ configuration

    def resolve_model(self, model: str) -> str:
        """The full LiteLLM id: a bare model name gets this backend's provider prefix."""
        if not model or not self.PROVIDER:
            return model
        head = model.split("/", 1)[0]
        if known_provider(head):
            return model
        return f"{self.PROVIDER}/{model}"

    def validate_model(self):
        """Fail at build time when LiteLLM cannot tell which provider the model id belongs to."""
        try:
            litellm_module().get_llm_provider(self.model)
        except Exception as e:
            raise ValueError(f"{self}: LiteLLM does not recognize the model id {self.model!r}: {e}") from e

    def resolve_api_base(self) -> str:
        """The server URL for local/self-hosted providers, normalized with the backend's suffix."""
        host = self.settings.get("api_base") or self.settings.get("api_host") or self.DEFAULT_API_BASE
        if not host:
            return ""
        host = str(host)
        if not host.startswith(("http://", "https://")):
            host = f"http://{host}"
        host = host.rstrip("/")
        if self.API_BASE_SUFFIX and not host.endswith(self.API_BASE_SUFFIX):
            host = f"{host}{self.API_BASE_SUFFIX}"
        return host

    def response_format(self) -> Optional[dict]:
        schema = self.settings.get("schema")
        if schema:
            if isinstance(schema, str):
                import json

                schema = json.loads(schema)
            return {"type": "json_schema", "json_schema": {"name": "chai_output", "schema": schema, "strict": True}}
        if self.settings.get("json_mode") or self.settings.get("response_mime_type") == "application/json":
            return {"type": "json_object"}
        return None

    def build_tools(self) -> list:
        names = self.settings.get("tools", self.DEFAULT_TOOLS)
        if isinstance(names, str):
            names = [n.strip() for n in names.split(",") if n.strip()]
        tools = []
        for name in names or []:
            if isinstance(name, dict):
                tools.append(name)
            elif name in GEMINI_TOOLS:
                tools.append(dict(GEMINI_TOOLS[name]))
            else:
                raise ValueError(f"{self}: unknown tool {name!r}; choose from {sorted(GEMINI_TOOLS)} or pass a tool dict")
        return tools

    def safety_settings(self) -> Optional[list]:
        default = self.settings.get("safety", self.DEFAULT_SAFETY)
        per_category = {k: self.settings[k] for k in HARM_CATEGORIES if k in self.settings}
        if default is None and not per_category:
            return None
        return [
            {"category": category, "threshold": per_category.get(key, default or "OFF")}
            for key, category in HARM_CATEGORIES.items()
        ]

    def build_base_kwargs(self) -> dict:
        """The ``litellm.completion`` keyword arguments shared by every call (all but ``messages``)."""
        kw: dict = {"model": self.model, "timeout": self.timeout}
        if self.api_base:
            kw["api_base"] = self.api_base
        if self.api_key:
            kw["api_key"] = self.api_key
        if self.temperature is not None:
            kw["temperature"] = float(self.temperature)
        if self.top_p is not None:
            kw["top_p"] = float(self.top_p)
        if self.max_output_tokens:
            kw["max_tokens"] = int(self.max_output_tokens)
        reasoning = self.settings.get("reasoning_effort")
        if reasoning:
            kw["reasoning_effort"] = str(reasoning)
        budget = self.settings.get("thinking_budget")
        if budget is not None:
            budget = int(budget)
            kw["thinking"] = {"type": "enabled" if budget > 0 else "disabled", "budget_tokens": budget}
        fmt = self.response_format()
        if fmt:
            kw["response_format"] = fmt
        tools = self.build_tools()
        if tools:
            kw["tools"] = tools
        safety = self.safety_settings()
        if safety:
            kw["safety_settings"] = safety
        if self.settings.get("extra_body"):
            kw["extra_body"] = dict(self.settings["extra_body"])
        kw.update(self.settings.get("litellm_params") or {})
        return kw

    # ------------------------------------------------------------------ building the request

    @staticmethod
    def peel(source):
        """The Result that carries the media or text, looking through ``ItemResult`` wrappers.

        ``Iterator`` and ``ListResult`` iteration can wrap an existing Result in an untyped
        ``ItemResult``; each wrapper is looked through until a Result whose value is not itself a
        Result. A ``FileItemResult`` is never looked into (its value would read the file).
        """
        while (
            isinstance(source, Result)
            and not isinstance(source, FileItemResult)
            and not source.metadata.get("type")
            and isinstance(source.value, Result)
        ):
            source = source.value
        return source

    @classmethod
    def unwrap_typed(cls, item: Result) -> Tuple[str, Any]:
        """``(type, value)`` for a result, looking through untyped wrappers (see ``peel``).

        Untyped strings are TEXT, untyped bytes are IMAGE.
        """
        typ = item.metadata.get("type", "")
        if typ:
            return typ, item.value
        inner = cls.peel(item)
        if inner is not item:
            inner_type = inner.metadata.get("type", "")
            if inner_type:
                return inner_type, inner.value
        value = inner.value if isinstance(inner, Result) and not isinstance(inner, FileItemResult) else inner
        if isinstance(value, (bytes, bytearray)):
            return "IMAGE", value
        if isinstance(value, str):
            return "TEXT", value
        return typ, item.value

    def _media_bytes(self, source, mime_types: dict) -> Tuple[bytes, Optional[str]]:
        """``(bytes, mime)`` for a media source; *mime* is ``None`` when no file name says what it is."""
        if isinstance(source, FileItemResult):
            return source.value, mime_types.get(source.file_path.suffix.lower())
        if isinstance(source, Result):
            return self._media_bytes(source.value, mime_types)
        if isinstance(source, Image.Image):
            buf = io.BytesIO()
            source.save(buf, format="PNG")
            return buf.getvalue(), "image/png"
        if isinstance(source, (str, Path)):
            path = Path(source)
            with open(path, "rb") as fh:
                return fh.read(), mime_types.get(path.suffix.lower())
        if isinstance(source, (bytes, bytearray)):
            return bytes(source), None
        raise ValueError(f"Unsupported media source type: {type(source)}")

    def image_part(self, source) -> dict:
        """An OpenAI-style ``image_url`` content part (LiteLLM converts it per provider).

        The type declared for the bytes is the file's extension when there is one, else the
        ``image_mime_type`` setting, else what the bytes' signature says, else PNG.
        """
        source = self.peel(source)
        raw = source.value if isinstance(source, Result) and not isinstance(source, FileItemResult) else source
        if isinstance(raw, str) and raw.startswith(("gs://", "http://", "https://")):
            url = raw  # Gemini file URIs / remote images are passed as references
        else:
            img_bytes, mime = self._media_bytes(source, IMAGE_MIME_TYPES)
            mime = mime or self.image_mime_type or sniff_image_mime(img_bytes) or "image/png"
            max_size = int(self.settings.get("max_image_size", 0) or 0)
            if max_size:
                img_bytes = bytes_from_image(scale(image_from_bytes(img_bytes), max_size))
                mime = "image/png"
            url = f"data:{mime};base64,{base64.b64encode(img_bytes).decode('ascii')}"
        part: dict = {"type": "image_url", "image_url": {"url": url}}
        detail = self.settings.get("image_detail")
        if detail:
            part["image_url"]["detail"] = detail
        return part

    def audio_part(self, source) -> dict:
        """A generic ``file`` content part carrying the audio as a data URL (provider-dependent)."""
        audio_bytes, mime = self._media_bytes(self.peel(source), AUDIO_MIME_TYPES)
        return {
            "type": "file",
            "file": {"file_data": f"data:{mime or 'audio/mpeg'};base64,{base64.b64encode(audio_bytes).decode('ascii')}"},
        }

    def build_contents(self, input) -> Tuple[str, List[dict]]:
        """Render the prompt and collect the content parts (attached texts, images, audio)."""
        parts: List[dict] = []
        format_vars: dict = {"step_name": self.name}
        format_vars.update(self.substitutions)
        prompt_text = self.prompt_text

        if not isinstance(input, Result):
            # Raw workflow input (text typed into a test run) flows through the TEXT path
            input = ItemResult(input, metadata={"type": "TEXT" if isinstance(input, str) else "DATA"})
        items = [input] if isinstance(input, ItemResult) else list(input)

        attachments = 0
        for i, item in enumerate(items):
            if not isinstance(item, Result):
                item = ItemResult(item, metadata={"type": "TEXT" if isinstance(item, str) else "DATA"})
            typ, value = self.unwrap_typed(item)
            if typ in ("DATA", "TEXT"):
                text = value if isinstance(value, str) else str(value)
                slot = f"{{text_input_{i}}}"
                if slot in prompt_text:
                    format_vars[f"text_input_{i}"] = text
                else:
                    parts.append({"type": "text", "text": text})
            elif typ == "IMAGE":
                parts.append(self.image_part(item))
                attachments += 1
            elif typ == "AUDIO":
                parts.append(self.audio_part(item))
                attachments += 1
            else:
                raise NotImplementedError(f"Unsupported result type {typ!r} for {self.ENGINE_NAME}: {item}")

        format_vars["input_length"] = attachments
        first, last = (items[0], items[-1]) if items else (None, None)
        format_vars["first_input"] = first.file_path.name if isinstance(first, FileItemResult) else ""
        format_vars["last_input"] = last.file_path.name if isinstance(last, FileItemResult) else ""

        # Substitute only the known slots so literal braces (JSON examples) survive.
        rendered = prompt_text
        for key, value in format_vars.items():
            rendered = rendered.replace("{" + key + "}", str(value))
        if not rendered:
            raise ValueError(f"Prompt text in {self} is empty")
        return rendered, parts

    def build_messages(self, input) -> List[dict]:
        """The chat messages for *input*: an optional system message, then the user turn with the
        attachments first and the rendered prompt last."""
        prompt, parts = self.build_contents(input)
        messages: List[dict] = []
        if self.system_instruction:
            messages.append({"role": "system", "content": self.system_instruction})
        messages.append({"role": "user", "content": [*parts, {"type": "text", "text": prompt}]})
        return messages

    def request_kwargs(self, input) -> dict:
        """The complete ``litellm.completion`` call for *input* -- what ``_process`` sends."""
        return {**self.base_kwargs, "messages": self.build_messages(input)}

    # ------------------------------------------------------------------ the call and its reply

    def generate_content(self, messages: List[dict]):
        """Send *messages* with the configured request options; returns LiteLLM's ``ModelResponse``."""
        return litellm_module().completion(messages=messages, **self.base_kwargs)

    @staticmethod
    def extract_text(response) -> str:
        """The reply text, with a reasoning model's inline ``<think>`` block removed."""
        try:
            text = response.choices[0].message.content or ""
        except (AttributeError, IndexError):
            return ""
        if "</think>" in text:
            text = text.rsplit("</think>", 1)[1]
        return text.strip()

    @staticmethod
    def get_usage(response) -> dict:
        """Token counts in chai's shape; ``None`` where the provider reported nothing."""
        usage = getattr(response, "usage", None)

        def field(obj, name):
            return getattr(obj, name, None) if obj is not None else None

        return {
            "total": field(usage, "total_tokens"),
            "prompt": field(usage, "prompt_tokens"),
            "result": field(usage, "completion_tokens"),
            "thinking": field(field(usage, "completion_tokens_details"), "reasoning_tokens"),
            "images": field(field(usage, "prompt_tokens_details"), "image_tokens"),
        }

    def get_cost(self, response, usage: dict) -> Optional[float]:
        """What the call cost in USD: the ``pricing`` setting when given, else LiteLLM's figure,
        else ``None`` (unpriceable)."""
        pricing = self.settings.get("pricing")
        if pricing:
            prompt = usage.get("prompt") or 0
            result = usage.get("result") or 0
            return (
                prompt * float(pricing.get("input_per_million", 0)) + result * float(pricing.get("output_per_million", 0))
            ) / 1_000_000
        hidden = getattr(response, "_hidden_params", None) or {}
        cost = hidden.get("response_cost")
        if cost is None:
            try:
                cost = litellm_module().completion_cost(completion_response=response, model=self.model)
            except Exception:
                cost = None
        return float(cost) if cost is not None else None

    def parse_output(self, text: str) -> Tuple[Any, str]:
        """``(value, type)`` for the reply text according to ``expected_output``."""
        if self.expects == "json":
            return extract_json(text), "DATA"
        if self.expects == "yaml":
            return extract_yaml(text), "DATA"
        return text, "TEXT"

    def _process(self, input):
        messages = self.build_messages(input)
        start = time.time()
        response = self.generate_content(messages)
        duration = time.time() - start

        value, data_type = self.parse_output(self.extract_text(response))
        usage = self.get_usage(response)
        metadata = {
            "token_usage": usage,
            "cost": self.get_cost(response, usage),
            "duration": duration,
            "type": data_type,
            "engine": self.ENGINE_NAME,
            "provider": self.provider,
            "model": self.model,
        }
        return ItemResult(value, metadata=metadata)
