"""Google Gemini backend for chai.

Talks to the Gemini API via the ``google-genai`` client, either directly (``GEMINI_API_KEY`` /
``GOOGLE_API_KEY``) or through Vertex AI when ``GOOGLE_CLOUD_PROJECT`` is set. Role mixins such as
``GeminiTranscriber`` are generated from ``GeminiComponent`` by ``chai.ai.create_all_components``.
"""

import io
import logging
import os
import re
import time
from pathlib import Path
from typing import Union

from google import genai
from google.genai import types
from PIL import Image

# from tenacity import retry
from ..core import Component
from ..result import FileItemResult, ItemResult, Result
from .ai_utils import extract_json

logger = logging.getLogger("chai")


class GeminiComponent(Component):
    """Component that sends its input to Google's Gemini API and returns the response.

    Input is an ``ItemResult`` or list-shaped Result whose entries carry ``type`` metadata: TEXT/DATA
    values are formatted into ``{text_input_<i>}`` prompt slots or attached as text parts, IMAGE
    entries are attached as image parts (other binary types are not supported yet). Output is an
    ``ItemResult`` whose value is the parsed JSON (when ``expected_output`` is 'json') or the raw text,
    with ``token_usage``/``duration``/``type`` metadata. Authentication comes from the environment:
    ``GOOGLE_CLOUD_PROJECT`` selects Vertex AI, otherwise ``GEMINI_API_KEY``/``GOOGLE_API_KEY`` is
    required.

    Reasoning is configured per model generation. Gemini 2.5 models take a token budget
    (``thinking_budget``, default 0 = thinking off). Gemini 3+ models (including 3.7) replace the
    budget with ``thinking_level`` ('low' | 'medium' | 'high', default 'low' here to keep pipeline
    latency and cost close to the old thinking-off default -- the API's own default is 'high').

    Settings:
        - model: Gemini model id (default 'gemini-3.1-flash-lite-preview')
        - prompt: prompt template; supports {step_name} and {text_input_<i>} substitutions (default '')
        - expected_output: 'json' to parse the reply as JSON, anything else for raw text (default 'json')
        - temperature: sampling temperature (default 0.4 for pre-3 models; for Gemini 3+ no default
          is sent -- Google recommends the API default -- but an explicit setting is always honored)
        - top_p: nucleus sampling threshold (same per-generation rules as temperature; default 0.9)
        - max_output_tokens: response token cap (default 8192)
        - location: Vertex AI location when using a GCP project (default 'global')
        - hate_speech_safety: HARM_CATEGORY_HATE_SPEECH threshold (default 'OFF')
        - dangerous_content_safety: HARM_CATEGORY_DANGEROUS_CONTENT threshold (default 'OFF')
        - sexually_explicit_safety: HARM_CATEGORY_SEXUALLY_EXPLICIT threshold (default 'OFF')
        - harassment_safety: HARM_CATEGORY_HARASSMENT threshold (default 'OFF')
        - tools: tool names to enable: 'search', 'url', 'code', 'maps' (default ['search']; pass [] to disable grounding)
        - system_instruction: standing instructions sent in the model's system slot (default: none)
        - response_mime_type: e.g. 'application/json' to make the API guarantee JSON (default: none)
        - thinking_budget: thinking-token budget for 2.5 models only (default 0 = thinking off)
        - thinking_level: reasoning effort for Gemini 3+ models: 'low', 'medium' or 'high' (default 'low')
    """

    def __init__(self, tree, workflow, parent=None):
        super().__init__(tree, workflow, parent)

        self.client = None

        # treat project name as sensitive information
        # if project is set, then use vertex, otherwise gemini API
        self.project = os.getenv("GOOGLE_CLOUD_PROJECT", "")
        self.api_key = os.getenv("GEMINI_API_KEY", os.getenv("GOOGLE_API_KEY", ""))
        if not self.project and not self.api_key:
            raise ValueError("Either GEMINI_API_KEY or GOOGLE_API_KEY must be set")

        self.location = self.settings.get("location", "global")
        self.model = self.settings.get("model", "gemini-3.1-flash-lite-preview")
        self.model_version = self._model_version(self.model)
        self.temperature = self.settings.get("temperature", 0.4)
        self.top_p = self.settings.get("top_p", 0.9)
        self.max_output_tokens = self.settings.get("max_output_tokens", 8192)
        self.prompt_text = self.settings.get("prompt", "")
        self.expects = self.settings.get("expected_output", "json")

        hate = self.settings.get("hate_speech_safety", "OFF")
        danger = self.settings.get("dangerous_content_safety", "OFF")
        sex = self.settings.get("sexually_explicit_safety", "OFF")
        harass = self.settings.get("harassment_safety", "OFF")

        self.safety_settings = [
            types.SafetySetting(category="HARM_CATEGORY_HATE_SPEECH", threshold=hate),
            types.SafetySetting(category="HARM_CATEGORY_DANGEROUS_CONTENT", threshold=danger),
            types.SafetySetting(category="HARM_CATEGORY_SEXUALLY_EXPLICIT", threshold=sex),
            types.SafetySetting(category="HARM_CATEGORY_HARASSMENT", threshold=harass),
        ]

        # A system instruction is not the same thing as a longer prompt: it is
        # the slot the model treats as standing instructions, and it is what a
        # finetuning export preserves separately from the user turn.
        self.system_instruction = self.settings.get("system_instruction", "") or None
        # Ask the API itself to guarantee JSON rather than hoping the reply
        # parses; the model may otherwise wrap it in prose or a code fence.
        self.response_mime_type = self.settings.get("response_mime_type", "") or None

        config_kwargs = {
            "max_output_tokens": self.max_output_tokens,
            "safety_settings": self.safety_settings,
            "system_instruction": self.system_instruction,
            "response_mime_type": self.response_mime_type,
        }

        # Gemini 3 models tune their own sampling and Google recommends leaving
        # temperature at its API default, so for 3.x the knobs are sent only
        # when the workflow explicitly sets them. An explicit setting is always
        # honored (3.7-flash on Vertex accepts them, despite what the migration
        # docs imply) -- a model that truly rejects them should fail loudly,
        # never silently sample where the workflow asked for determinism.
        if self.model_version < (3, 0):
            config_kwargs["temperature"] = self.temperature
            config_kwargs["top_p"] = self.top_p
        else:
            if "temperature" in self.settings:
                config_kwargs["temperature"] = self.temperature
            if "top_p" in self.settings:
                config_kwargs["top_p"] = self.top_p

        if self.model_version >= (3, 0):
            if "thinking_budget" in self.settings and "thinking_level" not in self.settings:
                logger.warning(
                    "%s: %s takes `thinking_level` (low/medium/high), not `thinking_budget`; "
                    "using the default level 'low'",
                    self,
                    self.model,
                )
            self.thinking_config = types.ThinkingConfig(
                thinking_level=self.settings.get("thinking_level", "low")
            )
            config_kwargs["thinking_config"] = self.thinking_config
        elif self.model_version == (2, 5):
            # Default to turning off thinking, as it can go wild and ignore the budget
            self.thinking_config = types.ThinkingConfig(
                thinking_budget=self.settings.get("thinking_budget", 0)
            )
            config_kwargs["thinking_config"] = self.thinking_config

        self.base_config = types.GenerateContentConfig(**config_kwargs)

        # Google Search grounding is on by default; disable with tools: [] or
        # pick the exact set with e.g. tools: ["search", "url"].
        self.tools = []
        tls = self.settings.get("tools", ["search"])
        if "search" in tls:
            self.tools.append(types.Tool(google_search=types.GoogleSearch()))
        if "url" in tls:
            self.tools.append({"url_context": {}})
        if "code" in tls:
            self.tools.append(types.Tool(code_execution=types.ToolCodeExecution))
        if "maps" in tls:
            self.tools.append(types.Tool(google_maps=types.GoogleMaps()))
        if self.tools:
            self.base_config.tools = self.tools

        self.retry_options = types.HttpRetryOptions(attempts=3)
        self.http_options = types.HttpOptions(api_version="v1")

        self.substitutions = {"ADDITIONAL_CONTEXT": ""}

        self.connect_to_client()

        # if telemetry:
        # self.run = task(name=name)(self.run)

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

    def connect_to_client(self):
        """
        Connects to Google Cloud Platform.
        Returns: None
        """
        if self.project:
            self.client = genai.Client(
                vertexai=True, project=self.project, location=self.location, http_options=self.http_options
            )
        else:
            self.client = genai.Client(api_key=self.api_key)

    def request_config(self, system_instruction=None):
        """The configured request, optionally with a per-call system instruction.

        Components whose system instruction varies from call to call (a staged
        pipeline, say) pass it here rather than rebuilding the component; when
        it is None the configured one stands.
        """
        if system_instruction is None:
            return self.base_config
        return self.base_config.model_copy(update={"system_instruction": system_instruction})

    def generate_content(self, contents: Union[str, list], system_instruction=None):
        """Synchronous wrapper for generate_content."""
        if not self.client:
            raise RuntimeError("Gemini client not initialized.")

        return self.client.models.generate_content(
            model=self.model, contents=contents, config=self.request_config(system_instruction)
        )

    async def generate_content_async(self, contents: Union[str, list], system_instruction=None):
        """Asynchronous wrapper for generate_content."""
        if not self.client:
            raise RuntimeError("Gemini client not initialized.")
        return await self.client.aio.models.generate_content(
            model=self.model, contents=contents, config=self.request_config(system_instruction)
        )

    @staticmethod
    def extract_text(response) -> str:
        """Safely extract text from a Gemini response, handling candidate fallbacks."""
        try:
            if response.text:
                return response.text.strip()
        except ValueError:
            pass  # Fallback below

        text = ""
        if hasattr(response, "candidates") and response.candidates:
            for part in response.candidates[0].content.parts:
                if hasattr(part, "text") and part.text:
                    text += part.text
        return text.strip()

    @staticmethod
    def get_usage(response) -> dict:
        """Extract usage metadata (tokens) from response."""
        meta = getattr(response, "usage_metadata", None)
        if not meta:
            return {}

        try:
            t_output = meta.candidates_token_count
            t_thinking = meta.thoughts_token_count
            t_prompt = -1
            t_image = -1
            for ptd in meta.prompt_tokens_details:
                if ptd.modality.value == "TEXT":
                    t_prompt = ptd.token_count
                elif ptd.modality.value == "IMAGE":
                    t_image = ptd.token_count
            t_total = meta.total_token_count
            return {
                "total": t_total,
                "prompt": t_prompt,
                "images": t_image,
                "thinking": t_thinking,
                "result": t_output,
            }
        except Exception:
            return {}

    @staticmethod
    def image_to_part(image_source, mime_type: str = "image/png") -> types.Part:
        """Convert various image formats into a google.genai.types.Part."""

        ext_map = {
            ".jpg": "image/jpeg",
            ".jpeg": "image/jpeg",
            ".png": "image/png",
            ".webp": "image/webp",
            ".gif": "image/gif",
        }

        if isinstance(image_source, FileItemResult):
            fn = Path(image_source.file_name)
            mime_type = ext_map.get(fn.suffix.lower(), mime_type)
            img_bytes = image_source.value
        elif isinstance(image_source, str) and image_source.startswith("gs://"):
            path = Path(image_source)
            # Infer mime type from extension if standard
            mime_type = ext_map.get(path.suffix.lower(), mime_type)
            return types.Part.from_uri(file_uri=image_source, mime_type=mime_type)
        elif isinstance(image_source, Image.Image):
            buf = io.BytesIO()
            image_source.save(buf, format=mime_type.split("/")[-1].upper() if "/" in mime_type else "PNG")
            img_bytes = buf.getvalue()
        elif isinstance(image_source, (str, Path)):
            path = Path(image_source)
            # Infer mime type from extension if standard
            mime_type = ext_map.get(path.suffix.lower(), mime_type)
            with open(path, "rb") as f:
                img_bytes = f.read()
        elif isinstance(image_source, bytes):
            img_bytes = image_source
        else:
            raise ValueError(f"Unsupported image source type: {type(image_source)}")

        return types.Part.from_bytes(data=img_bytes, mime_type=mime_type)

    def build_contents(self, input):
        """Baseline processor for inputs to send to Gemini"""

        ### Process input into the API call

        inputs = []
        format_vars = {"step_name": self.name}
        format_vars.update(self.substitutions)
        prompt_text = self.prompt_text
        if not isinstance(input, Result):
            # Raw workflow input (e.g. text typed into a test run) -- wrap it
            # so it flows through the same TEXT path as a transcribed result.
            input = ItemResult(input, metadata={"type": "TEXT" if isinstance(input, str) else "DATA"})
        if isinstance(input, ItemResult):
            input = [input]

        for i, item in enumerate(input):
            if isinstance(item, Result):
                typ = item.metadata.get("type", "")
                if not typ and isinstance(item.value, str):
                    # untyped results holding a string are treated as text
                    typ = "TEXT"
                # DATA, TEXT, IMAGE, AUDIO
                if typ in ["DATA", "TEXT"]:
                    text = item.value if isinstance(item.value, str) else str(item.value)
                    # embed if slot, else attach
                    if f"{{text_input_{i}}}" in prompt_text:
                        format_vars[f"text_input_{i}"] = text
                    else:
                        p = types.Part.from_text(text=text)
                        inputs.append(p)
                elif typ in ["IMAGE", "AUDIO", "VIDEO", "BINARY"]:
                    # attach
                    if typ == "IMAGE":
                        p = self.image_to_part(item)
                        inputs.append(p)
                    else:
                        raise NotImplementedError(f"Unsupported type {typ} for gemini: {item}")
                else:
                    raise NotImplementedError(f"Unsupported type {typ} for gemini: {item}")

        # if type(input[0]) is str:
        #     format_vars["first_input"] = self.inputs[0]
        #     format_vars["last_input"] = self.inputs[-1]
        # elif isinstance(input[0], types.Part):
        #     format_vars["first_input"] = self.inputs[0].file_data.file_uri.rsplit("/", 1)[-1]
        #     format_vars["last_input"] = self.inputs[-1].file_data.file_uri.rsplit("/", 1)[-1]

        # Substitute only the known slots ({step_name}, {text_input_0}, ...) so
        # literal braces -- e.g. JSON examples in the prompt -- survive intact.
        p_text = prompt_text
        for k, v in format_vars.items():
            p_text = p_text.replace("{" + k + "}", str(v))

        if not p_text:
            raise ValueError(f"Prompt text in {self} is empty")

        prompt = types.Part.from_text(text=p_text)
        contents = [types.Content(role="user", parts=[*inputs, prompt])]
        return contents

    def _process(self, input):
        client = self.client
        if client is None:
            self.connect_to_client()

        contents = self.build_contents(input)

        start = time.time()
        resp = self.generate_content(contents=contents)
        duration = time.time() - start

        data_type = "DATA"
        if hasattr(resp, "parsed") and resp.parsed:
            result = resp.parsed.dict()
        else:
            txt = self.extract_text(resp)
            if self.expects == "json":
                result = extract_json(txt)
            else:
                result = txt
                data_type = "TEXT"

        toks = self.get_usage(resp)

        metadata = {"token_usage": toks, "duration": duration, "type": data_type}
        r = ItemResult(result, metadata=metadata)
        return r
