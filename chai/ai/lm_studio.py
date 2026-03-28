import io
import logging
import time
from pathlib import Path

import lmstudio as lms
from PIL import Image

# from tenacity import retry
from ..core import Component
from ..result import FileItemResult, ItemResult, Result
from .ai_utils import extract_json

logger = logging.getLogger("chai")

# image_path = "/Users/rs2668/Development/llm/chai-engine/downloads/wf2_1/image_0001.jpg"
# imgh = lms.prepare_image(image_path)
# model = lms.llm("qwen/qwen3.5-35b-a3b")
# chat = lms.Chat()
# msg = chat.add_user_message("Carefully transcribe each line of text in this image. Only return the transcription. Do not include any comments.", images=[imgh])
# pred = model.respond(chat, config)


class LMStudioComponent(Component):
    def __init__(self, tree, workflow, parent=None):
        super().__init__(tree, workflow, parent)
        self.api_host = self.settings.get("api_host", "localhost:1234")
        self.model = self.settings.get("model", "qwen/qwen3.5-9b")
        self.temperature = self.settings.get("temperature", 0.4)
        self.top_p = self.settings.get("top_p", 0.9)
        self.max_output_tokens = self.settings.get("max_output_tokens", 20000)
        self.prompt_text = self.settings.get("prompt", "")
        self.expects = self.settings.get("expected_output", "json")
        self.substitutions = {"ADDITIONAL_CONTEXT": ""}

        self.base_config = {"temperature": self.temperature, "maxTokens": self.max_output_tokens}

        self.client = None
        self.client_model = None
        self.connect_to_client()

    def connect_to_client(self):
        self.client = lms.Client(self.api_host)
        self.client_model = self.client.llm.model(self.model)

    def generate_content(self, contents):
        chat = lms.Chat()
        chat.add_user_message(contents[0], images=contents[1])
        resp = self.client_model.respond(chat)
        return resp

    @staticmethod
    def get_usage(response) -> dict:
        tokens_out = response.stats.predicted_tokens_count
        tokens_in = response.stats.prompt_tokens_count
        t_image = -1
        t_thinking = -1

        return {
            "total": tokens_out + tokens_in,
            "prompt": tokens_in,
            "images": t_image,
            "thinking": t_thinking,
            "result": tokens_out,
        }

    def image_to_part(self, image_source, mime_type: str = "image/png"):
        """Convert various image formats into a file part"""

        if isinstance(image_source, FileItemResult):
            img_bytes = image_source.value
        elif isinstance(image_source, Image.Image):
            buf = io.BytesIO()
            image_source.save(buf, format="PNG")
            img_bytes = buf.getvalue()
        elif isinstance(image_source, (str, Path)):
            path = Path(image_source)
            with open(path, "rb") as f:
                img_bytes = f.read()
        elif isinstance(image_source, bytes):
            img_bytes = image_source
        else:
            raise ValueError(f"Unsupported image source type: {type(image_source)}")

        imgh = self.client.files.prepare_image(img_bytes)
        return imgh

    def build_contents(self, input):
        """Baseline processor for inputs to send to LM Studio"""

        ### Process input into the API call

        inputs = []
        format_vars = {"step_name": self.name}
        format_vars.update(self.substitutions)
        prompt_text = self.prompt_text
        if isinstance(input, ItemResult):
            input = [input]

        for i, item in enumerate(input):
            if isinstance(item, Result):
                typ = item.metadata.get("type", "")
                # DATA, TEXT, IMAGE, AUDIO
                if typ in ["DATA", "TEXT"]:
                    # embed if slot, else attach
                    if f"{{text_input_{i}}}" in prompt_text:
                        format_vars[f"text_input_{i}"] = item.value
                    else:
                        # Can we attach?
                        pass
                elif typ in ["IMAGE"]:
                    # attach
                    p = self.image_to_part(item)
                    inputs.append(p)
                else:
                    raise NotImplementedError(f"Unsupported type {typ} for lm_studio: {item}")

        try:
            p_text = prompt_text.format(**format_vars)
        except KeyError as e:
            print(f"Missing substitution in prompt for {self}: {e}")

        if not p_text:
            raise ValueError(f"Prompt text in {self} is empty")

        return (p_text, inputs)

    def extract_text(self, resp):
        text = resp.content
        print(text)
        if "\n</think>\n" in text:
            thinking, text = text.split("\n</think>\n")
        return text.strip()

    def _process(self, input):
        client = self.client
        if client is None:
            self.connect_to_client()

        contents = self.build_contents(input)

        start = time.time()
        resp = self.generate_content(contents)
        duration = time.time() - start

        data_type = "DATA"
        # LM Studio sets .parsed to .content even if not .structured
        if hasattr(resp, "structured") and resp.structured:
            result = resp.parsed
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
