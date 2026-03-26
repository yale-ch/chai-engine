import io
import os
import time

import ujson as json
from google import genai
from google.genai import types
from pillow import Image

from ..core import Component

known_gemini_models = {
    "gemini-3.1-pro-preview": {},
    "gemini-3-flash-preview": {},
    "gemini-3.1-flash-lite-preview": {},
    "gemini-2.5-pro": {},
    "gemini-2.5-flash": {},
    "gemini-2.5-flash-lite": {},
}


class GeminiComponent(Component):
    def __init__(self, tree, workflow, parent=None):
        super().__init__(tree, workflow, parent)

        self.client = None
        self.project = os.getenv("GOOGLE_CLOUD_PROJECT", "")
        self.location = self.settings.get("location", "global")
        self.model = self.settings.get("model", "gemini-3.1-flash-lite-preview")
        self.temperature = self.settings.get("temperature", 0.4)
        self.top_p = self.settings.get("top_p", 0.9)
        self.max_output_tokens = self.settings.get("max_output_tokens", 8192)
        self.prompt = self.settings.get("prompt", "")

        self.safety_settings = [
            types.SafetySetting(category="HARM_CATEGORY_HATE_SPEECH", threshold="OFF"),
            types.SafetySetting(category="HARM_CATEGORY_DANGEROUS_CONTENT", threshold="OFF"),
            types.SafetySetting(category="HARM_CATEGORY_SEXUALLY_EXPLICIT", threshold="OFF"),
            types.SafetySetting(category="HARM_CATEGORY_HARASSMENT", threshold="OFF"),
        ]

        self.base_config = types.GenerateContentConfig(
            temperature=self.temperature,
            top_p=self.top_p,
            max_output_tokens=self.max_output_tokens,
            safety_settings=self.safety_settings,
        )

        self.tools = [types.Tool(google_search=types.GoogleSearch())]
        self.retry_options = types.HttpRetryOptions(attempts=3)
        self.http_options = types.HttpOptions(api_version="v1")
        # timeout = milliseconds
        # retry_options = self.retry_options
        # base_config.tools = self.tools

        if "2.5" in self.model:
            # Default to turning off thinking, as it can go wild and ignore the budget
            self.thinking_config = types.ThinkingConfig(thinking_budget=parent.ai_config.get("thinking_budget", 0))
            self.base_config.thinking_config = self.thinking_config

        self.use_cache = True
        self.connect_to_client()

        # if telemetry:
        # self.run = task(name=name)(self.run)

    def connect_to_client(self):
        """
        Connects to Google Cloud Platform and initializes Vertex AI and OpenAI clients.
        Returns: None
        """
        self.client = genai.Client(
            vertexai=True, project=self.project, location=self.location, http_options=self.http_options
        )

    def setup(self, prompt, schema=None):
        self.prompt_text = prompt
        self.prompt = types.Part.from_text(text=prompt)

    def setup_inputs(self, inputs):
        self.inputs = []
        for i in inputs:
            self.inputs.append(self.make_input(i))

    def make_input(self, input):
        return types.Part.from_uri(file_uri=f"gs://{input}", mime_type="image/jpeg")

    def make_image_input_part(self, fn):
        with Image.open(fn) as img:
            # Convert to bytes
            img_buffer = io.BytesIO()
            img.save(img_buffer, format="PNG")
            img_bytes = img_buffer.getvalue()

        image_part = types.Part.from_bytes(data=img_bytes, mime_type="image/png")
        return image_part

    def run(self):
        client = self.client
        if client is None:
            self.connect_to_client()

        format_vars = {"input_length": len(self.inputs), "step_name": self.name}
        if self.inputs:
            if type(self.inputs[0]) is str:
                format_vars["first_input"] = self.inputs[0]
                format_vars["last_input"] = self.inputs[-1]
            elif isinstance(self.inputs[0], types.Part):
                format_vars["first_input"] = self.inputs[0].file_data.file_uri.rsplit("/", 1)[-1]
                format_vars["last_input"] = self.inputs[-1].file_data.file_uri.rsplit("/", 1)[-1]

        format_vars.update(self.substitutions)
        self.prompt.text = self.prompt_text.format(**format_vars)

        contents = [types.Content(role="user", parts=[*self.inputs, self.prompt])]
        start = time.time()
        resp = client.models.generate_content(model=self.model, contents=contents, config=self.base_config)
        duration = time.time() - start

        if hasattr(resp, "parsed") and resp.parsed:
            result = resp.parsed.dict()
        else:
            try:
                txt = resp.candidates[0].content.parts[0].text
                result = {}
            except Exception:
                result = {"type": "ERROR", "cause": "Failed to retrieve response", "text": ""}
                txt = ""
                self.results = resp
                return

            txt = txt.strip()
            if "```json" in txt:
                txt = txt.split("```json", 1)[1]
                txt = txt.rsplit("```", 1)[0]
                txt = txt.strip()
                result = json.loads(txt)
            elif not txt:
                result = {"type": "ERROR", "cause": "No text returned", "text": ""}
                self.results = resp
                return
            elif (txt[0] == "{" and txt[-1] == "}") or (txt[0] == "[" and txt[-1] == "]"):
                result = json.loads(txt)
            elif not result:
                result = {"type": "TEXT", "text": txt}

        try:
            umd = resp.usage_metadata
            t_output = umd.candidates_token_count
            t_thinking = umd.thoughts_token_count
            t_prompt = -1
            t_image = -1
            for ptd in umd.prompt_tokens_details:
                if ptd.modality.value == "TEXT":
                    t_prompt = ptd.token_count
                elif ptd.modality.value == "IMAGE":
                    t_image = ptd.token_count
            t_total = umd.total_token_count
            toks = {
                "total": t_total,
                "prompt": t_prompt,
                "images": t_image,
                "thinking": t_thinking,
                "result": t_output,
            }
        except Exception:
            toks = {}
        try:
            inputs = [x.file_data.file_uri for x in self.inputs]
        except Exception:
            inputs = []

        cfd = self.base_config.dict()
        del cfd["response_schema"]
        del cfd["safety_settings"]

        result_js = {
            "step": self.name,
            "description": self.description,
            "model": self.model,
            "prompt": self.prompt.text,
            "config": cfd,
            "inputs": inputs,
            "timestamp": time.time(),
            "duration": duration,
            "tokens": toks,
            "result": result,
        }
        try:
            result_js["schema"] = (self.base_config.response_schema.schema(),)
        except Exception:
            result_js["schema"] = None

        self.results = result_js
        return result_js
