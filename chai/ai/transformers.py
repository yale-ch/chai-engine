import logging
import time

from transformers import AutoModelForCausalLM, AutoTokenizer

from ..core import Component
from ..result import ItemResult
from .ai_utils import extract_json, extract_yaml

logger = logging.getLogger("chai")


class TransformersComponent(Component):
    def __init__(self, tree, workflow, parent=None):
        super().__init__(tree, workflow, parent)
        model = self.settings.get("model", None)
        if model is None:
            raise ValueError(f"model setting not found for {self}")
        self.tokenizer = AutoTokenizer.from_pretrained(model)
        self.model = AutoModelForCausalLM.from_pretrained(model, torch_dtype="auto", device_map="auto")
        self.prompt_text = self.settings.get("prompt", "")
        self.max_new_tokens = self.settings.get("max_new_tokens", 2048)

    def generate_content(self, contents: str):
        messages = [{"role": "user", "content": contents}]
        text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
        )
        inputs = self.tokenizer([text], return_tensors="pt").to(self.model.device)
        in_tokens = inputs.input_ids.shape[1]
        output_ids = self.model.generate(**inputs, max_new_tokens=self.max_new_tokens)
        out_tokens = output_ids.shape[1] - in_tokens
        raw_output = self.tokenizer.decode(output_ids[0][inputs.input_ids.shape[1] :], skip_special_tokens=True)
        return {"text": raw_output, "input_tokens": in_tokens, "output_tokens": out_tokens}

    @staticmethod
    def extract_text(resp) -> str:
        return resp["text"]

    @staticmethod
    def get_usage(resp) -> dict:
        return {
            "total": resp["input_tokens"] + resp["output_tokens"],
            "prompt": resp["input_tokens"],
            "result": resp["output_tokens"],
            "thinking": 0,
            "images": 0,
        }

    def _process(self, input):
        contents = self.build_contents(input)
        start = time.time()
        resp = self.generate_content(contents=contents)
        duration = time.time() - start

        text = self.extract_text(resp)

        data_type = "DATA"
        if self.expects == "json":
            result = extract_json(text)
        elif self.expects == "yaml":
            result = extract_yaml(text)
        else:
            result = text
            data_type = "TEXT"

        toks = self.get_usage(resp)
        metadata = {"token_usage": toks, "duration": duration, "type": data_type}
        return ItemResult(result, metadata=metadata)
