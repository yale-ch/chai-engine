import logging
import time

from transformers import AutoModelForCausalLM, AutoTokenizer

from ..core import Component
from ..result import ItemResult, Result
from .ai_utils import extract_json, extract_yaml

logger = logging.getLogger("chai")


class TransformersComponent(Component):
    def __init__(self, tree, workflow, parent=None):
        super().__init__(tree, workflow, parent)
        model = self.settings.get("model", None)
        twe = self.settings.get("tie_word_embeddings", True)
        if model is None:
            raise ValueError(f"model setting not found for {self}")
        self.tokenizer = AutoTokenizer.from_pretrained(model)
        self.model = AutoModelForCausalLM.from_pretrained(
            model, tie_word_embeddings=twe, torch_dtype="auto", device_map="auto"
        )
        self.prompt_text = self.settings.get("prompt", "")
        self.max_new_tokens = self.settings.get("max_new_tokens", 2048)
        self.expects = self.settings.get("expected_output", "json")
        self.substitutions = {"ADDITIONAL_CONTEXT": ""}

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

    def build_contents(self, input):
        """Baseline processor for inputs to send to a transformers model"""

        ### Process input into the API call

        print(input)

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
                        raise ValueError(
                            f"Number of input slots in prompt doesn't match inputs in results for {self}"
                        )
                else:
                    raise NotImplementedError(f"Unsupported type {typ} for transformers: {item}")
        try:
            p_text = prompt_text.format(**format_vars)
        except KeyError as e:
            print(f"Missing substitution in prompt for {self}: {e}\n{prompt_text}\n{format_vars}")

        if not p_text:
            raise ValueError(f"Prompt text in {self} is empty")

        return p_text

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
