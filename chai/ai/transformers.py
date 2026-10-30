"""HuggingFace transformers backend for chai (in-process: text models and vision-language models).

Runs generation directly in the Python process -- no server -- on whatever this machine has: an
NVIDIA GPU (CUDA), Apple Silicon (MPS) or the CPU. A text model is loaded with
``AutoModelForCausalLM``/``AutoTokenizer``; a vision-language model (one whose config has a
``vision_config``, e.g. Chandra OCR 2 or Qwen3.5-VL) with ``AutoModelForImageTextToText``/
``AutoProcessor``, and then IMAGE results are passed to it alongside the prompt. Every call records
``local_compute`` (``chai.hardware``): the device, the time and the machine.

Listed in ``chai.ai.__more_than_all__``: it is skipped by the automatic role-mixin generation
because it loads heavy local weights at construction time, and the roles that use it are made
explicitly (``chai.transcriber.TransformersTranscriber``, ``chai.describer.TransformersDescriber``).
"""

import logging
import threading
import time

from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from ..core import Component
from ..hardware import local_compute, torch_device
from ..image_operations import image_from_bytes
from ..result import ItemResult, Result
from .ai_utils import check_text_slots, extract_json, extract_yaml, warn_prompt_order

logger = logging.getLogger("chai")

# one copy of a model's weights per process and device, however many components use it
_MODELS = {}
_MODELS_LOCK = threading.Lock()


class TransformersComponent(Component):
    """Component that runs a local HuggingFace model on text, or on images and text.

    Input is an ``ItemResult`` or list-shaped Result. TEXT/DATA entries fill ``{text_input_<i>}``
    prompt slots (a TEXT/DATA entry without a matching slot raises); IMAGE entries go to a
    vision-language model with the prompt, and are refused by a text-only one. Output is an
    ``ItemResult`` whose value is parsed JSON ('json'), parsed YAML ('yaml') or raw text, depending
    on ``expected_output``, with ``token_usage``/``duration``/``type``/``engine``/``model`` and
    ``local_compute`` metadata. The chat template is applied with thinking disabled.

    Settings:
        - model: HuggingFace model id or local path (required)
        - prompt: prompt template; supports {step_name} and {text_input_<i>} substitutions (default '')
        - expected_output: 'json', 'yaml' or anything else for raw text (default 'json')
        - max_new_tokens: generation token cap (default 2048)
        - device: 'cuda', 'cuda:1', 'mps' or 'cpu' (default: CUDA if present, then MPS, then CPU)
        - dtype: torch dtype for the weights, e.g. 'bfloat16' or 'float32' (default 'auto'; 'float32'
          on CPU, where half precision is slow or unsupported)
        - max_image_pixels: scale an image down to at most this many pixels before the processor
          sees it (default: leave it to the processor)
        - vision: force the vision-language path on or off (default: from the model's config)
        - tie_word_embeddings: passed to from_pretrained for text models (default true)
    """

    ENGINE_NAME: str = "transformers"

    def __init__(self, tree, workflow, parent=None):
        super().__init__(tree, workflow, parent)
        model = self.settings.get("model", None)
        if model is None:
            raise ValueError(f"model setting not found for {self}")
        # self.model is the loaded model object, so the name it was loaded from is kept separately
        self.model_name = model
        self.device = torch_device(self.settings.get("device"))
        vision = self.settings.get("vision")
        if vision is None:
            vision = getattr(AutoConfig.from_pretrained(model), "vision_config", None) is not None
        self.vision = bool(vision)
        self.model, self.processor = self.load()
        self.tokenizer = getattr(self.processor, "tokenizer", self.processor)
        self.prompt_text = self.settings.get("prompt", "")
        self.max_new_tokens = int(self.settings.get("max_new_tokens", 2048))
        self.expects = self.settings.get("expected_output", "json")
        self.substitutions = {"ADDITIONAL_CONTEXT": ""}
        warn_prompt_order(self)

    def load(self):
        """``(model, processor-or-tokenizer)``, loaded once per process for each model and device."""
        import torch

        dtype = self.settings.get("dtype") or ("float32" if self.device == "cpu" else "auto")
        dtype = dtype if dtype == "auto" else getattr(torch, dtype)
        key = (self.model_name, self.device, str(dtype), self.vision)
        with _MODELS_LOCK:
            if key not in _MODELS:
                logger.info(f"Loading {self.model_name} on {self.device} ({'vision' if self.vision else 'text'})")
                if self.vision:
                    from transformers import AutoModelForImageTextToText, AutoProcessor

                    model = AutoModelForImageTextToText.from_pretrained(self.model_name, dtype=dtype)
                    processor = AutoProcessor.from_pretrained(self.model_name)
                    processor.tokenizer.padding_side = "left"
                else:
                    twe = self.settings.get("tie_word_embeddings", True)
                    model = AutoModelForCausalLM.from_pretrained(self.model_name, tie_word_embeddings=twe, dtype=dtype)
                    processor = AutoTokenizer.from_pretrained(self.model_name)
                _MODELS[key] = (model.to(self.device).eval(), processor, threading.Lock())
            model, processor, self._lock = _MODELS[key]
        return model, processor

    def stop_token_ids(self):
        """The model's end-of-sequence ids plus the chat template's end-of-turn token."""
        eos = self.model.generation_config.eos_token_id
        eos = [eos] if isinstance(eos, int) else list(eos or [])
        im_end = self.tokenizer.convert_tokens_to_ids("<|im_end|>")
        if isinstance(im_end, int) and im_end >= 0 and im_end != self.tokenizer.unk_token_id and im_end not in eos:
            eos.append(im_end)
        return eos or None

    def generate_content(self, contents):
        """Generate from *contents*: the prompt text, or ``(prompt text, [PIL images])`` for a
        vision-language model."""
        text, images = contents if isinstance(contents, tuple) else (contents, [])
        if self.vision:
            content = [{"type": "image", "image": image} for image in images]
            content.append({"type": "text", "text": text})
            inputs = self.processor.apply_chat_template(
                [{"role": "user", "content": content}],
                tokenize=True,
                add_generation_prompt=True,
                return_dict=True,
                return_tensors="pt",
            )
        else:
            prompt = self.tokenizer.apply_chat_template(
                [{"role": "user", "content": text}], tokenize=False, add_generation_prompt=True, enable_thinking=False
            )
            inputs = self.tokenizer([prompt], return_tensors="pt")
        inputs = inputs.to(self.model.device)
        in_tokens = inputs["input_ids"].shape[1]
        # one generation at a time per loaded model: a worker pool shares the weights, not the device.
        # The clock starts once the model is ours, so a call is not charged for its turn in the queue.
        with self._lock:
            start = time.time()
            output_ids = self.model.generate(
                **inputs, max_new_tokens=self.max_new_tokens, eos_token_id=self.stop_token_ids()
            )
            seconds = time.time() - start
        new_ids = output_ids[0][in_tokens:]
        raw_output = self.tokenizer.decode(new_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
        return {
            "text": raw_output,
            "input_tokens": in_tokens,
            "output_tokens": len(new_ids),
            "images": len(images),
            "seconds": seconds,
        }

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
            # image tokens are counted in the prompt; the backend does not split them out
            "images": -1 if resp.get("images") else 0,
        }

    def to_image(self, item):
        """A PIL image from an IMAGE result, scaled to ``max_image_pixels`` when that is set."""
        image = image_from_bytes(item.value).convert("RGB")
        limit = self.settings.get("max_image_pixels")
        if limit and image.width * image.height > int(limit):
            factor = (int(limit) / (image.width * image.height)) ** 0.5
            image = image.resize((max(1, int(image.width * factor)), max(1, int(image.height * factor))))
        return image

    def build_contents(self, input):
        """The prompt text for a text model, or ``(prompt text, [images])`` for a vision-language model."""

        warn_prompt_order(self)
        format_vars = {"step_name": self.name}
        format_vars.update(self.substitutions)
        prompt_text = self.prompt_text
        if isinstance(input, ItemResult):
            input = [input]

        images = []
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
                elif typ == "IMAGE" and self.vision:
                    images.append(self.to_image(item))
                else:
                    raise NotImplementedError(
                        f"Unsupported type {typ} for transformers ({'vision' if self.vision else 'text-only'} model "
                        f"{self.model_name}): {item}"
                    )
        check_text_slots(self, prompt_text, format_vars, input)
        try:
            p_text = prompt_text.format(**format_vars)
        except KeyError as e:
            raise ValueError(f"Missing substitution in prompt for {self}: {e}") from e

        if not p_text:
            raise ValueError(f"Prompt text in {self} is empty")

        return (p_text, images) if self.vision else p_text

    def _process(self, input):
        contents = self.build_contents(input)
        start = time.time()
        resp = self.generate_content(contents=contents)
        # the time the model worked on this call, without waiting for another call to finish
        duration = resp.get("seconds", time.time() - start)

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
        metadata = {
            "token_usage": toks,
            "duration": duration,
            "type": data_type,
            "engine": self.ENGINE_NAME,
            "model": self.model_name,
            "local_compute": local_compute(duration, self.device),
        }
        return ItemResult(result, metadata=metadata)
