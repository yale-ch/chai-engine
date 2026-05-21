"""MLX-VLM backend for chai (Apple Silicon vision-language models).

`mlx-vlm <https://github.com/Blaizzy/mlx-vlm>`_ runs vision-language models
natively on Apple Silicon through Apple's MLX framework. Unlike the
``VLLMComponent`` / ``SGLangComponent`` backends, this is **in-process**: the
model is loaded into the Python process via ``mlx_vlm.load`` and inference
runs directly against the Metal GPU -- no HTTP server.

This is the only way (as of May 2026) to run multimodal models like
``Qwen/Qwen3-VL-4B-Instruct`` natively on an Apple Silicon Mac; both
vllm-metal and sglang's MLX backend are still text-only.

See ``docs/mlx_vlm.md`` for hardware/setup notes.
"""

from __future__ import annotations

import io
import logging
import os
import tempfile
import time
from pathlib import Path
from typing import Any, List, Tuple

from PIL import Image

from ..core import Component
from ..image_operations import bytes_from_image, image_from_bytes, scale
from ..result import FileItemResult, ItemResult, Result
from .ai_utils import extract_json

logger = logging.getLogger("chai")


class MLXVLMComponent(Component):
    """Run an MLX-VLM vision-language model in-process on Apple Silicon."""

    DEFAULT_MODEL = "mlx-community/Qwen3-VL-4B-Instruct-4bit"
    ENGINE_NAME = "mlx-vlm"

    _model = None
    _processor = None
    _config = None
    _loaded_model_name: str = ""

    def __init__(self, tree, workflow, parent=None):
        super().__init__(tree, workflow, parent)

        self.model_name = self.settings.get("model", self.DEFAULT_MODEL)
        self.temperature = self.settings.get("temperature", 0.4)
        self.top_p = self.settings.get("top_p", 0.9)
        self.max_output_tokens = self.settings.get("max_output_tokens", 2048)
        self.verbose = self.settings.get("verbose", False)

        self.prompt_text = self.settings.get("prompt", "")
        self.expects = self.settings.get("expected_output", "json")
        self.substitutions = {"ADDITIONAL_CONTEXT": ""}

        self._scratch_dir: Path | None = None
        self._scratch_files: List[Path] = []

        self.load_model()

    def load_model(self):
        if (
            MLXVLMComponent._model is not None
            and MLXVLMComponent._loaded_model_name == self.model_name
        ):
            self.model = MLXVLMComponent._model
            self.processor = MLXVLMComponent._processor
            self.config = MLXVLMComponent._config
            return

        try:
            from mlx_vlm import load
            from mlx_vlm.utils import load_config
        except ImportError as e:
            raise ImportError(
                "The 'mlx-vlm' package is required for the mlx-vlm backend. "
                "Install it with: pip install mlx-vlm (Apple Silicon only)"
            ) from e

        logger.info(f"Loading MLX-VLM model: {self.model_name}")
        model, processor = load(self.model_name)
        config = load_config(self.model_name)

        MLXVLMComponent._model = model
        MLXVLMComponent._processor = processor
        MLXVLMComponent._config = config
        MLXVLMComponent._loaded_model_name = self.model_name

        self.model = model
        self.processor = processor
        self.config = config

    def _image_path(self, image_source) -> str:
        """Return a filesystem path for an image, materializing temp files
        for in-memory / PIL inputs (mlx-vlm currently accepts paths).
        """

        if isinstance(image_source, FileItemResult):
            return str(image_source.file_path)
        if isinstance(image_source, (str, os.PathLike)):
            return os.fspath(image_source)
        if isinstance(image_source, Image.Image):
            buf = io.BytesIO()
            image_source.save(buf, format="PNG")
            return self._stash_bytes(buf.getvalue(), ".png")
        if isinstance(image_source, (bytes, bytearray)):
            return self._stash_bytes(bytes(image_source), ".png")
        raise ValueError(f"Unsupported image source type: {type(image_source)}")

    def _stash_bytes(self, data: bytes, suffix: str) -> str:
        if self._scratch_dir is None:
            self._scratch_dir = Path(tempfile.mkdtemp(prefix="chai-mlxvlm-"))
        path = self._scratch_dir / f"img_{len(self._scratch_files):04d}{suffix}"
        path.write_bytes(data)
        self._scratch_files.append(path)
        return str(path)

    def _maybe_downscale(self, image_path: str) -> str:
        max_size = self.settings.get("max_image_size", 0)
        if not max_size:
            return image_path
        with open(image_path, "rb") as fh:
            data = fh.read()
        img = image_from_bytes(data)
        img = scale(img, max_size)
        return self._stash_bytes(bytes_from_image(img), ".png")

    @staticmethod
    def _unwrap_typed(item: Result):
        """Same one-level-deep type unwrap as the OpenAI-compatible base.

        chai's ``Iterator`` re-wraps each iterated value in ``ItemResult`` with
        empty metadata, so we look one level into ``item.value`` to find the
        original ``type`` set by ``FileItemResult`` / explicit constructors.
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

    def build_contents(self, input) -> Tuple[str, List[str]]:
        """Resolve the chai inputs into ``(prompt_text, [image_paths])``."""

        image_paths: List[str] = []
        format_vars: dict[str, Any] = {"step_name": self.name}
        format_vars.update(self.substitutions)
        prompt_text = self.prompt_text

        if isinstance(input, ItemResult):
            input = [input]

        for i, item in enumerate(input):
            if not isinstance(item, Result):
                continue
            typ, value = self._unwrap_typed(item)
            if typ in ("DATA", "TEXT"):
                slot = f"{{text_input_{i}}}"
                if slot in prompt_text:
                    format_vars[f"text_input_{i}"] = value
            elif typ == "IMAGE":
                src = item if isinstance(item, FileItemResult) else value
                path = self._image_path(src)
                path = self._maybe_downscale(path)
                image_paths.append(path)
            else:
                raise NotImplementedError(
                    f"Unsupported result type {typ!r} for {self.ENGINE_NAME}: {item}"
                )

        format_vars["input_length"] = len(image_paths)
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

        return p_text, image_paths

    def generate_content(self, contents: Tuple[str, List[str]]):
        from mlx_vlm import apply_chat_template, generate

        prompt_text, image_paths = contents

        formatted = apply_chat_template(
            self.processor,
            self.config,
            prompt_text,
            num_images=len(image_paths),
        )

        kwargs = {
            "model": self.model,
            "processor": self.processor,
            "prompt": formatted,
            "image": image_paths if image_paths else None,
            "max_tokens": self.max_output_tokens,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "verbose": self.verbose,
        }
        return generate(**{k: v for k, v in kwargs.items() if v is not None})

    @staticmethod
    def get_usage(resp) -> dict:
        prompt_tokens = int(getattr(resp, "prompt_tokens", -1))
        gen_tokens = int(getattr(resp, "generation_tokens", -1))
        total = (
            prompt_tokens + gen_tokens
            if prompt_tokens >= 0 and gen_tokens >= 0
            else int(getattr(resp, "total_tokens", -1))
        )
        return {
            "total": total,
            "prompt": prompt_tokens,
            "images": -1,
            "thinking": -1,
            "result": gen_tokens,
            "prompt_tps": float(getattr(resp, "prompt_tps", 0.0)),
            "generation_tps": float(getattr(resp, "generation_tps", 0.0)),
            "peak_memory_gb": float(getattr(resp, "peak_memory", 0.0)),
        }

    @staticmethod
    def extract_text(resp) -> str:
        text = getattr(resp, "text", "") or ""
        if "\n</think>\n" in text:
            _, text = text.split("\n</think>\n", 1)
        return text.strip()

    def _cleanup_scratch(self):
        if self._scratch_dir is None:
            return
        for p in self._scratch_files:
            try:
                p.unlink(missing_ok=True)
            except Exception:
                pass
        try:
            self._scratch_dir.rmdir()
        except OSError:
            pass
        self._scratch_dir = None
        self._scratch_files = []

    def _process(self, input):
        contents = self.build_contents(input)
        try:
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
                "model": self.model_name,
            }
            return ItemResult(result, metadata=metadata)
        finally:
            self._cleanup_scratch()
