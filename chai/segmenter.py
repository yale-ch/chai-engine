"""Segmenters: components that break content into smaller pieces.

Covers text splitting (``TextSegmenter``/``WordSegmenter``), image region detection
(``YoloSegmenter``) and AI-generated segmentations (``GeminiSegmenter`` etc., generated from
``chai.ai``). Output is a list-shaped Result of segments, usually consumed by an ``Iterator``.
"""

import io
import re

from .ai import create_all_components
from .core import Component
from .result import FileItemResult, ListResult
from .utils import text_from_input


class Segmenter(Component):
    """Takes content and breaks it up into segments.

    Abstract base for the segmenter role: subclasses implement ``_process`` and return a list-shaped
    Result (``ListResult`` of text chunks, ``FileItemResult`` crops, ...) derived from the input. When
    no prompt is configured, AI-backed variants use the workflow's default ``segmentation`` prompt.
    """

    def __init__(self, tree, workflow, parent=None):
        super().__init__(tree, workflow, parent)
        if not getattr(self, "prompt_text", None):
            self.prompt_text = self.workflow.default_prompts.get("segmentation", "")
        self.expects = "data"

    def _process(self, input):
        raise NotImplementedError()


globals().update(create_all_components(Segmenter))


class TextSegmenter(Segmenter):
    """Splits text into segments deterministically -- no model required.

    Settings:
        - mode:    paragraph (default) | line | sentence | word | regex
        - pattern: split regex, used when mode is 'regex'
    """

    _MODES = {
        "paragraph": r"\n\s*\n",
        "line": r"\n",
        "sentence": r"(?<=[.!?])\s+",
        "word": r"\s+",
    }

    def __init__(self, tree, workflow, parent=None):
        super().__init__(tree, workflow, parent)
        mode = self.settings.get("mode", "paragraph")
        if mode == "regex":
            pattern = self.settings.get("pattern")
            if not pattern:
                raise ValueError(f"TextSegmenter ({self!r}) needs the `pattern` setting when mode is 'regex'")
        elif mode in self._MODES:
            pattern = self._MODES[mode]
        else:
            raise ValueError(f"TextSegmenter ({self!r}) unknown mode {mode!r}")
        self.splitter = re.compile(pattern)

    def _process(self, input):
        segments = [s.strip() for s in self.splitter.split(text_from_input(input))]
        segments = [s for s in segments if s]
        return ListResult(segments, input=input, processor=self)


class WordSegmenter(TextSegmenter):
    """Splits text into words (TextSegmenter in 'word' mode)."""

    def __init__(self, tree, workflow, parent=None):
        tree.setdefault("settings", {})["mode"] = "word"
        super().__init__(tree, workflow, parent)


class YoloSegmenter(Segmenter):
    """Runs a YOLO detection model and emits one FileItemResult per bounding box.

    Each emitted result carries the YOLO class, bbox, and confidence in its
    metadata; downstream gates/iterators can branch on those. When ``crop`` is
    true the cropped PNG bytes are populated in ``file_bytes`` so AI components
    (e.g. GeminiTranscriber) can consume the region without touching disk.

    Settings:
        - model:      path or name of a YOLO detection model (default 'yolo11n.pt')
        - confidence: minimum box confidence to keep (default 0.25)
        - crop:       if true (default), emit cropped PNG bytes; otherwise only metadata
        - classes:    optional list of class names to keep; others are dropped
    """

    def __init__(self, tree, workflow, parent=None):
        super().__init__(tree, workflow, parent)
        from ultralytics import YOLO

        self.model = YOLO(self.settings.get("model", "yolo11n.pt"))
        self.confidence = self.settings.get("confidence", 0.25)
        self.crop = self.settings.get("crop", True)
        self.keep_classes = set(self.settings.get("classes", []) or [])

    def _process(self, input):
        from PIL import Image

        if isinstance(input, FileItemResult):
            image_path = input.file_name
        else:
            image_path = input.value if hasattr(input, "value") else str(input)

        img = Image.open(image_path) if self.crop else None
        out = ListResult([], input=input, processor=self)

        for result in self.model(image_path, verbose=False):
            for box in result.boxes:
                conf = float(box.conf[0])
                if conf < self.confidence:
                    continue
                cls_name = result.names[int(box.cls[0])]
                if self.keep_classes and cls_name not in self.keep_classes:
                    continue
                x1, y1, x2, y2 = (float(v) for v in box.xyxy[0].cpu().numpy())

                crop = FileItemResult(
                    "crop.png",
                    input=input,
                    processor=self,
                    metadata={
                        "type": "IMAGE",
                        "yolo_class": cls_name,
                        "bbox": [x1, y1, x2, y2],
                        "confidence": conf,
                    },
                )
                if self.crop:
                    buf = io.BytesIO()
                    img.crop((x1, y1, x2, y2)).save(buf, format="PNG")
                    crop.file_bytes = buf.getvalue()
                out.append(crop)

        return out
