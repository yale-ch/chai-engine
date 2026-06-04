import io

from .ai import create_all_components
from .core import Component
from .result import FileItemResult, ItemResult, ListResult


class Segmenter(Component):
    """Takes content and breaks it up into segments"""

    def __init__(self, tree, workflow, parent=None):
        super().__init__(tree, workflow, parent)
        if not getattr(self, "prompt_text", None):
            self.prompt_text = self.workflow.default_prompts.get("segmentation", "")
        self.expects = "data"

    def _process(self, input):
        raise NotImplementedError()


globals().update(create_all_components(Segmenter))


class MockSegmenter(Segmenter):
    def _process(self, input):
        # Could be a file ref, or extracted pixels
        try:
            filename = input.file_name
        except Exception:
            filename = repr(input)
        return ItemResult(f"segmentation of {filename}", metadata={"effort": 0}, input=input, processor=self)


class WordSegmenter(Segmenter):
    def __init__(self, tree, workflow, parent=None):
        super().__init__(tree, workflow, parent)

    def _process(self, input):
        return ListResult(input.split())


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
