from random import randint

from .core import Component
from .result import FileItemResult, LabelListResult


class Classifier(Component):
    """Takes an input and runs a classification on it to return one or more labels"""

    def _process(self, input):
        # Will assign one or more labels to the input
        return LabelListResult([], input=input, processor=self)


class SampleClassifier(Classifier):
    """Sample a given percentage of inputs by tagging them"""

    def _process(self, input):
        pc = randint(0, 10000) / 100
        flags = ["flagged"] if pc < self.percentage else []
        return LabelListResult(flags, input=input, processor=self)


class HumanClassifier(Classifier):
    """A human will assign the classification in the web API"""

    def _process(self, input):
        return LabelListResult(["NOT_DONE"], input=input, processor=self)


class FileTypeClassifier(Classifier):
    """Classifies input by its filetype metadata, labelling TEXT or IMAGE results as 'accepted'."""

    def _process(self, input):
        file_type = input.metadata.get("type", "")
        labels = ["accepted"] if file_type in ("TEXT", "IMAGE") else ["rejected"]
        return LabelListResult(labels, input=input, processor=self)


class YoloClassifier(Classifier):
    """Classifies images using a YOLO model from ultralytics.

    Returns detected class names as labels. Configure via settings:
        - model: YOLO model name (default: "yolo11n-cls.pt")
        - confidence: minimum confidence threshold (default: 0.5)
    """

    def __init__(self, tree, workflow, parent=None):
        super().__init__(tree, workflow, parent)
        from ultralytics import YOLO

        model_name = self.settings.get("model", "yolo11n-cls.pt")
        self.confidence = self.settings.get("confidence", 0.5)
        self.model = YOLO(model_name)

    def _process(self, input):
        if isinstance(input, FileItemResult):
            source = input.file_name
        else:
            source = input.value

        results = self.model(source, verbose=False)

        labels = []
        for result in results:
            if hasattr(result, "probs") and result.probs is not None:
                # Classification model — get top classes above threshold
                for idx, conf in enumerate(result.probs.data.tolist()):
                    if conf >= self.confidence:
                        labels.append(result.names[idx])
            elif hasattr(result, "boxes") and result.boxes is not None:
                # Detection model — get unique class names above threshold
                for box in result.boxes:
                    if box.conf.item() >= self.confidence:
                        cls_id = int(box.cls.item())
                        name = result.names[cls_id]
                        if name not in labels:
                            labels.append(name)

        return LabelListResult(labels, input=input, processor=self)
