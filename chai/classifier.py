from random import randint

from .core import Component
from .result import FileItemResult, LabelListResult


class Classifier(Component):
    """Takes an input and runs a classification on it to return one or more labels"""

    def __init__(self, tree, workflow, parent=None):
        super().__init__(tree, workflow, parent)
        if not getattr(self, "prompt_text", None):
            self.prompt_text = self.workflow.default_prompts.get("classification", "")
        self.expects = "text"

    def _process(self, input):
        # Will assign one or more labels to the input
        # return LabelListResult([], input=input, processor=self)
        raise NotImplementedError()


class MockClassifier(Classifier):
    def _process(self, input):
        return LabelListResult(["MOCK"], input=input, processor=self)


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
    """Classifies images using a YOLO classification model from ultralytics.

    Returns the top class names above the confidence threshold as labels.
    Strictly for classification heads (i.e. models with ``result.probs``);
    for detection models with bounding boxes, use ``segmenter.YoloSegmenter``.

    Settings:
        - model:      YOLO classification model (default: "yolo11n-cls.pt")
        - confidence: minimum probability to emit a label (default: 0.5)
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

        labels = []
        for result in self.model(source, verbose=False):
            if getattr(result, "probs", None) is None:
                raise ValueError(
                    f"{self.__class__.__name__} requires a YOLO classification model; "
                    "for detection models use segmenter.YoloSegmenter."
                )
            for idx, conf in enumerate(result.probs.data.tolist()):
                if conf >= self.confidence:
                    labels.append(result.names[idx])

        return LabelListResult(labels, input=input, processor=self)
