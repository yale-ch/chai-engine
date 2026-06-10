"""Classifiers: components that assign one or more labels to their input.

All classifiers return a ``LabelListResult`` of string labels for the input Result; combined with
``register_on`` and a gate (``LabelTestGate`` / ``ConditionGate``) they drive conditional branching.
"""

import re
from random import randint

from .core import Component
from .result import FileItemResult, LabelListResult
from .utils import text_from_input


class Classifier(Component):
    """Takes an input and runs a classification on it to return one or more labels.

    Abstract base for the classifier role: subclasses implement ``_process`` and return a
    ``LabelListResult`` (a list of string labels) for the input Result. AI-backed variants
    (``GeminiClassifier``, ``OllamaClassifier``, ...) are generated via ``chai.ai`` mixins; when no
    prompt is configured the workflow's default ``classification`` prompt is used.
    """

    def __init__(self, tree, workflow, parent=None):
        super().__init__(tree, workflow, parent)
        if not getattr(self, "prompt_text", None):
            self.prompt_text = self.workflow.default_prompts.get("classification", "")
        self.expects = "text"

    def _process(self, input):
        # Will assign one or more labels to the input
        # return LabelListResult([], input=input, processor=self)
        raise NotImplementedError()


class KeywordClassifier(Classifier):
    """Labels text by keyword (or regex) matches -- no model required.

    Settings:
        - labels:         dict of {label: [patterns]} (a bare string or list is
                          also accepted per label) (required)
        - regex:          if true, patterns are regular expressions (default false)
        - case_sensitive: default false
    """

    def __init__(self, tree, workflow, parent=None):
        super().__init__(tree, workflow, parent)
        raw = self.settings.get("labels")
        if not isinstance(raw, dict) or not raw:
            raise ValueError(f"KeywordClassifier ({self!r}) needs the `labels` setting: {{label: [patterns]}}")
        self.use_regex = bool(self.settings.get("regex", False))
        self.case_sensitive = bool(self.settings.get("case_sensitive", False))
        flags = 0 if self.case_sensitive else re.IGNORECASE
        self.label_patterns = {}
        for label, patterns in raw.items():
            if isinstance(patterns, str):
                patterns = [patterns]
            self.label_patterns[label] = [
                re.compile(p if self.use_regex else re.escape(p), flags) for p in patterns
            ]

    def _process(self, input):
        text = text_from_input(input)
        labels = [
            label
            for label, patterns in self.label_patterns.items()
            if any(p.search(text) for p in patterns)
        ]
        return LabelListResult(labels, input=input, processor=self)


class SampleClassifier(Classifier):
    """Sample a given percentage of inputs by tagging them.

    For each input, draws a random percentage and returns ``LabelListResult(["flagged"])`` when it
    falls below ``self.percentage``, otherwise an empty label list -- useful for spot-checking a
    fraction of a corpus. Note: ``percentage`` must be set on the instance by the caller; the class
    itself does not read it from ``settings``.
    """

    def _process(self, input):
        pc = randint(0, 10000) / 100
        flags = ["flagged"] if pc < self.percentage else []
        return LabelListResult(flags, input=input, processor=self)


class HumanClassifier(Classifier):
    """A human will assign the classification in the web API.

    Placeholder for human-in-the-loop labelling: always returns ``LabelListResult(["NOT_DONE"])`` so a
    downstream gate can park the result until a person reviews it.
    """

    def _process(self, input):
        return LabelListResult(["NOT_DONE"], input=input, processor=self)


class FileTypeClassifier(Classifier):
    """Classifies input by its filetype metadata, labelling TEXT or IMAGE results as 'accepted'.

    Reads the input Result's ``metadata["type"]`` (set by e.g. ``FileItemResult``) and returns
    ``LabelListResult(["accepted"])`` for TEXT/IMAGE, ``["rejected"]`` for anything else -- a simple
    filter for mixed directories.
    """

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
