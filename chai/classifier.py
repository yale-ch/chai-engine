"""Classifiers: components that assign one or more labels to their input.

All classifiers return a ``LabelListResult`` of string labels for the input Result; combined with
``register_on`` and a gate (``LabelTestGate`` / ``ConditionGate``) they drive conditional branching.
"""

import json
import re
from random import randint

from .ai import create_all_components
from .core import Component
from .result import FileItemResult, LabelListResult
from .utils import text_from_input


class Classifier(Component):
    """Takes an input and runs a classification on it to return one or more labels.

    Abstract base for the classifier role: subclasses implement ``_process`` and return a
    ``LabelListResult`` (a list of string labels) for the input Result. AI-backed variants
    (``GeminiClassifier``, ``OllamaClassifier``, ``LLMClassifier``, ...) are generated via ``chai.ai``
    mixins: the model's reply becomes the labels (``labels_from``), so a ``LabelTestGate`` can route
    on them. When no prompt is configured the workflow's default ``classification`` prompt is used.

    Settings (AI-backed classifiers):
        - labels: the labels the model may answer with, as a list. The reply -- plain text (one label,
          or several separated by commas or new lines) or JSON (``{"label": ...}``,
          ``{"labels": [...]}`` or a list) -- is matched to them ignoring case, spaces, hyphens and
          punctuation, and a reply that names none of them raises, so ``retries`` can ask again.
          Without it, the reply's labels are kept as they are.
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

    @staticmethod
    def _norm_label(label):
        return re.sub(r"[^0-9a-z]+", "_", str(label).lower()).strip("_")

    def labels_from(self, reply):
        """The labels a model's *reply* names, matched to the ``labels`` setting when there is one."""
        if isinstance(reply, str):
            text = reply.strip().strip("`")
            if text[:1] in "[{":
                try:
                    reply = json.loads(text)
                except ValueError:
                    pass
        if isinstance(reply, dict):
            reply = next((reply[k] for k in ("labels", "label", "class", "category") if k in reply), list(reply.values()))
        if isinstance(reply, str):
            found = [x for x in re.split(r"[,\n;]+", reply)]
        elif isinstance(reply, (list, tuple)):
            found = [str(x) for x in reply]
        else:
            found = [] if reply is None else [str(reply)]
        found = [x.strip().strip("\"'`*.:- ") for x in found]
        found = [x for x in found if x]

        allowed = self.settings.get("labels")
        if isinstance(allowed, str):
            allowed = [x.strip() for x in allowed.split(",") if x.strip()]
        if not isinstance(allowed, (list, tuple)) or not allowed:
            return found
        by_norm = {self._norm_label(a): a for a in allowed}
        # the whole reply first: "Typed document, complex layout" is one label, not two
        whole = by_norm.get(self._norm_label(" ".join(found)))
        if whole is not None:
            return [whole]
        labels = []
        for x in found:
            label = by_norm.get(self._norm_label(x))
            if label is not None and label not in labels:
                labels.append(label)
        if not labels:
            # a sentence around the label: accept it when exactly one allowed label appears in it
            text = "_" + self._norm_label(" ".join(found)) + "_"
            hits = [a for n, a in by_norm.items() if f"_{n}_" in text]
            longest = max(hits, key=lambda a: len(self._norm_label(a)), default=None)
            hits = [a for a in hits if a == longest or self._norm_label(a) not in self._norm_label(longest)]
            if len(hits) == 1:
                labels = hits
        if not labels:
            raise ValueError(f"{self}: the reply {reply!r} names none of the labels {list(allowed)}")
        return labels


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
    falls below the configured percentage, otherwise an empty label list -- useful for spot-checking
    a fraction of a corpus.

    Settings:
        - percentage: percentage of inputs to flag, 0-100 (default 10)
    """

    def __init__(self, tree, workflow, parent=None):
        super().__init__(tree, workflow, parent)
        self.percentage = float(self.settings.get("percentage", 10))

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
        - model:      YOLO classification model, or ``hf://owner/repo/file.pt`` (default: "yolo11n-cls.pt")
        - confidence: minimum probability to emit a label (default: 0.5)
    """

    def __init__(self, tree, workflow, parent=None):
        super().__init__(tree, workflow, parent)
        from .segmenter import yolo_model

        model_name = self.settings.get("model", "yolo11n-cls.pt")
        self.confidence = self.settings.get("confidence", 0.5)
        self.model = yolo_model(model_name)

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


def _labelling(cls):
    """Make an AI-backed classifier return its reply as a ``LabelListResult``."""
    backend = cls._process

    def _process(self, input):
        out = backend(self, input)
        metadata = dict(out.metadata or {}, type="DATA", reply=out.value)
        return LabelListResult(self.labels_from(out.value), input=input, processor=self, metadata=metadata)

    cls._process = _process
    return cls


globals().update({name: _labelling(cls) for name, cls in create_all_components(Classifier).items()})
