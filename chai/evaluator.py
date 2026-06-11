"""Evaluators: score workflow output against ground truth.

The missing "is the pipeline any good?" piece: ``TextMetricsEvaluator``
computes exact match / CER / WER between a produced text and a reference
(transcription quality), and ``RecordFieldEvaluator`` computes per-field
precision/recall/F1 between an extracted record and an expected one (Darwin
Core-style extraction quality). Both emit a DATA ItemResult of metrics --
gate on them (``ValueTestGate`` over ``metadata`` or fields), store them, or
show them in a Record viewer.
"""

import json

from .core import Component
from .result import ItemResult, Result
from .utils import text_from_input


def levenshtein(a, b):
    """Edit distance between two sequences (strings or token lists)."""
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def text_metrics(prediction, reference, case_sensitive=False):
    """Exact match, character error rate, and word error rate for *prediction* vs *reference*."""
    pred, ref = str(prediction), str(reference)
    if not case_sensitive:
        pred, ref = pred.lower(), ref.lower()
    pred, ref = " ".join(pred.split()), " ".join(ref.split())
    cer = levenshtein(pred, ref) / max(1, len(ref))
    wer = levenshtein(pred.split(), ref.split()) / max(1, len(ref.split()))
    return {"exact": pred == ref, "cer": round(cer, 4), "wer": round(wer, 4), "ref_chars": len(ref)}


def record_metrics(predicted, expected, fields=None, case_sensitive=False):
    """Field-level precision/recall/F1 between two flat dicts.

    A field counts as correct when both sides have it and the normalized
    values match. *fields* limits scoring to those keys (default: the union).
    """

    def norm(v):
        s = " ".join(str(v).split())
        return s if case_sensitive else s.lower()

    predicted = predicted or {}
    expected = expected or {}
    keys = list(fields) if fields else sorted(set(predicted) | set(expected))
    per_field = {}
    tp = fp = fn = 0
    for k in keys:
        has_p = k in predicted and predicted[k] not in (None, "")
        has_e = k in expected and expected[k] not in (None, "")
        if has_p and has_e:
            ok = norm(predicted[k]) == norm(expected[k])
            per_field[k] = "correct" if ok else "wrong"
            tp += ok
            fp += not ok
            fn += not ok
        elif has_p:
            per_field[k] = "spurious"
            fp += 1
        elif has_e:
            per_field[k] = "missing"
            fn += 1
    precision = tp / max(1, tp + fp)
    recall = tp / max(1, tp + fn)
    f1 = 2 * precision * recall / max(1e-9, precision + recall) if (precision + recall) else 0.0
    return {
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "fields": per_field,
    }


class Evaluator(Component):
    """Abstract base for the evaluation role: compare output to ground truth.

    Subclasses read the reference from settings (inline value or a file) and
    return a DATA ItemResult of metrics with ``type: METRICS`` metadata.
    """

    def __init__(self, tree, workflow, parent=None):
        super().__init__(tree, workflow, parent)
        self.expects = "data"

    def _reference(self):
        if "reference_file" in self.settings:
            with open(self.settings["reference_file"]) as fh:
                return fh.read()
        return self.settings.get("reference")

    def _process(self, input):
        raise NotImplementedError()


class TextMetricsEvaluator(Evaluator):
    """Scores produced text against a reference: exact match, CER, WER.

    Wire after a transcriber (or any text-producing step). The reference comes
    from settings -- inline or from a file beside your eval set.

    Settings:
        - reference: the ground-truth text (or use reference_file)
        - reference_file: path to a file holding the ground-truth text
        - case_sensitive: compare with case (default false)
    """

    def _process(self, input):
        reference = self._reference()
        if reference is None:
            raise ValueError(f"{self} needs a `reference` (or `reference_file`) setting")
        metrics = text_metrics(
            text_from_input(input), reference, case_sensitive=bool(self.settings.get("case_sensitive", False))
        )
        return ItemResult(metrics, metadata={"type": "METRICS"}, input=input, processor=self)


class RecordFieldEvaluator(Evaluator):
    """Scores an extracted record (dict) against an expected record:
    per-field correct/wrong/missing/spurious plus precision/recall/F1.

    Wire after an Extractor. JSON-string input is parsed first.

    Settings:
        - expected: the ground-truth record as a dict or JSON string (or use reference_file)
        - reference_file: path to a JSON file holding the ground-truth record
        - fields: optional comma-separated field names to score (default: all)
        - case_sensitive: compare values with case (default false)
    """

    def _process(self, input):
        expected = self.settings.get("expected") or self._reference()
        if expected is None:
            raise ValueError(f"{self} needs an `expected` (or `reference_file`) setting")
        if isinstance(expected, str):
            expected = json.loads(expected)

        predicted = input.value if isinstance(input, Result) else input
        if isinstance(predicted, (bytes, bytearray)):
            predicted = predicted.decode("utf-8", "replace")
        if isinstance(predicted, str):
            predicted = json.loads(predicted)
        if not isinstance(predicted, dict):
            raise ValueError(f"{self} expects a record-shaped (dict) input, got {type(predicted).__name__}")

        fields = self.settings.get("fields")
        if isinstance(fields, str):
            fields = [f.strip() for f in fields.split(",") if f.strip()]
        metrics = record_metrics(
            predicted, expected, fields=fields, case_sensitive=bool(self.settings.get("case_sensitive", False))
        )
        return ItemResult(metrics, metadata={"type": "METRICS"}, input=input, processor=self)
