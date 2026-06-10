"""JSON-friendly conditions evaluated against Results.

This is the shared test language for conditional branching (see
``chai.gate``). A condition is a plain dict, so it can live in workflow
configs, and it is evaluated against whatever Result a component produced --
abstract over the component types.

Leaf condition::

    {"source": "metadata.confidence", "op": "gte", "value": 0.8}

Combinators::

    {"all": [cond, ...]}     every sub-condition must hold
    {"any": [cond, ...]}     at least one sub-condition must hold
    {"not": cond}            negation

Sources (where the tested value comes from):

* ``value`` -- the result's value (or the raw input itself when a gate is the
  first step and receives a non-Result).
* ``file_name`` -- a FileItemResult's file name.
* ``type`` -- shorthand for ``metadata.type`` (IMAGE/TEXT/AUDIO/DATA).
* ``labels`` -- labels carried by the result: a LabelListResult's values plus
  any LabelListResult derivatives registered on it.
* ``metadata.<dotted.path>`` / ``extra.<dotted.path>`` -- nested lookups.
* ``input.<source>`` -- evaluate ``<source>`` against the result's input,
  e.g. ``input.metadata.yolo_class``.

Ops: ``exists``, ``truthy`` (default), ``eq``, ``ne``, ``gt``, ``gte``,
``lt``, ``lte``, ``contains``, ``in``, ``intersects``, ``matches`` (regex
search). Comparisons against a missing source are False (``ne`` is True).
"""

import re

from .result import LabelListResult, Result

MISSING = object()


def _dig(obj, path):
    cur = obj
    for part in path.split("."):
        if isinstance(cur, dict) and part in cur:
            cur = cur[part]
        else:
            return MISSING
    return cur


def _collect_labels(result):
    labels = []
    if isinstance(result, LabelListResult) and type(result.value) is list:
        labels.extend(result.value)
    for results in result.derivative_results.values():
        for r in results:
            if isinstance(r, LabelListResult) and type(r.value) is list:
                labels.extend(r.value)
    return labels


def resolve_source(result, source):
    """Resolve *source* against *result*; returns MISSING when unavailable."""
    if source.startswith("input."):
        if isinstance(result, Result):
            if result.input is None:
                return MISSING
            return resolve_source(result.input, source[len("input.") :])
        return MISSING

    if not isinstance(result, Result):
        # Raw workflow input (e.g. a string typed into a test run): it has no
        # metadata or labels, only a value.
        return result if source == "value" else MISSING

    if source == "value":
        return result.value
    if source == "file_name":
        return getattr(result, "file_name", MISSING) or MISSING
    if source == "type":
        return _dig(result.metadata or {}, "type")
    if source == "labels":
        return _collect_labels(result)
    if source.startswith("metadata."):
        return _dig(result.metadata or {}, source[len("metadata.") :])
    if source.startswith("extra."):
        return _dig(result.extra or {}, source[len("extra.") :])
    raise ValueError(f"Unknown condition source: {source!r}")


def _as_number(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def _compare(op, actual, expected):
    a, b = _as_number(actual), _as_number(expected)
    if a is None or b is None:
        return False
    return {"gt": a > b, "gte": a >= b, "lt": a < b, "lte": a <= b}[op]


def _as_text(v):
    if isinstance(v, (bytes, bytearray)):
        return v.decode("utf-8", "replace")
    return v if isinstance(v, str) else str(v)


def apply_op(op, actual, expected):
    if op == "exists":
        return actual is not MISSING and actual is not None
    if op == "truthy":
        return actual is not MISSING and bool(actual)
    if actual is MISSING:
        return op == "ne"
    if op == "eq":
        return actual == expected
    if op == "ne":
        return actual != expected
    if op in ("gt", "gte", "lt", "lte"):
        return _compare(op, actual, expected)
    if op == "contains":
        if isinstance(actual, (list, tuple, set, dict)):
            return expected in actual
        return _as_text(expected) in _as_text(actual)
    if op == "in":
        return actual in (expected or [])
    if op == "intersects":
        actual_items = actual if isinstance(actual, (list, tuple, set)) else [actual]
        return bool(set(actual_items) & set(expected or []))
    if op == "matches":
        return re.search(str(expected), _as_text(actual)) is not None
    raise ValueError(f"Unknown condition op: {op!r}")


def evaluate(condition, result):
    """Evaluate a condition dict against a Result (or raw input value)."""
    if not isinstance(condition, dict):
        raise ValueError(f"Condition must be a dict, got {condition!r}")
    if "all" in condition:
        return all(evaluate(c, result) for c in condition["all"])
    if "any" in condition:
        return any(evaluate(c, result) for c in condition["any"])
    if "not" in condition:
        return not evaluate(condition["not"], result)

    source = condition.get("source", "value")
    op = condition.get("op", "truthy")
    actual = resolve_source(result, source)
    return apply_op(op, actual, condition.get("value"))
