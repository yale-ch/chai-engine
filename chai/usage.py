"""Run-wide accounting of model calls: tokens, cost and time, per component and per model.

Every model-backed component records what its call consumed in its result's metadata
(``token_usage``, ``cost``, ``duration``, ``model``); ``Component.process`` hands that metadata to the
workflow, whose ``UsageLedger`` keeps the running totals. ``Workflow.run`` resets the ledger at the
start and attaches ``summary()`` to its ``component_end`` event, so a harness can show what a run cost
as it finishes; ``workflow.usage.summary()`` gives the same numbers afterwards.

``cost`` is in US dollars. It is ``None`` on a result when nobody could price the call (a self-hosted
model LiteLLM has no table entry for and no ``pricing`` setting); the ledger counts such calls under
``unpriced_calls`` so a total is never mistaken for a complete one.

A call is counted once, against the component that made the result. Steps that hand a result on
unchanged -- storage returns its input, a wrapper may return its child's result -- pass the same
``Result`` through ``Component.process`` again; the ledger remembers the results it has counted and
skips them.
"""

import threading


def _num(value):
    """Token counts come back as ints, ``None`` (unknown) or, from older backends, -1 (unknown)."""
    if value is None:
        return 0
    try:
        value = int(value)
    except (TypeError, ValueError):
        return 0
    return value if value > 0 else 0


def _empty():
    return {
        "calls": 0,
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "thinking_tokens": 0,
        "total_tokens": 0,
        "cost": 0.0,
        "unpriced_calls": 0,
        "seconds": 0.0,
    }


def _add(into, entry):
    for key, value in entry.items():
        if isinstance(value, (int, float)) and key in into:
            into[key] += value


class UsageLedger:
    """Thread-safe tally of model usage for one workflow run.

    ``record(component, metadata)`` reads a result's ``token_usage`` dict (``prompt``, ``result``,
    ``thinking``, ``total`` -- the shape every chai AI backend emits), its ``cost`` and ``duration``,
    and adds them to the run totals, to the producing component's row (``by_component``, keyed by
    component id) and to the model's row (``by_model``). Safe to call from ``Iterator``/``FanOut``
    worker threads.
    """

    def __init__(self):
        self._lock = threading.Lock()
        self.reset()

    def reset(self):
        """Forget everything recorded so far (``Workflow.run`` calls this at the start of a run)."""
        with self._lock:
            self.totals = _empty()
            self.by_component = {}
            self.by_model = {}
            self._counted = set()

    @staticmethod
    def entry_from_metadata(metadata):
        """One ledger entry for a result's metadata, or ``None`` when it reports no model usage."""
        if not isinstance(metadata, dict):
            return None
        usage = metadata.get("token_usage")
        cost = metadata.get("cost")
        if usage is None and cost is None:
            return None
        usage = usage if isinstance(usage, dict) else {}
        prompt = _num(usage.get("prompt"))
        completion = _num(usage.get("result"))
        total = _num(usage.get("total")) or (prompt + completion)
        try:
            seconds = float(metadata.get("duration") or 0.0)
        except (TypeError, ValueError):
            seconds = 0.0
        return {
            "calls": 1,
            "prompt_tokens": prompt,
            "completion_tokens": completion,
            "thinking_tokens": _num(usage.get("thinking")),
            "total_tokens": total,
            "cost": float(cost) if cost is not None else 0.0,
            "unpriced_calls": 0 if cost is not None else 1,
            "seconds": seconds,
        }

    def record(self, component, metadata, key=None):
        """Add one result's usage to the ledger; returns the entry added (``None`` if nothing to add).

        *key* identifies the result the usage belongs to (``Component.process`` passes the result's
        id). Usage already recorded under the same key is not counted again, so a result handed on
        by a pass-through step is counted once, against the component that made it.
        """
        entry = self.entry_from_metadata(metadata)
        if entry is None:
            return None
        cid = getattr(component, "id", None) or "?"
        model = metadata.get("model") or "?"
        with self._lock:
            if key is not None:
                if key in self._counted:
                    return None
                self._counted.add(key)
            _add(self.totals, entry)
            row = self.by_component.setdefault(
                cid,
                {
                    "name": getattr(component, "name", cid),
                    "component_class": component.__class__.__name__,
                    **_empty(),
                },
            )
            _add(row, entry)
            _add(self.by_model.setdefault(model, _empty()), entry)
        return entry

    def summary(self):
        """A JSON-ready snapshot: run totals plus ``by_component`` and ``by_model`` breakdowns."""
        with self._lock:
            return {
                **self.totals,
                "by_component": {k: dict(v) for k, v in self.by_component.items()},
                "by_model": {k: dict(v) for k, v in self.by_model.items()},
            }

    def __repr__(self):
        t = self.totals
        return (
            f"<UsageLedger calls={t['calls']} tokens={t['total_tokens']} "
            f"cost=${t['cost']:.4f} unpriced={t['unpriced_calls']}>"
        )
