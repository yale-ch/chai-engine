"""Gates: components that branch the workflow based on a test of the input.

A Gate evaluates ``_test(input)`` and routes the input to its ``true_steps`` or ``false_steps``
children. Concrete gates test labels (``LabelTestGate``), JSON conditions (``ConditionGate`` and its
shortcuts), metadata thresholds, or file types.
"""

from typing import List

from .conditions import evaluate
from .core import Component
from .workflow import Workflow


class Gate(Component):
    """A Component that acts as a gating mechanism.

    Instead of ``steps``/``next_steps`` it builds two child branches from the config: ``true_steps``
    (run when ``_test(input)`` is truthy) and ``false_steps`` (run otherwise). The chosen branch's
    outputs are merged into a ``ListResult``; when the chosen branch is empty the gate returns ``None``
    (input is dropped). ``process`` deliberately bypasses ``Component.process`` -- no ``register_on``
    handling, no ``process_out`` -- but still emits the standard lifecycle events. The base ``_test``
    always returns ``True``; subclasses override it.
    """

    false_steps: List["Workflow"] = []
    true_steps: List["Workflow"] = []

    def __init__(self, tree, workflow, parent=None):
        super().__init__(tree, workflow, parent)
        self.true_steps = []
        for o in tree.get("true_steps", []):
            op = self._make_step(o, workflow)
            op.parent = self
            self.true_steps.append(op)
        self.false_steps = []
        for o in tree.get("false_steps", []):
            op = self._make_step(o, workflow)
            op.parent = self
            self.false_steps.append(op)

    def _test(self, input):
        return True

    def _process(self, input, case):
        steps = self.true_steps if case else self.false_steps
        if not steps:
            return None
        merged = self.outputResultClass([], input=input, processor=self)
        for step in steps:
            x = step.process(input)
            merged.append(x)
        return merged

    def process(self, input):
        # Gates bypass Component.process (no result registration), so they
        # emit their own lifecycle events.
        self._emit("component_start")
        try:
            result = self._process(input, self._test(input))
        except Exception as e:
            self._emit("component_error", error=str(e))
            raise
        self._emit("component_end")
        return result


class ConditionGate(Gate):
    """Routes on a JSON condition evaluated against the input Result.

    The condition language (see ``chai.conditions``) is component-agnostic:
    test the result's value, its metadata (e.g. a YOLO class or confidence),
    its file type, its labels, or fields of its input -- and combine tests
    with all/any/not. Matching results flow to ``true_steps``, the rest to
    ``false_steps``.

    Settings:
        - condition: condition dict, e.g. {"source": "metadata.confidence", "op": "gte", "value": 0.8}
    """

    def __init__(self, tree, workflow, parent=None):
        super().__init__(tree, workflow, parent)
        # Build eagerly so config mistakes fail at build time, not mid-run
        self.condition = self._build_condition()
        if not isinstance(self.condition, dict):
            raise ValueError(f"{self.__class__.__name__} ({self!r}) needs a condition dict")

    def _build_condition(self):
        if "condition" not in self.settings:
            raise ValueError(f"ConditionGate ({self!r}) needs the `condition` setting")
        condition = self.settings["condition"]
        if isinstance(condition, str):
            # Conveniently allow a JSON string (e.g. typed into the builder UI)
            import json

            condition = json.loads(condition)
        return condition

    def _test(self, input):
        return evaluate(self.condition, input)


class ValueTestGate(ConditionGate):
    """Tests the input's value with a single comparison.

    Settings:
        - op:     eq | ne | gt | gte | lt | lte | contains | in | matches | truthy (default)
        - value:  the value to compare against
        - source: where to read from (default 'value'; see chai.conditions)
    """

    def _build_condition(self):
        return {
            "source": self.settings.get("source", "value"),
            "op": self.settings.get("op", "truthy"),
            "value": self.settings.get("value"),
        }


class MetadataTestGate(ConditionGate):
    """Tests a (dotted) key in the input's metadata.

    Settings:
        - key:   metadata key to test, e.g. 'yolo_class' (required)
        - op:    comparison op (default 'eq'; see chai.conditions)
        - value: the value to compare against
    """

    def _build_condition(self):
        if "key" not in self.settings:
            raise ValueError(f"MetadataTestGate ({self!r}) needs the `key` setting")
        return {
            "source": f"metadata.{self.settings['key']}",
            "op": self.settings.get("op", "eq"),
            "value": self.settings.get("value"),
        }


class ThresholdGate(ConditionGate):
    """Passes results whose numeric metadata field meets a threshold --
    e.g. keep only confident YOLO detections.

    Settings:
        - threshold: minimum value (required)
        - key:       metadata key holding the number (default 'confidence')
    """

    def _build_condition(self):
        if "threshold" not in self.settings:
            raise ValueError(f"ThresholdGate ({self!r}) needs the `threshold` setting")
        return {
            "source": f"metadata.{self.settings.get('key', 'confidence')}",
            "op": "gte",
            "value": self.settings["threshold"],
        }


class FileTypeGate(ConditionGate):
    """Routes by declared file type (IMAGE/TEXT/AUDIO/DATA).

    Settings:
        - types: type name or list of type names that pass (required)
    """

    def _build_condition(self):
        types = self.settings.get("types", self.settings.get("type"))
        if not types:
            raise ValueError(f"FileTypeGate ({self!r}) needs the `types` setting")
        if isinstance(types, str):
            types = [t.strip() for t in types.split(",")]
        return {"source": "type", "op": "in", "value": [t.upper() for t in types]}


class LabelTestGate(Gate):
    """Routes on labels that a specific classifier assigned to the input.

    The classifier must run earlier with ``register_on`` so its ``LabelListResult`` is registered as a
    derivative on the tested result; this gate fetches those derivatives and passes the input to
    ``true_steps`` when any configured label is present, ``false_steps`` otherwise.

    Settings:
        - label:     label or list of labels to look for (required)
        - component: id of the classifier whose labels are tested (required)
    """

    def __init__(self, tree, workflow, parent=None):
        super().__init__(tree, workflow, parent)
        if "label" not in self.settings:
            raise ValueError(f"LabelTestGate ({repr(self)}) needs the `label` setting")
        elif "component" not in self.settings:
            raise ValueError(f"LabelTestGate ({repr(self)}) needs the `component` setting")
        else:
            lbls = self.settings["label"]
            cid = self.settings["component"]
            # replace with instance
            comp = self.workflow.get_component_by_id(cid)
            if comp is None:
                raise ValueError(f"LabelTestGate ({repr(self)}) component '{cid}' not found")
            self.settings["component"] = comp
            if type(lbls) is str:
                self.settings["label"] = set([lbls])
            elif type(lbls) is list:
                self.settings["label"] = set(lbls)
            else:
                raise ValueError(f"LabelTestGate ({repr(self)} label setting isn't a string or array")

    def _test(self, input):
        """Look for any of settings['label'] in input's labels"""
        # Fetch labels from tested component's results
        lbls = input.get_derivative_result(self.settings["component"])
        in_labels = []
        for lbl in lbls:
            in_labels.extend(lbl.value)
        return len(self.settings["label"].intersection(set(in_labels))) > 0
