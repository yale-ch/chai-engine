from typing import List

from .core import Component
from .workflow import Workflow


class Gate(Component):
    """A Component that acts as a gating mechanism"""

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
        # Where does the metadata about this function get stored?
        return self._process(input, self._test(input))


class LabelTestGate(Gate):
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
