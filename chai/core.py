import importlib
import logging
import os
import uuid
from typing import List, Optional

logger = logging.getLogger("chai")


def importClass(objectType):
    if not objectType:
        return None
    try:
        (modName, className) = objectType.rsplit(".", 1)
    except Exception:
        (modName, className) = ("core", objectType)
    if not modName.startswith("chai."):
        modName = f"chai.{modName}"

    try:
        m = importlib.import_module(modName)
    except ModuleNotFoundError as mnfe:
        logger.critical(f"Could not find module {modName}: {mnfe}")
        raise
    except Exception as e:
        logger.critical(f"Failed to import {modName}: {e}")
        raise
    try:
        parentClass = getattr(m, className)
    except AttributeError:
        raise
    return parentClass


class BaseThing(object):
    id: str = ""
    name: str = ""
    workflow: Optional["Workflow"] = None
    input = None


class Result(BaseThing):
    value = None
    input: Optional["Result"] = None
    processor: Optional["Provider"] = None
    metadata: dict = {}
    extra: dict = {}
    derivative_results: dict = {}

    def __init__(
        self, value=None, input=None, processor=None, register_on=None, workflow=None, metadata={}, extra={}
    ):
        self.id = str(uuid.uuid4())
        self.set_value(value)
        self.input = input or None
        self.workflow = workflow or None
        self.processor = processor or None
        self.metadata = metadata or {}
        self.extra = extra or {}
        self.derivative_results = {}
        if register_on is not None:
            self.input = register_on
            if processor is None:
                raise ValueError(f"Must set processor in order to set register_on to {register_on}")
            register_on.register_result(processor, self)

    def __repr__(self):
        return f"{self.__class__.__name__}(value={self.value!r})"

    def set_value(self, value):
        self.value = value

    def register_result(self, component, result):
        # register the result against the component
        if component not in self.derivative_results:
            self.derivative_results[component] = []
        if result not in self.derivative_results[component]:
            self.derivative_results[component].append(result)

    def get_derivative_result(self, component):
        return self.derivative_results.get(component, [])

    def _build_view(self, lines, indent):
        lines.append(f"{'  ' * indent}{self}")
        if isinstance(self.value, Result) or type(self.value) is list:
            for x in self.value:
                if hasattr(x, "_build_view"):
                    x._build_view(lines, indent + 1)
                elif type(self.value) is str and len(self.value) < 120:
                    lines.append(f"{'  ' * indent} {self.value}")
                else:
                    lines.append(f"{'  ' * indent} <unprintable value>")
        elif type(self.value) is str and len(self.value) < 120:
            lines.append(f"{'  ' * indent} {self.value}")
        else:
            lines.append(f"{'  ' * indent} <unprintable value>")
        return lines

    def view(self):
        for line in self._build_view([], 0):
            print(line)


class ResultIter(object):
    count = 0
    result = None

    def __init__(self, result):
        self.valueClass = result.valueClass
        self.result = result
        self.count = 0

    def __next__(self):
        if self.count == len(self.result.value):
            raise StopIteration()
        elif self.valueClass is not None:
            val = self.valueClass(self.result.value[self.count])
        else:
            val = self.result.value[self.count]
        self.count += 1
        return val


class ItemResult(Result):
    """A Result with a single value"""

    def __iter__(self):
        return iter([self.value])


class ListResult(Result):
    """A Result with multiple values"""

    # This is copied by ResultIter
    valueClass = ItemResult

    def append(self, value):
        try:
            self.value.append(value)
        except Exception as e:
            print(f"Failed to append to {self.value}: {e}")
            raise

    def __iter__(self):
        if not hasattr(self.value, "__getitem__"):
            # Uhoh. Can't index into value.
            if hasattr(self.value, "__iter__"):
                # But we can iterate through it, e.g. dict_keys()
                return iter(self.value)
            elif hasattr(self.value, "__next__"):
                # or it is an iter?
                return self.value
            else:
                raise ValueError(f"Don't know how to iterate through {self.value}")
        return ResultIter(self)

    def __repr__(self):
        return f"<{self.__class__.__name__}({len(self.value)} items)>"


class FileItemResult(ItemResult):
    """A result that mirrors an on-disk file"""

    file_bytes = b""
    file_name = ""

    @property
    def value(self):
        if not self.file_bytes:
            with open(self.file_name, "rb") as fh:
                bs = fh.read()
            self.file_bytes = bs
        return self.file_bytes

    @value.setter
    def value(self, value):
        self.file_name = value

    def __repr__(self):
        return f"<FileItemResult('{self.file_name}')>"


class DirectoryListResult(ListResult):
    """A result that mirrors an on-disk directory"""

    valueClass = FileItemResult

    def set_value(self, value):
        if value and type(value) is list:
            value.sort()
        self.value = value


class LabelListResult(ListResult):
    """Receives a list of strings as labels from a classifier"""

    pass


# ----


class Component(BaseThing):
    """A Component receives input, performs some computation on the Result and returns a new Result"""

    parent: Optional["Workflow"] = None
    steps: List["Workflow"] = []
    next_steps: List["Workflow"] = []
    outputResultClass = ListResult
    settings: dict = {}
    register_on: list = []
    config: dict = {}

    def _make_step(self, s, wf):
        t = s["type"]

        cl = importClass(t)

        if cl is None:
            raise ValueError(t)
        else:
            inst = cl(s, wf, self)
            return inst

    def __init__(self, tree, workflow, parent=None):
        # Walk tree and built components

        self.workflow = workflow
        self.id = tree.get("id", self.workflow.get_new_id() if self.workflow else str(uuid.uuid4()))

        if self.workflow is not None:
            self.workflow.register_component(self)

        self.parent = parent or None
        self.name = tree.get("name", f"{self.__class__.__name__}/{self.id}")
        self.input = tree.get("input", None)
        self.settings = tree.get("settings", {})
        self.config = tree

        cids = tree.get("register_on", [])
        comps = []
        for cid in cids:
            # resolve id to instance
            if cid == "parent":
                comp = self.parent
            else:
                comp = self.workflow.get_component_by_id(cid)
            if comp is not None:
                comps.append(comp)
            else:
                logger.warning(f"Unrecognized component id for result registration: {cid}")
        self.register_on = comps

        self.steps = []
        for s in tree.get("steps", []):
            branch = self._make_step(s, workflow)
            self.steps.append(branch)

        self.next_steps = []
        for o in tree.get("next_steps", []):
            op = self._make_step(o, workflow)
            self.next_steps.append(op)

    def __repr__(self):
        return f"<{self.name}>"

    def _process(self, input: Result) -> Result:
        # Default is pass down
        merged = self.outputResultClass([], input=input, processor=self)
        for step in self.steps:
            res = step.process(input)
            merged.append(res)
        return merged

    def process(self, input) -> Result:
        new_result = self._process(input)

        # Ensure the result always knows its input
        if isinstance(input, Result):
            if new_result.input is None:
                new_result.input = input
            # ... And which component created it
            if new_result.processor is None:
                new_result.processor = self
            # ... so that we can walk up the hierarchy to link results

            if self.register_on:
                # list of processors, on to which we register the new result for the input
                inp = input
                targets = self.register_on[:]
                while targets:
                    if inp.processor in targets:
                        inp.register_result(self, new_result)
                        targets.remove(inp.processor)
                    # walk up one step
                    inp = inp.input

        # and call process_out to get to the next step
        return self.process_out(new_result)

    def process_out(self, input) -> Result:
        if self.next_steps:
            merged = ListResult([], input=input, processor=self)
            for step in self.next_steps:
                x = step.process(input)
                merged.append(x)
            return merged
        else:
            return input


class Workflow(Component):
    """Workflows manage multiple Components in a tree structure, and global variables"""

    def __init__(self, tree, workflow=None):
        if workflow is None:
            workflow = self
        self.registry_ids = {}
        self.id_counter = -1
        super().__init__(tree, workflow)

    def get_component_by_id(self, cid):
        return self.registry_ids.get(cid, None)

    def register_component(self, component):
        if component.id in self.registry_ids:
            raise ValueError(f"{component.id} already in registry for {self.registry_ids[component.id]}")
        self.registry_ids[component.id] = component

    def get_new_id(self):
        self.id_counter += 1
        cid = f"{self.id}_{self.id_counter}"
        if cid in self.registry_ids:
            raise ValueError(f"Tried to create identifier that already exists: {cid}")
        return cid

    def run(self, input=None) -> Result:
        res = ListResult([], processor=self)
        for s in self.steps:
            if s.input is not None:
                res.append(s.run())
            elif input is not None:
                res.append(s.process(input))
            else:
                raise ValueError("No input value provided")
        return res


class Provider(Component):
    """A Provider generates a Result given a raw input"""

    def run(self) -> Result:
        if self.input:
            return self.process(self.input)
        else:
            raise ValueError("No input value set")


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


class IterateStep(Component):
    """Take a result and call further components for each entry in result to make a new result"""

    def _process(self, input: Result) -> Result:
        merged = self.outputResultClass([], input=input, processor=self)
        for x in input:
            x.input = input
            x.processor = self
            step_value = ListResult([], input=x, processor=self)
            for step in self.steps:
                res = step.process(x)
                step_value.append(res)
            merged.append(step_value)
        return merged


class ValueIter(Component):
    pass


class DebugStep(Component):
    def _process(self, input: Result) -> Result:
        print(f"{self.id}: {repr(input)}")
        return input


class DirFileProvider(Provider):
    """Take a director name and return a ListResult of ItemResults for each file"""

    def _process(self, input):
        if os.path.exists(input):
            files = os.listdir(input)
            d = DirectoryListResult([os.path.join(input, x) for x in files], input=input, processor=self)
            return super()._process(d)
        else:
            raise ValueError("input file path does not exist")


class IIIFDirFileProvider(DirFileProvider):
    pass


class Classifier(Component):
    """Takes an input and runs a classification on it to return one or more labels"""

    def _process(self, input):
        # Will assign one or more labels to the input
        return LabelListResult(["okay"], input=input, processor=self)


class Transcriber(Component):
    """Takes an image or audio and extracts the text from the binary content"""

    def _process(self, input):
        # Could be a file ref, or extracted pixels
        try:
            filename = input.file_name
        except Exception:
            filename = repr(input)
        return ItemResult(f"transcription of {filename}", metadata={"effort": 0}, input=input, processor=self)


class GeminiTranscriber(Transcriber):
    pass


class LocalTranscriber(Transcriber):
    pass


class Segmenter(Component):
    """Takes content and breaks it up into segments"""

    pass


class Describer(Component):
    """Takes content and generates text to describe it"""

    def _process(self, input):
        return ItemResult("A wonderful input", extra={"something": "else"}, input=input, processor=self)


class Translator(Component):
    """Takes linguistic content and translates it into one or more different languages"""

    pass


class Extractor(Component):
    """Takes content and extracts structured data from it"""

    pass


class Reducer(Component):
    """Take multiple results and combine them to one"""

    pass


class ClassificationReducer(Reducer):
    pass
