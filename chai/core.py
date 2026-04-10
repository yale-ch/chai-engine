import importlib
import logging
import uuid
from typing import List, Optional

from dotenv import find_dotenv, load_dotenv

# Set up the environment globally
logger = logging.getLogger("chai")
fn = find_dotenv(usecwd=True)
if fn:
    load_dotenv(fn)


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
    workflow: Optional["Component"] = None
    input = None


from .result import ListResult, Result  # noqa -- Prevent circular import


class Component(BaseThing):
    """A Component receives input, performs some computation on the Result and returns a new Result"""

    parent: Optional["Component"] = None
    steps: List["Component"] = []
    next_steps: List["Component"] = []
    outputResultClass = ListResult
    settings: dict = {}
    register_on: list = []
    config: dict = {}

    def _make_step(self, tree, wf):
        base = tree.get("base", "")
        if base:
            if not self.workflow:
                raise ValueError("Cannot use a library based configuration for or without a Workflow")
            # merge the library's configuration with tree
            base_config = self.workflow.library.get(base, {})
            if base_config:
                for k, v in tree.items():
                    if k == "settings":
                        # merge new settings
                        sets = base_config.get("settings", {})
                        sets.update(v)
                        base_config[k] = sets
                    else:
                        base_config[k] = v
                tree = base_config

        t = tree["type"]
        cl = importClass(t)

        if cl is None:
            raise ValueError(t)
        else:
            inst = cl(tree, wf, self)
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
