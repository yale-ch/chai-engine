"""Core building blocks of the chai engine.

Defines ``Component`` (the unit of processing that turns a ``Result`` into a new ``Result``) and
``importClass`` (string -> class resolution used to build component trees from JSON workflow configs).
``BaseThing``, the shared identity base, lives in ``chai.base``. Every other component type in chai
subclasses ``Component``.
"""

import importlib
import logging
import uuid
from typing import List, Optional

from dotenv import find_dotenv, load_dotenv

from .base import BaseThing  # noqa: F401 -- re-exported for backwards compatibility
from .result import ListResult, Result

# Set up the environment globally
logger = logging.getLogger("chai")
fn = find_dotenv(usecwd=True)
if fn:
    load_dotenv(fn)


def importClass(objectType):
    """Resolve a workflow-config ``type`` string (e.g. ``"provider.DirFileProvider"``) to a class.

    The string is split into module and class name; the module is looked up inside the ``chai`` package
    (a bare class name defaults to ``chai.core``). Import or attribute failures are logged and re-raised.
    """
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


def result_preview(result, limit=160, depth=0):
    """Short, human-readable summary of what a component produced.

    Sent with ``component_end`` lifecycle events so harnesses (e.g. the
    workflow-builder) can show what leaves each node without serializing the
    full result. Never triggers a FileItemResult's lazy file read.
    """
    if result is None:
        return "(no output)"
    if isinstance(result, Result):
        cls = result.__class__.__name__
        file_name = getattr(result, "file_name", "")
        if file_name:
            return f"{cls} '{file_name}'"
        value = result.value
        if type(value) is list:
            if depth >= 1:
                return f"{cls}({len(value)} items)"
            inner = ", ".join(result_preview(v, limit=40, depth=depth + 1) for v in value[:3])
            more = f", +{len(value) - 3} more" if len(value) > 3 else ""
            return f"{cls}({len(value)} items: [{inner}{more}])"[: limit + 40]
        result = value  # fall through to plain-value handling
    text = result if isinstance(result, str) else repr(result)
    text = " ".join(text.split())
    if len(text) > limit:
        text = text[:limit] + "…"
    return text


class Component(BaseThing):
    """A Component receives input, performs some computation on the Result and returns a new Result.

    Components form a tree that mirrors the JSON workflow config: ``__init__`` walks the config dict and
    instantiates each entry of ``steps`` (children run *inside* ``_process``, receiving this component's
    input) and ``next_steps`` (children run *after* ``process``, receiving this component's output via
    ``process_out``). Free-form per-component configuration lives in ``settings`` (a plain dict read by
    subclasses); the full config dict is kept on ``config``.

    The processing contract is split in two:

    * ``_process(input)`` -- the core logic, overridden by subclasses. Receives a ``Result`` (or raw
      input when first in a run) and returns a new ``Result`` (or ``None`` for no-op/debug steps). The
      default implementation passes the input to every child in ``steps`` and merges their outputs into
      an ``outputResultClass`` (``ListResult`` by default).
    * ``process(input)`` -- the wrapper called by parents. It emits ``component_start`` /
      ``component_end`` / ``component_error`` lifecycle events via ``_emit``, calls ``_process``, fills
      in the new result's ``input`` and ``processor`` back-references, handles ``register_on``
      bookkeeping, and finally forwards the result to ``next_steps`` through ``process_out``.

    ``register_on`` is a list of component ids (or the literal ``"parent"``) resolved to instances at
    build time: after ``_process`` returns, ``process`` walks up the input's ``Result.input`` chain and
    registers the new result as a derivative on each ancestor result produced by one of those
    components, so later steps (e.g. ``LabelTestGate``) can look up what this component said about an
    earlier result.
    """

    parent: Optional["Component"] = None
    steps: List["Component"] = []
    next_steps: List["Component"] = []
    outputResultClass = ListResult
    settings: dict = {}
    register_on: list = []
    config: dict = {}

    def _make_step(self, tree, wf):
        """Build one child component from its config dict.

        If the dict has a ``base`` key, the named entry from the workflow's ``library`` (see
        ``Workflow``) is used as a template: the step's own keys override the library's, except
        ``settings``, whose keys are merged into (and take precedence over) the library settings. The
        resulting ``type`` string is resolved via ``importClass`` and instantiated with this component
        as parent.
        """
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
        """Walk the config *tree* and build this component plus its ``steps``/``next_steps`` children.

        Assigns an ``id`` (from the config, or generated by the workflow), registers the component in
        the workflow's registry, resolves ``register_on`` ids to component instances, and recursively
        instantiates children. A second call on an already-initialized instance is a no-op (guards
        against double ``__init__`` in the AI mixin classes).
        """

        if self.workflow is not None or self.id != "":
            # Already initialized
            return
        self.workflow = workflow
        if "id" in tree:
            self.id = tree["id"]
        elif self.workflow:
            self.id = self.workflow.get_new_id()
        else:
            self.id = str(uuid.uuid4())

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
        """Default core logic: pass *input* to every child in ``steps`` and merge their outputs.

        Subclasses override this with their real computation; they should return a new ``Result``
        (or ``None`` to signal a no-op step).
        """
        # Default is pass down
        merged = self.outputResultClass([], input=input, processor=self)
        for step in self.steps:
            res = step.process(input)
            if res is not None:
                merged.append(res)
        return merged

    def _emit(self, event, **info):
        """Forward a lifecycle event to the workflow's run listeners."""
        if self.workflow is not None and self.workflow is not self:
            self.workflow.emit(event, self, **info)

    def process(self, input) -> Result:
        """Run this component on *input* and forward the output to ``next_steps``.

        Wraps ``_process`` with lifecycle events (``component_start`` / ``component_end``, or
        ``component_error`` before re-raising), back-fills ``input``/``processor`` on the new result,
        performs ``register_on`` derivative registration by walking up the input chain, and returns
        ``process_out``'s result. Returns ``None`` when ``_process`` produced no result.
        """
        self._emit("component_start")
        try:
            new_result = self._process(input)
        except Exception as e:
            self._emit("component_error", error=str(e))
            raise
        self._emit("component_end", preview=result_preview(new_result))

        if new_result is None:
            # debug or other no-op step
            return None
        elif isinstance(input, Result):
            # Ensure the result always knows its input
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
        """Forward this component's output to its ``next_steps``.

        With no ``next_steps``, *input* is returned unchanged. Otherwise each next step processes the
        output and the per-step results are merged into a ``ListResult``.
        """
        if self.next_steps:
            merged = ListResult([], input=input, processor=self)
            for step in self.next_steps:
                x = step.process(input)
                if x is not None:
                    merged.append(x)
            return merged
        else:
            return input
