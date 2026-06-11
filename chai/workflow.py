"""Workflow orchestration for chai.

A ``Workflow`` is the root ``Component`` of a run: it owns the component registry, loads shared
defaults (``prompts.json``) and reusable step templates (``library.json``), drives the top-level steps
via ``run``, and fans lifecycle events out to registered listeners.
"""

import logging
import os

import ujson as json

from .core import Component
from .result import ListResult, Result

logger = logging.getLogger("chai")


class Workflow(Component):
    """Workflows manage multiple Components in a tree structure, and global variables.

    As the root ``Component`` it provides run-wide services to its children:

    * **Registry** -- every component built from the config registers itself by ``id``
      (``register_component``); ``get_component_by_id`` resolves the ids used by ``register_on``,
      ``LabelTestGate`` and friends. ``get_new_id`` hands out sequential ids for components without one.
    * **Defaults** -- ``default_prompts`` is loaded from ``settings.defaults_path`` (or the package's
      ``data/prompts.json``); role components (Transcriber, Classifier, ...) fall back to these prompts
      when none is configured. ``library`` is loaded the same way from ``library.json`` and supplies the
      ``base`` templates merged by ``Component._make_step``.
    * **Events** -- listeners added with ``add_listener`` receive a payload dict for every
      ``component_start`` / ``component_end`` / ``component_error`` emitted during a run.

    ``run(input)`` is the entry point; see ``_run`` for how the input is fed to the top-level steps.
    Returns a ``ListResult`` collecting one entry per top-level step's output.

    Settings:
        - defaults_path: directory or .json file with default prompts (default: chai/data/prompts.json)
        - library_path:  directory or .json file with reusable step templates (default: chai/data/library.json)
    """

    def __init__(self, tree, workflow=None):
        if workflow is None:
            workflow = self
        self.registry_ids = {}
        self.id_counter = -1
        self.listeners = []

        # read in defaults from data/

        js = {}
        try:
            if "settings" in tree and "defaults_path" in tree["settings"]:
                dfp = tree["settings"]["defaults_path"]
            else:
                dfp = os.path.join(os.path.dirname(__file__), "data")
            if not dfp.endswith(".json"):
                dfp = os.path.join(dfp, "prompts.json")
            if os.path.exists(dfp):
                with open(dfp) as fh:
                    js = json.load(fh)
        except Exception:
            pass
        self.default_prompts = js

        lib = {}
        try:
            if "settings" in tree and "library_path" in tree["settings"]:
                dfp = tree["settings"]["library_path"]
            else:
                dfp = os.path.join(os.path.dirname(__file__), "data")
            if not dfp.endswith(".json"):
                dfp = os.path.join(dfp, "library.json")
            if os.path.exists(dfp):
                with open(dfp) as fh:
                    lib = json.load(fh)
        except Exception:
            pass
        self.library = lib

        super().__init__(tree, workflow)

    def get_component_by_id(self, cid):
        """Return the registered component with id *cid*, or ``None`` if unknown."""
        return self.registry_ids.get(cid, None)

    def register_component(self, component):
        """Add *component* to the registry; duplicate ids raise ``ValueError``."""
        if component.id in self.registry_ids:
            raise ValueError(f"{component.id} already in registry for {self.registry_ids[component.id]}")
        self.registry_ids[component.id] = component

    def get_new_id(self):
        """Generate a sequential component id of the form ``<workflow_id>_<n>``."""
        self.id_counter += 1
        cid = f"{self.id}_{self.id_counter}"
        if cid in self.registry_ids:
            raise ValueError(f"Tried to create identifier that already exists: {cid}")
        return cid

    def add_listener(self, fn):
        """Subscribe *fn* to run events; it is called with a dict per event
        (component_start / component_end / component_error). Lets harnesses
        like chai-workflow-builder show live progress."""
        self.listeners.append(fn)

    def remove_listener(self, fn):
        """Unsubscribe *fn* from run events; unknown listeners are ignored."""
        if fn in self.listeners:
            self.listeners.remove(fn)

    def emit(self, event, component, **info):
        """Notify listeners that *component* hit a lifecycle point. A failing
        listener is logged and skipped -- observers must never break a run."""
        if not self.listeners:
            return
        payload = {
            "event": event,
            "component_id": component.id,
            "component_name": component.name,
            "component_class": component.__class__.__name__,
            **info,
        }
        for fn in self.listeners:
            try:
                fn(payload)
            except Exception as e:
                logger.warning(f"Workflow listener failed on {event}: {e}")

    def run(self, input=None) -> Result:
        """Execute the workflow, wrapping ``_run`` in workflow-level lifecycle events."""
        self.emit("component_start", self)
        try:
            res = self._run(input)
        except Exception as e:
            self.emit("component_error", self, error=str(e))
            raise
        from .core import result_preview

        self.emit("component_end", self, preview=result_preview(res), result=res)
        return res

    def _run(self, input=None) -> Result:
        """Drive the top-level ``steps`` and collect their outputs in a ``ListResult``.

        Each step that has its own configured ``input`` (typically a Provider) is started via
        ``step.run()`` and ignores the caller-supplied *input*; its output becomes the *input* passed
        to the following step. Steps without their own input receive the current *input* via
        ``step.process``. A step without input when none is available raises ``ValueError``.
        """
        res = ListResult([], processor=self)
        if input is not None:
            res.input = input

        # Each step's output is reassigned to `input`, so it feeds the next top-level step
        for s in self.steps:
            if s.input is not None:
                input = s.run()
                if input is not None:
                    res.append(input)
            else:
                if input is None:
                    raise ValueError("No input value provided")
                input = s.process(input)
                if input is not None:
                    res.append(input)
        return res
