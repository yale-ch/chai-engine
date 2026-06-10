import logging
import os

import ujson as json

from .core import Component
from .result import ListResult, Result

logger = logging.getLogger("chai")


class Workflow(Component):
    """Workflows manage multiple Components in a tree structure, and global variables"""

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

    def add_listener(self, fn):
        """Subscribe *fn* to run events; it is called with a dict per event
        (component_start / component_end / component_error). Lets harnesses
        like chai-workflow-builder show live progress."""
        self.listeners.append(fn)

    def remove_listener(self, fn):
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
        self.emit("component_start", self)
        try:
            res = self._run(input)
        except Exception as e:
            self.emit("component_error", self, error=str(e))
            raise
        self.emit("component_end", self)
        return res

    def _run(self, input=None) -> Result:
        res = ListResult([], processor=self)
        if input is not None:
            res.input = input

        # input gets passed to all steps, not the result of the previous step
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
