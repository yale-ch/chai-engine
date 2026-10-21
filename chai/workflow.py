"""Workflow orchestration for chai.

A ``Workflow`` is the root ``Component`` of a run: it owns the component registry, loads shared
defaults (``prompts.json``) and reusable step templates (``library.json``), drives the top-level steps
via ``run``, and fans lifecycle events out to registered listeners.
"""

import logging
import os

import ujson as json

from .core import Component
from .records import Project, WorkflowRun, as_datetime, as_uuid, now
from .result import ListResult, Result
from .schema import column_types
from .usage import UsageLedger

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
    * **Usage** -- ``usage`` is a ``chai.usage.UsageLedger``: every model-backed component's
      ``token_usage``/``cost`` metadata is recorded as results are produced (``record_usage``), the
      ledger is reset at the start of each ``run``, and ``usage.summary()`` (also attached to the
      workflow's ``component_end`` event as ``usage``) totals calls, tokens and cost per run, per
      component and per model.
    * **CHAI-TEA rows** -- a workflow is a WORKFLOW row and carries its columns, read from its
      config: ``name``, ``description``, ``project`` (a name or a dict of PROJECT columns; see
      ``chai.records.Project``), ``previous_id`` (the workflow this is a new version of),
      ``is_active`` (default true), and ``created``/``modified`` (ISO-8601; default the time the
      workflow is built). ``uuid`` is its WORKFLOW id -- the config's ``uuid``, else derived from its
      ``id`` -- and ``to_row()`` renders the row, ``to_config()`` the config that keeps all of it.
    * **Runs** -- each ``run`` is a ``chai.records.WorkflowRun`` (``current_run``), started when the
      run begins and finished, successful or not, before the final event goes out. ``run_id`` is its
      id: the ``workflow_run_id`` the storages record on every row, so all the rows one run wrote, to
      however many files or databases, belong to the same WORKFLOW_RUN. ``records()`` gives the
      PROJECT, WORKFLOW and WORKFLOW_RUN rows those results hang under.

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
        self.usage = UsageLedger()
        # Made before the steps are built: a storage reads the run id when it is constructed
        self.current_run = WorkflowRun()

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

        # The WORKFLOW columns, from the config
        self.uuid = as_uuid(tree.get("uuid") or self.id)
        self.description = tree.get("description")
        self.project = Project.from_config(tree.get("project"))
        self.previous_id = as_uuid(tree.get("previous_id"))
        self.is_active = bool(tree.get("is_active", True))
        self.created = as_datetime(tree.get("created")) or now()
        self.modified = as_datetime(tree.get("modified")) or self.created
        self.current_run.workflow_id = self.uuid

    @property
    def run_id(self):
        """The id of the current (or, between runs, the last) run: its ``workflow_run_id``."""
        return self.current_run.id

    @property
    def project_id(self):
        return self.project.id

    def touch(self):
        """Mark the workflow changed: ``modified`` becomes now."""
        self.modified = now()

    def to_row(self):
        """The WORKFLOW row: exactly its columns, in order."""
        values = {
            "id": self.uuid,
            "project_id": self.project.id,
            "previous_id": self.previous_id,
            "name": self.name,
            "description": self.description,
            "is_active": self.is_active,
            "created": self.created,
            "modified": self.modified,
        }
        return {column: values[column] for column in column_types("WORKFLOW")}

    def records(self):
        """The rows this workflow's results hang under: ``{entity: row}`` for PROJECT, WORKFLOW, WORKFLOW_RUN."""
        return {
            "PROJECT": self.project.to_row(),
            "WORKFLOW": self.to_row(),
            "WORKFLOW_RUN": self.current_run.to_row(),
        }

    def to_config(self):
        """The workflow's config with its WORKFLOW columns filled in, so saving it keeps them all."""
        config = dict(self.config)
        config.update(
            {
                "id": self.id,
                "uuid": self.uuid,
                "name": self.name,
                "description": self.description,
                "project": self.project.to_config(),
                "is_active": self.is_active,
                "created": self.created.isoformat(),
                "modified": self.modified.isoformat(),
            }
        )
        if self.previous_id:
            config["previous_id"] = self.previous_id
        return {key: value for key, value in config.items() if value is not None}

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

    def record_usage(self, component, metadata, result=None):
        """Add a result's ``token_usage``/``cost`` metadata (if any) to this run's usage ledger.

        Pass the *result* itself when there is one: a result is counted once, so a pass-through
        step that returns an earlier component's result does not count that call again.
        """
        key = getattr(result, "id", None) if result is not None else None
        return self.usage.record(component, metadata, key=key)

    def usage_summary(self):
        """Calls, tokens and cost of the last (or current) run; see ``chai.usage.UsageLedger``."""
        return self.usage.summary()

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
        """Execute the workflow, wrapping ``_run`` in workflow-level lifecycle events.

        The usage ledger is reset first, so ``usage.summary()`` (and the ``usage`` carried by the
        final ``component_end`` event) describes this run alone. A new ``WorkflowRun`` is started,
        and finished -- with its duration, success and usage -- before the final event, so a storage
        closing up on that event sees the finished run.
        """
        self.usage.reset()
        self.current_run = WorkflowRun(workflow_id=self.uuid)
        self.current_run.start()
        self.emit("component_start", self)
        try:
            res = self._run(input)
        except Exception as e:
            self.current_run.finish(False, {"usage": self.usage.summary()}, error=e)
            self.emit("component_error", self, error=str(e))
            raise
        from .core import result_preview

        usage = self.usage.summary()
        self.current_run.finish(True, {"usage": usage})
        self.emit("component_end", self, preview=result_preview(res), result=res, usage=usage)
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
