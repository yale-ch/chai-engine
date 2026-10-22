"""The CHAI-TEA rows chai objects stand for: each carries its table's columns and can render its row.

Every object a run is made of corresponds to a row of the CHAI-TEA schema (``chai.persistence.schema``), and holds
that row's values itself rather than leaving them for a storage to make up:

* ``Project`` -- a PROJECT row: name, description, is_active, auth_key, planned_enddate
* ``Workflow`` (``chai.workflow``) -- a WORKFLOW row: name, description, project, previous version,
  is_active, created and modified, all settable in the workflow's config
* ``WorkflowRun`` -- a WORKFLOW_RUN row: one per ``Workflow.run``, recording when it ran, how long
  it took, whether it succeeded, and what it consumed
* ``Result`` (``chai.result``) -- a RESULT row (``persistence.rows.result_row`` renders it), with its own
  created/modified timestamps and editor

``Record.to_row()`` gives exactly the table's columns, in order -- which is what the storages write.
"""

import uuid
from datetime import datetime, timezone

from .schema import column_types

#: Namespace for ids derived from a name (a project's, a workflow's config id) -- the same namespace
#: ``as_uuid`` and ``persistence.legacy.record_to_result_row`` hash non-uuid ids with, so the derivations agree.
UUID_NAMESPACE = uuid.UUID("6ba7b810-9dad-11d1-80b4-00c04fd430c8")


def now():
    """The current time, as the timezone-aware UTC datetime every timestamp column holds."""
    return datetime.now(timezone.utc)


def as_uuid(value):
    """*value* as uuid text: used directly if it is one, else hashed into one deterministically."""
    if value is None or value == "":
        return None
    try:
        return str(uuid.UUID(str(value)))
    except ValueError:
        return str(uuid.uuid5(UUID_NAMESPACE, str(value)))


def as_datetime(value):
    """*value* as a UTC datetime -- from a datetime, an ISO-8601 string or epoch seconds -- or ``None``."""
    if value is None or value == "":
        return None
    if isinstance(value, datetime):
        return value if value.tzinfo else value.replace(tzinfo=timezone.utc)
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return datetime.fromtimestamp(value, tz=timezone.utc)
    try:
        return as_datetime(datetime.fromisoformat(str(value).strip().replace("Z", "+00:00")))
    except ValueError:
        return None


def _iso(value):
    return value.isoformat() if isinstance(value, datetime) else value


class Record:
    """Something that is a row of the CHAI-TEA schema; ``ENTITY`` names its table."""

    ENTITY = ""

    def to_row(self):
        """This object's row: exactly its table's columns, in the table's order."""
        return {column: getattr(self, column) for column in column_types(self.ENTITY)}

    def touch(self):
        """Mark the object changed: ``modified`` becomes now."""
        self.modified = now()


class Project(Record):
    """A PROJECT row: the project a workflow belongs to.

    Configured in a workflow's ``project`` key, either as a name or as a dict of the columns. The id is
    derived from the name unless given, so every workflow naming the same project lands under the same
    PROJECT row.
    """

    ENTITY = "PROJECT"

    def __init__(
        self,
        name="default",
        id=None,
        description=None,
        is_active=True,
        auth_key=None,
        planned_enddate=None,
        created=None,
        modified=None,
    ):
        self.name = str(name)
        self.id = as_uuid(id) or as_uuid(f"project:{self.name}")
        self.description = description
        self.is_active = bool(is_active)
        self.auth_key = auth_key
        self.planned_enddate = as_datetime(planned_enddate)
        self.created = as_datetime(created) or now()
        self.modified = as_datetime(modified) or self.created

    @classmethod
    def from_config(cls, config):
        """A project from a workflow's ``project`` setting: a name, a dict of columns, or nothing."""
        if isinstance(config, Project):
            return config
        if isinstance(config, dict):
            return cls(**{key: value for key, value in config.items() if key in column_types(cls.ENTITY)})
        if config:
            return cls(name=config)
        return cls()

    def to_config(self):
        """The ``project`` setting that recreates this project, timestamps included."""
        return {column: _iso(value) for column, value in self.to_row().items() if value is not None}


class WorkflowRun(Record):
    """A WORKFLOW_RUN row: one execution of a workflow.

    ``Workflow.run`` makes one per run and ``start``/``finish`` it; ``id`` is the ``workflow_run_id``
    every RESULT row of the run records. ``last_run`` is when the run started, ``duration`` how long
    it took in seconds, ``metadata`` what it consumed (the usage summary) and ``extra_data`` anything
    else -- the error, for a run that failed.
    """

    ENTITY = "WORKFLOW_RUN"

    def __init__(self, workflow_id=None, id=None, metadata=None, extra_data=None):
        self.id = as_uuid(id) or str(uuid.uuid4())
        self.workflow_id = workflow_id
        self.was_successful = None
        self.metadata = dict(metadata or {}) or None
        self.extra_data = dict(extra_data or {}) or None
        self.duration = None
        self.last_run = None
        self.created = now()
        self.modified = self.created

    def start(self):
        """Note that the run has begun."""
        self.last_run = now()
        self.touch()

    def finish(self, success, metadata=None, error=None):
        """Note how the run ended: whether it succeeded, how long it took, what it recorded."""
        end = now()
        self.was_successful = bool(success)
        if self.last_run is not None:
            self.duration = (end - self.last_run).total_seconds()
        if metadata:
            self.metadata = {**(self.metadata or {}), **metadata}
        if error is not None:
            self.extra_data = {**(self.extra_data or {}), "error": str(error)}
        self.modified = end
