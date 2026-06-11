"""Result classes: the data objects passed between chai components.

Every component consumes and produces ``Result`` instances. The hierarchy distinguishes single values
(``ItemResult``), lists (``ListResult``), on-disk files (``FileItemResult``), directories
(``DirectoryListResult``) and classifier labels (``LabelListResult``). Results form a linked chain via
``input`` (what they were computed from) and ``processor`` (which component made them), letting later
steps walk back up the provenance chain.
"""

import time
import uuid
from pathlib import Path
from typing import Optional

from .base import BaseThing


class Result(BaseThing):
    """Base class for all data passed between components.

    Attributes:

    * ``value`` -- the payload itself (text, bytes, dict, list of child Results, ...).
    * ``input`` -- the ``Result`` (or raw input value) this one was computed from; following ``input``
      repeatedly walks up the provenance chain to the original workflow input.
    * ``processor`` -- the ``Component`` that produced this result.
    * ``metadata`` -- run information: a ``timestamp`` is always added; components add e.g. ``type``
      (TEXT/IMAGE/AUDIO/DATA), ``token_usage``, ``duration``, ``confidence``, ``bbox``.
    * ``extra`` -- free-form extra information not interpreted by the engine.
    * ``derivative_results`` -- ``{component: [results]}`` registered via ``register_result``; this is
      how ``register_on`` makes e.g. a classifier's labels for a result discoverable by a later
      ``LabelTestGate``.

    ``to_json`` serializes the result (optionally recursing into child Results); ``view`` prints an
    indented tree of nested results for debugging.
    """

    value = None
    input: Optional["Result"] = None
    processor: Optional["BaseThing"] = None
    metadata: dict = {}
    extra: dict = {}
    derivative_results: dict = {}

    def __init__(
        self, value=None, input=None, processor=None, register_on=None, workflow=None, metadata={}, extra={}
    ):
        self.id = str(uuid.uuid4())
        self.input = input or None
        self.workflow = workflow or None
        self.processor = processor or None
        self.metadata = metadata or {}
        if "timestamp" not in metadata:
            self.metadata["timestamp"] = time.time()
        self.extra = extra or {}
        self.derivative_results = {}
        self.set_value(value)
        if register_on is not None:
            self.input = register_on
            if processor is None:
                raise ValueError(f"Must set processor in order to set register_on to {register_on}")
            register_on.register_result(processor, self)

    def __repr__(self):
        return f"{self.__class__.__name__}(value={self.value!r})"

    def set_value(self, value):
        """Assign the payload; subclasses hook this to normalize values (e.g. sorting file lists)."""
        self.value = value

    def to_json(self, recurse=False):
        """Return a JSON representation of the result, including metadata etc."""
        js = {
            "id": self.id,
            "type": self.__class__.__name__,
            "workflowId": self.workflow.id
            if self.workflow
            else (self.processor.workflow.id if self.processor else None),
            "processorId": self.processor.id if self.processor else None,
            "metadata": self.metadata,
            "extraInfo": self.extra,
            "input": self.input.id if isinstance(self.input, Result) else self.input,
        }

        if type(self.value) is list:
            js["value"] = []
            for j in self.value:
                if isinstance(j, Result):
                    if recurse:
                        js["value"].append(j.to_json(recurse))
                    else:
                        js["value"].append(j.id)
                else:
                    js["value"].append(j)
        elif isinstance(self.value, Result):
            if recurse:
                js["value"] = self.value.to_json(recurse)
            else:
                js["value"] = self.value.id
        else:
            js["value"] = self.value

        return js

    def register_result(self, component, result):
        """Record *result* as a derivative of this result made by *component* (no duplicates)."""
        # register the result against the component
        if component not in self.derivative_results:
            self.derivative_results[component] = []
        if result not in self.derivative_results[component]:
            self.derivative_results[component].append(result)

    def get_derivative_result(self, component):
        """Return the list of results *component* registered against this result (may be empty)."""
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


class ItemResult(Result):
    """A Result with a single value.

    Iterating over an ItemResult yields its one value, so code written against ListResults (e.g. the
    AI components' ``build_contents``) also accepts single items.
    """

    def __iter__(self):
        return iter([self.value])


class ResultIter(object):
    """Iterator over a ``ListResult``'s values, wrapping raw entries on the fly.

    Mirrors ``ListResult.__getitem__``: values that are already ``Result`` instances are passed through
    untouched (re-wrapping them would hide their metadata); raw values (paths, strings, ...) are wrapped
    in the list's ``valueClass`` (e.g. ``ItemResult``, or ``FileItemResult`` for directory listings).
    """

    count = 0
    result = None

    def __init__(self, result):
        self.valueClass = result.valueClass
        self.result = result
        self.count = 0

    def __next__(self):
        if self.count == len(self.result.value):
            raise StopIteration()
        val = self.result.value[self.count]
        # Match ListResult.__getitem__: pass existing Results through untouched
        # (re-wrapping them would hide their metadata), only wrap raw values.
        if not isinstance(val, Result) and self.valueClass is not None:
            val = self.valueClass(val)
        self.count += 1
        return val


class ListResult(Result):
    """A Result with multiple values.

    ``value`` is normally a list whose entries may be raw values or nested Results. Iteration and
    indexing wrap raw entries in ``valueClass`` while passing existing Results through unchanged (see
    ``ResultIter``). This is the default ``outputResultClass`` that components use when merging the
    outputs of their ``steps``/``next_steps``.
    """

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

    def __getitem__(self, idx):
        if isinstance(self.value[idx], Result):
            return self.value[idx]
        else:
            return self.valueClass(self.value[idx])


class FileItemResult(ItemResult):
    """A result that mirrors an on-disk file.

    Constructed with a file *path* as its value; ``file_name``/``file_path`` record the location and
    the ``type`` metadata (IMAGE/TEXT/AUDIO/DATA) is guessed from the extension. Reading is lazy: the
    ``value`` property returns ``file_bytes``, opening and caching the file on first access only.
    Components may also assign ``file_bytes`` directly (e.g. ``YoloSegmenter`` crops) to carry
    in-memory content that never touches disk.
    """

    file_bytes = b""
    file_name = ""
    file_path = None

    @property
    def value(self):
        if not self.file_bytes:
            with open(self.file_name, "rb") as fh:
                bs = fh.read()
            self.file_bytes = bs
        return self.file_bytes

    @value.setter
    def value(self, value):
        extensions = {
            ".jpg": "IMAGE",
            ".png": "IMAGE",
            ".tif": "IMAGE",
            ".tiff": "IMAGE",
            ".gif": "IMAGE",
            ".webp": "IMAGE",
            ".jpeg": "IMAGE",
            ".txt": "TEXT",
            ".md": "TEXT",
            ".html": "TEXT",
            ".mp3": "AUDIO",
            ".wav": "AUDIO",
            ".json": "DATA",
            ".xml": "DATA",
        }

        self.file_name = value
        self.file_path = Path(value)
        # Guess file type
        if "type" not in self.metadata:
            t = extensions.get(self.file_path.suffix.lower(), "")
            if t:
                self.metadata["type"] = t

    def __repr__(self):
        return f"<FileItemResult('{self.file_name}')>"


class DirectoryListResult(ListResult):
    """A result that mirrors an on-disk directory.

    ``value`` is a list of file paths (sorted on assignment for deterministic order); iterating or
    indexing wraps each path in a lazily-read ``FileItemResult``. Produced by ``DirFileProvider`` and
    friends.
    """

    valueClass = FileItemResult

    def set_value(self, value):
        if value and type(value) is list:
            value.sort()
        self.value = value


class LabelListResult(ListResult):
    """Receives a list of strings as labels from a classifier.

    Emitted by ``Classifier`` subclasses; gates (``LabelTestGate``, the ``labels`` condition source in
    ``chai.gate``) look specifically for this type when collecting labels from a result and its
    derivatives.
    """

    pass
