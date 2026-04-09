import uuid
from pathlib import Path
from typing import Optional

from .core import BaseThing


class Result(BaseThing):
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
        self.value = value

    def to_json(self):
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
                    js["value"].append(j.to_json())
                else:
                    js["value"].append(j)
        elif isinstance(self.value, Result):
            js["value"] = self.value.to_json()
        else:
            js["value"] = self.value

        return js

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


class ItemResult(Result):
    """A Result with a single value"""

    def __iter__(self):
        return iter([self.value])


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
    """A result that mirrors an on-disk directory"""

    valueClass = FileItemResult

    def set_value(self, value):
        if value and type(value) is list:
            value.sort()
        self.value = value


class LabelListResult(ListResult):
    """Receives a list of strings as labels from a classifier"""

    pass
