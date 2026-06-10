"""Iterator: fan a list-shaped Result out over child components, one entry at a time."""

from .core import Component, Result
from .result import ListResult


class Iterator(Component):
    """Take a result and call further components for each entry in result to make a new result.

    Input is any iterable ``Result`` (typically a ``ListResult``/``DirectoryListResult``); iteration
    wraps raw entries in the list's ``valueClass`` while passing nested Results through (see
    ``chai.result.ResultIter``). For every entry, each child in ``steps`` is run on that entry and the
    per-entry outputs are gathered in a ``ListResult``; the overall output is a ``ListResult`` of those
    per-entry lists, parallel to the input entries.
    """

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
