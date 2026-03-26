from .core import Component, Result
from .result import ListResult


class Iterator(Component):
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
