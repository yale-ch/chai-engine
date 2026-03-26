from .core import Component
from .result import Result


class DebugStep(Component):
    def _process(self, input: Result) -> Result:
        print(f"{self.id}: {repr(input)}")
        return input
