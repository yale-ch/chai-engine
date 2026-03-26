from .core import Component
from .result import ItemResult


class Describer(Component):
    """Takes content and generates text to describe it"""

    def _process(self, input):
        return ItemResult("A wonderful input", extra={"something": "else"}, input=input, processor=self)
