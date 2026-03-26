from .core import Component
from .result import LabelListResult


class Classifier(Component):
    """Takes an input and runs a classification on it to return one or more labels"""

    def _process(self, input):
        # Will assign one or more labels to the input
        return LabelListResult(["okay"], input=input, processor=self)
