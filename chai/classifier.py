from random import randint

from .core import Component
from .result import LabelListResult


class Classifier(Component):
    """Takes an input and runs a classification on it to return one or more labels"""

    def _process(self, input):
        # Will assign one or more labels to the input
        return LabelListResult([], input=input, processor=self)


class SampleClassifier(Classifier):
    """Sample a given percentage of inputs by tagging them"""

    def _process(self, input):
        pc = randint(0, 10000) / 100
        flags = ["flagged"] if pc < self.percentage else []
        return LabelListResult(flags, input=input, processor=self)


class HumanClassifier(Classifier):
    """A human will assign the classification in the web API"""

    def _process(self, input):
        return LabelListResult(["NOT_DONE"], input=input, processor=self)
