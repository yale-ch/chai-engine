from .core import Component, Result


class Storage(Component):
    """Take the input and store it in the backend"""

    def _process(self, input: Result) -> Result:
        # Do persistence here
        return input
