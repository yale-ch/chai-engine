import os

from .core import Component
from .result import DirectoryListResult, Result


class Provider(Component):
    """A Provider generates a Result given a raw input"""

    def run(self) -> Result:
        if self.input:
            return self.process(self.input)
        else:
            raise ValueError("No input value set")


class DirFileProvider(Provider):
    """Take a director name and return a ListResult of ItemResults for each file"""

    def _process(self, input):
        if os.path.exists(input):
            files = os.listdir(input)
            d = DirectoryListResult([os.path.join(input, x) for x in files], input=input, processor=self)
            return super()._process(d)
        else:
            raise ValueError("input file path does not exist")


class IIIFDirFileProvider(DirFileProvider):
    pass
