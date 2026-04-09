import os

import ujson as json

from .core import Component, Result


class Storage(Component):
    """Take the input and store it somewhere"""

    def build_json(self, input: Result):
        return input.to_json(recurse=False)

    def _process(self, input: Result) -> Result:
        # Do persistence here
        return input


class FileSystemStorage(Storage):
    def __init__(self, tree, workflow, parent=None):
        super().__init__(tree, workflow, parent)
        if "directory" not in self.settings:
            self.settings["directory"] = "results"
        dn = self.settings["directory"]
        if not os.path.exists(dn):
            os.makedirs(dn, exist_ok=True)

    def _process(self, input: Result) -> Result:
        """Write the result to the filesystem according to the processor id"""
        if input.processor is not None:
            # Store in directory per processor
            base = os.path.join(self.settings["directory"], input.processor.id)
        else:
            base = os.path.join(self.settings["directory"], "base")
        if not os.path.exists(base):
            os.makedirs(base, exist_ok=True)
        # Now make a pair-tree
        pair = os.path.join(base, input.id[0:2], input.id[2:4])
        if not os.path.exists(pair):
            os.makedirs(pair, exist_ok=True)
        fn = os.path.join(pair, f"{input.id}.json")
        if os.path.exists(fn):
            # make a new version
            vn = 1  # FIXME: Make this the count of files with this name
            fn = f"{fn}.{vn}"
        js = self.build_json(input)
        with open(fn, "w") as fh:
            json.dump(js, fh)

        return input


class PostgresStorage(Storage):
    pass


class SqliteStorage(Storage):
    pass
