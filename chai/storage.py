from .core import Component, Result


class Storage(Component):
    """Take the input and store it somewhere"""

    def build_json(self, input: Result):
        return input.to_json()

    def _process(self, input: Result) -> Result:
        # Do persistence here
        return input


class FileSystemStorage(Storage):
    def __init__(self, tree, workflow, parent=None):
        super().__init__(tree, workflow, parent)
        if "directory" not in self.settings:
            self.settings["directory"] = "results"


class PostgresStorage(Storage):
    pass


class SqliteStorage(Storage):
    pass
