from .core import Component
from .result import ListResult, Result


class Workflow(Component):
    """Workflows manage multiple Components in a tree structure, and global variables"""

    def __init__(self, tree, workflow=None):
        if workflow is None:
            workflow = self
        self.registry_ids = {}
        self.id_counter = -1
        super().__init__(tree, workflow)

    def get_component_by_id(self, cid):
        return self.registry_ids.get(cid, None)

    def register_component(self, component):
        if component.id in self.registry_ids:
            raise ValueError(f"{component.id} already in registry for {self.registry_ids[component.id]}")
        self.registry_ids[component.id] = component

    def get_new_id(self):
        self.id_counter += 1
        cid = f"{self.id}_{self.id_counter}"
        if cid in self.registry_ids:
            raise ValueError(f"Tried to create identifier that already exists: {cid}")
        return cid

    def run(self, input=None) -> Result:
        res = ListResult([], processor=self)
        for s in self.steps:
            if s.input is not None:
                res.append(s.run())
            elif input is not None:
                res.append(s.process(input))
            else:
                raise ValueError("No input value provided")
        return res
