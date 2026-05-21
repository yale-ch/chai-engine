from .ai import create_all_components
from .core import Component
from .result import ItemResult


class Describer(Component):
    """Takes content and generates text to describe it"""

    def __init__(self, tree, workflow, parent=None):
        super().__init__(tree, workflow, parent)
        if not getattr(self, "prompt_text", None):
            self.prompt_text = self.workflow.default_prompts.get("description", "")
        self.expects = "text"

    def _process(self, input):
        raise NotImplementedError()


class MockDescriber(Describer):
    def _process(self, input):
        return ItemResult("A wonderful input", extra={"something": "else"}, input=input, processor=self)


globals().update(create_all_components(Describer))
