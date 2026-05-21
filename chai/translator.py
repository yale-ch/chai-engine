from .ai import create_all_components
from .core import Component
from .result import ItemResult


class Translator(Component):
    """Takes linguistic content and translates it into one or more different languages"""

    def __init__(self, tree, workflow, parent=None):
        super().__init__(tree, workflow, parent)
        if not getattr(self, "prompt_text", None):
            self.prompt_text = self.workflow.default_prompts.get("translation", "")
        self.expects = "text"

    def _process(self, input):
        raise NotImplementedError()


globals().update(create_all_components(Translator))


class MockTranslator(Translator):
    def _process(self, input):
        return ItemResult(f"translation of {input}", metadata={"effort": 0}, input=input, processor=self)
