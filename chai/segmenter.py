from .ai import create_all_components
from .core import Component
from .result import ItemResult, ListResult


class Segmenter(Component):
    """Takes content and breaks it up into segments"""

    def __init__(self, tree, workflow, parent=None):
        super().__init__(tree, workflow, parent)
        if not getattr(self, "prompt_text", None):
            self.prompt_text = self.workflow.default_prompts.get("segmentation", "")
        self.expects = "data"

    def _process(self, input):
        raise NotImplementedError()


globals().update(create_all_components(Segmenter))


class MockSegmenter(Segmenter):
    def _process(self, input):
        # Could be a file ref, or extracted pixels
        try:
            filename = input.file_name
        except Exception:
            filename = repr(input)
        return ItemResult(f"segmentation of {filename}", metadata={"effort": 0}, input=input, processor=self)


class WordSegmenter(Segmenter):
    def __init__(self, tree, workflow, parent=None):
        super().__init__(tree, workflow, parent)

    def _process(self, input):
        return ListResult(input.split())
