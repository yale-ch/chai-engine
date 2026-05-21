from .ai import create_all_components
from .core import Component
from .result import ItemResult


class Transcriber(Component):
    """Takes an image or audio and extracts the text from the binary content"""

    def __init__(self, tree, workflow, parent=None):
        super().__init__(tree, workflow, parent)
        if not getattr(self, "prompt_text", None):
            self.prompt_text = self.workflow.default_prompts.get("transcription", "")
        self.expects = "text"

    def _process(self, input):
        raise NotImplementedError()


globals().update(create_all_components(Transcriber))


class MockTranscriber(Transcriber):
    def _process(self, input):
        # Could be a file ref, or extracted pixels
        try:
            filename = input.file_name
        except Exception:
            filename = repr(input)
        return ItemResult(f"transcription of {filename}", metadata={"effort": 0}, input=input, processor=self)
