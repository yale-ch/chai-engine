from .ai.gemini import GeminiComponent
from .core import Component
from .result import ItemResult


class Transcriber(Component):
    """Takes an image or audio and extracts the text from the binary content"""

    def _process(self, input):
        # Could be a file ref, or extracted pixels
        try:
            filename = input.file_name
        except Exception:
            filename = repr(input)
        return ItemResult(f"transcription of {filename}", metadata={"effort": 0}, input=input, processor=self)


class GeminiTranscriber(Transcriber, GeminiComponent):
    def __init__(self, tree, workflow, parent=None):
        GeminiComponent.__init__(self, tree, workflow, parent)

        if not self.prompt_text:
            self.prompt_text = self.workflow.default_prompts.get("transcription", "")
        self.expects = "text"

    def _process(self, input):
        return GeminiComponent._process(self, input)


class LocalTranscriber(Transcriber):
    pass
