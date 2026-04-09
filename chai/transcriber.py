from .ai.gemini import GeminiComponent
from .ai.lm_studio import LMStudioComponent
from .ai.ollama import OllamaComponent
from .core import Component
from .result import ItemResult


class Transcriber(Component):
    """Takes an image or audio and extracts the text from the binary content"""

    def __init__(self, tree, workflow, parent=None):
        # Component.__init__ will already be called from the engine
        if not self.prompt_text:
            self.prompt_text = self.workflow.default_prompts.get("transcription", "")
        self.expects = "text"

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
        Transcriber.__init__(self, tree, workflow, parent)

    def _process(self, input):
        return GeminiComponent._process(self, input)


class LMSTranscriber(Transcriber, LMStudioComponent):
    def __init__(self, tree, workflow, parent=None):
        LMStudioComponent.__init__(self, tree, workflow, parent)
        Transcriber.__init__(self, tree, workflow, parent)

    def _process(self, input):
        return LMStudioComponent._process(self, input)


class OllamaTranscriber(Transcriber, OllamaComponent):
    def __init__(self, tree, workflow, parent=None):
        OllamaComponent.__init__(self, tree, workflow, parent)
        Transcriber.__init__(self, tree, workflow, parent)

    def _process(self, input):
        return OllamaComponent._process(self, input)
