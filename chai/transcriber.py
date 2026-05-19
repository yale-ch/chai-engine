from .ai import create_ai_component
from .ai.gemini import GeminiComponent
from .ai.lm_studio import LMStudioComponent
from .ai.ollama import OllamaComponent
from .core import Component
from .result import ItemResult


class Transcriber(Component):
    """Takes an image or audio and extracts the text from the binary content"""

    def __init__(self, tree, workflow, parent=None):
        # Component.__init__ will already be called from the engine
        if not getattr(self, "prompt_text", None):
            self.prompt_text = self.workflow.default_prompts.get("transcription", "")
        self.expects = "text"

    def _process(self, input):
        # Could be a file ref, or extracted pixels
        try:
            filename = input.file_name
        except Exception:
            filename = repr(input)
        return ItemResult(f"transcription of {filename}", metadata={"effort": 0}, input=input, processor=self)


GeminiTranscriber = create_ai_component("GeminiTranscriber", Transcriber, GeminiComponent)
LMSTranscriber = create_ai_component("LMSTranscriber", Transcriber, LMStudioComponent)
OllamaTranscriber = create_ai_component("OllamaTranscriber", Transcriber, OllamaComponent)
