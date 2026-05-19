from .ai import create_ai_component
from .ai.gemini import GeminiComponent
from .ai.lm_studio import LMStudioComponent
from .ai.ollama import OllamaComponent
from .core import Component
from .result import ItemResult


class Segmenter(Component):
    """Takes content and breaks it up into segments"""

    def __init__(self, tree, workflow, parent=None):
        # Component.__init__ will already be called from the engine
        if not getattr(self, "prompt_text", None):
            self.prompt_text = self.workflow.default_prompts.get("segmentation", "")
        self.expects = "data"

    def _process(self, input):
        # Could be a file ref, or extracted pixels
        try:
            filename = input.file_name
        except Exception:
            filename = repr(input)
        return ItemResult(f"segmentation of {filename}", metadata={"effort": 0}, input=input, processor=self)


GeminiSegmenter = create_ai_component("GeminiSegmenter", Segmenter, GeminiComponent)
LMSSegmenter = create_ai_component("LMSSegmenter", Segmenter, LMStudioComponent)
OllamaSegmenter = create_ai_component("OllamaSegmenter", Segmenter, OllamaComponent)
