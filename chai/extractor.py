import json
from collections import Counter

from .ai.gemini import GeminiComponent
from .ai.lm_studio import LMStudioComponent
from .ai.ollama import OllamaComponent
from .ai.transformers import TransformersComponent
from .core import Component
from .result import ItemResult, Result


class Extractor(Component):
    """Takes content and extracts structured data from it"""

    pass


class WordCountExtractor(Extractor):
    """Extracts individual words from plain text and produces a JSON word-count dictionary."""

    def _process(self, input: Result) -> Result:
        text = input.value if isinstance(input.value, str) else str(input.value)
        words = text.lower().split()
        counts = dict(Counter(words))
        return ItemResult(json.dumps(counts), input=input, processor=self)


class GeminiExtractor(Extractor, GeminiComponent):
    def __init__(self, tree, workflow, parent=None):
        GeminiComponent.__init__(self, tree, workflow, parent)

        if not self.prompt_text:
            self.prompt_text = self.workflow.default_prompts.get("extraction", "")
        self.expects = "json"

    def _process(self, input):
        return GeminiComponent._process(self, input)


class LMSExtractor(Extractor, LMStudioComponent):
    def __init__(self, tree, workflow, parent=None):
        LMStudioComponent.__init__(self, tree, workflow, parent)

        if not self.prompt_text:
            self.prompt_text = self.workflow.default_prompts.get("extraction", "")
        self.expects = "json"

    def _process(self, input):
        return LMStudioComponent._process(self, input)


class OllamaExtractor(Extractor, OllamaComponent):
    def __init__(self, tree, workflow, parent=None):
        OllamaComponent.__init__(self, tree, workflow, parent)

        if not self.prompt_text:
            self.prompt_text = self.workflow.default_prompts.get("extraction", "")
        self.expects = "json"

    def _process(self, input):
        return OllamaComponent._process(self, input)


class NameExtractor(Extractor, TransformersComponent):
    def __init__(self, tree, workflow, parent=None):
        TransformersComponent.__init__(self, tree, workflow, parent)
        if not self.prompt_text:
            self.prompt_text = self.workflow.default_prompts.get("extraction", "")
        self.expects = "text"

    def _process(self, input):
        return TransformersComponent._process(self, input)
