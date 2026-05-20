import json
from collections import Counter

from .ai import create_all_components
from .core import Component
from .data_utils import extract_xpath
from .result import ItemResult, Result


class Extractor(Component):
    """Takes content and extracts structured data from it"""

    def __init__(self, tree, workflow, parent=None):
        if not getattr(self, "prompt_text", None):
            self.prompt_text = self.workflow.default_prompts.get("extraction", "")
        self.expects = "json"


globals().update(create_all_components(Extractor))


class WordCountExtractor(Extractor):
    """Extracts individual words from plain text and produces a JSON word-count dictionary."""

    def _process(self, input: Result) -> Result:
        text = input.value if isinstance(input.value, str) else str(input.value)
        words = text.lower().split()
        counts = dict(Counter(words))
        return ItemResult(json.dumps(counts), input=input, processor=self)


class JsonXpathExtractor(Extractor):
    def _process(self, input):
        """Take JSON from input and use an XPath in settings to get a sub-value"""

        js = input.value
        xp = self.settings.get("xpath", None)
        if not xp:
            raise ValueError(f"Missing 'xpath' setting in {self}")
        val = extract_xpath(js, xp)
        return ItemResult(val)
