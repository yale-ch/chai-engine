"""Extractors: components that pull structured data out of their input.

AI-backed variants (``GeminiExtractor``, ...) prompt a model for JSON; the deterministic ones here
count words or apply XPath expressions to JSON values.
"""

import json
from collections import Counter

from .ai import create_all_components
from .core import Component
from .data_utils import extract_xpath
from .result import ItemResult, ListResult, Result


class Extractor(Component):
    """Takes content and extracts structured data from it.

    Abstract base for the extractor role: subclasses implement ``_process`` and return an
    ``ItemResult`` carrying structured data (a dict/list, or JSON text) extracted from the input
    Result. When no prompt is configured, AI-backed variants use the workflow's default ``extraction``
    prompt and expect JSON output from the model.

    Settings:
        - schema: JSON-Schema dict (or JSON string) the extracted value must match -- type,
          properties, required, items, enum (see chai.schema). Invalid output counts as a failure,
          so combined with `retries` the model is re-asked until the record validates.
    """

    def __init__(self, tree, workflow, parent=None):
        super().__init__(tree, workflow, parent)
        if not getattr(self, "prompt_text", None):
            self.prompt_text = self.workflow.default_prompts.get("extraction", "")
        self.expects = "json"


    def _process(self, input):
        raise NotImplementedError()


globals().update(create_all_components(Extractor))


class DoubleExtractor(Extractor):
    """Multiplies the input value by 2.

    For an integer this doubles it; for a list it follows Python semantics
    and repeats the list (e.g. [1,2,3] * 2 == [1,2,3,1,2,3]).
    """

    def _process(self, input: Result) -> Result:
        val = input.value * 2
        if type(val) is list:
            return ListResult(val, input=input, processor=self)
        return ItemResult(val, input=input, processor=self)


class WordCountExtractor(Extractor):
    """Extracts individual words from plain text and produces a JSON word-count dictionary.

    Input is a Result with a text value (anything else is stringified); output is an ``ItemResult``
    whose value is a JSON object mapping lowercased whitespace-split words to their counts.
    """

    def _process(self, input: Result) -> Result:
        text = input.value if isinstance(input, Result) else input
        if not isinstance(text, str):
            text = str(text)
        words = text.lower().split()
        counts = dict(Counter(words))
        return ItemResult(json.dumps(counts), input=input, processor=self)


class JsonXpathExtractor(Extractor):
    """Extracts a sub-value from a JSON-shaped Result value using an XPath expression.

    Input is a Result whose value is a dict (e.g. an AI extractor's parsed JSON); the dict is rendered
    as XML internally (see ``chai.data_utils``) so the configured XPath can be applied, and the matched
    value is returned as an ``ItemResult``.

    Settings:
        - xpath: XPath expression evaluated against the JSON, e.g. '/people/name' (required)
    """

    def _process(self, input):
        """Take JSON from input and use an XPath in settings to get a sub-value"""

        js = input.value
        xp = self.settings.get("xpath", None)
        if not xp:
            raise ValueError(f"Missing 'xpath' setting in {self}")
        val = extract_xpath(js, xp)
        return ItemResult(val)
