"""Extractors: components that pull structured data out of their input.

AI-backed variants (``GeminiExtractor``, ...) prompt a model for JSON; the deterministic ones here
count words, pick out keywords as labelled entities, or apply XPath expressions to JSON values.
"""

import json
import re
from collections import Counter

from .ai import TransformersComponent, create_ai_component, create_all_components
from .core import Component
from .data_utils import extract_xpath
from .result import ItemResult, ListResult, Result
from .utils import text_from_input


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

    def _process(self, input):
        raise NotImplementedError()


globals().update(create_all_components(Extractor))

TransformersExtractor = create_ai_component(
    "TransformersExtractor", Extractor, TransformersComponent
)


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


class KeywordExtractor(Extractor):
    """Extracts the configured keywords (or regex matches) found in text, as labelled entities -- no model.

    The deterministic stand-in for an entity-extracting model: output is an ``ItemResult`` whose
    value is ``[{"label": ..., "text": ...}]``, one entry per distinct text found under each label,
    in the shape an AI extractor is usually asked for. It reports *what* was found, not where;
    ``segmenter.TextSpanLocator`` finds that.

    Settings:
        - labels:         dict of {label: [patterns]} (a bare string is also accepted) (required)
        - regex:          if true, patterns are regular expressions (default false)
        - case_sensitive: default false
    """

    def __init__(self, tree, workflow, parent=None):
        super().__init__(tree, workflow, parent)
        raw = self.settings.get("labels")
        if not isinstance(raw, dict) or not raw:
            raise ValueError(f"KeywordExtractor ({self!r}) needs the `labels` setting: {{label: [patterns]}}")
        flags = 0 if self.settings.get("case_sensitive", False) else re.IGNORECASE
        regex = bool(self.settings.get("regex", False))
        self.label_patterns = {
            label: [re.compile(p if regex else re.escape(p), flags) for p in ([ps] if isinstance(ps, str) else ps)]
            for label, ps in raw.items()
        }

    def _process(self, input):
        text = text_from_input(input)
        entities, seen = [], set()
        for label, patterns in self.label_patterns.items():
            for pattern in patterns:
                for match in pattern.finditer(text):
                    if match.group(0) and (label, match.group(0)) not in seen:
                        seen.add((label, match.group(0)))
                        entities.append({"label": label, "text": match.group(0)})
        return ItemResult(entities, input=input, processor=self, metadata={"type": "DATA"})


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
        # Type the output so the extracted value can be fed to a downstream AI component
        typ = "TEXT" if isinstance(val, str) else "DATA"
        return ItemResult(val, input=input, processor=self, metadata={"type": typ})
