from .ai import create_all_components
from .core import Component
from .result import ItemResult


class Translator(Component):
    """Takes linguistic content and translates it into one or more different languages"""

    def __init__(self, tree, workflow, parent=None):
        super().__init__(tree, workflow, parent)
        if not getattr(self, "prompt_text", None):
            self.prompt_text = self.workflow.default_prompts.get("translation", "")
        self.expects = "text"

    def _process(self, input):
        raise NotImplementedError()


globals().update(create_all_components(Translator))


class GlossaryTranslator(Translator):
    """Deterministic term-by-term translation from a configured glossary.
    For free-form translation use an AI translator (GeminiTranslator, ...).

    Settings:
        - glossary:       dict of {source_term: translated_term} (required)
        - case_sensitive: default false
        - language:       optional target-language tag recorded in metadata
    """

    def __init__(self, tree, workflow, parent=None):
        super().__init__(tree, workflow, parent)
        glossary = self.settings.get("glossary")
        if not isinstance(glossary, dict) or not glossary:
            raise ValueError(f"GlossaryTranslator ({self!r}) needs the `glossary` setting: {{term: translation}}")
        self.case_sensitive = bool(self.settings.get("case_sensitive", False))
        import re

        flags = 0 if self.case_sensitive else re.IGNORECASE
        # Longest terms first so multi-word entries win over their substrings
        terms = sorted(glossary, key=len, reverse=True)
        self.pattern = re.compile("|".join(re.escape(t) for t in terms), flags)
        self.glossary = glossary if self.case_sensitive else {k.lower(): v for k, v in glossary.items()}

    def _lookup(self, match):
        key = match.group(0) if self.case_sensitive else match.group(0).lower()
        return self.glossary.get(key, match.group(0))

    def _process(self, input):
        from .utils import text_from_input

        translated = self.pattern.sub(self._lookup, text_from_input(input))
        metadata = {"type": "TEXT"}
        if self.settings.get("language"):
            metadata["language"] = self.settings["language"]
        return ItemResult(translated, metadata=metadata, input=input, processor=self)
