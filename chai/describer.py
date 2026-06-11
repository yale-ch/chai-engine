"""Describers: components that generate descriptive text about their input.

AI-backed variants (``GeminiDescriber``, ``OllamaDescriber``, ...) are generated from the backends in
``chai.ai``; ``FileInfoDescriber`` produces deterministic file statistics without a model.
"""

from .ai import create_all_components
from .core import Component
from .result import ItemResult


class Describer(Component):
    """Takes content and generates text to describe it.

    Abstract base for the describer role: subclasses implement ``_process`` and return an
    ``ItemResult`` whose value is a textual description of the input Result (image, text, data...).
    When no prompt is configured, AI-backed variants use the workflow's default ``description`` prompt.
    """

    def __init__(self, tree, workflow, parent=None):
        super().__init__(tree, workflow, parent)
        if not getattr(self, "prompt_text", None):
            self.prompt_text = self.workflow.default_prompts.get("description", "")
        self.expects = "text"

    def _process(self, input):
        raise NotImplementedError()


class FileInfoDescriber(Describer):
    """Describes a result deterministically: file name, declared type, size,
    and pixel dimensions for images. For semantic descriptions use an AI
    describer (GeminiDescriber, OllamaDescriber, ...)."""

    def _process(self, input):
        from .result import FileItemResult, Result

        details = {}
        if isinstance(input, FileItemResult):
            details["file"] = input.file_name
            details["type"] = input.metadata.get("type", "UNKNOWN")
            data = input.file_bytes
            if not data:
                try:
                    data = input.value
                except OSError:
                    data = b""
            if data:
                details["bytes"] = len(data)
            if data and details["type"] == "IMAGE":
                try:
                    from .image_operations import image_from_bytes

                    img = image_from_bytes(data)
                    details["width"], details["height"] = img.size
                    details["format"] = img.format
                except Exception:
                    pass
        elif isinstance(input, Result):
            value = input.value
            details["type"] = input.metadata.get("type", type(value).__name__)
            if isinstance(value, (str, bytes, list, tuple, dict)):
                details["length"] = len(value)
        else:
            details["type"] = type(input).__name__
            if isinstance(input, (str, bytes, list, tuple, dict)):
                details["length"] = len(input)

        text = ", ".join(f"{k}: {v}" for k, v in details.items())
        return ItemResult(text, metadata=dict(details), input=input, processor=self)


globals().update(create_all_components(Describer))
