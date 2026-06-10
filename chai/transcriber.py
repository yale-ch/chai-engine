"""Transcribers: components that extract text from binary content (images, audio, files).

AI-backed variants (``GeminiTranscriber``, ``LMStudioTranscriber``, ...) are generated from the
backends in ``chai.ai``; ``TextFileTranscriber`` handles plain text files without a model.
"""

from .ai import create_all_components
from .core import Component
from .result import ItemResult


class Transcriber(Component):
    """Takes an image or audio and extracts the text from the binary content.

    Abstract base for the transcriber role: subclasses implement ``_process``, take a Result carrying
    binary or file content (typically a ``FileItemResult`` with IMAGE/AUDIO type metadata) and return
    an ``ItemResult`` whose value is the transcribed text. When no prompt is configured, AI-backed
    variants use the workflow's default ``transcription`` prompt.
    """

    def __init__(self, tree, workflow, parent=None):
        super().__init__(tree, workflow, parent)
        if not getattr(self, "prompt_text", None):
            self.prompt_text = self.workflow.default_prompts.get("transcription", "")
        self.expects = "text"

    def _process(self, input):
        raise NotImplementedError()


globals().update(create_all_components(Transcriber))


class TextFileTranscriber(Transcriber):
    """Extracts the text of a TEXT result (plain files, prior results) without
    a model. For images or audio use an AI transcriber (GeminiTranscriber,
    LMStudioTranscriber, ...).

    Settings:
        - encoding: text encoding for file bytes (default 'utf-8')
    """

    def _process(self, input):
        from .utils import text_from_input

        text = text_from_input(input, encoding=self.settings.get("encoding", "utf-8"))
        return ItemResult(text, metadata={"type": "TEXT"}, input=input, processor=self)
