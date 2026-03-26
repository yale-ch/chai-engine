from .core import Component
from .result import ItemResult


class Transcriber(Component):
    """Takes an image or audio and extracts the text from the binary content"""

    def _process(self, input):
        # Could be a file ref, or extracted pixels
        try:
            filename = input.file_name
        except Exception:
            filename = repr(input)
        return ItemResult(f"transcription of {filename}", metadata={"effort": 0}, input=input, processor=self)


class GeminiTranscriber(Transcriber):
    pass


class LocalTranscriber(Transcriber):
    pass
