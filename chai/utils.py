from .core import Component
from .result import Result


def text_from_input(input, encoding="utf-8"):
    """Best-effort plain text from a Result or raw input value.

    Components that operate on text receive a Result whose value may be a
    string, file bytes (FileItemResult), or -- when they are the first step of
    a test run -- the raw input itself.
    """
    value = input.value if isinstance(input, Result) else input
    if isinstance(value, (bytes, bytearray)):
        return value.decode(encoding, "replace")
    return value if isinstance(value, str) else str(value)


class DebugStep(Component):
    def _process(self, input: Result) -> Result:
        print(f"{self.id}: {repr(input)}")
        return None
