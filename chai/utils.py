"""Small shared helpers: input-to-text coercion and a debugging pass-through step."""

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
    """Prints the input's repr and returns ``None`` (a no-op step).

    Because ``_process`` returns ``None``, ``Component.process`` skips registration and ``next_steps``
    entirely -- drop a DebugStep anywhere in a tree to inspect what flows through without altering it.
    """

    def _process(self, input: Result) -> Result:
        print(f"{self.id}: {repr(input)}")
        return None


class FanOut(Component):
    """Runs every child step on the SAME input and merges their outputs into one ListResult.

    The explicit "two different sets of runs" node: put the parallel branches in ``steps`` and a
    reducer in ``next_steps`` to reunite them::

        {"type": "utils.FanOut",
         "steps": [analysisA, analysisB],
         "next_steps": [{"type": "reducer.MergeDictReducer"}]}

    With ``workers`` the branches run concurrently in a thread pool -- two AI calls overlap instead
    of queueing. Outputs always keep step order, so reducers see the same shape either way.

    Settings:
        - workers: run child steps concurrently with this many threads (default 1 = sequential)
    """

    def _process(self, input: Result) -> Result:
        workers = int(self.settings.get("workers", 1) or 1)
        if workers <= 1 or len(self.steps) <= 1:
            return super()._process(input)
        from concurrent.futures import ThreadPoolExecutor

        merged = self.outputResultClass([], input=input, processor=self)
        with ThreadPoolExecutor(max_workers=workers) as pool:
            for res in pool.map(lambda step: step.process(input), self.steps):  # keeps step order
                if res is not None:
                    merged.append(res)
        return merged
