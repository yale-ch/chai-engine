"""Sequential context: carry part of one entry's output into the next entry's input.

Some work is not entry-independent. Transcribing a page of a manuscript is easier when the model can
see the last line of the page before it (a sentence, or even a word, runs across the page break);
translating a page is better when the model has read the page before it. ``ContextWindow`` is the
component that carries that text forward: it wraps the steps that need the context, prepends what it
kept from the previous entry to their input, and then keeps a slice of what they produced for the
next one.

It is deliberately component-agnostic -- it injects a plain TEXT ``Result`` alongside the input and
reads text back out of a result, so it works in front of any component that accepts a list-shaped
input (every AI backend does: see ``GeminiComponent.build_contents``), whatever the input carries.

Because the carry-over is sequential, the entries must be processed in order: an ``Iterator`` around
a ``ContextWindow`` must run with ``workers: 1`` (the default), and the input cannot be divided
between parallel runs with a ``SliceIterator``. The component warns when it finds itself inside a
concurrent iterator.
"""

import logging

from .core import Component
from .result import FileItemResult, ItemResult, ListResult, Result
from .utils import text_from_input

logger = logging.getLogger("chai")

#: Result types that carry no text to keep as context.
BINARY_TYPES = ("IMAGE", "AUDIO", "VIDEO", "BINARY")


def _made_by(node, component_id):
    """Was *node* produced by the component with id *component_id*?"""
    return getattr(getattr(node, "processor", None), "id", None) == component_id


def _search_down(node, component_id, seen):
    """Find a result made by *component_id* in *node* or nested inside its value."""
    if not isinstance(node, Result) or id(node) in seen:
        return None
    seen.add(id(node))
    if _made_by(node, component_id):
        return node
    if getattr(node, "file_name", ""):
        # Never touch a FileItemResult's value; reading it would pull the file off disk
        return None
    value = node.value
    if type(value) is list:
        for child in value:
            found = _search_down(child, component_id, seen)
            if found is not None:
                return found
    return None


def result_by_processor(result, component_id):
    """The result component *component_id* made, from *result*, its children, or its provenance chain.

    Looks down first (the component ran as one of the steps just executed) and then up the ``input``
    chain (the component ran earlier and its result is what these steps were given), so a
    ``capture_from`` id can name a component on either side of the window.
    """
    found = _search_down(result, component_id, set())
    if found is not None:
        return found
    node = result
    seen = set()
    while isinstance(node, Result) and id(node) not in seen:
        seen.add(id(node))
        if _made_by(node, component_id):
            return node
        node = node.input
    return None


def _file_of(result):
    """The name of the nearest file up *result*'s provenance chain, or '' if there is none."""
    node = result
    seen = set()
    while isinstance(node, Result) and id(node) not in seen:
        seen.add(id(node))
        if isinstance(node, FileItemResult) and node.file_name:
            return node.file_name
        node = node.input
    return ""


class ContextWindow(Component):
    """Run this component's ``steps`` with the text kept from the previous entry prepended to the input.

    Each time it runs it does three things:

    1. renders the text it kept from the last run into a TEXT ``Result`` (``template``, or ``empty``
       the first time through, when there is nothing to carry);
    2. runs every child in ``steps`` on a ``ListResult`` holding that context result and the input
       (context first, unless ``position`` is 'after'), so a prompt can address the context as
       ``{text_input_0}`` and the real input as ``{text_input_1}``;
    3. keeps a slice of what came out (``capture``, from the last step or from ``capture_from``) for
       the next entry.

    With a single step the step's own result is returned unchanged, so the window is transparent to
    whatever follows it; with several, their outputs are merged into a ``ListResult`` as usual.

    The carry-over lives on the component, so the entries must be processed in order: keep the
    surrounding ``Iterator`` at ``workers: 1`` and do not divide the input between parallel runs.
    ``reset()`` clears it (and is what a second run of the same workflow object needs).

    Settings:
        - capture: how much of the captured text to carry over -- 'last_line' (default), 'last_lines'
          (the last ``lines`` non-empty lines), 'first_line', 'all', or 'tail_chars' (the last
          ``chars`` characters)
        - lines: how many lines 'last_lines' keeps (default 3)
        - chars: how many characters 'tail_chars' keeps (default 500)
        - max_chars: cap on the carried text however it was captured; the tail is kept (default 0, no cap)
        - capture_from: id of the component whose result is captured (default: the last step's output).
          It may name a component inside the window or one that produced an earlier result in the
          input's chain -- e.g. capturing the text a translator was given rather than its translation.
        - template: how the carried text is rendered into the injected result; ``{context}`` is the
          text (default '{context}')
        - empty: text injected when there is nothing to carry yet, i.e. for the first entry (default
          '(none)'). Something is always injected, so the ``{text_input_<i>}`` slots do not shift on
          the first entry.
        - position: 'before' (default) or 'after' -- where the context goes relative to the input
        - initial: text to start with, as if a previous entry had produced it (default '')
        - ignore: values that are not worth carrying (e.g. a no-text marker); captured text equal to
          one of these is treated as empty (default [])
        - on_empty: what to do when nothing was captured -- 'clear' (default) or 'keep' the text from
          the entry before
    """

    def __init__(self, tree, workflow, parent=None):
        super().__init__(tree, workflow, parent)
        self.capture_mode = self.settings.get("capture", "last_line")
        self.capture_lines = int(self.settings.get("lines", 3) or 3)
        self.capture_chars = int(self.settings.get("chars", 500) or 500)
        self.max_chars = int(self.settings.get("max_chars", 0) or 0)
        self.capture_from = self.settings.get("capture_from", "") or ""
        self.template = self.settings.get("template", "{context}")
        self.empty_text = self.settings.get("empty", "(none)")
        self.position = self.settings.get("position", "before")
        self.ignore = [str(x).strip() for x in self.settings.get("ignore", [])]
        self.on_empty = self.settings.get("on_empty", "clear")
        self.initial = self.settings.get("initial", "")
        self.context = self.initial
        self.context_source = ""
        self._warn_if_concurrent()

    def _warn_if_concurrent(self):
        """Carrying context forward only means anything if the entries are processed in order."""
        node = self.parent
        while node is not None:
            if int(getattr(node, "settings", {}).get("workers", 1) or 1) > 1:
                logger.warning(
                    f"{self} carries context between entries, but {node} processes them "
                    f"concurrently (workers > 1): the context will not be the previous entry's. "
                    f"Set workers to 1."
                )
                return
            node = getattr(node, "parent", None)

    def reset(self):
        """Forget the carried text (back to ``initial``), so the next entry starts a new sequence."""
        self.context = self.initial
        self.context_source = ""

    def context_result(self, input):
        """The TEXT Result holding what was carried over, injected alongside *input*."""
        if self.context:
            text = self.template.replace("{context}", self.context)
        else:
            text = self.empty_text
        metadata = {"type": "TEXT", "context": True}
        if self.context_source:
            metadata["context_source"] = self.context_source
        return ItemResult(text, metadata=metadata, input=input, processor=self)

    def combine(self, context, input):
        """The list the steps are run on: the context result and *input*, in ``position`` order."""
        if isinstance(input, ListResult):
            items = list(input)
        elif isinstance(input, Result):
            items = [input]
        else:
            # Raw input (e.g. text typed into a test run): wrap it the way the AI backends do
            items = [ItemResult(input, metadata={"type": "TEXT" if isinstance(input, str) else "DATA"})]
        values = [context] + items if self.position != "after" else items + [context]
        return ListResult(values, input=input, processor=self)

    def extract(self, result):
        """The text to carry forward out of *result*, according to ``capture``."""
        if result is None:
            return ""
        if isinstance(result, FileItemResult) or result.metadata.get("type", "") in BINARY_TYPES:
            logger.warning(f"{self} cannot keep context from a non-text result: {result!r}")
            return ""
        text = text_from_input(result).strip()
        if not text or text in self.ignore:
            return ""
        lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
        if self.capture_mode == "all":
            kept = text
        elif self.capture_mode == "tail_chars":
            kept = text[-self.capture_chars :].strip()
        elif self.capture_mode == "first_line":
            kept = lines[0] if lines else ""
        elif self.capture_mode == "last_lines":
            kept = "\n".join(lines[-self.capture_lines :])
        else:  # last_line
            kept = lines[-1] if lines else ""
        if self.max_chars and len(kept) > self.max_chars:
            kept = kept[-self.max_chars :].strip()
        if kept.strip() in self.ignore:
            # The slice itself may be the value that is not worth carrying -- a no-text marker is the
            # last line of a result whose injected context makes the text as a whole look different
            return ""
        return kept

    def capture(self, produced, last):
        """Keep a slice of what the steps produced for the next entry."""
        source = last
        if self.capture_from:
            source = result_by_processor(produced, self.capture_from)
            if source is None:
                logger.warning(f"{self} found no result from '{self.capture_from}' to keep as context")
        kept = self.extract(source)
        if kept:
            self.context = kept
            self.context_source = _file_of(source)
        elif self.on_empty != "keep":
            self.context = ""
            self.context_source = ""
        return kept

    def _process(self, input: Result) -> Result:
        carried = self.context
        context = self.context_result(input)
        step_input = self.combine(context, input)

        merged = self.outputResultClass([], input=input, processor=self)
        last = None
        for step in self.steps:
            res = step.process(step_input)
            if res is not None:
                merged.append(res)
                last = res

        self.capture(merged, last)
        if len(merged.value) == 1:
            # A single step's result passes through, so the window is transparent to what follows it
            out = merged.value[0]
        else:
            out = merged
        out.metadata["context_used"] = bool(carried)
        return out
