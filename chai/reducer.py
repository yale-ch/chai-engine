"""Reducers: components that combine multiple results into a single one.

This is how branches reunite in a chai workflow. The core convergence pattern
is built into ``Component``: a parent's ``steps`` fan OUT (every child gets the
same input, their outputs are merged into one ``ListResult``), and the parent's
``next_steps`` receive that merged list -- so "run A and B, then merge" is::

    {"type": "core.Component", "name": "fan out",
     "steps": [A, B],
     "next_steps": [{"type": "reducer.MergeDictReducer"}]}

When the results to merge are scattered deeper in the tree (inside gate
branches or per-item iterators), ``CollectReducer`` gathers everything that
specific components produced, wherever it landed. The deterministic reducers
(``FlattenReducer``, ``CollectReducer``, ``MergeDictReducer``,
``TextJoinReducer``) run without models; AI-backed variants (``GeminiReducer``,
...) prompt a model with the collected inputs instead.
"""

import json

from .ai import create_all_components
from .core import Component
from .result import FileItemResult, ItemResult, ListResult, Result


class Reducer(Component):
    """Take multiple results and combine them to one.

    Abstract base for the reducer role: subclasses implement ``_process``, take a list-shaped Result
    (typically the merged fan-out of a parent's ``steps``, or an ``Iterator``'s per-entry outputs)
    and return a single combined Result. When no prompt is configured, AI-backed variants use the
    workflow's default ``reduction`` prompt.
    """

    def __init__(self, tree, workflow, parent=None):
        super().__init__(tree, workflow, parent)
        if not getattr(self, "prompt_text", None):
            self.prompt_text = self.workflow.default_prompts.get("reduction", "")
        self.expects = "data"

    def _process(self, input):
        raise NotImplementedError()


globals().update(create_all_components(Reducer))


def walk_results(result, visit, _seen=None):
    """Depth-first walk over a Result and every Result nested in list values.

    FileItemResults are visited but never descended into (their lazy ``value``
    would read the file from disk).
    """
    seen = _seen if _seen is not None else set()
    if not isinstance(result, Result) or result.id in seen:
        return
    seen.add(result.id)
    visit(result)
    if isinstance(result, FileItemResult):
        return
    value = result.value
    if type(value) is list:
        for v in value:
            walk_results(v, visit, seen)
    elif isinstance(value, Result):
        walk_results(value, visit, seen)


class FlattenReducer(Reducer):
    """Flattens nested ListResults into one flat list of leaf results.

    Gates inside iterators inside steps produce deeply nested lists; this
    collapses them so downstream components see one clean sequence. Empty
    branches and empty lists are dropped.

    Settings:
        - keep_empty: keep empty leaves instead of dropping them (default false)
    """

    def _process(self, input):
        keep_empty = bool(self.settings.get("keep_empty", False))
        leaves = []

        def visit(r):
            if r is input:
                return
            if not isinstance(r, FileItemResult) and type(r.value) is list:
                return  # containers contribute their children, not themselves
            leaves.append(r)

        walk_results(input, visit)
        if not keep_empty:
            leaves = [r for r in leaves if isinstance(r, FileItemResult) or r.value not in (None, "", [])]
        return ListResult(leaves, input=input, processor=self)


class CollectReducer(Reducer):
    """Collects every result that specific components produced, wherever those
    results landed in the input's subtree -- gate branches, iterator entries,
    parallel steps -- and merges them into one flat ListResult.

    This is the explicit join for "two different sets of runs feed one step":
    name the components whose outputs you want, and place this reducer
    anywhere downstream of them. Results registered as derivatives (via
    ``register_on``) on subtree results are gathered too.

    Settings:
        - components: component id or list of ids whose results to collect (required)
    """

    def __init__(self, tree, workflow, parent=None):
        super().__init__(tree, workflow, parent)
        raw = self.settings.get("components")
        if not raw:
            raise ValueError(f"CollectReducer ({self!r}) needs the `components` setting (list of ids)")
        if isinstance(raw, str):
            raw = [c.strip() for c in raw.split(",") if c.strip()]
        self.component_ids = list(raw)

    def _process(self, input):
        wanted = set(self.component_ids)
        targets = [self.workflow.get_component_by_id(cid) for cid in self.component_ids]
        targets = [t for t in targets if t is not None]
        collected = []
        seen_ids = set()

        def take(r):
            if r.id not in seen_ids:
                seen_ids.add(r.id)
                collected.append(r)

        def visit(r):
            if r.processor is not None and getattr(r.processor, "id", None) in wanted:
                take(r)
            for target in targets:
                for derived in r.get_derivative_result(target):
                    take(derived)

        walk_results(input, visit)
        return ListResult(collected, input=input, processor=self)


class MergeDictReducer(Reducer):
    """Merges record-shaped child results into a single dict -- e.g. two
    extractors each contribute fields, the reducer emits one combined record.

    Each child of the input list is coerced to a record: dict values merge
    key-by-key (later children win conflicts), JSON-string values are parsed
    first, and non-dict values (labels, plain text) appear under the producing
    component's id. With ``namespaced`` every child is kept under its
    producer's id instead of merging at the top level.

    Settings:
        - namespaced: key every child's record by its producing component id (default false)
    """

    def _process(self, input):
        namespaced = bool(self.settings.get("namespaced", False))
        merged = {}

        items = list(input) if type(input.value) is list else [input]
        for i, item in enumerate(items):
            value = item.value if isinstance(item, Result) else item
            if isinstance(value, (bytes, bytearray)):
                value = value.decode("utf-8", "replace")
            if isinstance(value, str):
                try:
                    value = json.loads(value)
                except ValueError:
                    pass
            key = None
            if isinstance(item, Result) and item.processor is not None:
                key = getattr(item.processor, "id", None)
            key = key or f"input_{i}"

            if namespaced:
                merged[key] = value
            elif isinstance(value, dict):
                merged.update(value)
            else:
                merged[key] = value

        return ItemResult(merged, metadata={"type": "DATA"}, input=input, processor=self)


class TextJoinReducer(Reducer):
    """Joins the text of every leaf result into one text Result.

    Walks nested lists (so it can directly follow an Iterator or gate tree
    without a FlattenReducer first) and concatenates the string values.

    Settings:
        - separator: string placed between texts (default a blank line)
    """

    def _process(self, input):
        separator = self.settings.get("separator", "\n\n")
        texts = []

        def visit(r):
            if isinstance(r, FileItemResult):
                return
            value = r.value
            if isinstance(value, (bytes, bytearray)):
                value = value.decode("utf-8", "replace")
            if isinstance(value, str) and value.strip():
                texts.append(value)

        walk_results(input, visit)
        if not texts and not isinstance(input, Result) and str(input).strip():
            texts = [str(input)]
        return ItemResult(separator.join(texts), metadata={"type": "TEXT"}, input=input, processor=self)
