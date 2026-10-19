"""Iterator: fan a list-shaped Result out over child components, one entry at a time."""

import logging
from concurrent.futures import ThreadPoolExecutor

from .core import Component, Result
from .result import ItemResult, ListResult

logger = logging.getLogger("chai")


class Iterator(Component):
    """Take a result and call further components for each entry in result to make a new result.

    Input is any iterable ``Result`` (typically a ``ListResult``/``DirectoryListResult``); iteration
    wraps raw entries in the list's ``valueClass`` while passing nested Results through (see
    ``chai.result.ResultIter``). For every entry, each child in ``steps`` is run on that entry and the
    per-entry outputs are gathered in a ``ListResult``; the overall output is a ``ListResult`` of those
    per-entry lists, parallel to the input entries (order is preserved even with workers).

    Settings:
        - workers: process this many entries concurrently in a thread pool (default 1, sequential).
          Child components and AI clients are shared across threads -- most API-backed components are
          safe; local models (YOLO, transformers) may not be.
        - continue_on_error: when an entry fails, record an ERROR result for it and keep going
          instead of aborting the whole run (default false)
        - retain_results: keep the per-entry results in the output (default true). Set it false for a
          long run whose results are persisted as they are made (see ``storage.JsonLinesStorage``):
          each entry's results are then dropped once its steps have finished, so memory does not grow
          with the number of entries. The output is an empty list, and the entry counts move to the
          result's metadata. Note that results a step registers on an ancestor result (``register_on``)
          are held by that ancestor, so they stay in memory either way.

    Either way, the output records how many entries were processed in its ``processed`` metadata, and
    how many of those produced an ERROR result in its ``errors`` metadata.
    """

    def _run_entry(self, x, input):
        x.input = input
        x.processor = self
        step_value = ListResult([], input=x, processor=self)
        for step in self.steps:
            res = step.process(x)
            step_value.append(res)
        return step_value

    def _entry_error(self, x, e):
        logger.warning(f"{self} entry failed (continue_on_error): {e}")
        self._emit("iterator_item_error", error=str(e))
        return ItemResult(
            None,
            metadata={"type": "ERROR", "error": str(e), "error_class": e.__class__.__name__},
            input=x if isinstance(x, Result) else None,
            processor=self,
        )

    def select_items(self, input: Result) -> list:
        """The entries of *input* to run the child steps over; subclasses may narrow this."""
        return list(input)

    def _process(self, input: Result) -> Result:
        workers = int(self.settings.get("workers", 1) or 1)
        continue_on_error = bool(self.settings.get("continue_on_error", False))
        retain_results = bool(self.settings.get("retain_results", True))

        def run_one(x):
            try:
                return self._run_entry(x, input)
            except Exception as e:
                if not continue_on_error:
                    raise
                return self._entry_error(x, e)

        items = self.select_items(input)
        merged = self.outputResultClass([], input=input, processor=self)
        counts = {"processed": 0, "errors": 0}

        def collect(step_value):
            """Count an entry's output and, unless results are being discarded, keep it."""
            counts["processed"] += 1
            if step_value is not None and step_value.metadata.get("type", "") == "ERROR":
                counts["errors"] += 1
            if retain_results:
                merged.append(step_value)

        if workers > 1 and len(items) > 1:
            with ThreadPoolExecutor(max_workers=workers) as pool:
                for step_value in pool.map(run_one, items):  # map preserves input order
                    collect(step_value)
        else:
            for x in items:
                collect(run_one(x))
        merged.metadata.update(counts)
        return merged


class SliceIterator(Iterator):
    """Process only every nth entry of a list, so one input can be split across parallel runs.

    Identical to ``Iterator`` except that an entry is processed only when its zero-based position
    satisfies ``position % max_slices == slice``: with ``max_slices`` of 24, slice 0 takes entries
    0, 24, 48, ... and slice 1 takes 1, 25, 49, ... The slices of a given ``max_slices`` are disjoint
    and together cover every entry exactly once, so the same workflow can be run once per slice --
    in separate processes or on separate machines -- to divide the work.

    The output holds one entry per *selected* input entry, not per input entry, so positions in the
    result are the slice's own. Where the original position matters, take it from the entry (a
    ``CsvFileProvider`` row, for instance, records its own ``row`` in its metadata).

    Settings:
        - max_slices: how many slices to divide the input into (default 1: every entry, i.e. the
          same behaviour as a plain Iterator)
        - slice: zero-based index of the slice this run processes; must be less than max_slices
          (default 0)
        - workers, continue_on_error, retain_results: as Iterator
    """

    def select_items(self, input: Result) -> list:
        max_slices = int(self.settings.get("max_slices", 1) or 1)
        offset = int(self.settings.get("slice", 0) or 0)
        if max_slices < 1:
            raise ValueError(f"max_slices must be 1 or more in {self}, got {max_slices}")
        if not 0 <= offset < max_slices:
            raise ValueError(f"slice must be between 0 and {max_slices - 1} in {self}, got {offset}")
        return [x for i, x in enumerate(input) if i % max_slices == offset]
