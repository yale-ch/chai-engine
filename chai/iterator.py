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
        - cache: path to a SQLite run-cache database. Each entry's output is stored after
          processing and replayed on later runs, so an interrupted corpus run resumes where it
          stopped. Editing the iterator's child configuration invalidates the cache.
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

    def _process(self, input: Result) -> Result:
        workers = int(self.settings.get("workers", 1) or 1)
        continue_on_error = bool(self.settings.get("continue_on_error", False))

        cache = cfg_hash = None
        if self.settings.get("cache"):
            from .cache import RunCache, config_hash

            cache = RunCache(self.settings["cache"])
            cfg_hash = config_hash(
                {"steps": self.config.get("steps", []), "type": self.config.get("type", "")}
            )

        def run_one(x):
            if cache is not None:
                from .cache import item_key
                from .storage import _json_safe

                key = item_key(x)
                hit = cache.get(cfg_hash, key)
                if hit is not None:
                    self._emit("iterator_cache_hit", preview=key)
                    return ListResult(
                        [ItemResult(v["value"], metadata=v.get("metadata") or {}) for v in hit],
                        input=x if isinstance(x, Result) else None,
                        processor=self,
                    )
            try:
                step_value = self._run_entry(x, input)
            except Exception as e:
                if not continue_on_error:
                    raise
                return self._entry_error(x, e)
            if cache is not None:
                payload = []
                for r in step_value.value:
                    if isinstance(r, Result) and not getattr(r, "file_bytes", None):
                        payload.append({"value": _json_safe(r.value), "metadata": _json_safe(r.metadata)})
                cache.put(cfg_hash, key, payload)
            return step_value

        items = list(input)
        merged = self.outputResultClass([], input=input, processor=self)
        if workers > 1 and len(items) > 1:
            with ThreadPoolExecutor(max_workers=workers) as pool:
                for step_value in pool.map(run_one, items):  # map preserves input order
                    merged.append(step_value)
        else:
            for x in items:
                merged.append(run_one(x))
        return merged
