"""JSON-Lines files: reading them, the run records kept beside a file of RESULT rows, and token totals.

A ``JsonLinesStorage`` file ``run.jsonl`` keeps the PROJECT/WORKFLOW/WORKFLOW_RUN rows its RESULT
rows hang under in ``run.records.json`` (``run_records_file``); ``token_usage_summary`` totals the
tokens an AI run recorded in such a file.
"""

import logging
import os
import threading

import ujson as json

from .schema import TABLE_ORDER
from .values import _isoformat_rows, _json_safe

logger = logging.getLogger("chai")


_line_locks = {}
_line_locks_guard = threading.Lock()


def _line_lock(path):
    """The write lock shared by everything appending to *path* (one per file, not per component)."""
    with _line_locks_guard:
        lock = _line_locks.get(path)
        if lock is None:
            lock = _line_locks[path] = threading.Lock()
        return lock


def run_records_file(jsonl_file):
    """Where the run rows of a JSON-Lines file of RESULT rows are kept: ``run.records.json`` for ``run.jsonl``."""
    return os.path.splitext(jsonl_file)[0] + ".records.json"


def read_run_records(path):
    """The ``{entity: [rows]}`` recorded in a run records file, or ``{}`` if there is none."""
    if not os.path.exists(path):
        return {}
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def add_run_records(path, records):
    """Add ``{entity: row or [rows]}`` to the run records file *path*, replacing rows of the same id.

    Rewritten whole and swapped into place, so a reader never sees half a file.
    """
    current = read_run_records(path)
    for entity, rows in records.items():
        if isinstance(rows, dict):
            rows = [rows]
        by_id = {row["id"]: row for row in current.get(entity, [])}
        for row in _json_safe(_isoformat_rows({entity: rows})[entity]):
            by_id[row["id"]] = row
        current[entity] = list(by_id.values())
    ordered = {entity: current[entity] for entity in TABLE_ORDER if entity in current}
    temp = f"{path}.{os.getpid()}.tmp"
    with open(temp, "w", encoding="utf-8") as fh:
        json.dump(ordered, fh, ensure_ascii=False, indent=2)
    os.replace(temp, path)
    return ordered



def read_jsonl(path):
    """Yield each record of a JSON-Lines file, skipping blank lines and lines that are not JSON.

    A run killed mid-write can leave a partial last line; the rest of the file is still good.
    """
    with open(path, encoding="utf-8") as fh:
        for number, line in enumerate(fh, 1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except ValueError as e:
                logger.warning("%s line %d is not valid JSON, skipped (%s)", path, number, e)


#: The zero value of a usage total, and the order the counts are reported in.
_USAGE_FIELDS = ("calls", "input", "text", "images", "thinking", "output", "total", "duration")


def _usage_of(record, metadata_key="metadata"):
    """The ``token_usage`` dict and duration recorded on one stored record, or ``None``.

    Reads the metadata an AI component puts on its result (see ``GeminiComponent.get_usage``),
    wherever the storage put it: the record's own ``metadata``, or the whole-result ``@record``
    column the tabular storages write.
    """
    for holder in (record.get(metadata_key), record.get("@record"), record.get("value_json")):
        if isinstance(holder, str):
            try:
                holder = json.loads(holder)
            except ValueError:
                continue
        if isinstance(holder, dict):
            meta = holder.get("metadata", holder) if "token_usage" not in holder else holder
            if isinstance(meta, dict) and isinstance(meta.get("token_usage"), dict):
                return meta["token_usage"], meta.get("duration", 0) or 0
    return None


def _add_usage(totals, usage, duration):
    """Add one call's counts to a running total, filling in what the API did not break down."""
    def count(key):
        value = usage.get(key, 0)
        return value if isinstance(value, (int, float)) and value > 0 else 0

    text, images = count("prompt"), count("images")
    thinking, output, total = count("thinking"), count("result"), count("total")
    sent = text + images
    if not sent and total:
        # No per-modality breakdown in the response: what is left of the total is what went in
        sent = max(total - thinking - output, 0)
    if not total:
        total = sent + thinking + output
    totals["calls"] += 1
    totals["input"] += sent
    totals["text"] += text
    totals["images"] += images
    totals["thinking"] += thinking
    totals["output"] += output
    totals["total"] += total
    totals["duration"] += duration


def token_usage_summary(
    jsonl_file, step_key=None, metadata_key="metadata", input_price=None, output_price=None
):
    """Total the tokens (and optionally the cost) an AI run recorded, per component and overall.

    Every AI component records what a call used in its result's ``token_usage`` metadata, so a run
    that persisted its results -- ``JsonLinesStorage`` is the cheapest way, one line per result as it
    is produced -- can be totalled afterwards without re-reading anything from the API. Lines with no
    token usage (deterministic steps, and calls whose response carried no usage metadata) are counted
    in ``without_usage`` rather than silently ignored, so a suspiciously cheap-looking run is visible.

    Returns ``{"components": {id: counts}, "total": counts, "records", "without_usage"}``, where each
    counts dict holds ``calls``, ``input`` (everything sent: ``text`` + ``images``), ``thinking``,
    ``output``, ``total`` and ``duration`` in seconds. *input_price* and *output_price*, in dollars
    per million tokens, add a ``cost`` to each: thinking tokens are billed as output. Take the prices
    from the provider's own pricing page for the exact model -- they are not guessed here.
    """
    components = {}
    records = 0
    without_usage = 0
    with open(jsonl_file, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            records += 1
            record = json.loads(line)
            found = _usage_of(record, metadata_key)
            if found is None:
                without_usage += 1
                continue
            step = record.get(step_key) if step_key else None
            if step is None:
                step = record.get("process_id", record.get("processorId", record.get("step", "unknown")))
            totals = components.setdefault(str(step), dict.fromkeys(_USAGE_FIELDS, 0))
            _add_usage(totals, *found)

    total = dict.fromkeys(_USAGE_FIELDS, 0)
    for counts in components.values():
        for key in _USAGE_FIELDS:
            total[key] += counts[key]
    if input_price is not None or output_price is not None:
        for counts in list(components.values()) + [total]:
            counts["cost"] = (counts["input"] * (input_price or 0) / 1_000_000) + (
                (counts["output"] + counts["thinking"]) * (output_price or 0) / 1_000_000
            )
    return {
        "components": components,
        "total": total,
        "records": records,
        "without_usage": without_usage,
    }
