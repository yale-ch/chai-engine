"""Where a result came from: the input it was made from, that input's md5, and where in it the result sits.

``input_provenance`` gives a RESULT row its ``input``/``input_hash``/``input_segment``,
``result_sequence`` its ``input_sequence``, ``correction_target`` its ``previous_id``, and
``source_value`` the value of an ancestor result a storage's ``sources`` setting names.
"""

import hashlib
import os
import re
import threading

import ujson as json

from ..result import FileItemResult, Result
from .values import _json_safe


def source_value(result: Result, component_id: str):
    """Value of the nearest result in *result*'s provenance chain that component *component_id* made.

    Walks up the ``Result.input`` chain, so a result can record what it was computed from -- the CSV
    row, the file, the segment of text -- rather than just the id of a result that lives elsewhere.
    Returns ``None`` when no ancestor came from that component. A ``FileItemResult`` contributes its
    ``file_name``, never the file's content.

    Note that an ``Iterator`` stamps itself onto each entry it processes, so the entry a step ran on
    is found under the *iterator's* id rather than that of the component that first made it.
    """
    found = _source_result(result, component_id)
    if found is None:
        return None
    if isinstance(found, FileItemResult):
        return found.file_name
    return found.value


_URI_SCHEME = re.compile(r"^[a-zA-Z][a-zA-Z0-9+.-]*://")

_FILE_HASHES = {}
_file_hash_guard = threading.Lock()


def _file_md5(path):
    """The md5 of a file's content, remembered while the file's size and mtime stay the same.

    A run stores one result per page, per row or per crop of the same file, so the file is read and
    hashed once rather than once per result.
    """
    try:
        stat = os.stat(path)
    except OSError:
        return None
    key = (path, stat.st_size, stat.st_mtime_ns)
    with _file_hash_guard:
        digest = _FILE_HASHES.get(key)
    if digest is not None:
        return digest
    digest = hashlib.md5()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    digest = digest.hexdigest()
    with _file_hash_guard:
        if len(_FILE_HASHES) >= 4096:  # a long run over many files should not grow without bound
            _FILE_HASHES.clear()
        _FILE_HASHES[key] = digest
    return digest


def content_md5(value):
    """The md5 of *value*'s content: bytes as they are, text as UTF-8, anything else as sorted JSON.

    Sorting the keys makes the digest of a dict (a CSV row, a parsed record) independent of the order
    its keys happen to be in, so the same content always hashes the same way.
    """
    if value is None:
        return None
    if isinstance(value, bytes):
        return hashlib.md5(value).hexdigest()
    if not isinstance(value, str):
        value = json.dumps(_json_safe(value), sort_keys=True, ensure_ascii=False)
    return hashlib.md5(value.encode("utf-8")).hexdigest()


def result_md5(value):
    """The md5 of the content *value* holds -- a ``Result``, a file path or a plain value.

    A ``FileItemResult`` hashes the file (or the bytes it is carrying), a path names its file's
    content, and any other Result hashes its value. The digest is remembered on the Result, so a
    result derived from the same input by several steps is only hashed once.
    """
    if isinstance(value, Result):
        cached = getattr(value, "_content_md5", None)
        if cached:
            return cached
        if isinstance(value, FileItemResult):
            digest = content_md5(value.file_bytes) if value.file_bytes else _file_md5(value.file_name)
        else:
            digest = content_md5(value.value)
        try:
            value._content_md5 = digest
        except AttributeError:  # pragma: no cover - Results are ordinary objects
            pass
        return digest
    if isinstance(value, str) and os.path.isfile(value):
        return _file_md5(value)
    return content_md5(value)


def _source_result(result: Result, component_id: str):
    """The nearest result in *result*'s provenance chain that component *component_id* produced."""
    for node in _provenance(result):
        processor = node.processor
        if processor is not None and getattr(processor, "id", None) == component_id:
            return node
    return None


def _provenance(result):
    """Yield *result* and each Result up its ``input`` chain, stopping if the chain loops back."""
    seen = set()
    while isinstance(result, Result) and id(result) not in seen:
        seen.add(id(result))
        yield result
        result = result.input


def _uri_for(node):
    """Walk *node*'s provenance chain for the file or URI it came from, or ``None`` if there is none."""
    last = None
    for last in _provenance(node):
        if isinstance(last, FileItemResult) and last.file_name:
            return last.file_name
        uri = (last.metadata or {}).get("uri")  # e.g. what a UrlJsonProvider fetched
        if isinstance(uri, str) and uri:
            return uri
    # The end of the chain is the raw input a provider was given: a path or a URI, but not text
    raw = last.input if isinstance(last, Result) else node
    if isinstance(raw, str) and (_URI_SCHEME.match(raw) or os.path.exists(raw)):
        return raw
    return None


def recorded_source(result: Result, source_id=None):
    """The result a stored row should name as what it was generated from, or ``None`` if unmarked.

    A run's real source is rarely the raw input at the bottom of the chain -- that may be a directory
    of scans, iterated into pages, segmented into regions and iterated again. The component whose
    results are the source is marked ``source: true`` in the config (see ``Component``), and this
    returns the nearest such result up *result*'s provenance chain, so every row a page gives rise to,
    however many steps later, records that page. *source_id* names a component explicitly instead.
    """
    if source_id:
        return _source_result(result, source_id)
    for node in _provenance(result):
        if getattr(node.processor, "is_source", False):
            return node
    return None


def input_locator(result: Result, source=None):
    """How to find *result* inside its source: the locators between it and *source*, outermost first.

    Each component that carves a piece out of something records where that piece is (``Result.locator``
    -- a bounding box, a character range), and this collects those frames along the chain, so a stored
    row says where in the source it came from without any of the pieces being materialized. A region
    of a page whose transcript was split into sentences yields two frames,
    ``[{"bbox": [...]}, {"start": 120, "end": 168}]``: the region within the page, then the sentence
    within the region's text. ``None`` when nothing on the way asserted a position.
    """
    frames = []
    for node in _provenance(result):
        if node is source:
            break
        locator = node.extra.get("locator", None) if isinstance(node.extra, dict) else None
        if locator:
            frames.append(_json_safe(locator))
    frames.reverse()  # outermost (nearest the source) first, so they read source -> result
    return frames or None


def input_provenance(result: Result, source_id=None, with_hash=True):
    """What a stored row records about the input *result* was generated from.

    Returns ``{"uri", "hash", "locator"}``: the file or URI of the recorded source (see
    ``recorded_source``), the md5 of that source's content, so rows made from the same input can be
    found together whatever the run or the file name, and the locators saying where in it this result
    is (see ``input_locator``). With no component marked as the source, the input is the result's
    immediate ``input`` and the URI is the nearest file or URI up the chain; a *source_id* naming a
    component that made nothing in this chain records nothing at all. *with_hash* false skips the
    digest, and any file read it would need.
    """
    source = recorded_source(result, source_id)
    start = source
    if source is None and not source_id:
        # Nothing marked, and no component named: the input is what the result was made from, and the
        # nearest file to the result itself is the best URI there is. A *named* source that is not in
        # the chain records nothing rather than quietly standing in something else.
        source = result.input if isinstance(result, Result) else None
        start = result
    return {
        "uri": _uri_for(start),
        "hash": result_md5(source) if with_hash else None,
        "locator": input_locator(result, source),
    }


def correction_target(result: Result):
    """The id of the result *result* corrects, or ``None`` when it is not a correction.

    A component that produces a corrected version of an earlier result says so by putting that
    result (or its id) in the new result's ``extra['corrects']`` -- ``metadata['corrects']`` is read
    too -- and the storages record it as the row's ``previous_id``, which is what makes the
    correction an entry in its own right rather than an overwrite of the original.
    """
    for holder in (getattr(result, "extra", None), getattr(result, "metadata", None)):
        if isinstance(holder, dict) and holder.get("corrects"):
            target = holder["corrects"]
            return target.id if isinstance(target, Result) else str(target)
    return None


def _sequence_of(locator):
    """The last integer frame of an input locator (an index, a page number), if it has one."""
    if not isinstance(locator, list):
        return None
    for frame in reversed(locator):
        if isinstance(frame, int) and not isinstance(frame, bool):
            return frame
        if isinstance(frame, dict):
            for key in ("sequence", "index", "page", "n"):
                if isinstance(frame.get(key), int):
                    return frame[key]
    return None


def result_sequence(result: Result):
    """*result*'s position among its siblings: the nearest ``sequence`` up its chain, or ``None``.

    An ``Iterator`` stamps each entry with its position, so this is the position of the entry the
    result was made from -- the page a transcription and its translation came from, the line within
    an image. Failing that, an integer frame of the input locator counts (see ``_sequence_of``).
    """
    for node in _provenance(result):
        sequence = node.extra.get("sequence") if isinstance(node.extra, dict) else None
        if isinstance(sequence, int) and not isinstance(sequence, bool):
            return sequence
    return _sequence_of(input_locator(result))
