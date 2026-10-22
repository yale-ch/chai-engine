"""Segmenters: components that break content into smaller pieces.

Covers text splitting (``TextSegmenter``/``WordSegmenter``), finding extracted values in their text
(``TextSpanLocator``), image region detection (``YoloSegmenter``) and AI-generated segmentations
(``GeminiSegmenter`` etc., generated from ``chai.ai``). Output is a list-shaped Result of segments, usually consumed by an ``Iterator``.
"""

import io
import json
import re

from .ai import create_all_components
from .core import Component
from .persistence.locators import bbox_frame, text_frame
from .result import FileItemResult, ItemResult, ListResult, Result
from .utils import text_from_input


class Segmenter(Component):
    """Takes content and breaks it up into segments.

    Abstract base for the segmenter role: subclasses implement ``_process`` and return a list-shaped
    Result (``ListResult`` of text chunks, ``FileItemResult`` crops, ...) derived from the input. When
    no prompt is configured, AI-backed variants use the workflow's default ``segmentation`` prompt.
    """

    def __init__(self, tree, workflow, parent=None):
        super().__init__(tree, workflow, parent)
        if not getattr(self, "prompt_text", None):
            self.prompt_text = self.workflow.default_prompts.get("segmentation", "")
        self.expects = "data"

    def _process(self, input):
        raise NotImplementedError()


globals().update(create_all_components(Segmenter))


class TextSegmenter(Segmenter):
    """Splits text into segments deterministically -- no model required.

    Settings:
        - mode:    paragraph (default) | line | sentence | word | regex
        - pattern: split regex, used when mode is 'regex'
        - locate:  emit each segment as a Result carrying its character range in the text it was cut
                   from (``locator`` of ``{"start": ..., "end": ...}``, plus the ``result`` whose text
                   it indexes when that is not a file's content; see ``chai.persistence.locators``)
                   instead of a bare string, so a stored row can point back into the source text
                   rather than a copy of the segment (default false: the segments are plain strings)
    """

    _MODES = {
        "paragraph": r"\n\s*\n",
        "line": r"\n",
        "sentence": r"(?<=[.!?])\s+",
        "word": r"\s+",
    }

    def __init__(self, tree, workflow, parent=None):
        super().__init__(tree, workflow, parent)
        mode = self.settings.get("mode", "paragraph")
        if mode == "regex":
            pattern = self.settings.get("pattern")
            if not pattern:
                raise ValueError(f"TextSegmenter ({self!r}) needs the `pattern` setting when mode is 'regex'")
        elif mode in self._MODES:
            pattern = self._MODES[mode]
        else:
            raise ValueError(f"TextSegmenter ({self!r}) unknown mode {mode!r}")
        self.splitter = re.compile(pattern)

    def _process(self, input):
        text = text_from_input(input)
        if not self.settings.get("locate", False):
            segments = [s.strip() for s in self.splitter.split(text)]
            return ListResult([s for s in segments if s], input=input, processor=self)
        # Locating mode: keep each segment's character range in the text it was cut from. The
        # offsets are of the stripped segment, so they address the segment's own characters.
        out = ListResult([], input=input, processor=self)
        for match in self._spans(text):
            start, end = match
            segment = ItemResult(text[start:end], input=input, processor=self)
            segment.locator = text_frame(start, end, result=input)
            out.append(segment)
        return out

    def _spans(self, text):
        """The ``(start, end)`` of each non-empty segment of *text*, stripped of surrounding space."""
        spans, position = [], 0
        for piece in self.splitter.split(text):
            start = text.index(piece, position) if piece else position
            end = start + len(piece)
            position = end
            stripped_start = start + (len(piece) - len(piece.lstrip()))
            stripped_end = end - (len(piece) - len(piece.rstrip()))
            if stripped_end > stripped_start:
                spans.append((stripped_start, stripped_end))
        return spans


class WordSegmenter(TextSegmenter):
    """Splits text into words (TextSegmenter in 'word' mode)."""

    def __init__(self, tree, workflow, parent=None):
        tree.setdefault("settings", {})["mode"] = "word"
        super().__init__(tree, workflow, parent)


class TextSpanLocator(Segmenter):
    """Finds where each value an extractor reported sits in the text it was extracted from -- no model.

    Wire this as a ``next_step`` of an extractor: it reads the extracted values (JSON text, a dict or
    a list), walks up the input chain to the nearest text (the transcription the extractor read, or
    a text file), and emits one result per occurrence of each value, whose value is the entity
    ``{"label": ..., "text": ...}`` and whose locator is the character range it occupies
    (``{"start", "end", "result", "exact"}``; see ``chai.persistence.locators``). Nothing is
    rendered: a stored row says what was found and where, and a front end decides how to show it.
    Hand the list to an ``Iterator`` with a storage in its steps for one RESULT row per span.

    Entity-shaped dicts (``{"type": "Person", "text": "Tom"}``) are labelled with their entity type
    rather than the JSON field name; label/value key names are auto-detected (label/type/class/
    category/entity and text/value/span/name/match) and overridable. Any other value is labelled
    with the key it was found under. Values are matched longest first, so "Ada Lovelace" wins over a
    bare "Ada", and two labels given the same text each get the span. A value that cannot be found in
    the text is still emitted, with no locator and ``located: false`` in its metadata -- a value the
    text does not contain is worth knowing about.

    Settings:
        - label_field:    dict key holding each entity's label (default: auto-detect)
        - value_field:    dict key holding each entity's text (default: auto-detect)
        - fields:         labels to keep, a list or comma-separated (default: all)
        - occurrences:    'all' (default) or 'first' -- every place a value occurs, or only the first
        - case_sensitive: default true
        - text_from:      id of the component whose result is the text (default: the nearest text
                          up the input chain)
    """

    _LABEL_KEYS = ("label", "type", "class", "category", "entity")
    _VALUE_KEYS = ("text", "value", "span", "name", "match")

    def __init__(self, tree, workflow, parent=None):
        super().__init__(tree, workflow, parent)
        self.occurrences = self.settings.get("occurrences", "all")
        if self.occurrences not in ("all", "first"):
            raise ValueError(f"{self!r}: occurrences must be 'all' or 'first', not {self.occurrences!r}")
        self.case_sensitive = bool(self.settings.get("case_sensitive", True))

    def _process(self, input):
        pairs = self.entities(input)
        found = self.source_text(input)
        if found is None:
            raise ValueError(f"{self} could not find source text in the input chain of {input}")
        text, text_result = found
        out = ListResult([], input=input, processor=self)
        located = set()
        for start, end, label, value in self.locate(text, pairs):
            span = ItemResult(
                {"label": label, "text": value},
                input=input,
                processor=self,
                metadata={"type": "DATA", "label": label, "located": True},
            )
            span.locator = text_frame(start, end, result=text_result, exact=text[start:end])
            out.append(span)
            located.add((label, value))
        for label, value in pairs:
            if (label, value) not in located:
                out.append(
                    ItemResult(
                        {"label": label, "text": value},
                        input=input,
                        processor=self,
                        metadata={"type": "DATA", "label": label, "located": False},
                    )
                )
        return out

    def entities(self, input):
        """The ``(label, text)`` pairs in *input*'s value, longest text first."""
        value = input.value if isinstance(input, Result) else input
        if isinstance(value, (bytes, bytearray)):
            value = value.decode("utf-8", "replace")
        if isinstance(value, str):
            try:
                value = json.loads(value)
            except ValueError:
                value = {"match": value}

        keep = self.settings.get("fields", []) or []
        if isinstance(keep, str):
            keep = [f.strip() for f in keep.split(",") if f.strip()]
        keep = {k.lower() for k in keep}
        pairs, seen = [], set()

        def add(label, text):
            text = str(text).strip()
            key = (str(label), text)
            if text and key not in seen:
                seen.add(key)
                pairs.append(key)

        def entity_pair(d):
            """(label, text) when the dict looks like one extracted entity."""
            lf = self.settings.get("label_field") or next((k for k in self._LABEL_KEYS if k in d), None)
            vf = self.settings.get("value_field") or next(
                (k for k in self._VALUE_KEYS if k in d and k != lf), None
            )
            if lf and vf and isinstance(d.get(vf), (str, int, float)) and isinstance(d.get(lf), str):
                return (d[lf], d[vf])
            return None

        def flatten(label, v):
            if isinstance(v, dict):
                pair = entity_pair(v)
                if pair:
                    add(*pair)
                    return
                for k, sub in v.items():
                    flatten(k, sub)
            elif isinstance(v, list):
                for sub in v:
                    flatten(label, sub)
            elif v is not None and not isinstance(v, bool):
                add(label, v)

        flatten("value", value)
        if keep:
            pairs = [(label, v) for label, v in pairs if label.lower() in keep]
        pairs.sort(key=lambda p: len(p[1]), reverse=True)
        return pairs

    def source_text(self, input):
        """``(text, result)`` for the text the values were extracted from, or ``None``.

        *result* is the result whose value the text is -- ``None`` for the raw workflow input -- so the
        frames can name it (a file's content is left to the row's ``input``; see ``text_frame``).
        """
        from .persistence.provenance import _source_result

        text_from = self.settings.get("text_from")
        if text_from:
            node = _source_result(input, text_from)
            return (text_from_input(node), node) if node is not None else None
        node = input.input if isinstance(input, Result) else None
        while node is not None:
            if not isinstance(node, Result):
                # raw workflow input, e.g. text typed into a test run
                text = str(node)
                return (text, None) if text.strip() else None
            if isinstance(node, FileItemResult):
                if node.metadata.get("type") == "TEXT":
                    return text_from_input(node), node
            elif isinstance(node.value, str) and node.value.strip():
                return node.value, node
            node = node.input
        return None

    def locate(self, text, pairs):
        """``(start, end, label, value)`` for each place a value occurs in *text*, in text order.

        One pass of one pattern over the text, longest value first, so a match is never found inside
        a longer one already taken.
        """
        if not pairs:
            return []
        labels = {}
        for label, value in pairs:  # pairs arrive longest-first
            key = value if self.case_sensitive else value.lower()
            labels.setdefault(key, []).append((label, value))
        flags = 0 if self.case_sensitive else re.IGNORECASE
        pattern = re.compile("|".join(re.escape(v) for v in labels), flags)
        spans, taken = [], set()
        for match in pattern.finditer(text):
            key = match.group(0) if self.case_sensitive else match.group(0).lower()
            if self.occurrences == "first" and key in taken:
                continue
            taken.add(key)
            for label, value in labels[key]:
                spans.append((match.start(), match.end(), label, value))
        return spans


#: The orders ``YoloSegmenter`` can put its regions in.
ORDERS = ("reading", "top_to_bottom", "detected")


def reading_order(boxes, order="reading"):
    """The indexes of *boxes* (``[x1, y1, x2, y2]``) in the order a page is read.

    * ``reading`` (the default) -- columns left to right, and top to bottom within each column. Boxes
      are grouped into a column when they overlap it horizontally by more than half the narrower of
      the two. The narrowest boxes are placed first and a column only widens for a box of similar
      width, so the lines of a column define it; a box spanning several columns (a heading above
      two) joins the leftmost of them, so it is read before both rather than merging them into one.
      A spanning box partway down the page is read within the left column, not between the blocks.
    * ``top_to_bottom`` -- by vertical centre, then left edge: a single column of lines.
    * ``detected`` -- as the detector returned them (roughly by confidence).
    """
    if order not in ORDERS:
        raise ValueError(f"Unknown order {order!r}; use one of {', '.join(ORDERS)}")
    indexes = list(range(len(boxes)))
    if order == "detected":
        return indexes

    def down(i):
        x1, y1, _, y2 = boxes[i]
        return ((y1 + y2) / 2, x1)

    if order == "top_to_bottom":
        return sorted(indexes, key=down)
    columns = []  # [left, right, members]
    for i in sorted(indexes, key=lambda i: boxes[i][2] - boxes[i][0]):
        x1, _, x2, _ = boxes[i]
        width = max(x2 - x1, 1e-9)
        # A box spanning several columns (a heading over two) belongs to the leftmost, read first
        overlapping = [
            column
            for column in columns
            if min(x2, column[1]) - max(x1, column[0]) > 0.5 * min(width, column[1] - column[0])
        ]
        if overlapping:
            column = min(overlapping, key=lambda c: c[0])
            column[2].append(i)
            if width <= 1.5 * (column[1] - column[0]):
                column[0], column[1] = min(column[0], x1), max(column[1], x2)
        else:
            columns.append([x1, x2, [i]])
    columns.sort(key=lambda column: column[0])
    return [i for column in columns for i in sorted(column[2], key=down)]


class YoloSegmenter(Segmenter):
    """Runs a YOLO detection model and emits one FileItemResult per bounding box.

    Each emitted result carries the YOLO class, bbox, and confidence in its
    metadata; downstream gates/iterators can branch on those. When ``crop`` is
    true the cropped PNG bytes are populated in ``file_bytes`` so AI components
    (e.g. GeminiTranscriber) can consume the region without touching disk.

    The regions are emitted in reading order (see ``reading_order``) and each
    records its position as its ``sequence``, so the rows stored for the lines
    of a page sort into the order they are read in (``input_sequence``).

    Settings:
        - model:      path or name of a YOLO detection model (default 'yolo11n.pt')
        - confidence: minimum box confidence to keep (default 0.25)
        - crop:       if true (default), emit cropped PNG bytes; otherwise only metadata
        - classes:    optional list of class names to keep; others are dropped
        - order:      'reading' (default: columns left to right, each top to
                      bottom), 'top_to_bottom', or 'detected' (the model's own order)
    """

    def __init__(self, tree, workflow, parent=None):
        super().__init__(tree, workflow, parent)
        from ultralytics import YOLO

        self.model = YOLO(self.settings.get("model", "yolo11n.pt"))
        self.confidence = self.settings.get("confidence", 0.25)
        self.crop = self.settings.get("crop", True)
        self.keep_classes = set(self.settings.get("classes", []) or [])
        self.order = self.settings.get("order", "reading")
        if self.order not in ORDERS:
            raise ValueError(f"{self!r}: unknown order {self.order!r}; use one of {', '.join(ORDERS)}")

    def _process(self, input):
        from PIL import Image

        if isinstance(input, FileItemResult):
            image_path = input.file_name
        else:
            image_path = input.value if hasattr(input, "value") else str(input)

        img = Image.open(image_path) if self.crop else None
        out = ListResult([], input=input, processor=self)

        for result in self.model(image_path, verbose=False):
            for box in result.boxes:
                conf = float(box.conf[0])
                if conf < self.confidence:
                    continue
                cls_name = result.names[int(box.cls[0])]
                if self.keep_classes and cls_name not in self.keep_classes:
                    continue
                x1, y1, x2, y2 = (float(v) for v in box.xyxy[0].cpu().numpy())

                crop = FileItemResult(
                    "crop.png",
                    input=input,
                    processor=self,
                    metadata={
                        "type": "IMAGE",
                        "yolo_class": cls_name,
                        "bbox": [x1, y1, x2, y2],
                        "confidence": conf,
                    },
                )
                # Where the crop is in the page, so a stored row can point at the region even when
                # the crop itself is never written anywhere
                crop.locator = bbox_frame(x1, y1, x2, y2)
                if self.crop:
                    buf = io.BytesIO()
                    img.crop((x1, y1, x2, y2)).save(buf, format="PNG")
                    crop.file_bytes = buf.getvalue()
                out.append(crop)

        # Detection order is roughly confidence order; the rows want the order the page is read in
        crops = list(out.value)
        ordered = [crops[i] for i in reading_order([c.metadata["bbox"] for c in crops], self.order)]
        for position, crop in enumerate(ordered):
            crop.sequence = position
        out.value = ordered
        return out
