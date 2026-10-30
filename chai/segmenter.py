"""Segmenters: components that break content into smaller pieces.

Covers text splitting (``TextSegmenter``/``WordSegmenter``), finding extracted values in their text
(``TextSpanLocator``), image region detection (``YoloSegmenter``), mapping the boxes a vision model
reported onto the image's pixels (``BoxLocator``) and AI-generated segmentations
(``GeminiSegmenter`` etc., generated from ``chai.ai``). Output is a list-shaped Result of segments, usually consumed by an ``Iterator``.
"""

import io
import json
import logging
import re

from .ai import create_all_components
from .core import Component
from .persistence.locators import bbox_frame, text_frame
from .result import FileItemResult, ItemResult, ListResult, Result
from .utils import text_from_input

logger = logging.getLogger("chai")


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


#: How ``BoxLocator`` reads the four numbers of a box: the corner order a model reports them in.
BOX_FORMATS = ("yxyx", "xyxy", "xywh")


class BoxLocator(Segmenter):
    """Turns the boxes a vision model reported into regions of the image, in its pixels -- no model.

    Wire this as a ``next_step`` of a model asked to detect things (a ``GeminiSegmenter`` returning
    ``[{"label": "ruler", "box_2d": [ymin, xmin, ymax, xmax]}, ...]``): it reads the detections, walks
    up the input chain to the image they were found in, and emits one result per box whose value is
    the detection with its box in pixels (``{"label", "bbox": [x1, y1, x2, y2], ...}``, any other keys
    the model gave kept) and whose locator is the ``bbox`` frame with the image's size (see
    ``chai.persistence.locators``) -- what ``YoloSegmenter`` writes for its own detections, so the
    two are interchangeable downstream (``measurer.ScaleMeasurer``, a storage). Nothing is drawn.

    The image's size is taken upright -- with its EXIF orientation applied -- since that is the image
    a viewer shows and the one a model is sent when the image is re-encoded (``max_image_size``). A
    camera JPEG stored on its side and sent as raw bytes may be seen on its side by the model; give
    the model a ``max_image_size`` so it sees the upright image the boxes are mapped onto.

    The regions are emitted in reading order (``order``, as for ``YoloSegmenter``) and each records
    its position as its ``sequence``. A box that is not four numbers, or has no area, raises: a reply
    the component cannot read is an error of the call, for ``retries``/``on_error`` to deal with.

    Settings:
        - box_field:   key holding each box (default: auto-detect box_2d/bbox/box/bounding_box)
        - label_field: key holding each label (default: auto-detect label/class/name/type/category)
        - box_format:  'yxyx' (default -- Gemini's ``box_2d``), 'xyxy', or 'xywh' (left, top, width,
                       height)
        - normalized:  what the coordinates are out of: 1000 (default, Gemini), 1 for fractions, or
                       0 for coordinates already in the image's pixels
        - list_field:  key of the detection list when the reply is an object rather than a list
                       (default: the only list in it)
        - labels:      labels to keep, a list (default: all)
        - order:       'reading' (default), 'top_to_bottom' or 'detected'
        - image_from:  id of the component whose result is the image (default: the nearest image up
                       the input chain)
    """

    _BOX_KEYS = ("box_2d", "bbox", "box", "bounding_box")
    _LABEL_KEYS = ("label", "class", "name", "type", "category")

    def __init__(self, tree, workflow, parent=None):
        super().__init__(tree, workflow, parent)
        self.box_format = self.settings.get("box_format", "yxyx")
        if self.box_format not in BOX_FORMATS:
            raise ValueError(f"{self!r}: box_format must be one of {', '.join(BOX_FORMATS)}, not {self.box_format!r}")
        self.normalized = float(self.settings.get("normalized", 1000))
        if self.normalized < 0:
            raise ValueError(f"{self!r}: normalized must be 1000, 1, or 0 for pixels, not {self.normalized}")
        self.order = self.settings.get("order", "reading")
        if self.order not in ORDERS:
            raise ValueError(f"{self!r}: unknown order {self.order!r}; use one of {', '.join(ORDERS)}")
        keep = self.settings.get("labels") or []
        self.keep = {keep} if isinstance(keep, str) else set(keep)

    def _process(self, input):
        detections = self.detections(input)
        image = self.source_image(input)
        if image is None:
            raise ValueError(f"{self} could not find the image in the input chain of {input}")
        width, height = image_size(image)
        regions, dropped = [], []
        for detection in detections:
            label = self.label_of(detection)
            if self.keep and label is None:
                # a reply that lost its labels (a model writing its label under the box's key) would
                # otherwise be filtered away to nothing without a word
                raise ValueError(f"{self}: a detection has no label to keep or drop: {detection!r}")
            if self.keep and label not in self.keep:
                dropped.append(label)
                continue
            x1, y1, x2, y2 = self.pixels(self.box_of(detection), width, height)
            value = {k: v for k, v in detection.items() if k != self.box_key(detection)}
            value["label"] = label
            value["bbox"] = [x1, y1, x2, y2]
            region = ItemResult(value, input=input, processor=self, metadata={"type": "DATA", "label": label})
            region.locator = bbox_frame(x1, y1, x2, y2, size=(width, height))
            regions.append(region)
        ordered = [regions[i] for i in reading_order([r.value["bbox"] for r in regions], self.order)]
        for position, region in enumerate(ordered):
            region.sequence = position
        out = ListResult(ordered, input=input, processor=self)
        if dropped:
            logger.warning(f"{self}: dropped detections labelled {sorted(set(dropped))}, not in labels")
            out.metadata["dropped"] = dropped
        return out

    def detections(self, input):
        """The list of detection dicts in *input*'s value (a list, an object holding one, or JSON text)."""
        value = input.value if isinstance(input, Result) else input
        if isinstance(value, (bytes, bytearray)):
            value = value.decode("utf-8", "replace")
        if isinstance(value, str):
            # A model's reply as text (a segmenter's is), fenced or not
            from .ai.ai_utils import extract_json

            parsed = extract_json(value)
            if parsed is None:
                raise ValueError(f"{self}: the reply is not JSON: {value[:200]!r}")
            value = parsed
        if isinstance(value, dict):
            field = self.settings.get("list_field")
            if field:
                value = value[field]
            elif any(k in value for k in self._BOX_KEYS):
                value = [value]  # a single detection
            else:
                lists = [v for v in value.values() if isinstance(v, list)]
                if len(lists) != 1:
                    raise ValueError(f"{self}: say which list holds the detections (list_field) in {value!r}")
                value = lists[0]
        if not isinstance(value, list) or not all(isinstance(d, dict) for d in value):
            raise ValueError(f"{self}: expected a list of detections, got {value!r}")
        return value

    def box_key(self, detection):
        return self.settings.get("box_field") or next((k for k in self._BOX_KEYS if k in detection), None)

    def box_of(self, detection):
        key = self.box_key(detection)
        box = detection.get(key) if key else None
        if not isinstance(box, (list, tuple)) or len(box) != 4 or not all(isinstance(v, (int, float)) for v in box):
            raise ValueError(f"{self}: a detection's box must be four numbers, got {detection!r}")
        return [float(v) for v in box]

    def label_of(self, detection):
        field = self.settings.get("label_field") or next((k for k in self._LABEL_KEYS if k in detection), None)
        return str(detection[field]) if field and detection.get(field) is not None else None

    def pixels(self, box, width, height):
        """*box* in the model's format and scale as ``[x1, y1, x2, y2]`` pixels, clamped to the image."""
        a, b, c, d = box
        if self.box_format == "yxyx":
            x1, y1, x2, y2 = b, a, d, c
        elif self.box_format == "xyxy":
            x1, y1, x2, y2 = a, b, c, d
        else:  # xywh
            x1, y1, x2, y2 = a, b, a + c, b + d
        if self.normalized:
            x1, x2 = x1 / self.normalized * width, x2 / self.normalized * width
            y1, y2 = y1 / self.normalized * height, y2 / self.normalized * height
        x1, x2 = sorted((min(max(x1, 0.0), width), min(max(x2, 0.0), width)))
        y1, y2 = sorted((min(max(y1, 0.0), height), min(max(y2, 0.0), height)))
        if x2 <= x1 or y2 <= y1:
            raise ValueError(f"{self}: box {box} has no area in a {width}x{height} image")
        return [round(x1, 1), round(y1, 1), round(x2, 1), round(y2, 1)]

    def source_image(self, input):
        """The result holding the image the boxes were found in, or ``None``."""
        from .persistence.provenance import _source_result

        image_from = self.settings.get("image_from")
        if image_from:
            node = _source_result(input, image_from)
            return node.value if node is not None and isinstance(node.value, FileItemResult) else node
        node = input.input if isinstance(input, Result) else None
        while isinstance(node, Result):
            if isinstance(node, FileItemResult) and node.metadata.get("type") == "IMAGE":
                return node
            if isinstance(node.value, FileItemResult) and node.value.metadata.get("type") == "IMAGE":
                return node.value  # an iterator's entry wrapping the file
            if node.metadata.get("type") == "IMAGE" and isinstance(node.value, (bytes, bytearray)):
                return node
            node = node.input
        return None


def image_size(image):
    """``(width, height)`` of an image result as it is shown: upright, its EXIF orientation applied."""
    from PIL import Image

    if isinstance(image, FileItemResult) and not image.file_bytes:
        img = Image.open(image.file_name)
    else:
        img = Image.open(io.BytesIO(image.value))
    with img:
        width, height = img.size
        # EXIF orientations 5-8 turn the picture a quarter turn: the shown image is the stored one on its side
        if img.getexif().get(0x0112, 1) in (5, 6, 7, 8):
            width, height = height, width
    return width, height


def yolo_model(name):
    """An ultralytics ``YOLO`` model from a local path, a stock name, or ``hf://owner/repo/file.pt``.

    A Hugging Face reference is downloaded once into the Hugging Face cache, so a workflow can name a
    published model (``hf://small-models-for-glam/medieval-manuscript-yolov11/medieval-yolov11m.pt``)
    rather than a file someone has to fetch first.
    """
    from ultralytics import YOLO

    name = str(name)
    if name.startswith("hf://"):
        parts = name[len("hf://") :].split("/")
        if len(parts) < 3:
            raise ValueError(f"A Hugging Face model is hf://owner/repo/file.pt, not {name!r}")
        from huggingface_hub import hf_hub_download

        name = hf_hub_download("/".join(parts[:2]), "/".join(parts[2:]))
    return YOLO(name)


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
        - model:      path or name of a YOLO detection model, or ``hf://owner/repo/file.pt`` for one
                      published on Hugging Face (default 'yolo11n.pt')
        - confidence: minimum box confidence to keep (default 0.25)
        - crop:       if true (default), emit cropped PNG bytes; otherwise only metadata
        - classes:    optional list of class names to keep; others are dropped
        - order:      'reading' (default: columns left to right, each top to
                      bottom), 'top_to_bottom', or 'detected' (the model's own order)
    """

    def __init__(self, tree, workflow, parent=None):
        super().__init__(tree, workflow, parent)
        self.model = yolo_model(self.settings.get("model", "yolo11n.pt"))
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
            height, width = result.orig_shape[:2]  # the boxes are in this image's pixels
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
                # Where the crop is in the page, and the page's size, so a stored row can point at
                # the region on any rendition of the page even when the crop is never written anywhere
                crop.locator = bbox_frame(x1, y1, x2, y2, size=(width, height))
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
