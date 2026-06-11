"""Annotators render Results as human-reviewable artifacts.

Where most components transform data for the *next* component, an Annotator
produces something for a *person*: the original image with detection boxes
burned in, or source text with extracted values highlighted. They are intended
as ``next_steps`` of detection/extraction components so a test run can be
inspected visually rather than as raw JSON.
"""

import json

from .core import Component
from .result import FileItemResult, ItemResult, Result


def collect_detections(result):
    """Walk a Result tree and gather detection dicts (bbox/label/confidence).

    YoloSegmenter emits one child result per box with ``bbox``, ``yolo_class``
    and ``confidence`` in its metadata; this flattens those (or any result
    carrying a ``bbox``) into plain dicts that annotate_image_bytes accepts.
    """
    detections = []

    def walk(r):
        if not isinstance(r, Result):
            return
        md = r.metadata or {}
        if "bbox" in md:
            detections.append(
                {
                    "bbox": [float(v) for v in md["bbox"]],
                    "label": md.get("yolo_class", md.get("label", "")),
                    "confidence": md.get("confidence"),
                }
            )
        # Don't trigger FileItemResult's lazy file read by touching .value
        if not isinstance(r, FileItemResult) and type(r.value) is list:
            for v in r.value:
                walk(v)

    walk(result)
    return detections


def find_source_image(result):
    """Walk up the input chain to the nearest on-disk/in-memory IMAGE result.

    Accepts a Result or a raw input value: the chain may bottom out at the raw
    workflow input (e.g. a path string typed into a test run rather than a
    Provider-made result), which gets wrapped in a FileItemResult when it
    points at an image file.
    """
    r = result
    while r is not None:
        if not isinstance(r, Result):
            try:
                candidate = FileItemResult(str(r))
                if candidate.metadata.get("type") == "IMAGE" and candidate.file_path.exists():
                    return candidate
            except (OSError, ValueError):
                pass
            return None
        if isinstance(r, FileItemResult) and r.metadata.get("type") == "IMAGE":
            return r
        r = r.input
    return None


def annotate_image_bytes(image_bytes, detections, thickness=None, with_labels=True, with_confidence=True):
    """Draw *detections* onto *image_bytes* with supervision; return PNG bytes.

    ``thickness`` defaults to scaling with image size (a 2px line on a 5000px
    herbarium scan disappears when the preview is downscaled); pass an int to
    force a specific line width. Label text scales the same way.
    """
    import numpy as np
    import supervision as sv

    from .image_operations import bytes_from_image, image_from_bytes

    img = image_from_bytes(image_bytes).convert("RGB")
    if not detections:
        return bytes_from_image(img)

    long_edge = max(img.size)
    if thickness is None:
        thickness = max(2, round(long_edge / 400))
    text_scale = max(0.5, long_edge / 1500)

    class_names = sorted({d["label"] for d in detections})
    name_to_id = {name: i for i, name in enumerate(class_names)}
    sv_detections = sv.Detections(
        xyxy=np.array([d["bbox"] for d in detections], dtype=float),
        confidence=np.array([d.get("confidence") or 0.0 for d in detections], dtype=float),
        class_id=np.array([name_to_id[d["label"]] for d in detections], dtype=int),
    )

    scene = np.asarray(img)
    scene = sv.BoxAnnotator(thickness=thickness).annotate(scene=scene.copy(), detections=sv_detections)
    if with_labels:
        labels = []
        for d in detections:
            label = d["label"] or "?"
            if with_confidence and d.get("confidence") is not None:
                label = f"{label} {d['confidence']:.2f}"
            labels.append(label)
        scene = sv.LabelAnnotator(
            text_scale=text_scale, text_thickness=max(1, thickness // 2), text_padding=max(4, thickness)
        ).annotate(scene=scene, detections=sv_detections, labels=labels)

    from PIL import Image

    return bytes_from_image(Image.fromarray(scene))


class Annotator(Component):
    """Renders the input as a human-reviewable artifact (annotated image or text).

    Abstract base for the annotator role: subclasses implement ``_process``, take a detection or
    extraction Result, and return a new Result (image or text) with ``annotated: true`` metadata,
    leaving the original data untouched. Typically wired as a ``next_step`` of the component whose
    output should be reviewed.
    """

    def __init__(self, tree, workflow, parent=None):
        super().__init__(tree, workflow, parent)
        self.expects = "data"

    def _process(self, input):
        raise NotImplementedError()


class ImageBoxAnnotator(Annotator):
    """Draws detection bounding boxes onto the source image using supervision.

    Wire this as a next_step of ``segmenter.YoloSegmenter``: it collects the
    bbox/yolo_class/confidence metadata from the detection results, walks up
    the input chain to the original image, and emits a new IMAGE
    FileItemResult with the boxes and labels burned in.

    Settings:
        - labels:     if true (default), draw class labels on each box
        - confidence: if true (default), append confidence scores to labels
        - thickness:  box line thickness in pixels (default: scales with image size)
        - output:     optional file path to also write the annotated PNG
    """

    def _process(self, input):
        detections = collect_detections(input)
        source = find_source_image(input)
        if source is None:
            raise ValueError(f"{self} could not find a source IMAGE result in the input chain of {input}")

        thickness = self.settings.get("thickness")
        png = annotate_image_bytes(
            source.value,
            detections,
            thickness=int(thickness) if thickness else None,  # None = scale with image size
            with_labels=bool(self.settings.get("labels", True)),
            with_confidence=bool(self.settings.get("confidence", True)),
        )

        out_name = self.settings.get("output", f"{source.file_path.stem}_annotated.png")
        annotated = FileItemResult(
            out_name,
            input=input,
            processor=self,
            metadata={
                "type": "IMAGE",
                "annotated": True,
                "source": source.file_name,
                "detections": detections,
            },
        )
        annotated.file_bytes = png
        if self.settings.get("output"):
            with open(self.settings["output"], "wb") as fh:
                fh.write(png)
        return annotated


class TextHighlightAnnotator(Annotator):
    """Highlights extracted values in their source text -- ONE visualization
    for the whole task, with each value labelled by its entity type
    ("Tom PERSON, D.C. PLACE"), so wire a single annotator per extractor.

    Walks up the input chain to the nearest plain-text result (e.g. a
    transcription or the raw test input), then wraps every extracted value in
    a labelled highlight. Entity-shaped dicts ({"type": "Person", "text":
    "Tom"}) are labelled with their entity type, not the JSON field name;
    label/value key names are auto-detected (label/type/class/category and
    text/value/span/name) and overridable. Output is an ``annotated: true``
    TEXT result whose value is HTML ``<mark data-label=...>`` markup (or
    ``**bold**`` markdown); viewers color each distinct label differently.

    Settings:
        - format: 'html' (default) or 'markdown'
        - label_field: dict key holding each entity's label (default: auto-detect)
        - value_field: dict key holding each entity's text (default: auto-detect)
        - fields: optional labels to keep, comma-separated (default: all)
    """

    _LABEL_KEYS = ("label", "type", "class", "category", "entity")
    _VALUE_KEYS = ("text", "value", "span", "name", "match")

    def _process(self, input):
        spans = self._extract_spans(input)
        source_text = self._find_source_text(input)
        if source_text is None:
            raise ValueError(f"{self} could not find source text in the input chain of {input}")

        fmt = self.settings.get("format", "html")
        annotated_text, matched = self._highlight(source_text, spans, fmt)
        return ItemResult(
            annotated_text,
            input=input,
            processor=self,
            metadata={
                "type": "TEXT",
                "annotated": True,
                "format": fmt,
                "highlights": matched,
            },
        )

    def _extract_spans(self, input):
        """Pull {label: value} pairs out of the input -- JSON extractor output,
        a dict value, or a list of labels."""
        value = input.value
        if isinstance(value, (bytes, bytearray)):
            value = value.decode("utf-8", "replace")
        if isinstance(value, str):
            try:
                value = json.loads(value)
            except Exception:
                value = {"match": value}

        keep = self.settings.get("fields", []) or []
        if isinstance(keep, str):
            keep = [f.strip() for f in keep.split(",") if f.strip()]
        keep = {k.lower() for k in keep}
        pairs = []
        seen = set()

        def add(label, text):
            text = str(text).strip()
            if not text:
                return
            key = (str(label), text)
            if key not in seen:
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
                ep = entity_pair(v)
                if ep:
                    add(*ep)
                    return
                for k, sub in v.items():
                    flatten(k, sub)
            elif isinstance(v, list):
                for sub in v:
                    flatten(label, sub)
            elif v is not None:
                add(label, v)

        flatten("value", value)
        if keep:
            pairs = [(label, v) for label, v in pairs if label.lower() in keep]
        # Longest values first so "Ada Lovelace" wins over a bare "Ada"
        pairs.sort(key=lambda p: len(p[1]), reverse=True)
        return pairs

    def _find_source_text(self, input):
        r = input.input
        while r is not None:
            if not isinstance(r, Result):
                # raw workflow input, e.g. text typed into a test run
                text = str(r)
                return text if text.strip() else None
            v = r.value
            if isinstance(v, str) and v.strip():
                return v
            r = r.input
        return None

    def _highlight(self, text, pairs, fmt):
        import re

        if not pairs:
            return text, []
        # One regex pass over the original text: longer values win overlaps
        # and nothing gets re-wrapped inside an earlier replacement.
        by_value = {}
        for label, value in pairs:  # pairs arrive longest-first
            by_value.setdefault(value, label)
        pattern = re.compile("|".join(re.escape(v) for v in by_value))
        matched = []
        seen = set()

        def repl(m):
            value = m.group(0)
            label = by_value[value]
            if (label, value) not in seen:
                seen.add((label, value))
                matched.append({"label": label, "value": value})
            if fmt == "markdown":
                return f"**{value}** _[{label}]_"
            return f'<mark title="{label}" data-label="{label}">{value}</mark>'

        return pattern.sub(repl, text), matched
