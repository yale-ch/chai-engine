"""Measurers: components that turn the regions of an image into real-world sizes -- no model.

A photograph says nothing about how big anything in it is until something of known size is in the
picture with it: a ruler laid along the object, a scale bar, a colour chart. ``ScaleMeasurer`` takes
the regions a detector found in one image (``YoloSegmenter`` crops, ``BoxLocator`` regions -- anything
with a ``bbox`` locator and a label), treats the ones named in its ``references`` as those objects of
known size, works out how many pixels make a unit along each axis, and gives every other region its
width and height in that unit. It draws nothing: the size goes into each region's value and the box
stays in its locator, for a front end to show however it likes.
"""

from .core import Component
from .result import FileItemResult, ItemResult, ListResult, Result

#: The dimensions a reference may give: its width, its height, or the length of its long side.
REFERENCE_DIMENSIONS = ("width", "height", "length")


class ScaleMeasurer(Component):
    """Sizes the regions of an image from the objects of known size detected in it.

    Its input is the list of regions found in one image; each needs a label (the value's ``label``,
    else the metadata's ``label`` or ``yolo_class``) and a ``bbox`` locator in the image's pixels.
    ``references`` maps the labels of the objects of known size to that size, along the axis it was
    measured: ``{"vertical-ruler": {"height": 92}, "horizontal-ruler": {"width": 61}}`` for a yardstick
    standing up and a ruler lying down, ``{"scale_bar": {"length": 5}}`` for a bar whose long side is
    5 units whichever way it lies, ``{"color_chart": {"width": 21.6, "height": 27.9}}`` for both.

    A detector may find one reference twice (a second ruler at the edge of the frame, a ruler boxed in
    two pieces): by default only the most confident detection of each reference label is used
    (``duplicates: best``) and the others are emitted with ``reference_used: false``; ``mean`` uses
    them all. Each reference gives pixels-per-unit along its axis; different references along one
    axis are averaged. A
    region's width is converted with the horizontal scale and its height with the vertical one; when
    only one axis has a reference its scale is used for both (square pixels, as a camera's are) and
    the region says so. The output is a list of one result per region, in the input's order, carrying
    the region's ``bbox`` locator and ``sequence`` (so its row keeps the box in ``input_segment``,
    however the list is iterated afterwards), whose value is the
    region's value plus ``width``, ``height``, ``unit``, ``width_px`` and ``height_px``, and whose
    metadata records ``measured``, ``reference``, the ``scale`` used (pixels per unit, ``x``/``y``) and
    which references ``calibrated`` each dimension. A reference's own row carries its size in pixels,
    its known size and its scale. An image with no reference in it is not an error -- its regions are
    emitted with ``measured: false`` and no size -- and the output's metadata holds the image's
    ``scale``, so a run can be checked for images that could not be measured.

    Settings:
        - references: {label: {"width"|"height"|"length": size, ...}} (required)
        - unit:       the unit the reference sizes are in (default 'cm')
        - measure:    labels to measure, a list (default: every region that is not a reference)
        - duplicates: 'best' (default) -- the most confident detection of each reference -- or 'mean'
        - references_out: emit the reference regions too (default true)
        - precision:  decimal places for the sizes (default 2)
    """

    def __init__(self, tree, workflow, parent=None):
        super().__init__(tree, workflow, parent)
        self.references = self.settings.get("references") or {}
        if not isinstance(self.references, dict) or not self.references:
            raise ValueError(f"{self!r}: references must map each label of known size to its size")
        for label, size in self.references.items():
            if not isinstance(size, dict) or not size:
                raise ValueError(f"{self!r}: reference {label!r} must give its width, height or length")
            for dimension, value in size.items():
                if dimension not in REFERENCE_DIMENSIONS:
                    raise ValueError(
                        f"{self!r}: reference {label!r} has {dimension!r}; give one of {', '.join(REFERENCE_DIMENSIONS)}"
                    )
                if not isinstance(value, (int, float)) or isinstance(value, bool) or value <= 0:
                    raise ValueError(f"{self!r}: reference {label!r} {dimension} must be a positive number, not {value!r}")
            if "length" in size and len(size) > 1:
                raise ValueError(f"{self!r}: reference {label!r} gives a length; it cannot also give width or height")
        self.unit = self.settings.get("unit", "cm")
        measure = self.settings.get("measure") or []
        self.measure = {measure} if isinstance(measure, str) else set(measure)
        self.duplicates = self.settings.get("duplicates", "best")
        if self.duplicates not in ("best", "mean"):
            raise ValueError(f"{self!r}: duplicates must be 'best' or 'mean', not {self.duplicates!r}")
        self.references_out = bool(self.settings.get("references_out", True))
        self.precision = int(self.settings.get("precision", 2))

    def _process(self, input):
        regions = [r for r in (input.value if isinstance(input, Result) else input) or [] if isinstance(r, Result)]
        boxes = [(region, region_label(region), region_box(region)) for region in regions]
        scale, calibrated, readings = self.calibrate(boxes)

        out = ListResult([], input=input, processor=self)
        for region, label, box in boxes:
            is_reference = label in self.references
            if is_reference and not self.references_out:
                continue
            if not is_reference and self.measure and label not in self.measure:
                continue
            # A YOLO crop's value is its image (or, uncropped, a file that was never written): the row
            # wants the label, the box and the detector's confidence, not the pixels
            value = dict(region.value) if region_data(region) else {"label": label}
            if box is not None:
                value.setdefault("bbox", [round(v, 1) for v in box])
            if region.metadata.get("confidence") is not None:
                value.setdefault("confidence", round(float(region.metadata["confidence"]), 3))
            metadata = {"type": "DATA", "label": label, "reference": is_reference}
            if box is None:
                metadata["measured"] = False
                out.append(self.measurement(value, input, region, metadata))
                continue
            width_px, height_px = box[2] - box[0], box[3] - box[1]
            value.update(width_px=round(width_px, 1), height_px=round(height_px, 1))
            if is_reference:
                value["known_size"] = dict(self.references[label], unit=self.unit)
                metadata["readings"] = [reading for reading in readings if reading["region"] == region.id]
                metadata["reference_used"] = bool(metadata["readings"])
            sx = scale["x"] or scale["y"]
            sy = scale["y"] or scale["x"]
            if sx and sy:
                value.update(
                    width=round(width_px / sx, self.precision),
                    height=round(height_px / sy, self.precision),
                    unit=self.unit,
                )
                metadata.update(
                    measured=True,
                    scale={"x": sx, "y": sy},
                    calibrated={
                        "width": calibrated["x"] or calibrated["y"],
                        "height": calibrated["y"] or calibrated["x"],
                    },
                )
                if not (scale["x"] and scale["y"]):
                    metadata["single_axis"] = "x" if scale["x"] else "y"
            else:
                metadata["measured"] = False
            out.append(self.measurement(value, input, region, metadata))
        out.metadata.update(scale=scale, calibrated=calibrated, unit=self.unit)
        return out

    def measurement(self, value, input, region, metadata):
        """The result for *region*: made from the list it came in, as a segmenter's regions are, and
        located where the region is -- an ``Iterator`` over the output re-parents each entry, so the
        box has to travel on the result itself."""
        result = ItemResult(value, input=input, processor=self, metadata=metadata)
        result.locator = region.locator
        result.sequence = region.sequence
        return result

    def calibrate(self, boxes):
        """``(scale, calibrated, readings)`` from the reference regions among *boxes*.

        *scale* is pixels per unit along ``x`` and ``y`` (``None`` for an axis no reference measures);
        *calibrated* the labels of the references behind each; *readings* one entry per reference
        dimension read, so a reference that disagrees with the others can be found.
        """
        used = None
        if self.duplicates == "best":
            best = {}
            for region, label, box in boxes:
                if label in self.references and box is not None:
                    if label not in best or region_confidence(region) > region_confidence(best[label]):
                        best[label] = region
            used = {region.id for region in best.values()}
        readings = []
        for region, label, box in boxes:
            size = self.references.get(label)
            if size is None or box is None or (used is not None and region.id not in used):
                continue
            width_px, height_px = box[2] - box[0], box[3] - box[1]
            if "length" in size:
                axis = "x" if width_px >= height_px else "y"
                pixels = width_px if axis == "x" else height_px
                readings.append(dict(region=region.id, label=label, axis=axis, pixels=pixels, size=size["length"]))
            if "width" in size:
                readings.append(dict(region=region.id, label=label, axis="x", pixels=width_px, size=size["width"]))
            if "height" in size:
                readings.append(dict(region=region.id, label=label, axis="y", pixels=height_px, size=size["height"]))
        for reading in readings:
            reading["scale"] = reading["pixels"] / reading["size"]
        scale, calibrated = {}, {}
        for axis in ("x", "y"):
            along = [r for r in readings if r["axis"] == axis]
            scale[axis] = sum(r["scale"] for r in along) / len(along) if along else None
            calibrated[axis] = sorted({r["label"] for r in along})
        return scale, calibrated, readings


def region_data(region):
    """Whether *region*'s value is a dict of what was found -- never read from a file to find out."""
    return not isinstance(region, FileItemResult) and isinstance(region.value, dict)


def region_confidence(region):
    """The detector's confidence in *region* (``-1`` when it gave none, so any stated one wins)."""
    confidence = region.metadata.get("confidence")
    if confidence is None and region_data(region):
        confidence = region.value.get("confidence")
    return float(confidence) if isinstance(confidence, (int, float)) else -1.0


def region_label(region):
    """The label a detector gave *region*: its value's ``label``, else its metadata's."""
    if region_data(region) and region.value.get("label") is not None:
        return str(region.value["label"])
    for key in ("label", "yolo_class"):
        if region.metadata.get(key) is not None:
            return str(region.metadata[key])
    return None


def region_box(region):
    """*region*'s ``[x1, y1, x2, y2]`` in its image's pixels, from its locator, or ``None``."""
    locator = region.locator
    box = locator.get("bbox") if isinstance(locator, dict) else None
    if isinstance(box, (list, tuple)) and len(box) == 4:
        return [float(v) for v in box]
    return None
