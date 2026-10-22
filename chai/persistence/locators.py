"""Locator frames: how a stored row says where in its input it sits, for a UI to draw.

A component that carves a piece out of something records where the piece is (``Result.locator``)
rather than rendering it, and ``input_locator`` stacks the frames between a result and its recorded
source into the RESULT row's ``input_segment``, outermost first. What a front end makes of them --
a box over an image, a highlight in a text -- is up to the front end; the engine stores only the
data. These are the frames the engine's own components write:

* **bbox** -- ``{"bbox": [x1, y1, x2, y2]}``: a region of an image, in that image's pixels, so an
  overlay drawn in the image's own coordinates (an SVG ``viewBox`` of its natural size) lines up
  however the image is scaled. Written by ``YoloSegmenter``.
* **text** -- ``{"start": 120, "end": 132}``: a character range, ``text[start:end]``. Two optional
  keys say more: ``result``, the id of the result whose (text) value the offsets index, when that
  is not a file's content -- a transcription of a page image, say, whose row's ``input`` is the
  image rather than the text; and ``exact``, the characters themselves, when the row's value is not
  them (an extracted entity), so a UI can tell when the text has since been corrected and the
  offsets no longer fit. Written by ``TextSegmenter`` (``locate: true``) and ``TextSpanLocator``.
* **index** -- ``{"index": 7}``, ``{"page": 3}``: a position in a sequence.

The ``result`` key is how text frames are tied to the text they index for now; it may change with
the way storage and the segmenters come to handle images and texts. These map onto W3C Web
Annotation selectors (``FragmentSelector`` ``xywh=``, ``TextPositionSelector``,
``TextQuoteSelector``) should rows ever need exporting as annotations.
"""

import ujson as json

from ..result import FileItemResult, Result


def bbox_frame(x1, y1, x2, y2):
    """A region of an image, in its pixels: ``{"bbox": [x1, y1, x2, y2]}``."""
    return {"bbox": [float(x1), float(y1), float(x2), float(y2)]}


def text_frame(start, end, result=None, exact=None):
    """A character range ``[start, end)`` of a text, and optionally the result it indexes and its text.

    *result* is a ``Result`` (or its id) whose value is the text; it is left out when the text is a
    file's content (a ``FileItemResult``) or the raw workflow input, which the row's ``input`` names.
    """
    frame = {"start": int(start), "end": int(end)}
    result_id = text_result_id(result)
    if result_id:
        frame["result"] = result_id
    if exact is not None:
        frame["exact"] = exact
    return frame


def text_result_id(result):
    """The id a text frame records for *result*: ``None`` for a file's content or a raw input."""
    if isinstance(result, FileItemResult) or result is None:
        return None
    if isinstance(result, Result):
        return result.id
    return str(result)


def frame_kind(frame):
    """``"bbox"``, ``"text"`` or ``"index"`` for a frame of one of those kinds, else ``None``."""
    if not isinstance(frame, dict):
        return None
    if isinstance(frame.get("bbox"), (list, tuple)) and len(frame["bbox"]) == 4:
        return "bbox"
    if isinstance(frame.get("start"), int) and isinstance(frame.get("end"), int):
        return "text"
    if any(isinstance(frame.get(key), int) for key in ("index", "page")):
        return "index"
    return None


def segment_frames(input_segment):
    """The frames of a RESULT row's ``input_segment`` (JSON text, or already parsed); ``[]`` if none."""
    if input_segment is None or input_segment == "":
        return []
    if isinstance(input_segment, str):
        try:
            input_segment = json.loads(input_segment)
        except ValueError:
            return []
    if isinstance(input_segment, dict):
        return [input_segment]
    return [frame for frame in input_segment if isinstance(frame, dict)] if isinstance(input_segment, list) else []
