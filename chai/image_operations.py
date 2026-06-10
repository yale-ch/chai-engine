"""Small PIL-based image helpers shared by the AI backends and annotators.

Conversions between bytes and ``PIL.Image``, EXIF-aware rotation, normalized-coordinate cropping and
long-edge downscaling (used to shrink images before sending them to vision models).
"""

from io import BytesIO

from PIL import Image, ImageOps


def image_from_bytes(data):
    """Open raw image *data* (bytes) as a ``PIL.Image``."""
    return Image.open(BytesIO(data))


def bytes_from_image(img):
    """Serialize a ``PIL.Image`` to PNG bytes."""
    buf = BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def exif_rotate(img):
    """Apply the image's EXIF orientation so pixels match how the photo was taken."""
    img = ImageOps.exif_transpose(img)
    return img


def crop(img, crop_region):
    """Crop *img* to *crop_region*, a dict of normalized (0-1) x/y/width/height values."""
    # Convert normalized coords to pixel coords
    w, h = img.size
    left = int(crop_region["x"] * w)
    top = int(crop_region["y"] * h)
    right = int((crop_region["x"] + crop_region["width"]) * w)
    bottom = int((crop_region["y"] + crop_region["height"]) * h)
    img = img.crop((left, top, right, bottom))
    return img


def scale(img, long_edge):
    """Downscale *img* so its longest edge is at most *long_edge* pixels (no-op if already smaller)."""
    width, height = img.size

    # If image is already smaller than max_size, return as is
    if width <= long_edge and height <= long_edge:
        return img

    # Calculate new dimensions maintaining aspect ratio
    if width > height:
        new_width = long_edge
        new_height = int((height * long_edge) / width)
    else:
        new_height = long_edge
        new_width = int((width * long_edge) / height)

    return img.resize((new_width, new_height), Image.Resampling.LANCZOS)
