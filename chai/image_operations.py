from PIL import Image, ImageOps


def exif_rotate(img):
    img = ImageOps.exif_transpose(img)
    return img


def crop(img, crop_region):
    # Convert normalized coords to pixel coords
    w, h = img.size
    left = int(crop_region["x"] * w)
    top = int(crop_region["y"] * h)
    right = int((crop_region["x"] + crop_region["width"]) * w)
    bottom = int((crop_region["y"] + crop_region["height"]) * h)
    img = img.crop((left, top, right, bottom))
    return img


def scale(img, long_edge):
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
