from PIL import ImageOps


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
