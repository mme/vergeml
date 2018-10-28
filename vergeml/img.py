import os.path
from PIL import Image
from PIL.Image import Image as ImageType


INPUT_PATTERNS = ["**/*.jpg", "**/*.jpeg", "**/*.png", "**/*.bmp"]


def fixext(path, img):
    """Change the format of files with the wrong extension."""
    path, ext = os.path.splitext(path)

    if img.format:
        return path + "." + img.format.lower()
    elif img.mode == 'RGBA':
        return path + ".png"
    elif ext.lower() not in [".jpg", ".jpeg", ".png", ".bmp"]:
        return path + ".png"
    else:
        return path + ext


def open_image(path):
    """Open image at path.

    PIL lazily opens the image, which can lead to a 'too many open files' error.
    This workaround reads the file into memory immediately."""
    img1 = Image.open(path)
    img2 = img1.copy()
    img1.close()
    return img2
