import os
from PIL import Image, ImageOps, ImageFilter, ImageEnhance

def enhance_image(input_path: str, output_path: str) -> None:
    """
    Enhance an input image and save the result to output_path.
    Steps: autocontrast, smoothing, unsharp mask, slight contrast & sharpness boost.
    """
    img = Image.open(input_path).convert("RGB")
    # Improve contrast and reduce dullness
    img = ImageOps.autocontrast(img, cutoff=1)
    # Light smoothing to reduce minor noise
    img = img.filter(ImageFilter.SMOOTH)
    # Sharpen edges to highlight lesion boundaries
    img = img.filter(ImageFilter.UnsharpMask(radius=1.5, percent=150, threshold=3))
    # Subtle global adjustments
    img = ImageEnhance.Contrast(img).enhance(1.15)
    img = ImageEnhance.Sharpness(img).enhance(1.2)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    img.save(output_path, format="JPEG", quality=92)