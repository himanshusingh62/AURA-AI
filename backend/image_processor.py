"""
Image processing module for AURA.AI Multimodal AI Research Assistant.
Handles image loading, visual inspection, thumbnail generation, security checks, and metadata extraction.
"""

from __future__ import annotations

import io
from dataclasses import dataclass
from pathlib import Path
from PIL import Image

from utils.helpers import image_to_base64_thumbnail, validate_image


class ImageProcessingError(Exception):
    """Base exception for image processing and validation errors."""
    pass


@dataclass
class ProcessedImage:
    """Structured representation of an uploaded and processed image."""

    filename: str
    format: str
    width: int
    height: int
    mode: str
    size_mb: float
    thumbnail_base64: str
    raw_bytes: bytes
    pil_image: Image.Image


# 40 Megapixels maximum to prevent decompression bomb denial-of-service
MAX_IMAGE_PIXELS: int = 40_000_000


def process_uploaded_image(file_bytes: bytes, filename: str) -> ProcessedImage:
    """Validate, inspect and extract metadata from an uploaded image.

    Security: Sanitizes filename to protect against path traversal and checks pixel bounds.

    Raises:
        ImageProcessingError / ValueError: if validation or loading fails.
    """
    safe_filename = Path(filename).name or "image.png"

    is_valid, error_msg = validate_image(file_bytes, safe_filename)
    if not is_valid:
        raise ValueError(error_msg)

    try:
        # Open with PIL for metadata and conversion
        img = Image.open(io.BytesIO(file_bytes))
        
        # Verify resolution bounds against decompression bombs
        if img.width * img.height > MAX_IMAGE_PIXELS:
            raise ValueError(
                f"Image resolution ({img.width}x{img.height} px) exceeds maximum allowed limit ({MAX_IMAGE_PIXELS:,} pixels)."
            )

        # Ensure image data is loaded safely
        img.load()

        # Convert palette/grayscale/CMYK images if needed for standard display & Gemini vision
        if img.mode not in ("RGB", "RGBA"):
            display_img = img.convert("RGB")
        else:
            display_img = img.copy()

        fmt = img.format or Path(safe_filename).suffix.upper().replace(".", "")
        size_mb = round(len(file_bytes) / (1024 * 1024), 2)
        thumb_b64 = image_to_base64_thumbnail(file_bytes, max_size=(280, 280))

        return ProcessedImage(
            filename=safe_filename,
            format=fmt,
            width=img.width,
            height=img.height,
            mode=img.mode,
            size_mb=size_mb,
            thumbnail_base64=thumb_b64,
            raw_bytes=file_bytes,
            pil_image=display_img,
        )
    except Exception as exc:
        if isinstance(exc, ValueError):
            raise
        raise ImageProcessingError(f"Failed to process image '{safe_filename}': {exc}") from exc
