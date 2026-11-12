"""
Unit tests for Image processing, validation, thumbnail generation, and decompression bomb protection.
"""

from __future__ import annotations

import io
import pytest
from PIL import Image

from backend.image_processor import (
    MAX_IMAGE_PIXELS,
    process_uploaded_image,
)
from utils.helpers import is_image, is_pdf, validate_image


def _create_test_image(fmt: str = "PNG", size: tuple[int, int] = (120, 120), color: str = "blue") -> bytes:
    img = Image.new("RGB", size, color=color)
    buf = io.BytesIO()
    img.save(buf, format=fmt)
    return buf.getvalue()


def test_is_image_and_is_pdf():
    assert is_image("photo.png") is True
    assert is_image("chart.jpg") is True
    assert is_image("diagram.jpeg") is True
    assert is_image("banner.webp") is True
    assert is_image("report.pdf") is False

    assert is_pdf("report.pdf") is True
    assert is_pdf("photo.png") is False


def test_process_uploaded_image_valid_png():
    raw_png = _create_test_image(fmt="PNG", size=(150, 100), color="green")
    proc = process_uploaded_image(raw_png, "sample.png")

    assert proc.filename == "sample.png"
    assert proc.width == 150
    assert proc.height == 100
    assert proc.format == "PNG"
    assert proc.thumbnail_base64.startswith("data:image/")


def test_process_uploaded_image_valid_jpeg():
    raw_jpg = _create_test_image(fmt="JPEG", size=(80, 80), color="red")
    proc = process_uploaded_image(raw_jpg, "test.jpg")

    assert proc.filename == "test.jpg"
    assert proc.width == 80
    assert proc.height == 80
    assert proc.format == "JPEG"


def test_validate_image_corrupted_header():
    is_valid, err = validate_image(b"CORRUPTED_HEADER_DATA", "corrupted.png")
    assert is_valid is False
    assert "Corrupted or spoofed PNG" in err


def test_validate_image_oversized_pixel_bounds():
    # Attempt an image exceeding MAX_IMAGE_PIXELS
    raw_png = _create_test_image(fmt="PNG", size=(50, 50))
    # Test function with mocked image bounds
    with pytest.raises(ValueError) as exc_info:
        # Pass a mock image with width * height > MAX_IMAGE_PIXELS
        from unittest.mock import patch
        with patch("PIL.Image.open") as mock_open:
            mock_img = Image.new("RGB", (10000, 10000))  # 100M pixels > 40M limit
            mock_open.return_value = mock_img
            process_uploaded_image(raw_png, "giant.png")
    assert "exceeds maximum allowed limit" in str(exc_info.value)
