"""
Utility helpers — PDF validation, text cleaning, file management.
"""

import hashlib
import re
from pathlib import Path

from config.settings import ALLOWED_EXTENSIONS, MAX_UPLOAD_SIZE_MB


def validate_pdf(file_bytes: bytes, filename: str) -> tuple[bool, str]:
    """Validate an uploaded PDF file.

    Returns:
        (is_valid, error_message)
    """
    # Check extension
    ext = Path(filename).suffix.lower()
    if ext not in ALLOWED_EXTENSIONS:
        return False, f"Invalid file type '{ext}'. Only PDF files are allowed."

    # Check magic bytes (PDF header)
    if not file_bytes[:5] == b"%PDF-":
        return False, "The file does not appear to be a valid PDF."

    # Check size
    size_mb = len(file_bytes) / (1024 * 1024)
    if size_mb > MAX_UPLOAD_SIZE_MB:
        return False, f"File size ({size_mb:.1f} MB) exceeds the {MAX_UPLOAD_SIZE_MB} MB limit."

    return True, ""


def clean_text(text: str) -> str:
    """Normalize extracted text — collapse whitespace, strip control chars."""
    text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f]", "", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def file_hash(file_bytes: bytes) -> str:
    """Return a SHA-256 hex digest for deduplication."""
    return hashlib.sha256(file_bytes).hexdigest()
