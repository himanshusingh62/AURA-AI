"""
Utility helpers for AURA.AI — PDF & Image validation, text cleaning, file hashing,
base64 conversion, path security, and error message / API key sanitization for enterprise security.
"""

from __future__ import annotations

import base64
import hashlib
import io
import re
from pathlib import Path
from PIL import Image

from config.settings import (
    ALLOWED_IMAGE_EXTENSIONS,
    ALLOWED_PDF_EXTENSIONS,
    MAX_IMAGE_SIZE_MB,
    MAX_UPLOAD_SIZE_MB,
)


def safe_filename(filename: str) -> str:
    """Sanitize and return only the base filename to prevent path traversal attacks.

    Strips directory separators, relative path markers (../), and control characters.
    Ensures identical behavior across Windows and POSIX/Linux platforms.
    """
    if not filename:
        return "unnamed_file"
    # Normalize backslashes to forward slashes for cross-platform POSIX/Windows consistency
    normalized = filename.replace("\\", "/")
    base = Path(normalized).name
    # Strip any remaining directory separators
    base = base.replace("/", "_").replace("\\", "_")
    # Remove null bytes and control chars
    base = re.sub(r"[\x00-\x1f\x7f]", "", base)
    # Avoid hidden files or current directory references
    base = base.lstrip(".")
    return base if base else "unnamed_file"


def sanitize_error_message(error: Exception | str) -> str:
    """Sanitize error messages and exception strings to prevent leaking sensitive API keys or tokens.

    Replaces:
    - Google API keys (e.g. AIzaSy... 39 characters)
    - URL query parameters (key=..., api_key=..., token=...)
    - Bearer authorization headers
    - Server internal absolute filesystem paths
    """
    if error is None:
        return ""
    text = str(error)

    # 1. Redact Google API keys (starts with AIza or AQ followed by alphanumeric/_/-/. chars)
    text = re.sub(r"(AIza[0-9A-Za-z_\-]{35}|AQ[0-9A-Za-z_\-\.]{25,70})", "[REDACTED_API_KEY]", text)

    # 2. Redact URL query parameter / key-value credentials (?key=... or &key=...)
    text = re.sub(
        r"(?i)(key|api_key|token|access_token|auth|secret)=([A-Za-z0-9_\-\.]+)",
        r"\1=[REDACTED]",
        text,
    )

    # 3. Redact Authorization Bearer tokens
    text = re.sub(r"(?i)Bearer\s+[A-Za-z0-9_\-\.]+", "Bearer [REDACTED]", text)

    # 4. Redact potential Windows/POSIX absolute home paths to protect user privacy
    text = re.sub(r"[A-Za-z]:\\(?:Users|home)\\[^\s\"\'\`\)]+", "[REDACTED_PATH]", text)
    text = re.sub(r"/(?:home|Users)/[^\s\"\'\`\)]+", "[REDACTED_PATH]", text)

    return text


def mask_api_key(api_key: str) -> str:
    """Securely mask an API key for safe UI display (e.g. AIza••••••••1234).

    Avoids exposing excessive key characters in logs or frontend text.
    """
    if not api_key:
        return ""
    cleaned = api_key.strip()
    if len(cleaned) <= 8:
        return "••••••••"
    return f"{cleaned[:4]}••••••••{cleaned[-4:]}"


def is_pdf(filename: str) -> bool:
    """Return True if the filename has a supported PDF extension."""
    return Path(filename).suffix.lower() in ALLOWED_PDF_EXTENSIONS


def is_image(filename: str) -> bool:
    """Return True if the filename has a supported image extension."""
    return Path(filename).suffix.lower() in ALLOWED_IMAGE_EXTENSIONS


def validate_pdf(file_bytes: bytes, filename: str) -> tuple[bool, str]:
    """Validate an uploaded PDF file for type, header signature, and size.

    Returns:
        (is_valid, error_message)
    """
    if not file_bytes:
        return False, "The uploaded file is empty (0 bytes)."

    # Check extension
    clean_name = safe_filename(filename)
    ext = Path(clean_name).suffix.lower()
    if ext not in ALLOWED_PDF_EXTENSIONS:
        return False, f"Invalid file type '{ext}'. Supported PDF extensions: {ALLOWED_PDF_EXTENSIONS}"

    # Check magic bytes (PDF header: %PDF-)
    if not file_bytes.startswith(b"%PDF-"):
        return False, "The file does not appear to be a valid PDF (missing %PDF- header signature)."

    # Check size limit
    size_mb = len(file_bytes) / (1024 * 1024)
    if size_mb > MAX_UPLOAD_SIZE_MB:
        return False, f"File size ({size_mb:.1f} MB) exceeds the maximum allowed {MAX_UPLOAD_SIZE_MB} MB limit."

    return True, ""


def validate_image(file_bytes: bytes, filename: str) -> tuple[bool, str]:
    """Validate an uploaded image file (.png, .jpg, .jpeg, .webp).

    Checks extension, magic bytes header, file size, and PIL image format integrity.

    Returns:
        (is_valid, error_message)
    """
    if not file_bytes:
        return False, "The uploaded image is empty (0 bytes)."

    clean_name = safe_filename(filename)
    ext = Path(clean_name).suffix.lower()
    if ext not in ALLOWED_IMAGE_EXTENSIONS:
        return (
            False,
            f"Unsupported image format '{ext}'. Supported formats: {', '.join(ALLOWED_IMAGE_EXTENSIONS)}",
        )

    # Check file size
    size_mb = len(file_bytes) / (1024 * 1024)
    if size_mb > MAX_IMAGE_SIZE_MB:
        return False, f"Image size ({size_mb:.1f} MB) exceeds the maximum allowed {MAX_IMAGE_SIZE_MB} MB limit."

    # Verify Magic Bytes signatures
    if ext == ".png" and not file_bytes.startswith(b"\x89PNG\r\n\x1a\n"):
        return False, "Corrupted or spoofed PNG image (invalid magic bytes header)."
    elif ext in (".jpg", ".jpeg") and not file_bytes.startswith(b"\xff\xd8"):
        return False, "Corrupted or spoofed JPEG image (invalid magic bytes header)."
    elif ext == ".webp" and not (file_bytes[:4] == b"RIFF" and file_bytes[8:12] == b"WEBP"):
        return False, "Corrupted or spoofed WEBP image (invalid RIFF/WEBP header signature)."

    # Deep verification with Pillow
    try:
        with Image.open(io.BytesIO(file_bytes)) as img:
            img.verify()
    except Exception as exc:
        return False, f"Invalid or corrupted image file structure: {exc}"

    return True, ""


def image_to_base64_thumbnail(file_bytes: bytes, max_size: tuple[int, int] = (260, 260)) -> str:
    """Generate a sanitized base64-encoded thumbnail for lightweight UI display."""
    try:
        with Image.open(io.BytesIO(file_bytes)) as img:
            thumb = img.copy()
            thumb.thumbnail(max_size, Image.Resampling.LANCZOS)
            buffered = io.BytesIO()
            if thumb.mode in ("RGBA", "LA") or (thumb.mode == "P" and "transparency" in thumb.info):
                thumb.save(buffered, format="PNG")
                mime = "image/png"
            else:
                thumb = thumb.convert("RGB")
                thumb.save(buffered, format="JPEG", quality=85)
                mime = "image/jpeg"
            encoded = base64.b64encode(buffered.getvalue()).decode("utf-8")
            return f"data:{mime};base64,{encoded}"
    except Exception:
        return ""


def clean_text(text: str) -> str:
    """Normalize extracted text — collapse excessive whitespace, strip control chars."""
    text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f]", "", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def file_hash(file_bytes: bytes) -> str:
    """Return a SHA-256 hex digest for file deduplication."""
    return hashlib.sha256(file_bytes).hexdigest()
