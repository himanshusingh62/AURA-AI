"""Tests for password-protected and encrypted PDF handling."""

from __future__ import annotations

from unittest.mock import MagicMock, patch
import pytest

from backend.pdf_processor import (
    PDFProcessingError,
    PasswordProtectedPDFError,
    inspect_and_extract_pdf,
)


def test_password_protected_pdf_error_hierarchy():
    """Verify PasswordProtectedPDFError inherits from PDFProcessingError."""
    assert issubclass(PasswordProtectedPDFError, PDFProcessingError)


def test_encrypted_pdf_raises_password_protected_error():
    """Verify inspect_and_extract_pdf raises PasswordProtectedPDFError when doc is encrypted."""
    mock_doc = MagicMock()
    mock_doc.is_encrypted = True
    mock_doc.needs_pass = True

    dummy_pdf_bytes = b"%PDF-1.4\n%fake-encrypted-content"

    with patch("backend.pdf_processor.fitz.open", return_value=mock_doc):
        with pytest.raises(PasswordProtectedPDFError) as exc_info:
            inspect_and_extract_pdf(dummy_pdf_bytes, source_name="secret_report.pdf")

    assert "secret_report.pdf" in str(exc_info.value)
    assert "password-protected or encrypted" in str(exc_info.value)
    mock_doc.close.assert_called_once()


def test_unencrypted_normal_pdf_does_not_raise_password_error():
    """Verify unencrypted PDF proceeds normally without password error."""
    mock_page = MagicMock()
    mock_page.get_text.return_value = "Normal text content extracted from unencrypted document."
    mock_page.get_images.return_value = []

    mock_doc = MagicMock()
    mock_doc.is_encrypted = False
    mock_doc.needs_pass = False
    mock_doc.__len__.return_value = 1
    mock_doc.load_page.return_value = mock_page

    dummy_pdf_bytes = b"%PDF-1.4\n%normal-content"

    with patch("backend.pdf_processor.fitz.open", return_value=mock_doc):
        docs, meta = inspect_and_extract_pdf(dummy_pdf_bytes, source_name="public_doc.pdf")

    assert len(docs) == 1
    assert "Normal text" in docs[0].text
    assert meta.total_pages == 1
