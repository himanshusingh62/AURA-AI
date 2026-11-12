"""
Unit tests for PyMuPDF PDF extraction, validation, and scanned document detection.
"""

from __future__ import annotations

import io
import pytest
import pymupdf

from backend.pdf_processor import (
    CorruptedPDFError,
    EmptyPDFError,
    ScannedPDFError,
    extract_documents_from_pdf,
    inspect_and_extract_pdf,
)
from utils.helpers import validate_pdf


def _create_sample_pdf(pages_text: list[str]) -> bytes:
    doc = pymupdf.open()
    for text in pages_text:
        page = doc.new_page()
        page.insert_text((50, 72), text)
    pdf_bytes = doc.tobytes()
    doc.close()
    return pdf_bytes


def test_validate_pdf_valid_header():
    pdf_bytes = _create_sample_pdf(["Page 1 Content"])
    is_valid, err = validate_pdf(pdf_bytes, "report.pdf")
    assert is_valid is True
    assert err == ""


def test_validate_pdf_invalid_magic_bytes():
    is_valid, err = validate_pdf(b"NOT_A_PDF_STREAM", "report.pdf")
    assert is_valid is False
    assert "missing %PDF-" in err


def test_validate_pdf_invalid_extension():
    pdf_bytes = _create_sample_pdf(["Page 1 Content"])
    is_valid, err = validate_pdf(pdf_bytes, "report.docx")
    assert is_valid is False
    assert "Invalid file type" in err


def test_extract_documents_multi_page():
    pages = [
        "First section covering deep learning and transformers.",
        "Second section covering multimodal vision embeddings.",
        "Third section covering benchmark evaluation results.",
    ]
    pdf_bytes = _create_sample_pdf(pages)
    docs, meta = inspect_and_extract_pdf(pdf_bytes, source_name="research_paper.pdf")

    assert len(docs) == 3
    assert meta.total_pages == 3
    assert meta.text_pages == 3
    assert meta.char_count > 100
    assert meta.is_scanned is False

    for idx, doc in enumerate(docs, start=1):
        assert doc.metadata["page_number"] == idx
        assert doc.metadata["total_pages"] == 3
        assert doc.metadata["source"] == "research_paper.pdf"


def test_extract_documents_empty_bytes():
    with pytest.raises(EmptyPDFError):
        inspect_and_extract_pdf(b"", "empty.pdf")


def test_extract_documents_corrupted_data():
    with pytest.raises(CorruptedPDFError):
        inspect_and_extract_pdf(b"%PDF-1.4 CORRUPTED DATA WITHOUT VALID XREF", "broken.pdf")


def test_extract_documents_blank_page():
    doc = pymupdf.open()
    doc.new_page()  # totally blank page without text or images
    blank_bytes = doc.tobytes()
    doc.close()

    with pytest.raises(EmptyPDFError):
        inspect_and_extract_pdf(blank_bytes, "blank.pdf")


def test_extract_documents_scanned_pdf_detected():
    # Create a PDF with an embedded raster image but no text
    from PIL import Image
    doc = pymupdf.open()
    page = doc.new_page()
    img = Image.new("RGB", (100, 100), color=(0, 128, 255))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    page.insert_image(pymupdf.Rect(50, 50, 150, 150), stream=buf.getvalue())
    scanned_bytes = doc.tobytes()
    doc.close()

    with pytest.raises(ScannedPDFError) as exc_info:
        inspect_and_extract_pdf(scanned_bytes, "scan.pdf")
    assert "Scanned PDF detected" in str(exc_info.value)
