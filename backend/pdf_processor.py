"""
PDF processing module for AURA.AI — multi-page text extraction with PyMuPDF.
Includes scanned PDF detection, image count inspection, and path traversal protection.
"""

from __future__ import annotations

from pathlib import Path
from typing import NamedTuple

try:
    import pymupdf as fitz
except ImportError:
    import fitz

from llama_index.core.schema import Document
from utils.helpers import clean_text


class PDFProcessingError(ValueError):
    """Base exception for PDF processing errors."""
    pass


class EmptyPDFError(PDFProcessingError):
    """Raised when PDF file is empty or contains zero pages."""
    pass


class CorruptedPDFError(PDFProcessingError):
    """Raised when PDF byte stream cannot be parsed by PyMuPDF."""
    pass


class ScannedPDFError(PDFProcessingError):
    """Raised when PDF consists of scanned raster images without a digital text layer."""
    pass


class PasswordProtectedPDFError(PDFProcessingError):
    """Raised when PDF file is encrypted or password-protected."""
    pass


class PDFExtractionMetadata(NamedTuple):
    """Metadata summary of the extracted PDF document."""
    total_pages: int
    text_pages: int
    char_count: int
    image_count: int
    is_scanned: bool


def inspect_and_extract_pdf(
    pdf_bytes: bytes,
    source_name: str = "uploaded.pdf",
    file_id: str | None = None,
) -> tuple[list[Document], PDFExtractionMetadata]:
    """Parse a PDF from bytes, extract text page-by-page, and detect scanned documents.

    Security: Sanitizes source_name to prevent path traversal attacks.

    Raises:
        EmptyPDFError: if the file is 0 bytes or has 0 pages.
        CorruptedPDFError: if the PDF format is invalid or corrupted.
        ScannedPDFError: if the PDF contains images without extractable digital text.
    """
    if not pdf_bytes or len(pdf_bytes) == 0:
        raise EmptyPDFError("The uploaded PDF file is empty (0 bytes).")

    # Secure against directory traversal: retain basename only
    safe_source_name = Path(source_name).name or "uploaded.pdf"
    effective_file_id = file_id or safe_source_name

    try:
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    except Exception as exc:
        raise CorruptedPDFError(f"Corrupted or invalid PDF file: {exc}") from exc

    if getattr(doc, "is_encrypted", False) or getattr(doc, "needs_pass", False):
        doc.close()
        raise PasswordProtectedPDFError(
            f"The PDF '{safe_source_name}' is password-protected or encrypted. "
            "Please remove the password protection before uploading."
        )

    total_pages = len(doc)
    if total_pages == 0:
        doc.close()
        raise EmptyPDFError("The PDF document contains 0 pages.")

    documents: list[Document] = []
    total_chars = 0
    total_images = 0

    for page_num in range(total_pages):
        page = doc.load_page(page_num)
        raw_text = page.get_text("text")
        text = clean_text(raw_text)

        images = page.get_images()
        total_images += len(images)

        if text:
            total_chars += len(text)
            documents.append(
                Document(
                    text=text,
                    metadata={
                        "page_number": page_num + 1,
                        "source": safe_source_name,
                        "file_id": effective_file_id,
                        "total_pages": total_pages,
                    },
                )
            )

    doc.close()

    # Detect scanned documents (e.g. pages have images or zero extractable text)
    if not documents or total_chars < 30:
        if total_images > 0:
            raise ScannedPDFError(
                f"Scanned PDF detected in '{safe_source_name}' ({total_pages} pages, {total_images} images, but no digital text layer). "
                "Please upload a text-searchable PDF or export it with OCR enabled."
            )
        raise EmptyPDFError(
            f"The PDF '{safe_source_name}' contains no extractable text or characters."
        )

    meta = PDFExtractionMetadata(
        total_pages=total_pages,
        text_pages=len(documents),
        char_count=total_chars,
        image_count=total_images,
        is_scanned=False,
    )

    return documents, meta


def extract_documents_from_pdf(
    pdf_bytes: bytes,
    source_name: str = "uploaded.pdf",
    file_id: str | None = None,
) -> list[Document]:
    """Backward-compatible entry point returning list of LlamaIndex Document objects."""
    docs, _ = inspect_and_extract_pdf(pdf_bytes, source_name=source_name, file_id=file_id)
    return docs
