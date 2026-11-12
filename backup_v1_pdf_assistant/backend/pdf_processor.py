"""
PDF processing — extract text page-by-page using PyMuPDF.

Produces a list of LlamaIndex Document objects, one per page,
with page-number metadata attached.
"""

from __future__ import annotations

try:
    import pymupdf as fitz
except ImportError:
    import fitz

from llama_index.core.schema import Document
from utils.helpers import clean_text


def extract_documents_from_pdf(
    pdf_bytes: bytes,
    source_name: str = "uploaded.pdf",
) -> list[Document]:
    """Parse a PDF from raw bytes and return one Document per page.

    Each Document carries metadata:
        - page_number  (1-based)
        - source       (filename)
        - total_pages

    Raises:
        ValueError: if the PDF has zero extractable text or is empty.
    """
    if not pdf_bytes:
        raise ValueError("The provided PDF file is empty (0 bytes).")

    try:
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    except Exception as exc:
        raise ValueError(f"Failed to parse PDF document: {exc}") from exc

    documents: list[Document] = []
    total_pages = len(doc)

    if total_pages == 0:
        doc.close()
        raise ValueError("The PDF contains no pages.")

    for page_num in range(total_pages):
        page = doc.load_page(page_num)
        raw_text = page.get_text("text")
        text = clean_text(raw_text)

        if not text:
            continue  # skip blank pages

        documents.append(
            Document(
                text=text,
                metadata={
                    "page_number": page_num + 1,
                    "source": source_name,
                    "total_pages": total_pages,
                },
            )
        )

    doc.close()

    if not documents:
        raise ValueError(
            "The PDF contains no extractable text. "
            "It may be a scanned image or contain non-searchable content."
        )

    return documents
