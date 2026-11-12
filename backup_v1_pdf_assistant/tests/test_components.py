"""
Automated unit and integration test suite for the AI PDF Research Assistant.
"""

from __future__ import annotations

import io
import os
import unittest
import pymupdf

from backend.pdf_processor import extract_documents_from_pdf
from backend.query_engine import QueryResult
from config.settings import (
    ALLOWED_EXTENSIONS,
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    GEMINI_CHAT_MODEL,
    GEMINI_EMBEDDING_MODEL,
    MAX_UPLOAD_SIZE_MB,
    TOP_K_RESULTS,
    get_google_api_key,
)
from utils.helpers import clean_text, file_hash, validate_pdf


class TestUtils(unittest.TestCase):
    """Test helper functions in utils/helpers.py."""

    def test_clean_text_normalizes_whitespace(self):
        raw = "Hello   world!\n\n\n\nThis is   a   test.\t\t"
        cleaned = clean_text(raw)
        self.assertEqual(cleaned, "Hello world!\n\nThis is a test.")

    def test_clean_text_removes_control_characters(self):
        raw = "Text with \x00 null \x07 bell \x0b vertical tab."
        cleaned = clean_text(raw)
        self.assertEqual(cleaned, "Text with null bell vertical tab.")

    def test_file_hash(self):
        content = b"PDF Assistant Test Document"
        digest = file_hash(content)
        self.assertIsInstance(digest, str)
        self.assertEqual(len(digest), 64)  # SHA-256 is 64 hex characters

    def test_validate_pdf_valid(self):
        fake_pdf = b"%PDF-1.4\n1 0 obj\n<<>>\nendobj\ntrailer\n<<>>\n%%EOF"
        is_valid, err = validate_pdf(fake_pdf, "test.pdf")
        self.assertTrue(is_valid)
        self.assertEqual(err, "")

    def test_validate_pdf_invalid_extension(self):
        is_valid, err = validate_pdf(b"%PDF-1.4...", "test.docx")
        self.assertFalse(is_valid)
        self.assertIn("Invalid file type", err)

    def test_validate_pdf_invalid_magic_bytes(self):
        is_valid, err = validate_pdf(b"NOT_A_PDF_FILE", "test.pdf")
        self.assertFalse(is_valid)
        self.assertIn("not appear to be a valid PDF", err)

    def test_validate_pdf_size_limit(self):
        # Create bytes larger than MAX_UPLOAD_SIZE_MB
        oversized = b"%PDF-" + b"0" * (MAX_UPLOAD_SIZE_MB * 1024 * 1024 + 1024)
        is_valid, err = validate_pdf(oversized, "test.pdf")
        self.assertFalse(is_valid)
        self.assertIn("exceeds", err)


class TestPDFProcessor(unittest.TestCase):
    """Test PDF text extraction in backend/pdf_processor.py."""

    def _create_sample_pdf_bytes(self, pages: list[str]) -> bytes:
        """Helper to create a PDF in memory with custom text on each page."""
        doc = pymupdf.open()
        for text in pages:
            page = doc.new_page()
            page.insert_text((50, 72), text)
        pdf_bytes = doc.tobytes()
        doc.close()
        return pdf_bytes

    def test_extract_documents_multi_page(self):
        pages = [
            "This is page one discussing artificial intelligence and RAG.",
            "This is page two discussing vector embeddings and search.",
            "This is page three summarizing research findings.",
        ]
        pdf_bytes = self._create_sample_pdf_bytes(pages)
        docs = extract_documents_from_pdf(pdf_bytes, source_name="research_paper.pdf")

        self.assertEqual(len(docs), 3)
        for i, doc in enumerate(docs, start=1):
            self.assertEqual(doc.metadata["page_number"], i)
            self.assertEqual(doc.metadata["total_pages"], 3)
            self.assertEqual(doc.metadata["source"], "research_paper.pdf")
            self.assertIn("page", doc.text.lower())

    def test_extract_documents_empty_bytes(self):
        with self.assertRaises(ValueError) as ctx:
            extract_documents_from_pdf(b"", "empty.pdf")
        self.assertIn("empty", str(ctx.exception).lower())

    def test_extract_documents_blank_pages_raises_error(self):
        doc = pymupdf.open()
        doc.new_page()  # blank page without text
        blank_pdf = doc.tobytes()
        doc.close()

        with self.assertRaises(ValueError) as ctx:
            extract_documents_from_pdf(blank_pdf, "blank.pdf")
        self.assertIn("no extractable text", str(ctx.exception).lower())


class TestConfig(unittest.TestCase):
    """Test application settings in config/settings.py."""

    def test_settings_constants(self):
        self.assertGreater(CHUNK_SIZE, 0)
        self.assertGreater(CHUNK_OVERLAP, 0)
        self.assertLess(CHUNK_OVERLAP, CHUNK_SIZE)
        self.assertGreater(TOP_K_RESULTS, 0)
        self.assertIn("gemini", GEMINI_CHAT_MODEL.lower())
        self.assertIn("embedding", GEMINI_EMBEDDING_MODEL.lower())
        self.assertIn(".pdf", ALLOWED_EXTENSIONS)


class TestQueryEngineContract(unittest.TestCase):
    """Test data contracts in backend/query_engine.py."""

    def test_query_result_structure(self):
        result = QueryResult(
            answer="Quantum computing is an emerging field.",
            sources=[
                {
                    "page_number": 1,
                    "source": "quantum.pdf",
                    "text_snippet": "Quantum computing is...",
                    "relevance_score": 0.8921,
                }
            ],
        )
        self.assertEqual(result.answer, "Quantum computing is an emerging field.")
        self.assertEqual(len(result.sources), 1)
        self.assertEqual(result.sources[0]["page_number"], 1)


if __name__ == "__main__":
    unittest.main()
