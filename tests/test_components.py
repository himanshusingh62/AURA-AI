"""
Automated unit and integration test suite for the Multimodal AI Research Assistant.
Includes validation, security sanitization, XSS prevention, and component contract tests.
"""

from __future__ import annotations

import io
import os
import unittest
import warnings

# Suppress upstream SDK deprecation FutureWarning from google.generativeai and llama_index
warnings.filterwarnings("ignore", category=FutureWarning, module=r".*generativeai.*")
warnings.filterwarnings("ignore", category=FutureWarning, module=r".*llama_index.*")

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
from frontend.components import render_citation_card, render_image_citation_card
from utils.helpers import (
    clean_text,
    file_hash,
    mask_api_key,
    sanitize_error_message,
    validate_pdf,
)


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
    """Test data contracts in backend/query_engine.py and multimodal engine."""

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


class TestImageProcessing(unittest.TestCase):
    """Test image validation and metadata processing."""

    def test_validate_image_valid_png(self):
        from utils.helpers import validate_image, is_image, is_pdf
        from backend.image_processor import process_uploaded_image
        from PIL import Image

        img = Image.new("RGB", (100, 100), color=(255, 0, 0))
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        raw = buf.getvalue()

        self.assertTrue(is_image("photo.png"))
        self.assertFalse(is_pdf("photo.png"))

        is_valid, err = validate_image(raw, "photo.png")
        self.assertTrue(is_valid)
        self.assertEqual(err, "")

        processed = process_uploaded_image(raw, "photo.png")
        self.assertEqual(processed.width, 100)
        self.assertEqual(processed.height, 100)
        self.assertEqual(processed.format, "PNG")
        self.assertTrue(processed.thumbnail_base64.startswith("data:image/"))

    def test_validate_image_corrupted_data(self):
        from utils.helpers import validate_image
        is_valid, err = validate_image(b"CORRUPTED_NON_IMAGE_DATA", "corrupted.jpg")
        self.assertFalse(is_valid)
        self.assertIn("invalid", err.lower())


class TestSecurityAndSanitization(unittest.TestCase):
    """Test API key security, redaction, and error sanitization."""

    def test_sanitize_error_google_api_key(self):
        # Google API key format: AIza followed by 35 alphanumeric/_/- chars
        raw_error = "API error: https://generativelanguage.googleapis.com/v1beta/models?key=AIzaSyA1234567890abcdefghijklmnopqrstuv"
        sanitized = sanitize_error_message(raw_error)
        self.assertNotIn("AIzaSyA1234567890abcdefghijklmnopqrstuv", sanitized)
        self.assertIn("[REDACTED", sanitized)

    def test_sanitize_error_url_query_key(self):
        raw_error = "Failed to connect to host?token=secret123456&api_key=my_private_gemini_token"
        sanitized = sanitize_error_message(raw_error)
        self.assertNotIn("secret123456", sanitized)
        self.assertNotIn("my_private_gemini_token", sanitized)
        self.assertIn("token=[REDACTED]", sanitized)
        self.assertIn("api_key=[REDACTED]", sanitized)

    def test_sanitize_error_bearer_token(self):
        raw_error = "HTTP 401 Unauthorized: Authorization Bearer ya29.a0ARrda8-long_token_here"
        sanitized = sanitize_error_message(raw_error)
        self.assertNotIn("ya29.a0ARrda8", sanitized)
        self.assertIn("Bearer [REDACTED]", sanitized)

    def test_mask_api_key_standard(self):
        key = "AIzaSyD7XYZ9876543210FEDCBA0123456789"
        masked = mask_api_key(key)
        self.assertEqual(masked, "AIza••••••••6789")

    def test_mask_api_key_short_or_empty(self):
        self.assertEqual(mask_api_key(""), "")
        self.assertEqual(mask_api_key("12345"), "••••••••")


class TestXSSPrevention(unittest.TestCase):
    """Test HTML injection & Cross-Site Scripting (XSS) defenses in UI components."""

    def test_citation_card_escapes_xss_in_snippet(self):
        malicious_src = {
            "page_number": 3,
            "source": "safe_doc.pdf",
            "text_snippet": "<script>alert('xss')</script><img src=x onerror=alert(1)>",
            "relevance_score": 0.95,
        }
        card_html = render_citation_card(malicious_src)
        self.assertNotIn("<script>", card_html)
        self.assertNotIn("<img src=x", card_html)
        self.assertIn("&lt;script&gt;", card_html)
        self.assertIn("&lt;img src=x", card_html)

    def test_citation_card_escapes_xss_in_source(self):
        malicious_src = {
            "page_number": 1,
            "source": "<iframe src='evil.com'></iframe>",
            "text_snippet": "Safe snippet",
        }
        card_html = render_citation_card(malicious_src)
        self.assertNotIn("<iframe", card_html)
        self.assertIn("&lt;iframe", card_html)

    def test_image_citation_card_escapes_xss(self):
        malicious_img = {
            "filename": "<svg onload=alert(1)>.png",
            "format": "<b onclick=evil()>PNG</b>",
            "dimensions": "800x600",
            "thumbnail_base64": "data:image/png;base64,iVBORw0KGgoAAAANSUhEUg==",
        }
        card_html = render_image_citation_card(malicious_img)
        self.assertNotIn("<svg", card_html)
        self.assertNotIn("<b onclick", card_html)
        self.assertIn("&lt;svg onload=alert(1)&gt;.png", card_html)

    def test_image_citation_card_rejects_unsafe_thumbnail_scheme(self):
        malicious_img = {
            "filename": "chart.png",
            "format": "PNG",
            "dimensions": "400x400",
            # Attempt javascript: pseudo-protocol or HTML breakout
            "thumbnail_base64": "javascript:alert(document.cookie)",
        }
        card_html = render_image_citation_card(malicious_img)
        self.assertNotIn("javascript:", card_html)
        self.assertIn("🖼️", card_html)


if __name__ == "__main__":
    unittest.main()
