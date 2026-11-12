"""Tests for failed upload state management, error classification, and retry workflows."""

from __future__ import annotations

from unittest.mock import MagicMock, patch
import pytest
import streamlit as st

from frontend.app import (
    handle_image_upload,
    handle_pdf_upload,
    init_session_state,
    process_incoming_files,
    retry_failed_upload,
)
from utils.helpers import file_hash


@pytest.fixture(autouse=True)
def setup_streamlit_session():
    """Reset streamlit session state before each test."""
    st.session_state.clear()
    init_session_state()
    yield
    st.session_state.clear()


@pytest.fixture
def sample_valid_pdf_bytes():
    """Generate a minimal valid 1-page PDF byte string with extractable text."""
    import pymupdf as fitz
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((50, 50), "AURA.AI sample document content for test verification.")
    pdf_bytes = doc.tobytes()
    doc.close()
    return pdf_bytes


@pytest.fixture
def sample_valid_png_bytes():
    """Generate minimal valid 10x10 PNG bytes."""
    import io
    from PIL import Image
    buf = io.BytesIO()
    img = Image.new("RGB", (10, 10), color="blue")
    img.save(buf, format="PNG")
    return buf.getvalue()


class MockUploadedFile:
    def __init__(self, name: str, data: bytes):
        self.name = name
        self._data = data
        self._pos = 0

    def read(self, *args):
        return self._data

    def seek(self, pos: int):
        self._pos = pos


def test_missing_api_key_followed_by_successful_retry(sample_valid_pdf_bytes):
    """Test that missing API key causes temporary failure and subsequent retry succeeds."""
    h = file_hash(sample_valid_pdf_bytes)

    # 1. Attempt upload with NO API key configured
    with patch("frontend.app.resolve_api_key", return_value=""):
        ok = handle_pdf_upload(sample_valid_pdf_bytes, "test_doc.pdf")
        assert ok is False

    # Verify not marked as successfully processed, but stored in failed_uploads as temporary
    assert h not in st.session_state.processed_upload_hashes
    assert h in st.session_state.failed_uploads
    assert st.session_state.failed_uploads[h]["status"] == "temporary_error"
    assert st.session_state.failed_uploads[h]["file_bytes"] == sample_valid_pdf_bytes

    # 2. Configure API key and retry
    mock_index = MagicMock()
    mock_index.docstore.docs = {"n1": MagicMock()}

    with patch("frontend.app.resolve_api_key", return_value="AIzaSyTestKey123456789"), \
         patch("frontend.app.build_index", return_value=mock_index):
        retry_ok = retry_failed_upload(h)
        assert retry_ok is True

    # After retry success, marked as processed and removed from failed_uploads
    assert h in st.session_state.processed_upload_hashes
    assert h not in st.session_state.failed_uploads
    assert h in st.session_state.uploaded_docs


def test_temporary_network_failure_followed_by_successful_retry(sample_valid_pdf_bytes):
    """Test that transient network/embedding error allows retry without renaming file."""
    h = file_hash(sample_valid_pdf_bytes)

    # 1. Attempt upload with network drop during embedding
    with patch("frontend.app.resolve_api_key", return_value="AIzaSyTestKey123456789"), \
         patch("frontend.app.build_index", side_effect=RuntimeError("Connection reset by peer")):
        ok = handle_pdf_upload(sample_valid_pdf_bytes, "network_test.pdf")
        assert ok is False

    assert h not in st.session_state.processed_upload_hashes
    assert h in st.session_state.failed_uploads
    assert st.session_state.failed_uploads[h]["status"] == "temporary_error"

    # 2. Network recovers; retry
    mock_index = MagicMock()
    mock_index.docstore.docs = {"n1": MagicMock()}

    with patch("frontend.app.resolve_api_key", return_value="AIzaSyTestKey123456789"), \
         patch("frontend.app.build_index", return_value=mock_index):
        retry_ok = retry_failed_upload(h)
        assert retry_ok is True

    assert h in st.session_state.processed_upload_hashes
    assert h not in st.session_state.failed_uploads
    assert h in st.session_state.uploaded_docs


def test_gemini_429_quota_error_followed_by_successful_retry(sample_valid_pdf_bytes):
    """Test that HTTP 429 quota exhaustion marks as temporary and can be retried later."""
    from google.genai import errors
    h = file_hash(sample_valid_pdf_bytes)

    # Simulate 429 RESOURCE_EXHAUSTED ClientError
    mock_429 = errors.ClientError(
        429,
        {"error": {"message": "RESOURCE_EXHAUSTED: Quota exceeded", "code": 429, "status": "RESOURCE_EXHAUSTED"}},
    )

    with patch("frontend.app.resolve_api_key", return_value="AIzaSyTestKey123456789"), \
         patch("frontend.app.build_index", side_effect=mock_429):
        ok = handle_pdf_upload(sample_valid_pdf_bytes, "quota_test.pdf")
        assert ok is False

    assert h not in st.session_state.processed_upload_hashes
    assert h in st.session_state.failed_uploads
    assert st.session_state.failed_uploads[h]["status"] == "temporary_error"

    # Quota resets; retry succeeds
    mock_index = MagicMock()
    mock_index.docstore.docs = {"n1": MagicMock()}

    with patch("frontend.app.resolve_api_key", return_value="AIzaSyTestKey123456789"), \
         patch("frontend.app.build_index", return_value=mock_index):
        retry_ok = retry_failed_upload(h)
        assert retry_ok is True

    assert h in st.session_state.processed_upload_hashes
    assert h not in st.session_state.failed_uploads


def test_invalid_pdf_rejected_as_permanent_error():
    """Test that corrupt/invalid PDF is rejected permanently and does not loop."""
    bad_pdf_bytes = b"This is plain text pretending to be a PDF"
    h = file_hash(bad_pdf_bytes)

    with patch("frontend.app.resolve_api_key", return_value="AIzaSyTestKey123456789"):
        ok = handle_pdf_upload(bad_pdf_bytes, "invalid.pdf")
        assert ok is False

    assert h in st.session_state.processed_upload_hashes
    assert h in st.session_state.failed_uploads
    assert st.session_state.failed_uploads[h]["status"] == "permanent_error"
    assert st.session_state.failed_uploads[h]["file_bytes"] is None


def test_successful_upload_is_not_processed_twice(sample_valid_pdf_bytes):
    """Test that an already processed file is skipped on subsequent ingest cycles."""
    h = file_hash(sample_valid_pdf_bytes)
    mock_index = MagicMock()
    mock_index.docstore.docs = {"n1": MagicMock()}

    with patch("frontend.app.resolve_api_key", return_value="AIzaSyTestKey123456789"), \
         patch("frontend.app.build_index", return_value=mock_index):
        ok = handle_pdf_upload(sample_valid_pdf_bytes, "doc1.pdf")
        assert ok is True

    assert h in st.session_state.processed_upload_hashes

    # Process batch with same file again
    upload_file = MockUploadedFile("doc1.pdf", sample_valid_pdf_bytes)
    with patch("frontend.app.handle_pdf_upload") as mock_handle:
        changed = process_incoming_files([upload_file])
        assert changed is False
        mock_handle.assert_not_called()


def test_multiple_uploads_one_failed_and_one_succeeded(sample_valid_pdf_bytes, sample_valid_png_bytes):
    """Test uploading a batch where one file succeeds and one fails."""
    mock_index = MagicMock()
    mock_index.docstore.docs = {"n1": MagicMock()}

    good_pdf = MockUploadedFile("good.pdf", sample_valid_pdf_bytes)
    bad_file = MockUploadedFile("bad.txt", b"plain text unsupported file")

    with patch("frontend.app.resolve_api_key", return_value="AIzaSyTestKey123456789"), \
         patch("frontend.app.build_index", return_value=mock_index):
        changed = process_incoming_files([good_pdf, bad_file])

    assert changed is True
    good_h = file_hash(sample_valid_pdf_bytes)
    bad_h = file_hash(b"plain text unsupported file")

    assert good_h in st.session_state.processed_upload_hashes
    assert good_h in st.session_state.uploaded_docs

    assert bad_h in st.session_state.processed_upload_hashes
    assert st.session_state.failed_uploads[bad_h]["status"] == "permanent_error"


def test_image_upload_retry(sample_valid_png_bytes):
    """Test that image processing error allows retry."""
    h = file_hash(sample_valid_png_bytes)

    # 1. Image fails due to transient exception
    with patch("frontend.app.process_uploaded_image", side_effect=OSError("Temporary I/O error")):
        ok = handle_image_upload(sample_valid_png_bytes, "test.png")
        assert ok is False

    assert h not in st.session_state.processed_upload_hashes
    assert h in st.session_state.failed_uploads
    assert st.session_state.failed_uploads[h]["status"] == "temporary_error"

    # 2. Retry succeeds using the real process_uploaded_image
    retry_ok = retry_failed_upload(h)
    assert retry_ok is True

    assert h in st.session_state.processed_upload_hashes
    assert h not in st.session_state.failed_uploads
    assert "test.png" in st.session_state.uploaded_images


def test_streamlit_rerun_does_not_duplicate_processing(sample_valid_pdf_bytes):
    """Test that multiple simulated reruns with the same file list do not reprocess files."""
    mock_index = MagicMock()
    mock_index.docstore.docs = {"n1": MagicMock()}

    upload_file = MockUploadedFile("doc.pdf", sample_valid_pdf_bytes)

    # Run 1: initial upload
    with patch("frontend.app.resolve_api_key", return_value="AIzaSyTestKey123456789"), \
         patch("frontend.app.build_index", return_value=mock_index):
        changed1 = process_incoming_files([upload_file])
        assert changed1 is True
        assert len(st.session_state.uploaded_docs) == 1

    # Run 2: simulated Streamlit rerun with the same uploader state
    with patch("frontend.app.resolve_api_key", return_value="AIzaSyTestKey123456789"), \
         patch("frontend.app.build_index", return_value=mock_index):
        changed2 = process_incoming_files([upload_file])
        assert changed2 is False
        assert len(st.session_state.uploaded_docs) == 1
