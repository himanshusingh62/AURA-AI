"""Unit tests verifying support for uploading multiple PDFs with the same filename.

Verifies Item 13: Two same-name PDFs with different content indexed simultaneously
by hash without dictionary key collision or data overwrite.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch
import fitz
import streamlit as st
from llama_index.core.embeddings import MockEmbedding

from frontend.app import handle_pdf_upload, init_session_state
from utils.helpers import file_hash


def _create_simple_pdf_bytes(text: str) -> bytes:
    """Create an in-memory PDF with specified text."""
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((72, 72), text)
    pdf_bytes = doc.tobytes()
    doc.close()
    return pdf_bytes


def test_two_same_name_pdfs_indexed_without_collision():
    """Verify that uploading two distinct PDFs sharing the filename 'report.pdf'
    keys both by hash, keeps both active in session state, and prevents collisions.
    """
    init_session_state()
    st.session_state.uploaded_docs.clear()
    st.session_state.processed_upload_hashes.clear()
    st.session_state.failed_uploads.clear()
    st.session_state.index = None

    pdf1_bytes = _create_simple_pdf_bytes("Quarterly Financial Report Q1: Revenue exceeded projections by 15 percent.")
    pdf2_bytes = _create_simple_pdf_bytes("Quarterly Engineering Report Q1: Infrastructure uptime was maintained at 99.99 percent.")

    hash1 = file_hash(pdf1_bytes)
    hash2 = file_hash(pdf2_bytes)
    assert hash1 != hash2, "Test PDF hashes must be distinct"

    mock_embed = MockEmbedding(embed_dim=16)

    with patch("frontend.app.resolve_api_key", return_value="AIzaSyValidTestKey1234567890abcdef"), \
         patch("frontend.app.build_index") as mock_build:

        # Mock vector store index
        mock_index = MagicMock()
        mock_index.docstore.docs = {"node1": MagicMock(), "node2": MagicMock()}
        mock_build.return_value = mock_index

        # Upload first 'report.pdf'
        ok1 = handle_pdf_upload(pdf1_bytes, "report.pdf")
        assert ok1 is True
        assert hash1 in st.session_state.uploaded_docs
        assert len(st.session_state.uploaded_docs) == 1

        # Upload second 'report.pdf' (same filename, different content)
        ok2 = handle_pdf_upload(pdf2_bytes, "report.pdf")
        assert ok2 is True
        assert hash2 in st.session_state.uploaded_docs
        assert len(st.session_state.uploaded_docs) == 2

        # Verify both exist with distinct hashes and distinct sources
        doc1 = st.session_state.uploaded_docs[hash1]
        doc2 = st.session_state.uploaded_docs[hash2]

        assert doc1["name"] == "report.pdf"
        assert doc2["name"] == "report.pdf"
        assert doc1["source"] != doc2["source"], "Same-name PDFs must have distinct source keys for RAG citations"
        assert doc2["source"].startswith("report_")

        # Total chunks reflect both documents
        assert st.session_state.doc_chunks == doc1["chunks"] + doc2["chunks"]
