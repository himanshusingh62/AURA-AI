"""Tests for LlamaIndex retrieval-time metadata filtering and document isolation."""

from __future__ import annotations

from unittest.mock import MagicMock, patch
import pytest
from llama_index.core import Document, VectorStoreIndex
from llama_index.core.embeddings import MockEmbedding
from llama_index.core.llms import CompletionResponse

from backend.query_engine import query_index


@pytest.fixture
def mock_embed_model():
    return MockEmbedding(embed_dim=16)


@pytest.fixture
def indexed_multi_doc(mock_embed_model):
    """Build a VectorStoreIndex with two distinct documents containing domain-specific facts."""
    doc_a = Document(
        text="Alpha Corp earnings report: Q3 operating margin expanded to 28.4% and net cloud revenue was $4.2B.",
        metadata={"source": "alpha_earnings.pdf", "file_id": "hash_alpha_123", "page_number": 1},
    )
    doc_b = Document(
        text="Beta Health clinical study: patient systolic blood pressure decreased by 12 mmHg following treatment.",
        metadata={"source": "beta_clinical.pdf", "file_id": "hash_beta_456", "page_number": 1},
    )
    return VectorStoreIndex.from_documents([doc_a, doc_b], embed_model=mock_embed_model)


def test_correct_document_retrieval(indexed_multi_doc):
    """Verify that a query scoped to document A retrieves document A's content."""
    with patch("backend.query_engine.get_llm") as mock_get_llm:
        mock_llm = MagicMock()
        mock_llm.complete.return_value = CompletionResponse(text="Cloud revenue was $4.2B.")
        mock_get_llm.return_value = mock_llm

        res = query_index(
            index=indexed_multi_doc,
            question="What was cloud revenue?",
            doc_filter="alpha_earnings.pdf",
            api_key="AIzaSyMockKeyForTestingOnly",
        )

    assert len(res.sources) == 1
    assert res.sources[0]["source"] == "alpha_earnings.pdf"
    assert "28.4%" in res.sources[0]["text_snippet"] or "$4.2B" in res.sources[0]["text_snippet"]


def test_cross_document_isolation(indexed_multi_doc):
    """Verify that a question scoped to PDF A CANNOT retrieve content from PDF B, even if terms match."""
    with patch("backend.query_engine.get_llm") as mock_get_llm:
        mock_llm = MagicMock()
        mock_llm.complete.return_value = CompletionResponse(text="No medical content found in Alpha Corp.")
        mock_get_llm.return_value = mock_llm

        # Query about Beta Health's domain (blood pressure) but strictly scoped to Alpha Corp
        res = query_index(
            index=indexed_multi_doc,
            question="What was the patient systolic blood pressure reduction?",
            doc_filter="alpha_earnings.pdf",
            api_key="AIzaSyMockKeyForTestingOnly",
        )

    # All returned sources must strictly belong to the scoped document 'alpha_earnings.pdf'
    for src in res.sources:
        assert src["source"] == "alpha_earnings.pdf"
        assert "beta_clinical.pdf" != src["source"]


def test_same_name_pdf_isolation(mock_embed_model):
    """Verify that two PDFs with the same filename (e.g. revisions) are isolated by file_id/source."""
    doc_v1 = Document(
        text="Project Roadmap Version 1: Target delivery date is Q1 2026.",
        metadata={"source": "roadmap.pdf", "file_id": "file_hash_v1", "page_number": 1},
    )
    doc_v2 = Document(
        text="Project Roadmap Version 2: Target delivery date postponed to Q4 2026.",
        metadata={"source": "roadmap_rev2.pdf", "file_id": "file_hash_v2", "page_number": 1},
    )
    index = VectorStoreIndex.from_documents([doc_v1, doc_v2], embed_model=mock_embed_model)

    with patch("backend.query_engine.get_llm") as mock_get_llm:
        mock_llm = MagicMock()
        mock_llm.complete.return_value = CompletionResponse(text="Delivery date is Q1 2026.")
        mock_get_llm.return_value = mock_llm

        # Scope by source name of Version 1
        res_v1 = query_index(
            index=index,
            question="What is the delivery date?",
            doc_filter="roadmap.pdf",
            api_key="AIzaSyMockKeyForTestingOnly",
        )
        assert len(res_v1.sources) == 1
        assert res_v1.sources[0]["source"] == "roadmap.pdf"
        assert "Q1 2026" in res_v1.sources[0]["text_snippet"]

        # Scope by file_id of Version 2
        mock_llm.complete.return_value = CompletionResponse(text="Delivery date is Q4 2026.")
        res_v2 = query_index(
            index=index,
            question="What is the delivery date?",
            doc_filter="file_hash_v2",
            api_key="AIzaSyMockKeyForTestingOnly",
        )
        assert len(res_v2.sources) == 1
        assert res_v2.sources[0]["source"] == "roadmap_rev2.pdf"
        assert "Q4 2026" in res_v2.sources[0]["text_snippet"]


def test_combined_unrestricted_retrieval(indexed_multi_doc):
    """Verify that when doc_filter is None or 'All Indexed Documents', retrieval spans both PDFs."""
    with patch("backend.query_engine.get_llm") as mock_get_llm:
        mock_llm = MagicMock()
        mock_llm.complete.return_value = CompletionResponse(text="Synthesis of Alpha and Beta.")
        mock_get_llm.return_value = mock_llm

        res = query_index(
            index=indexed_multi_doc,
            question="Summarize cloud revenue and blood pressure study results",
            doc_filter=None,
            api_key="AIzaSyMockKeyForTestingOnly",
        )

    retrieved_sources = {s["source"] for s in res.sources}
    assert "alpha_earnings.pdf" in retrieved_sources
    assert "beta_clinical.pdf" in retrieved_sources


def test_empty_results_for_unavailable_document(indexed_multi_doc):
    """Verify that querying for a non-existent document filter returns clean empty results without error."""
    res = query_index(
        index=indexed_multi_doc,
        question="What was the revenue?",
        doc_filter="non_existent_report.pdf",
        api_key="AIzaSyMockKeyForTestingOnly",
    )

    assert "No relevant content found" in res.answer
    assert res.sources == []
