"""Unit tests verifying scoped RAG query filtering behavior across multi-document indexes.

Verifies Item 12: Scoped queries return up to top_k passages from the targeted document
even when the target document is a minority or overshadowed in the global index.
"""

from __future__ import annotations

import pytest
from unittest.mock import MagicMock, patch
from llama_index.core import Document, VectorStoreIndex
from llama_index.core.embeddings import MockEmbedding

from backend.query_engine import query_index


@pytest.fixture
def mock_embed_model():
    return MockEmbedding(embed_dim=16)


def test_scoped_query_retrieves_target_document_passages(mock_embed_model):
    """Verify that when doc_filter is specified, passages from the filtered document are
    retrieved up to top_k, even when another document dominates the index.
    """
    # Document A dominates the index with 15 chunks
    doc_a_chunks = [
        Document(
            text=f"Astrophysics passage {i}: Study of cosmic microwave background and stellar evolution.",
            metadata={"page_number": i, "source": "astrophysics.pdf"},
        )
        for i in range(1, 16)
    ]

    # Document B has only 3 chunks
    doc_b_chunks = [
        Document(
            text="Machine learning transformers utilize self-attention mechanisms for sequence modeling.",
            metadata={"page_number": 1, "source": "transformers.pdf"},
        ),
        Document(
            text="Multi-head attention allows the model to jointly attend to information from different representation subspaces.",
            metadata={"page_number": 2, "source": "transformers.pdf"},
        ),
        Document(
            text="Positional encodings inject information about the relative or absolute position of tokens in the sequence.",
            metadata={"page_number": 3, "source": "transformers.pdf"},
        ),
    ]

    index = VectorStoreIndex.from_documents(doc_a_chunks + doc_b_chunks, embed_model=mock_embed_model)

    # Mock the LLM generation so we can test the retrieval pipeline offline
    mock_llm = MagicMock()
    mock_resp = MagicMock()
    mock_resp.text = "Transformers rely on self-attention and positional encodings."
    mock_llm.complete.return_value = mock_resp

    with patch("backend.query_engine.get_llm", return_value=mock_llm):
        result = query_index(
            index=index,
            question="Explain self-attention in transformers",
            top_k=3,
            doc_filter="transformers.pdf",
        )

        assert len(result.sources) > 0
        # All returned sources must strictly belong to the scoped document
        for src in result.sources:
            assert src["source"] == "transformers.pdf"

        # Content from astrophysics must NEVER appear in the context or citations
        for src in result.sources:
            assert "astrophysics" not in src["source"]
