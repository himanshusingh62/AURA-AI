"""Unit and integration tests for multi-PDF upload, partial deletion, and vector store integrity.

Verifies that:
- Deleted PDF nodes and vectors are removed from both docstore and vector store data structures.
- Deleted PDF content cannot appear in subsequent retrieval results.
- Remaining PDFs remain fully searchable and undisturbed.
- Repeated deletion and nonexistent document deletion are completely safe.
- Deleting all PDFs leaves no stale index state.
"""

from __future__ import annotations

import pytest
from llama_index.core import Document, VectorStoreIndex
from llama_index.core.embeddings import MockEmbedding

from backend.embeddings import remove_document_from_index


class TestMultiPdfDeletion:
    """Test suite verifying safe node removal from unified vector store upon document deletion."""

    @pytest.fixture
    def mock_embed_model(self):
        return MockEmbedding(embed_dim=16)

    def test_upload_three_pdfs_delete_one_and_query(self, mock_embed_model):
        """Simulate uploading 3 PDFs, removing 1, and verifying retrieval results and vector store."""
        pdf1_docs = [
            Document(
                text="Quantum computing utilizes quantum superposition and entanglement to execute complex calculations.",
                metadata={"page_number": 1, "source": "quantum_physics.pdf", "file_id": "pdf1_hash"},
            ),
            Document(
                text="Qubits are fragile and susceptible to quantum decoherence and environmental noise.",
                metadata={"page_number": 2, "source": "quantum_physics.pdf", "file_id": "pdf1_hash"},
            ),
        ]
        pdf2_docs = [
            Document(
                text="Coral reefs are diverse underwater ecosystems held together by calcium carbonate structures produced by corals.",
                metadata={"page_number": 1, "source": "marine_biology.pdf", "file_id": "pdf2_hash"},
            ),
            Document(
                text="Deep sea hydrothermal vents host chemoautotrophic bacteria that utilize hydrogen sulfide.",
                metadata={"page_number": 2, "source": "marine_biology.pdf", "file_id": "pdf2_hash"},
            ),
        ]
        pdf3_docs = [
            Document(
                text="Solar photovoltaic cells convert solar photon radiation directly into electricity using the photoelectric effect.",
                metadata={"page_number": 1, "source": "renewable_energy.pdf", "file_id": "pdf3_hash"},
            ),
            Document(
                text="Wind turbines convert kinetic energy from airflow into electrical energy via electromagnetic induction.",
                metadata={"page_number": 2, "source": "renewable_energy.pdf", "file_id": "pdf3_hash"},
            ),
        ]

        all_docs = pdf1_docs + pdf2_docs + pdf3_docs

        # 1. Build unified index across all 3 PDFs
        index = VectorStoreIndex.from_documents(all_docs, embed_model=mock_embed_model)
        assert len(index.docstore.docs) == 6
        assert len(index.vector_store._data.embedding_dict) == 6
        assert len(index.vector_store._data.text_id_to_ref_doc_id) == 6

        # 2. Query initial index: marine biology is present
        retriever = index.as_retriever(similarity_top_k=4)
        nodes_before = retriever.retrieve("coral reefs and hydrothermal vents")
        sources_before = [n.node.metadata.get("source") for n in nodes_before]
        assert "marine_biology.pdf" in sources_before

        # 3. Delete PDF 2 ('marine_biology.pdf')
        remaining_docs = pdf1_docs + pdf3_docs
        pdf2_doc_ids = [d.doc_id for d in pdf2_docs]

        updated_index = remove_document_from_index(
            index=index,
            filename="marine_biology.pdf",
            doc_ids=pdf2_doc_ids,
            remaining_documents=remaining_docs,
        )

        assert updated_index is not None
        # Verify docstore count
        assert len(updated_index.docstore.docs) == 4
        # Verify vector store data structure counts
        assert len(updated_index.vector_store._data.embedding_dict) == 4
        assert len(updated_index.vector_store._data.text_id_to_ref_doc_id) == 4

        # 4. Confirm 'marine_biology.pdf' is completely absent from docstore and vector store metadata
        docstore_sources = {n.metadata.get("source") for n in updated_index.docstore.docs.values()}
        assert "marine_biology.pdf" not in docstore_sources
        assert docstore_sources == {"quantum_physics.pdf", "renewable_energy.pdf"}

        vs_sources = {
            m.get("source")
            for m in updated_index.vector_store._data.metadata_dict.values()
            if m.get("source")
        }
        assert "marine_biology.pdf" not in vs_sources

        # 5. Query again for marine biology terms: must NEVER return marine_biology.pdf
        retriever_after = updated_index.as_retriever(similarity_top_k=4)
        nodes_after = retriever_after.retrieve("coral reefs and hydrothermal vents")
        sources_after = [n.node.metadata.get("source") for n in nodes_after]
        assert "marine_biology.pdf" not in sources_after, "Deleted PDF content still appeared in query results!"

        # 6. Remaining PDFs remain searchable
        q_nodes = retriever_after.retrieve("qubits superposition and quantum computing")
        assert "quantum_physics.pdf" in [n.node.metadata.get("source") for n in q_nodes]

        r_nodes = retriever_after.retrieve("solar photovoltaic cells and wind turbines")
        assert "renewable_energy.pdf" in [n.node.metadata.get("source") for n in r_nodes]

        # 7. Delete remaining PDFs one by one
        index_after_q = remove_document_from_index(
            index=updated_index,
            filename="quantum_physics.pdf",
            doc_ids=[d.doc_id for d in pdf1_docs],
            remaining_documents=pdf3_docs,
        )
        assert index_after_q is not None
        assert len(index_after_q.docstore.docs) == 2
        assert len(index_after_q.vector_store._data.embedding_dict) == 2

        # 8. Delete final PDF: returns None and leaves no stale index state
        final_index = remove_document_from_index(
            index=index_after_q,
            filename="renewable_energy.pdf",
            doc_ids=[d.doc_id for d in pdf3_docs],
            remaining_documents=[],
        )
        assert final_index is None

    def test_repeated_deletion_is_safe(self, mock_embed_model):
        """Deleting the same document twice in succession must not corrupt remaining index."""
        docs_a = [Document(text="Alpha text", metadata={"source": "a.pdf"})]
        docs_b = [Document(text="Beta text", metadata={"source": "b.pdf"})]
        index = VectorStoreIndex.from_documents(docs_a + docs_b, embed_model=mock_embed_model)

        # First deletion
        index = remove_document_from_index(
            index=index,
            filename="a.pdf",
            doc_ids=[d.doc_id for d in docs_a],
            remaining_documents=docs_b,
        )
        assert index is not None
        assert len(index.docstore.docs) == 1

        # Second repeated deletion of already deleted document
        index = remove_document_from_index(
            index=index,
            filename="a.pdf",
            doc_ids=[d.doc_id for d in docs_a],
            remaining_documents=docs_b,
        )
        assert index is not None
        assert len(index.docstore.docs) == 1
        assert "b.pdf" in {n.metadata.get("source") for n in index.docstore.docs.values()}

    def test_remove_nonexistent_document_leaves_index_intact(self, mock_embed_model):
        """Removing a document name not in the index should be safe and leave the index intact."""
        docs = [Document(text="Sample content", metadata={"page_number": 1, "source": "sample.pdf"})]
        index = VectorStoreIndex.from_documents(docs, embed_model=mock_embed_model)
        assert len(index.docstore.docs) == 1

        updated = remove_document_from_index(
            index=index,
            filename="nonexistent.pdf",
            doc_ids=["fake-id"],
            remaining_documents=docs,
        )
        assert updated is not None
        assert len(updated.docstore.docs) == 1
        assert len(updated.vector_store._data.embedding_dict) == 1

    def test_remove_from_none_index(self):
        """Removing from None index returns None safely without raising an exception."""
        result = remove_document_from_index(None, "doc.pdf")
        assert result is None
