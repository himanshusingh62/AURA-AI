"""
Embedding & vector-index layer.

Uses Google Gemini embeddings via LlamaIndex and an in-memory
VectorStoreIndex for similarity search.
"""

from __future__ import annotations

from llama_index.core import Settings, VectorStoreIndex
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.schema import Document
from llama_index.embeddings.gemini import GeminiEmbedding

from config.settings import (
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    GEMINI_EMBEDDING_MODEL,
    get_google_api_key,
)


def _get_embedding_model() -> GeminiEmbedding:
    """Instantiate the Gemini embedding model."""
    api_key = get_google_api_key()
    if not api_key:
        raise EnvironmentError(
            "GOOGLE_API_KEY is not set. "
            "Please configure your Google Gemini API key in the .env file "
            "or set it as an environment variable."
        )
    return GeminiEmbedding(
        model_name=GEMINI_EMBEDDING_MODEL,
        api_key=api_key,
    )


def build_index(documents: list[Document]) -> VectorStoreIndex:
    """Chunk documents, embed them, and return a VectorStoreIndex.

    Steps:
        1. Split each Document into smaller nodes using SentenceSplitter.
        2. Generate vector embeddings via Gemini.
        3. Build an in-memory VectorStoreIndex for fast similarity search.
    """
    if not documents:
        raise ValueError("Cannot build index from an empty list of documents.")

    embed_model = _get_embedding_model()

    # Configure global LlamaIndex settings
    Settings.embed_model = embed_model
    Settings.chunk_size = CHUNK_SIZE
    Settings.chunk_overlap = CHUNK_OVERLAP

    splitter = SentenceSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )

    nodes = splitter.get_nodes_from_documents(documents, show_progress=False)

    index = VectorStoreIndex(
        nodes=nodes,
        embed_model=embed_model,
        show_progress=False,
    )

    return index
