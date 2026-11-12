"""
Query engine — RAG pipeline that answers questions using the vector index.

Combines retrieval (vector search) with generation (Gemini chat model)
and returns both the answer text and the source references.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from llama_index.core import Settings, VectorStoreIndex
from llama_index.llms.gemini import Gemini

from config.settings import (
    GEMINI_CHAT_MODEL,
    TOP_K_RESULTS,
    get_google_api_key,
)


@dataclass
class QueryResult:
    """Structured response from the RAG pipeline."""

    answer: str
    sources: list[dict] = field(default_factory=list)
    # Each source dict: {"page_number": int, "source": str, "text_snippet": str, "relevance_score": float | None}


_SYSTEM_PROMPT = (
    "You are a helpful and precise research assistant. "
    "Answer the user's question based ONLY on the provided context extracted from the PDF document. "
    "If the context does not contain enough information to answer truthfully, state clearly that "
    "the document does not contain that information. Do NOT extrapolate or fabricate facts. "
    "Whenever appropriate, mention the specific page number(s) where information was found."
)


def _get_llm() -> Gemini:
    """Instantiate the Gemini chat LLM."""
    api_key = get_google_api_key()
    if not api_key:
        raise EnvironmentError(
            "GOOGLE_API_KEY is not set. "
            "Please configure your Google Gemini API key in the .env file "
            "or set it as an environment variable."
        )
    return Gemini(
        model=GEMINI_CHAT_MODEL,
        api_key=api_key,
        system_prompt=_SYSTEM_PROMPT,
    )


def query_index(
    index: VectorStoreIndex,
    question: str,
    top_k: int = TOP_K_RESULTS,
) -> QueryResult:
    """Run a RAG query against the vector index.

    1. Retrieve the top-K most relevant chunks.
    2. Pass them as context to Gemini along with the user question.
    3. Return the grounded answer and source references.
    """
    if not question.strip():
        return QueryResult(answer="Please enter a valid non-empty question.")

    llm = _get_llm()
    Settings.llm = llm

    query_engine = index.as_query_engine(
        similarity_top_k=top_k,
        response_mode="compact",
        verbose=False,
    )

    response = query_engine.query(question)

    # Extract source references from the retrieved nodes
    sources: list[dict] = []
    seen_pages: set[int] = set()

    if getattr(response, "source_nodes", None):
        for node in response.source_nodes:
            meta = getattr(node, "metadata", {}) or {}
            page = meta.get("page_number", 0)
            if page in seen_pages:
                continue
            seen_pages.add(page)
            score = getattr(node, "score", None)
            snippet = node.text[:300] + ("…" if len(node.text) > 300 else "")
            sources.append(
                {
                    "page_number": page,
                    "source": meta.get("source", "uploaded.pdf"),
                    "text_snippet": snippet,
                    "relevance_score": round(score, 4) if score is not None else None,
                }
            )

    return QueryResult(
        answer=str(response),
        sources=sorted(sources, key=lambda s: s["page_number"]),
    )
