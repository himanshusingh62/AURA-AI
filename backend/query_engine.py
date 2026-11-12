"""
Query Engine — Grounded RAG Pipeline using the modern official google-genai SDK.
Implements a custom LlamaIndex LLM wrapper that connects directly to Google Gen AI Client.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Generator

from google import genai
from google.genai import errors, types
from llama_index.core import Settings, VectorStoreIndex
from llama_index.core.llms import (
    CompletionResponse,
    CompletionResponseGen,
    LLMMetadata,
)
from llama_index.core.llms.custom import CustomLLM
from pydantic import Field, PrivateAttr

from backend.gemini_client import get_genai_client, translate_genai_error
from backend.rate_limit import parse_quota_error
from config.settings import (
    GEMINI_CHAT_MODEL,
    TOP_K_RESULTS,
)
from utils.helpers import sanitize_error_message
from utils.usage_tracker import record_api_request


@dataclass
class QueryResult:
    """Structured response from the RAG pipeline."""

    answer: str
    sources: list[dict] = field(default_factory=list)
    # Each source dict: {"page_number": int, "source": str, "text_snippet": str, "relevance_score": float | None}


_DEFAULT_SYSTEM_PROMPT = (
    "You are a helpful, rigorous, and precise AI Research Assistant. "
    "Answer the user's question based strictly and truthfully on the provided context extracted from the PDF document(s). "
    "If the context does not contain enough information to answer truthfully, state clearly that "
    "the document does not contain that information. Do NOT extrapolate, hallucinate, or fabricate facts. "
    "Whenever appropriate, mention the specific page number(s) and document name(s) where information was found."
)


class GoogleGenAILLM(CustomLLM):
    """Native LlamaIndex LLM implementation using the modern official google-genai SDK."""

    model_name: str = Field(default=GEMINI_CHAT_MODEL)
    temperature: float = Field(default=0.2)
    system_prompt_str: str | None = Field(default=_DEFAULT_SYSTEM_PROMPT)
    max_retries: int = Field(default=3)
    _client: genai.Client = PrivateAttr()

    def __init__(
        self,
        api_key: str | None = None,
        model_name: str = GEMINI_CHAT_MODEL,
        temperature: float = 0.2,
        system_prompt: str | None = _DEFAULT_SYSTEM_PROMPT,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            model_name=model_name,
            temperature=temperature,
            system_prompt_str=system_prompt,
            **kwargs,
        )
        self._client = get_genai_client(api_key=api_key)

    @property
    def metadata(self) -> LLMMetadata:
        return LLMMetadata(
            context_window=1000000,
            num_output=8192,
            model_name=self.model_name,
            is_chat_model=False,
        )

    def complete(self, prompt: str, **kwargs: Any) -> CompletionResponse:
        """Synchronously complete a prompt using google-genai with retry logic."""
        from utils.usage_tracker import reserve_api_request
        reservation_id = reserve_api_request("chat", self.model_name)

        backoff = 1.0
        last_error: Exception | None = None

        config = types.GenerateContentConfig(
            system_instruction=self.system_prompt_str,
            temperature=self.temperature,
        )

        last_error_msg: str | None = None

        for attempt in range(1, self.max_retries + 1):
            t0 = time.time()
            try:
                response = self._client.models.generate_content(
                    model=self.model_name,
                    contents=prompt,
                    config=config,
                )
                text = response.text.strip() if hasattr(response, "text") and response.text else "Unable to generate an answer."
                record_api_request("chat", "success", latency=time.time() - t0, model_name=self.model_name, reservation_id=reservation_id)
                return CompletionResponse(text=text)
            except errors.ClientError as exc:
                latency = time.time() - t0
                quota_info = parse_quota_error(exc, model_name=self.model_name)
                last_error_msg = quota_info.friendly_message

                if quota_info.is_429:
                    record_api_request(
                        "chat",
                        "429_quota_exceeded",
                        latency=latency,
                        is_429=True,
                        error_message=quota_info.friendly_message,
                        model_name=self.model_name,
                        reservation_id=reservation_id,
                    )
                    reservation_id = None
                    # Never repeatedly retry if daily quota is exhausted
                    if quota_info.is_daily_limit:
                        raise RuntimeError(quota_info.friendly_message) from None

                    delay = quota_info.retry_delay_seconds
                    if delay is not None and delay <= 6.0 and attempt < self.max_retries:
                        time.sleep(delay)
                        continue
                    elif delay is None and attempt < self.max_retries:
                        time.sleep(backoff)
                        backoff *= 2.0
                        continue
                    else:
                        raise RuntimeError(quota_info.friendly_message) from None
                else:
                    record_api_request("chat", "error", latency=latency, error_message=str(exc), model_name=self.model_name, reservation_id=reservation_id)
                    reservation_id = None
                    raise RuntimeError(translate_genai_error(exc, model_name=self.model_name)) from None
            except Exception as exc:
                latency = time.time() - t0
                quota_info = parse_quota_error(exc, model_name=self.model_name)
                if quota_info.is_429:
                    record_api_request(
                        "chat",
                        "429_quota_exceeded",
                        latency=latency,
                        is_429=True,
                        error_message=quota_info.friendly_message,
                        model_name=self.model_name,
                        reservation_id=reservation_id,
                    )
                    reservation_id = None
                    if quota_info.is_daily_limit or attempt >= self.max_retries:
                        raise RuntimeError(quota_info.friendly_message) from None
                    time.sleep(backoff)
                    backoff *= 2.0
                    continue

                record_api_request("chat", "error", latency=latency, error_message=str(exc), model_name=self.model_name, reservation_id=reservation_id)
                reservation_id = None
                raise RuntimeError(translate_genai_error(exc, model_name=self.model_name)) from None

        raise RuntimeError(last_error_msg or f"Chat generation failed after {self.max_retries} attempts.")

    def stream_complete(self, prompt: str, **kwargs: Any) -> CompletionResponseGen:
        """Stream completion fallback."""
        resp = self.complete(prompt, **kwargs)
        yield resp


def get_llm(
    api_key: str | None = None,
    model_name: str | None = None,
    system_prompt: str | None = None,
) -> GoogleGenAILLM:
    """Instantiate a GoogleGenAILLM model with explicit key and model support."""
    return GoogleGenAILLM(
        api_key=api_key,
        model_name=model_name or GEMINI_CHAT_MODEL,
        system_prompt=system_prompt or _DEFAULT_SYSTEM_PROMPT,
    )


def query_index(
    index: VectorStoreIndex,
    question: str,
    top_k: int = TOP_K_RESULTS,
    api_key: str | None = None,
    doc_filter: str | None = None,
    model_name: str | None = None,
) -> QueryResult:
    """Run a grounded RAG query against the vector index.

    1. Retrieve the top-K most relevant chunks across all uploaded PDFs.
    2. Format context with document and page citations.
    3. Synthesize the grounded answer via configured Gemini model using official google-genai SDK.
    4. Return the answer and source citations.
    """
    if not question.strip():
        return QueryResult(answer="Please enter a valid non-empty question.")

    # Retrieval-time document filtering via LlamaIndex MetadataFilters
    filters = None
    if doc_filter and doc_filter != "All Indexed Documents":
        try:
            from llama_index.core.vector_stores import (
                ExactMatchFilter,
                FilterCondition,
                MetadataFilters,
            )
            filters = MetadataFilters(
                filters=[
                    ExactMatchFilter(key="source", value=doc_filter),
                    ExactMatchFilter(key="file_id", value=doc_filter),
                    ExactMatchFilter(key="doc_id", value=doc_filter),
                ],
                condition=FilterCondition.OR,
            )
        except Exception:
            filters = None

    try:
        if filters is not None:
            retriever = index.as_retriever(similarity_top_k=top_k, filters=filters)
        else:
            retriever = index.as_retriever(similarity_top_k=top_k)
        source_nodes = retriever.retrieve(question)
    except Exception as exc:
        # Fallback to broader retrieval if vector store backend does not support filters
        try:
            retriever = index.as_retriever(similarity_top_k=max(top_k * 5, 50) if doc_filter else top_k)
            source_nodes = retriever.retrieve(question)
        except Exception as inner_exc:
            raise RuntimeError(f"Retrieval error: {translate_genai_error(inner_exc)}") from None

    # Extract source references from the retrieved nodes
    sources: list[dict] = []
    seen_keys: set[tuple[str, int]] = set()
    context_blocks: list[str] = []

    for node_with_score in source_nodes:
        node = getattr(node_with_score, "node", node_with_score)
        score = getattr(node_with_score, "score", None)
        meta = getattr(node, "metadata", {}) or {}
        page = meta.get("page_number", 0)
        src_doc = meta.get("source", "uploaded.pdf")

        # Defense-in-depth: enforce document isolation at extraction time
        if doc_filter and doc_filter != "All Indexed Documents":
            src_match = (src_doc == doc_filter)
            fid_match = (meta.get("file_id") == doc_filter)
            did_match = (meta.get("doc_id") == doc_filter)
            if not (src_match or fid_match or did_match):
                continue

        dedup_key = (src_doc, page)
        if dedup_key not in seen_keys:
            seen_keys.add(dedup_key)
            snippet = node.text[:300] + ("…" if len(node.text) > 300 else "")
            sources.append(
                {
                    "page_number": page,
                    "source": src_doc,
                    "text_snippet": snippet,
                    "relevance_score": round(score, 4) if score is not None else None,
                }
            )

        context_blocks.append(f"--- [From Document: {src_doc}, Page: {page}] ---\n{node.text}")
        if len(context_blocks) >= top_k:
            break

    if not context_blocks:
        return QueryResult(
            answer="No relevant content found in the indexed documents to answer this question.",
            sources=[],
        )

    context_str = "\n\n".join(context_blocks)
    prompt = (
        f"DOCUMENT CONTEXT:\n{context_str}\n\n"
        f"USER QUESTION:\n{question}\n\n"
        f"Please provide a complete, well-reasoned answer based strictly on the context above. "
        f"Cite the relevant document name(s) and page number(s) wherever facts are referenced."
    )

    llm = get_llm(api_key=api_key, model_name=model_name)
    try:
        response = llm.complete(prompt)
    except Exception as exc:
        raise RuntimeError(translate_genai_error(exc)) from None

    return QueryResult(
        answer=response.text.strip(),
        sources=sorted(sources, key=lambda s: (s["source"], s["page_number"])),
    )
