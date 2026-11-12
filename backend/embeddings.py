"""
Embeddings Layer for AURA.AI using the modern official google-genai SDK.
Implements a native BaseEmbedding provider for LlamaIndex without any legacy dependencies.

Includes:
- Bounded LRU Embedding Cache with TTL and tenant isolation.
- Atomic server-side daily request reservation.
- Robust multi-document index construction and document node removal.
"""

from __future__ import annotations

import collections
import hashlib
import logging
import threading
import time
from typing import Any

from google import genai
from google.genai import errors
from llama_index.core import VectorStoreIndex
from llama_index.core.base.embeddings.base import BaseEmbedding
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.schema import Document
from pydantic import Field, PrivateAttr

logger = logging.getLogger(__name__)

from backend.gemini_client import get_genai_client, resolve_api_key, translate_genai_error
from backend.rate_limit import parse_quota_error
from config.settings import (
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    EMBEDDING_CACHE_TTL_SECONDS,
    GEMINI_EMBEDDING_MODEL,
    MAX_EMBEDDING_CACHE_SIZE,
)
from utils.helpers import sanitize_error_message
from utils.usage_tracker import complete_api_request, record_api_request, reserve_api_request

# Maximum texts to embed in a single batch request to Gemini API
EMBEDDING_BATCH_SIZE: int = 64
MAX_RETRIES: int = 2
INITIAL_BACKOFF_SECONDS: float = 1.5


def _compute_key_fingerprint(api_key: str | None) -> str:
    """Compute a one-way 16-character SHA-256 fingerprint for tenant cache isolation.
    Raw API keys are NEVER retained or stored in cache keys.
    """
    if api_key and api_key.strip():
        return hashlib.sha256(api_key.strip().encode("utf-8")).hexdigest()[:16]
    return "default_tenant"


def _hash_text(model_name: str, text: str, api_key_fingerprint: str = "default_tenant") -> str:
    """Compute deterministic cache key incorporating model name, tenant fingerprint, and text hash."""
    chunk_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()
    return f"{model_name}:{api_key_fingerprint}:{chunk_hash}"


class EmbeddingLRUCache:
    """Thread-safe bounded LRU cache for text embeddings with optional TTL and eviction."""

    def __init__(
        self,
        max_size: int = MAX_EMBEDDING_CACHE_SIZE,
        default_ttl_seconds: float | None = EMBEDDING_CACHE_TTL_SECONDS,
    ) -> None:
        self.max_size = max(1, max_size)
        self.default_ttl_seconds = max(0.0, default_ttl_seconds) if default_ttl_seconds is not None else 0.0
        self._cache: collections.OrderedDict[str, tuple[list[float], float, float]] = collections.OrderedDict()
        self._lock = threading.Lock()
        self._hits: int = 0
        self._misses: int = 0

    def get(self, key: str) -> list[float] | None:
        """Retrieve embedding if present and unexpired."""
        with self._lock:
            if key not in self._cache:
                self._misses += 1
                return None

            embedding, inserted_at, ttl = self._cache[key]
            now = time.time()
            if ttl > 0.0 and (now - inserted_at) > ttl:
                del self._cache[key]
                self._misses += 1
                return None

            self._cache.move_to_end(key)
            self._hits += 1
            return list(embedding)

    def set(self, key: str, embedding: list[float], ttl: float | None = None) -> None:
        """Store embedding in LRU cache with eviction if capacity exceeded."""
        effective_ttl = self.default_ttl_seconds if ttl is None else max(0.0, ttl)
        with self._lock:
            if key in self._cache:
                self._cache.move_to_end(key)
            self._cache[key] = (list(embedding), time.time(), effective_ttl)
            if len(self._cache) > self.max_size:
                self._cache.popitem(last=False)

    def clear(self) -> None:
        """Clear all cached embeddings."""
        with self._lock:
            self._cache.clear()
            self._hits = 0
            self._misses = 0

    def stats(self) -> dict[str, Any]:
        """Return cache statistics."""
        with self._lock:
            return {
                "cached_chunks": len(self._cache),
                "max_size": self.max_size,
                "hits": self._hits,
                "misses": self._misses,
            }


# Global thread-safe bounded embedding cache
_GLOBAL_EMBED_CACHE = EmbeddingLRUCache(
    max_size=MAX_EMBEDDING_CACHE_SIZE,
    default_ttl_seconds=EMBEDDING_CACHE_TTL_SECONDS,
)


def clear_embedding_cache() -> None:
    """Clear cached embeddings (useful for tests or cache reset)."""
    _GLOBAL_EMBED_CACHE.clear()


def get_embedding_cache_stats() -> dict[str, Any]:
    """Retrieve cache size and hit/miss statistics."""
    return _GLOBAL_EMBED_CACHE.stats()


class GoogleGenAIEmbedding(BaseEmbedding):
    """Native LlamaIndex Embedding implementation powered by the official google-genai SDK."""

    model_name: str = Field(default=GEMINI_EMBEDDING_MODEL)
    _client: genai.Client = PrivateAttr()
    _api_key_fingerprint: str = PrivateAttr(default="default_tenant")

    def __init__(
        self,
        api_key: str | None = None,
        model_name: str = GEMINI_EMBEDDING_MODEL,
        **kwargs: Any,
    ) -> None:
        super().__init__(model_name=model_name, **kwargs)
        resolved_key = resolve_api_key(api_key)
        self._api_key_fingerprint = _compute_key_fingerprint(resolved_key)
        self._client = get_genai_client(api_key=resolved_key)

    def _call_embed_with_retry(self, contents: str | list[str]) -> list[list[float]]:
        """Call client.models.embed_content with atomic daily request reservation and retry logic."""
        # Atomically reserve a daily request slot before network dispatch
        reservation_id = reserve_api_request("embedding", self.model_name)

        backoff = INITIAL_BACKOFF_SECONDS
        last_error_msg: str | None = None

        for attempt in range(1, MAX_RETRIES + 1):
            t0 = time.time()
            try:
                response = self._client.models.embed_content(
                    model=self.model_name,
                    contents=contents,
                )
                if not response.embeddings:
                    raise RuntimeError("Gemini returned empty embeddings.")
                latency = time.time() - t0
                complete_api_request(
                    reservation_id=reservation_id,
                    status="success",
                    latency=latency,
                    model_name=self.model_name,
                    request_type="embedding",
                )
                return [emb.values for emb in response.embeddings]
            except errors.ClientError as exc:
                latency = time.time() - t0
                quota_info = parse_quota_error(exc, model_name=self.model_name)
                last_error_msg = quota_info.friendly_message

                if quota_info.is_429:
                    complete_api_request(
                        reservation_id=reservation_id,
                        status="429_quota_exceeded",
                        latency=latency,
                        is_429=True,
                        error_message=quota_info.friendly_message,
                        model_name=self.model_name,
                        request_type="embedding",
                    )
                    # Never retry if daily quota is spent
                    if quota_info.is_daily_limit:
                        raise RuntimeError(quota_info.friendly_message) from None

                    delay = quota_info.retry_delay_seconds
                    if delay is not None and delay <= 6.0 and attempt < MAX_RETRIES:
                        time.sleep(delay)
                        continue
                    elif delay is None and attempt < MAX_RETRIES:
                        time.sleep(backoff)
                        backoff *= 2.0
                        continue
                    else:
                        raise RuntimeError(quota_info.friendly_message) from None
                else:
                    complete_api_request(
                        reservation_id=reservation_id,
                        status="error",
                        latency=latency,
                        error_message=str(exc),
                        model_name=self.model_name,
                        request_type="embedding",
                    )
                    raise RuntimeError(translate_genai_error(exc, model_name=self.model_name)) from None
            except Exception as exc:
                latency = time.time() - t0
                quota_info = parse_quota_error(exc, model_name=self.model_name)
                if quota_info.is_429:
                    complete_api_request(
                        reservation_id=reservation_id,
                        status="429_quota_exceeded",
                        latency=latency,
                        is_429=True,
                        error_message=quota_info.friendly_message,
                        model_name=self.model_name,
                        request_type="embedding",
                    )
                    if quota_info.is_daily_limit or attempt >= MAX_RETRIES:
                        raise RuntimeError(quota_info.friendly_message) from None
                    time.sleep(backoff)
                    backoff *= 2.0
                    continue

                complete_api_request(
                    reservation_id=reservation_id,
                    status="error",
                    latency=latency,
                    error_message=str(exc),
                    model_name=self.model_name,
                    request_type="embedding",
                )
                raise RuntimeError(translate_genai_error(exc, model_name=self.model_name)) from None

        raise RuntimeError(last_error_msg or f"Embedding generation failed after {MAX_RETRIES} attempts.")

    def _get_query_embedding(self, query: str) -> list[float]:
        """Embed a single search query, with caching to eliminate duplicate API requests."""
        h = _hash_text(self.model_name, query, self._api_key_fingerprint)
        cached = _GLOBAL_EMBED_CACHE.get(h)
        if cached is not None:
            return cached

        results = self._call_embed_with_retry(query)
        embedding = results[0]
        _GLOBAL_EMBED_CACHE.set(h, embedding)
        return embedding

    def _get_text_embedding(self, text: str) -> list[float]:
        """Embed a single chunk of text, using LRU cache to prevent duplicate calls."""
        h = _hash_text(self.model_name, text, self._api_key_fingerprint)
        cached = _GLOBAL_EMBED_CACHE.get(h)
        if cached is not None:
            return cached

        results = self._call_embed_with_retry(text)
        embedding = results[0]
        _GLOBAL_EMBED_CACHE.set(h, embedding)
        return embedding

    def _get_text_embeddings(self, texts: list[str]) -> list[list[float]]:
        """Embed multiple text chunks in batches, reusing cached embeddings to minimize API quota."""
        if not texts:
            return []

        # 1. Identify cached vs uncached texts
        results: list[list[float] | None] = [None] * len(texts)
        missing_indices: list[int] = []
        missing_texts: list[str] = []

        for idx, text in enumerate(texts):
            h = _hash_text(self.model_name, text, self._api_key_fingerprint)
            cached = _GLOBAL_EMBED_CACHE.get(h)
            if cached is not None:
                results[idx] = cached
            else:
                missing_indices.append(idx)
                missing_texts.append(text)

        # If all texts were cached, return immediately without any API calls!
        if not missing_texts:
            return [res for res in results if res is not None]

        # 2. Embed only the missing texts in batches
        all_new_embeddings: list[list[float]] = []
        for i in range(0, len(missing_texts), EMBEDDING_BATCH_SIZE):
            batch = missing_texts[i : i + EMBEDDING_BATCH_SIZE]
            batch_embeddings = self._call_embed_with_retry(batch)
            all_new_embeddings.extend(batch_embeddings)

        # 3. Store new embeddings in cache and assemble full results
        for missing_idx, text, emb in zip(missing_indices, missing_texts, all_new_embeddings):
            h = _hash_text(self.model_name, text, self._api_key_fingerprint)
            _GLOBAL_EMBED_CACHE.set(h, emb)
            results[missing_idx] = emb

        return [res for res in results if res is not None]

    async def _aget_query_embedding(self, query: str) -> list[float]:
        """Asynchronous query embedding fallback."""
        return self._get_query_embedding(query)

    async def _aget_text_embedding(self, text: str) -> list[float]:
        """Asynchronous text chunk embedding fallback."""
        return self._get_text_embedding(text)


def get_embedding_model(api_key: str | None = None, model_name: str | None = None) -> GoogleGenAIEmbedding:
    """Instantiate a GoogleGenAIEmbedding model with explicit key and model support."""
    return GoogleGenAIEmbedding(api_key=api_key, model_name=model_name or GEMINI_EMBEDDING_MODEL)


def build_index(
    documents: list[Document],
    api_key: str | None = None,
    existing_index: VectorStoreIndex | None = None,
    embedding_model: str | None = None,
) -> VectorStoreIndex:
    """Chunk documents, generate 3,072-dim embeddings via google-genai, and return a VectorStoreIndex.

    If existing_index is passed, new document nodes are inserted into the existing index,
    allowing seamless multi-document retrieval across multiple uploaded PDFs.
    """
    if not documents:
        raise ValueError("Cannot build index from an empty list of documents.")

    embed_model = get_embedding_model(api_key=api_key, model_name=embedding_model)

    splitter = SentenceSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )

    try:
        nodes = splitter.get_nodes_from_documents(documents, show_progress=False)

        if existing_index is not None:
            # Multi-document append to unified vector store
            existing_index.insert_nodes(nodes)
            return existing_index
        else:
            index = VectorStoreIndex(
                nodes=nodes,
                embed_model=embed_model,
                show_progress=False,
            )
            return index
    except Exception as exc:
        raise RuntimeError(f"Embedding generation error: {sanitize_error_message(exc)}") from None


def remove_document_from_index(
    index: VectorStoreIndex | None,
    filename: str,
    doc_ids: list[str] | None = None,
    remaining_documents: list[Document] | None = None,
    api_key: str | None = None,
) -> VectorStoreIndex | None:
    """Remove all nodes and vectors belonging to a specific PDF filename from the vector index.

    Ensures that content from the removed document can never appear in subsequent retrieval results.
    If direct node deletion fails or encounters an issue, safely rebuilds the index
    using only the remaining active documents.

    Returns:
        The updated VectorStoreIndex, or None if no documents remain.
    """
    if index is None:
        return None

    try:
        # 1. Identify all doc_ids and node_ids to delete
        target_doc_ids = set(doc_ids or [])

        # Inspect docstore for any nodes where metadata['source'] == filename or file_id == filename
        node_ids_to_delete = []
        if hasattr(index, "docstore") and hasattr(index.docstore, "docs"):
            for node_id, node in list(index.docstore.docs.items()):
                meta = getattr(node, "metadata", {}) or {}
                if meta.get("source") == filename or meta.get("file_id") == filename or meta.get("doc_id") == filename:
                    node_ids_to_delete.append(node_id)
                    ref_id = getattr(node, "ref_doc_id", None)
                    if ref_id:
                        target_doc_ids.add(ref_id)

        # 2. Delete ref documents from the index
        for ref_id in target_doc_ids:
            try:
                index.delete_ref_doc(ref_id, delete_from_docstore=True)
            except Exception as e:
                logger.warning("delete_ref_doc failed for ref_id %s: %s", ref_id, e)

        # 3. Delete any remaining individual nodes
        if node_ids_to_delete:
            try:
                index.delete_nodes(node_ids_to_delete, delete_from_docstore=True)
            except Exception as e:
                logger.warning("delete_nodes failed for nodes %s: %s", node_ids_to_delete, e)

        # Ensure underlying vector store structures are fully purged
        if hasattr(index, "vector_store"):
            vs = index.vector_store
            if hasattr(vs, "_data"):
                meta_dict = getattr(vs._data, "metadata_dict", {})
                lingering_vs = [
                    tid
                    for tid, m in list(meta_dict.items())
                    if m.get("source") == filename or m.get("file_id") == filename or m.get("doc_id") == filename
                ]
                for tid in lingering_vs:
                    try:
                        vs.delete(tid)
                    except Exception:
                        pass

        # 4. Check remaining node count
        remaining_count = len(index.docstore.docs) if hasattr(index, "docstore") else 0
        if remaining_count == 0 or (hasattr(index, "vector_store") and hasattr(index.vector_store, "_data") and len(index.vector_store._data.embedding_dict) == 0):
            return None

        # Verify that no nodes with this filename remain
        if hasattr(index, "docstore") and hasattr(index.docstore, "docs"):
            lingering = [
                n_id
                for n_id, n in list(index.docstore.docs.items())
                if (getattr(n, "metadata", {}) or {}).get("source") == filename
            ]
            if lingering:
                logger.warning("Lingering nodes found after deletion for %s: %s", filename, lingering)
                for n_id in lingering:
                    try:
                        index.docstore.delete_document(n_id)
                    except Exception as e:
                        logger.warning("docstore.delete_document failed for %s: %s", n_id, e)

                # If still lingering after direct docstore deletion and remaining_docs exists, rebuild
                still_lingering = [
                    n_id
                    for n_id, n in list(index.docstore.docs.items())
                    if (getattr(n, "metadata", {}) or {}).get("source") == filename
                ]
                if still_lingering and remaining_documents is not None:
                    return build_index(documents=remaining_documents, api_key=api_key)

        return index
    except Exception as exc:
        logger.warning("Error during document removal from index: %s", exc)
        # Fallback to full rebuild if node deletion is unsupported
        if remaining_documents:
            return build_index(documents=remaining_documents, api_key=api_key)
        return None
