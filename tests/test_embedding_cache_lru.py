"""Tests for Bounded LRU Embedding Cache, TTL eviction, and tenant key isolation."""

from __future__ import annotations

import time
from unittest.mock import MagicMock, patch
import pytest

from backend.embeddings import (
    EmbeddingLRUCache,
    GoogleGenAIEmbedding,
    _compute_key_fingerprint,
    _hash_text,
    clear_embedding_cache,
    get_embedding_cache_stats,
)


def test_lru_cache_bounded_eviction():
    """Verify LRU cache does not exceed max_size and evicts least recently used items."""
    cache = EmbeddingLRUCache(max_size=3, default_ttl_seconds=0.0)

    cache.set("k1", [1.0, 0.0])
    cache.set("k2", [2.0, 0.0])
    cache.set("k3", [3.0, 0.0])

    assert cache.stats()["cached_chunks"] == 3

    # Access k1 so k2 becomes the least recently used
    assert cache.get("k1") == [1.0, 0.0]

    # Add k4 -> should evict k2
    cache.set("k4", [4.0, 0.0])

    assert cache.stats()["cached_chunks"] == 3
    assert cache.get("k2") is None  # Evicted!
    assert cache.get("k1") == [1.0, 0.0]  # Kept
    assert cache.get("k3") == [3.0, 0.0]
    assert cache.get("k4") == [4.0, 0.0]


def test_lru_cache_ttl_expiration():
    """Verify expired items return None and are purged from cache."""
    cache = EmbeddingLRUCache(max_size=10, default_ttl_seconds=0.1)

    cache.set("fast_expire", [9.0, 9.0], ttl=0.05)
    assert cache.get("fast_expire") == [9.0, 9.0]

    # Wait for TTL to elapse
    time.sleep(0.08)
    assert cache.get("fast_expire") is None
    # Verify it was removed on access
    assert cache.stats()["cached_chunks"] == 0


def test_tenant_api_key_fingerprint_isolation():
    """Verify different tenant API keys produce isolated cache keys without leaking raw keys."""
    key_a = "AIzaSyFakeKeyTenantAlpha1111111111111"
    key_b = "AIzaSyFakeKeyTenantBeta22222222222222"

    fp_a = _compute_key_fingerprint(key_a)
    fp_b = _compute_key_fingerprint(key_b)

    assert fp_a != fp_b
    assert len(fp_a) == 16
    assert key_a not in fp_a  # Never store raw key

    chunk = "Sample confidential chunk text"
    hash_a = _hash_text("models/gemini-embedding-001", chunk, fp_a)
    hash_b = _hash_text("models/gemini-embedding-001", chunk, fp_b)

    # Cache keys are completely separated
    assert hash_a != hash_b


def test_failed_embedding_call_never_caches():
    """Verify that an API error does not result in invalid cache entries."""
    clear_embedding_cache()
    initial_stats = get_embedding_cache_stats()

    mock_client = MagicMock()
    mock_client.models.embed_content.side_effect = RuntimeError("503 Service Unavailable")

    with patch("backend.embeddings.get_genai_client", return_value=mock_client), \
         patch("backend.embeddings.resolve_api_key", return_value="AIzaSyMockKeyForTestingOnly12345678"):
        embedder = GoogleGenAIEmbedding(api_key="AIzaSyMockKeyForTestingOnly12345678")

        with pytest.raises(RuntimeError):
            embedder._get_text_embedding("This call should fail and never be cached")

    # Cache must remain empty
    stats_after = get_embedding_cache_stats()
    assert stats_after["cached_chunks"] == initial_stats["cached_chunks"]
