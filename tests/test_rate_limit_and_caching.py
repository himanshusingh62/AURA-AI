"""Unit tests for rate limit parsing, 429 quota handling, backoff, and embedding caching.

Verifies:
1. Detection of HTTP 429 RESOURCE_EXHAUSTED errors.
2. Extraction of retry_delay from RPC details, headers, and text regex.
3. Distinction between daily quota exhaustion vs temporary per-minute rate limits.
4. Prevention of repeated retries when daily quota is spent.
5. Embedding caching (ensuring identical chunks are never embedded repeatedly).
6. Never claiming success when 429 is returned.
"""

from __future__ import annotations

import time
from unittest.mock import MagicMock, patch

import pytest
from google.genai import errors

from backend.embeddings import (
    GoogleGenAIEmbedding,
    clear_embedding_cache,
    get_embedding_cache_stats,
)
from backend.rate_limit import (
    QuotaErrorDetails,
    execute_with_quota_guard,
    parse_quota_error,
)
from backend.query_engine import GoogleGenAILLM
from backend.multimodal_engine import _generate_multimodal_content
import tempfile
from pathlib import Path
from config.settings import USAGE_DB_PATH
from utils.usage_tracker import (
    reset_api_usage_metrics,
    get_comprehensive_dashboard_metrics,
)


@pytest.fixture(autouse=True)
def clean_test_env():
    """Ensure clean caches and telemetry before and after each test."""
    clear_embedding_cache()
    reset_api_usage_metrics()
    yield
    clear_embedding_cache()
    reset_api_usage_metrics()


# ── 1. Quota & Retry-Delay Parser Tests ─────────────────────────────────────────

def test_parse_quota_error_per_minute_with_text_delay():
    """Test parsing temporary per-minute rate limit with regex retry delay."""
    exc = errors.ClientError(
        429,
        {"error": {"message": "Resource has been exhausted (check quota). Quota exceeded for 'Queries per minute'. Please retry after 14.5s."}},
        None,
    )
    info: QuotaErrorDetails = parse_quota_error(exc, model_name="models/gemini-3.5-flash-lite")

    assert info.is_429 is True
    assert info.is_daily_limit is False
    assert info.retry_delay_seconds == 14.5
    assert "15 seconds" in info.friendly_message or "14 seconds" in info.friendly_message


@pytest.mark.parametrize(
    "error_substring",
    [
        "generate_content_free_tier_requests",
        "free_tier_requests",
        "requests per day",
        "daily quota",
        "daily limit",
        "GenerateContent free tier requests per day",
    ],
)
def test_parse_quota_error_daily_indicators(error_substring):
    """Verify that every daily quota indicator is correctly detected as daily quota exhaustion."""
    exc = errors.ClientError(
        429,
        {"error": {"message": f"Quota exceeded for metric '{error_substring}' of service 'generativelanguage.googleapis.com'"}},
        None,
    )
    info = parse_quota_error(exc, model_name="models/gemini-3.5-flash-lite")
    assert info.is_429 is True
    assert info.is_daily_limit is True
    assert "daily request quota has been exhausted" in info.friendly_message.lower()


def test_parse_quota_error_generate_content_free_tier_requests_exact():
    """Verify exact Google GenAI error string with generate_content_free_tier_requests."""
    raw_msg = (
        "429 Resource has been exhausted (check quota). Quota exceeded for quota metric "
        "'generate_content_free_tier_requests' and limit 'GenerateContent free tier requests per day' "
        "of service 'generativelanguage.googleapis.com' for consumer 'project_number:12345'."
    )
    exc = errors.ClientError(429, {"error": {"message": raw_msg}}, None)
    info = parse_quota_error(exc, model_name="models/gemini-3.5-flash-lite")

    assert info.is_429 is True
    assert info.is_daily_limit is True
    assert "daily request quota has been exhausted" in info.friendly_message.lower()
    assert "midnight pacific time" in info.friendly_message.lower()


def test_execute_with_quota_guard_free_tier_metric_called_exactly_once():
    """Verify that an API call returning generate_content_free_tier_requests is invoked exactly once with 0 retries."""
    exc = errors.ClientError(
        429,
        {"error": {"message": "Quota exceeded for quota metric 'generate_content_free_tier_requests'"}},
        None,
    )
    fn = MagicMock(side_effect=exc)

    with pytest.raises(RuntimeError) as exc_info:
        execute_with_quota_guard(fn, request_type="chat", model_name="models/gemini-3.5-flash-lite", max_retries=4)

    # Must be called exactly ONCE: zero retries on free-tier daily quota exhaustion
    assert fn.call_count == 1
    assert "daily request quota has been exhausted" in str(exc_info.value).lower()

    # Verify telemetry: failed request is recorded as 429 and NEVER as success
    metrics = get_comprehensive_dashboard_metrics()
    assert metrics["quota_errors_429"] == 1
    assert metrics["failed_requests"] == 1
    assert metrics["successful_requests"] == 0


def test_google_genai_llm_free_tier_metric_called_exactly_once():
    """Verify GoogleGenAILLM does not retry when generate_content_free_tier_requests occurs."""
    mock_client = MagicMock()
    daily_exc = errors.ClientError(
        429,
        {"error": {"message": "Quota exceeded for quota metric 'generate_content_free_tier_requests'"}},
        None,
    )
    mock_client.models.generate_content.side_effect = daily_exc

    with patch("backend.query_engine.get_genai_client", return_value=mock_client):
        llm = GoogleGenAILLM(api_key="AIzaSyMockKeyForLLM1234567890", max_retries=3)

        with pytest.raises(RuntimeError) as exc_info:
            llm.complete("Explain quantum computing")

        # Called exactly once, no retries!
        assert mock_client.models.generate_content.call_count == 1
        assert "daily request quota has been exhausted" in str(exc_info.value).lower()

        # Telemetry check: failed request is NOT recorded as successful
        metrics = get_comprehensive_dashboard_metrics()
        assert metrics["successful_requests"] == 0
        assert metrics["quota_errors_429"] == 1


def test_embedding_free_tier_metric_called_exactly_once():
    """Verify GoogleGenAIEmbedding does not retry on generate_content_free_tier_requests."""
    mock_client = MagicMock()
    daily_exc = errors.ClientError(
        429,
        {"error": {"message": "Quota exceeded for quota metric 'generate_content_free_tier_requests'"}},
        None,
    )
    mock_client.models.embed_content.side_effect = daily_exc

    with patch("backend.embeddings.get_genai_client", return_value=mock_client):
        embedder = GoogleGenAIEmbedding(api_key="AIzaSyMockKeyForCaching1234567890")

        with pytest.raises(RuntimeError) as exc_info:
            embedder._call_embed_with_retry("test chunk")

        # Called exactly once, no retries!
        assert mock_client.models.embed_content.call_count == 1
        assert "daily request quota has been exhausted" in str(exc_info.value).lower()

        metrics = get_comprehensive_dashboard_metrics()
        assert metrics["successful_requests"] == 0
        assert metrics["quota_errors_429"] == 1


def test_multimodal_free_tier_metric_called_exactly_once():
    """Verify _generate_multimodal_content does not retry on generate_content_free_tier_requests."""
    mock_client = MagicMock()
    daily_exc = errors.ClientError(
        429,
        {"error": {"message": "Quota exceeded for quota metric 'generate_content_free_tier_requests'"}},
        None,
    )
    mock_client.models.generate_content.side_effect = daily_exc

    with patch("backend.multimodal_engine.get_genai_client", return_value=mock_client):
        with pytest.raises(RuntimeError) as exc_info:
            _generate_multimodal_content(
                contents=["image_stub"],
                model_name="models/gemini-3.5-flash-lite",
                api_key="AIzaSyMockKeyForMultimodal12345",
                max_retries=3,
            )

        assert mock_client.models.generate_content.call_count == 1
        assert "daily request quota has been exhausted" in str(exc_info.value).lower()

        metrics = get_comprehensive_dashboard_metrics()
        assert metrics["successful_requests"] == 0
        assert metrics["quota_errors_429"] == 1



def test_parse_quota_error_with_rpc_retry_info():
    """Test extraction of retryDelay from Google RPC details array."""
    class MockErrorWithDetails(Exception):
        code = 429
        details = [{"@type": "type.googleapis.com/google.rpc.RetryInfo", "retryDelay": "12s"}]

    exc = MockErrorWithDetails("Resource has been exhausted")
    info = parse_quota_error(exc, model_name="models/gemini-3.5-flash-lite")

    assert info.is_429 is True
    assert info.retry_delay_seconds == 12.0
    assert "12 seconds" in info.friendly_message


def test_parse_quota_error_non_429():
    """Verify non-429 exceptions are not misclassified as quota errors."""
    exc = errors.ClientError(400, {"error": {"message": "API_KEY_INVALID"}}, None)
    info = parse_quota_error(exc)

    assert info.is_429 is False
    assert info.is_daily_limit is False
    assert info.retry_delay_seconds is None


def test_parse_quota_error_rejects_non_429_containing_quota_substring():
    """Verify that a 400 Bad Request mentioning 'quota' or 'rate limit' is NOT classified as 429."""
    exc_400 = errors.ClientError(400, {"error": {"message": "Invalid argument: quota parameter must be positive"}}, None)
    info_400 = parse_quota_error(exc_400)
    assert info_400.is_429 is False
    assert info_400.is_daily_limit is False

    exc_non_429_str = RuntimeError("Disk quota exceeded on local worker node 429")
    info_disk = parse_quota_error(exc_non_429_str)
    assert info_disk.is_429 is False

    # Authentic 429 error with RESOURCE_EXHAUSTED
    exc_real = errors.ClientError(429, {"error": {"message": "RESOURCE_EXHAUSTED: Rate limit exceeded"}}, None)
    info_real = parse_quota_error(exc_real)
    assert info_real.is_429 is True


# ── 2. Retry Guard & Zero-Retry on Daily Quota Tests ──────────────────────────

def test_execute_with_quota_guard_success():
    """Verify clean execution when function succeeds."""
    fn = MagicMock(return_value="response_ok")
    result = execute_with_quota_guard(fn, request_type="chat", model_name="models/gemini-3.5-flash-lite")

    assert result == "response_ok"
    assert fn.call_count == 1
    metrics = get_comprehensive_dashboard_metrics()
    assert metrics["successful_requests"] == 1


def test_execute_with_quota_guard_daily_quota_never_retries():
    """Verify that daily quota errors immediately raise and do not retry."""
    daily_exc = errors.ClientError(
        429,
        {"error": {"message": "Quota exceeded for GenerateContent requests per day"}},
        None,
    )
    fn = MagicMock(side_effect=daily_exc)

    with pytest.raises(RuntimeError) as exc_info:
        execute_with_quota_guard(fn, request_type="chat", model_name="models/gemini-3.5-flash-lite", max_retries=3)

    # CRUCIAL: Must have called fn exactly ONCE. Zero repeated retries on daily quota!
    assert fn.call_count == 1
    assert "daily" in str(exc_info.value).lower()

    metrics = get_comprehensive_dashboard_metrics()
    assert metrics["quota_errors_429"] == 1
    assert metrics["successful_requests"] == 0


def test_execute_with_quota_guard_temporary_rate_limit_backoff():
    """Verify temporary rate limit retries with short backoff when delay is small."""
    temp_exc = errors.ClientError(
        429,
        {"error": {"message": "Queries per minute exceeded. Please retry after 0.1s."}},
        None,
    )
    # Fail once, then succeed
    fn = MagicMock(side_effect=[temp_exc, "recovered_answer"])

    result = execute_with_quota_guard(
        fn, request_type="chat", model_name="models/gemini-3.5-flash-lite", max_retries=2
    )
    assert result == "recovered_answer"
    assert fn.call_count == 2


# ── 3. Embedding Caching Tests ────────────────────────────────────────────────

def test_embedding_cache_eliminates_duplicate_api_calls():
    """Verify that identical text chunks are embedded only once and reused from cache."""
    mock_client = MagicMock()
    # Mock return 3,072-dim embeddings for two chunks
    mock_emb1 = MagicMock(values=[0.1] * 10)
    mock_emb2 = MagicMock(values=[0.2] * 10)
    mock_response = MagicMock(embeddings=[mock_emb1, mock_emb2])
    mock_client.models.embed_content.return_value = mock_response

    with patch("backend.embeddings.get_genai_client", return_value=mock_client):
        embedder = GoogleGenAIEmbedding(api_key="AIzaSyMockKeyForCaching1234567890")

        chunks = ["Machine learning is great.", "Vector databases optimize retrieval."]

        # First call: cache miss, invokes client.models.embed_content
        first_res = embedder._get_text_embeddings(chunks)
        assert len(first_res) == 2
        assert mock_client.models.embed_content.call_count == 1
        assert get_embedding_cache_stats()["cached_chunks"] == 2

        # Second call with the EXACT same chunks: MUST BE A CACHE HIT (0 new API calls!)
        second_res = embedder._get_text_embeddings(chunks)
        assert len(second_res) == 2
        assert first_res == second_res
        # CRUCIAL: Call count must STILL be 1 (zero duplicate API calls)
        assert mock_client.models.embed_content.call_count == 1


def test_embedding_cache_partial_hits():
    """Verify partial cache hits: only uncached chunks are sent to Gemini API."""
    mock_client = MagicMock()
    mock_client.models.embed_content.side_effect = [
        MagicMock(embeddings=[MagicMock(values=[0.1] * 5)]),  # first call: chunk A
        MagicMock(embeddings=[MagicMock(values=[0.2] * 5)]),  # second call: chunk B
    ]

    with patch("backend.embeddings.get_genai_client", return_value=mock_client):
        embedder = GoogleGenAIEmbedding(api_key="AIzaSyMockKeyForCaching1234567890")

        # Call 1: Embed chunk A
        embedder._get_text_embeddings(["Chunk A"])
        assert mock_client.models.embed_content.call_count == 1

        # Call 2: Embed Chunk A and Chunk B together
        res = embedder._get_text_embeddings(["Chunk A", "Chunk B"])
        assert len(res) == 2
        # Only Chunk B needed to be sent to Gemini!
        assert mock_client.models.embed_content.call_count == 2


# ── 4. Mocked 429 Responses on Engines ─────────────────────────────────────────

def test_google_genai_llm_complete_429_handling():
    """Verify GoogleGenAILLM does not claim success and raises friendly error on 429."""
    mock_client = MagicMock()
    daily_exc = errors.ClientError(429, {"error": {"message": "Requests per day exceeded"}}, None)
    mock_client.models.generate_content.side_effect = daily_exc

    with patch("backend.query_engine.get_genai_client", return_value=mock_client):
        llm = GoogleGenAILLM(api_key="AIzaSyMockKeyForLLM1234567890", max_retries=2)

        with pytest.raises(RuntimeError) as exc_info:
            llm.complete("What is artificial intelligence?")

        # Verifies 429 was detected and daily quota was not retried
        assert mock_client.models.generate_content.call_count == 1
        assert "daily request quota has been exhausted" in str(exc_info.value).lower()


def test_generate_multimodal_content_429_handling():
    """Verify multimodal engine handles 429 gracefully with retry advice."""
    mock_client = MagicMock()
    rate_exc = errors.ClientError(
        429, {"error": {"message": "Resource has been exhausted. Please retry after 18s."}}, None
    )
    mock_client.models.generate_content.side_effect = rate_exc

    with patch("backend.multimodal_engine.get_genai_client", return_value=mock_client):
        with pytest.raises(RuntimeError) as exc_info:
            _generate_multimodal_content(
                contents=["image_stub", "question_stub"],
                model_name="models/gemini-3.5-flash-lite",
                api_key="AIzaSyMockKeyForMultimodal12345",
                max_retries=1,
            )

        assert "18 seconds" in str(exc_info.value) or "rate limit reached" in str(exc_info.value).lower()
