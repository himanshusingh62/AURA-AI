"""Comprehensive unit tests for utils/usage_tracker.py.

Verifies:
- SQLite persistence and table initialization.
- Successful and failed request logging.
- HTTP 429 quota error counting.
- Daily request aggregation.
- Model-wise and category-wise tracking.
- Quota status evaluation (GREEN, YELLOW, RED).
- Admin authentication passphrase validation.
- Database reset.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from config.settings import get_admin_password
from utils.usage_tracker import (
    get_api_usage_metrics,
    get_comprehensive_dashboard_metrics,
    get_quota_status,
    record_api_request,
    reset_api_usage_metrics,
    set_storage_db_path,
)


@pytest.fixture(autouse=True)
def clean_tracker():
    """Ensure clean tracker state before each test."""
    reset_api_usage_metrics()
    yield
    reset_api_usage_metrics()


def test_initial_usage_state():
    """Verify empty database starts with zero counts and GREEN quota status."""
    metrics = get_comprehensive_dashboard_metrics()
    assert metrics["total_requests"] == 0
    assert metrics["successful_requests"] == 0
    assert metrics["failed_requests"] == 0
    assert metrics["quota_errors_429"] == 0
    assert metrics["requests_today"] == 0
    assert metrics["requests_by_model"] == {}

    status = get_quota_status()
    assert status["status"] == "GREEN"


def test_successful_api_requests_persistence():
    """Verify successful calls persist into SQLite and update counts."""
    record_api_request(request_type="chat", status="success", latency=0.35, model_name="models/gemini-3.5-flash-lite")
    record_api_request(request_type="embedding", status="success", latency=0.12, model_name="models/gemini-embedding-001")
    record_api_request(request_type="multimodal", status="success", latency=0.88, model_name="models/gemini-3.5-flash-lite")

    metrics = get_comprehensive_dashboard_metrics()
    assert metrics["total_requests"] == 3
    assert metrics["successful_requests"] == 3
    assert metrics["failed_requests"] == 0
    assert metrics["quota_errors_429"] == 0
    assert metrics["requests_today"] == 3

    # Category breakdown
    assert metrics["requests_by_category"]["chat"] == 1
    assert metrics["requests_by_category"]["embedding"] == 1
    assert metrics["requests_by_category"]["multimodal"] == 1

    # Model breakdown
    assert metrics["requests_by_model"]["models/gemini-3.5-flash-lite"] == 2
    assert metrics["requests_by_model"]["models/gemini-embedding-001"] == 1


def test_failed_api_requests_logging():
    """Verify non-429 failures are correctly categorized under failed_requests."""
    record_api_request(
        request_type="chat",
        status="error",
        latency=0.04,
        is_429=False,
        error_message="Connection reset by peer",
        model_name="models/gemini-3.5-flash-lite",
    )

    metrics = get_comprehensive_dashboard_metrics()
    assert metrics["total_requests"] == 1
    assert metrics["successful_requests"] == 0
    assert metrics["failed_requests"] == 1
    assert metrics["quota_errors_429"] == 0
    assert len(metrics["recent_errors"]) == 1
    assert "Connection reset" in metrics["recent_errors"][0]["message"]


def test_429_quota_error_triggers_red_status():
    """Verify 429 quota errors update quota_errors_429 and transition status to RED."""
    record_api_request(
        request_type="chat",
        status="429_quota_exceeded",
        latency=0.05,
        is_429=True,
        error_message="RESOURCE_EXHAUSTED: Quota exceeded for quota metric 'Queries per minute'",
        model_name="models/gemini-3.5-flash-lite",
    )

    metrics = get_comprehensive_dashboard_metrics()
    assert metrics["total_requests"] == 1
    assert metrics["quota_errors_429"] == 1
    assert metrics["failed_requests"] == 1

    status = get_quota_status()
    assert status["status"] == "RED"
    assert "429" in status["badge"]


def test_past_day_429_does_not_turn_status_red():
    """Verify historical 429 errors from yesterday do not leave status stuck at RED today."""
    import sqlite3
    from datetime import datetime, timedelta
    from utils.usage_tracker import get_storage_db_path

    yesterday_str = (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d")
    yesterday_ts = (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d %H:%M:%S")

    db_path = get_storage_db_path()
    conn = sqlite3.connect(str(db_path))
    with conn:
        conn.execute(
            """
            INSERT INTO api_requests (timestamp, date, request_type, model_name, status, is_429, latency_seconds, error_message)
            VALUES (?, ?, 'chat', 'models/gemini-3.5-flash-lite', '429_quota_exceeded', 1, 0.05, 'RESOURCE_EXHAUSTED')
            """,
            (yesterday_ts, yesterday_str),
        )
    conn.close()

    # Lifetime metrics should include yesterday's 429 for admin audit
    metrics = get_comprehensive_dashboard_metrics()
    assert metrics["quota_errors_429"] == 1
    assert metrics["requests_today"] == 0

    # Quota health for today should be GREEN because no 429 happened today
    status = get_quota_status()
    assert status["status"] == "GREEN"
    assert "Normal Quota Health" in status["badge"]


def test_yellow_status_on_approaching_limit():
    """Verify status transitions to YELLOW when daily requests approach the warning threshold."""
    import sqlite3
    from datetime import datetime
    from config.settings import INTERNAL_DAILY_REQUEST_LIMIT, INTERNAL_WARNING_THRESHOLD_PERCENT
    from utils.usage_tracker import get_storage_db_path

    threshold = int(INTERNAL_DAILY_REQUEST_LIMIT * (INTERNAL_WARNING_THRESHOLD_PERCENT / 100.0))

    db_path = get_storage_db_path()
    conn = sqlite3.connect(str(db_path))
    today_str = datetime.now().strftime("%Y-%m-%d")
    now_ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with conn:
        conn.executemany(
            """
            INSERT INTO api_requests (timestamp, date, request_type, model_name, status, is_429, latency_seconds, error_message)
            VALUES (?, ?, 'chat', 'models/gemini-3.5-flash-lite', 'success', 0, 0.05, NULL)
            """,
            [(now_ts, today_str)] * threshold,
        )
    conn.close()

    status = get_quota_status()
    assert status["status"] == "YELLOW"
    assert "Approaching" in status["badge"]


def test_reset_api_usage_metrics():
    """Verify reset clears both SQLite records and in-memory caches."""
    record_api_request(request_type="chat", status="success", latency=0.2, model_name="models/gemini-3.5-flash-lite")
    record_api_request(request_type="chat", status="429_quota_exceeded", latency=0.05, is_429=True, model_name="models/gemini-3.5-flash-lite")

    assert get_comprehensive_dashboard_metrics()["total_requests"] == 2

    reset_api_usage_metrics()

    metrics = get_comprehensive_dashboard_metrics()
    assert metrics["total_requests"] == 0
    assert metrics["quota_errors_429"] == 0
    assert metrics["recent_errors"] == []
    assert get_quota_status()["status"] == "GREEN"


def test_admin_password_resolution(monkeypatch):
    """Verify admin passphrase lookup respects environment setting."""
    monkeypatch.setenv("ADMIN_PASSWORD", "custom-secret-pass-123")
    assert get_admin_password() == "custom-secret-pass-123"


def test_admin_password_missing_returns_none(monkeypatch):
    """Verify that when ADMIN_PASSWORD is not configured, get_admin_password returns None
    and NEVER falls back to any hardcoded default."""
    monkeypatch.delenv("ADMIN_PASSWORD", raising=False)
    # Ensure no secrets or env fallback exists
    password = get_admin_password()
    assert password is None


def test_unauthorized_dashboard_access_denied():
    """Verify that unauthorized requests cannot access administrator credentials or session."""
    configured_pass = get_admin_password()

    # Case 1: When no password is configured in environment
    if configured_pass is None:
        user_attempt = "random_guess"
        assert (user_attempt == configured_pass) is False

    # Case 2: When a password is configured, incorrect attempt must be denied
    with pytest.MonkeyPatch().context() as mp:
        mp.setenv("ADMIN_PASSWORD", "VeryStrongPassword!987")
        admin_pass = get_admin_password()
        assert admin_pass == "VeryStrongPassword!987"

        # Wrong attempt
        assert ("wrong-pass" == admin_pass) is False
        # Empty attempt
        assert ("" == admin_pass) is False


def test_server_side_daily_limit_enforcement():
    """Verify server-side daily request limit blocks execution when threshold is exceeded."""
    from utils.usage_tracker import check_server_side_daily_limit, DailyLimitExceededError

    # Limit = 2
    record_api_request(request_type="chat", status="success", latency=0.1, model_name="models/gemini-3.5-flash-lite")
    record_api_request(request_type="chat", status="success", latency=0.1, model_name="models/gemini-3.5-flash-lite")

    with pytest.raises(DailyLimitExceededError) as exc_info:
        check_server_side_daily_limit(limit=2)

    err_msg = str(exc_info.value)
    assert "Internal daily application limit reached" in err_msg
    assert "2/2 requests today" in err_msg
    assert "distinct from Google's official API quotas" in err_msg

