"""Tests for daily trends query correctness and admin privacy gating in usage telemetry."""

from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path
import sqlite3
import pytest

from utils.usage_tracker import (
    get_comprehensive_dashboard_metrics,
    reset_api_usage_metrics,
    set_storage_db_path,
)


@pytest.fixture(autouse=True)
def setup_test_db(tmp_path: Path):
    test_db = tmp_path / "test_trends_usage.db"
    set_storage_db_path(test_db)
    reset_api_usage_metrics()
    yield
    reset_api_usage_metrics()


def test_daily_trends_queries_latest_14_days_chronologically(tmp_path: Path):
    """Verify that daily_trends returns the 14 most recent calendar days, ordered chronologically."""
    db_path = str(tmp_path / "test_trends_usage.db")
    conn = sqlite3.connect(db_path)

    # Insert 25 consecutive days of historical usage records
    base_date = datetime(2026, 1, 1)
    with conn:
        for day_offset in range(25):
            day_dt = base_date + timedelta(days=day_offset)
            date_str = day_dt.strftime("%Y-%m-%d")
            ts_str = day_dt.strftime("%Y-%m-%d 12:00:00")
            # Insert 2 requests per day
            conn.execute(
                """
                INSERT INTO api_requests (timestamp, date, request_type, model_name, status, is_429, latency_seconds)
                VALUES (?, ?, 'chat', 'models/gemini-3.5-flash-lite', 'success', 0, 0.5)
                """,
                (ts_str, date_str),
            )
    conn.close()

    metrics = get_comprehensive_dashboard_metrics(is_admin=True)
    trends = metrics["daily_trends"]

    # Must return exactly 14 days
    assert len(trends) == 14

    trend_dates = [t["date"] for t in trends]

    # Must be ordered chronologically (oldest of the 14 to newest)
    assert trend_dates == sorted(trend_dates)

    # The newest date must be the 25th day (offset 24: 2026-01-25)
    assert trend_dates[-1] == "2026-01-25"

    # The oldest date in the 14-day window must be offset 11: 2026-01-12
    assert trend_dates[0] == "2026-01-12"


def test_unauthenticated_visitor_cannot_view_raw_recent_errors(tmp_path: Path):
    """Verify that when is_admin is False, recent_errors are strictly omitted."""
    db_path = str(tmp_path / "test_trends_usage.db")
    conn = sqlite3.connect(db_path)
    with conn:
        conn.execute(
            """
            INSERT INTO api_requests (timestamp, date, request_type, model_name, status, is_429, latency_seconds, error_message)
            VALUES ('2026-01-01 12:00:00', '2026-01-01', 'chat', 'models/gemini-3.5-flash-lite', 'error', 0, 0.5, 'Sensitive internal database error message')
            """
        )
    conn.close()

    # Unauthenticated query
    public_metrics = get_comprehensive_dashboard_metrics(is_admin=False)
    assert public_metrics["recent_errors"] == []

    # Admin query
    admin_metrics = get_comprehensive_dashboard_metrics(is_admin=True)
    assert len(admin_metrics["recent_errors"]) == 1
    assert "Sensitive internal database error message" in admin_metrics["recent_errors"][0]["message"]
