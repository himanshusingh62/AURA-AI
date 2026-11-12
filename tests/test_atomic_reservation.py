"""Tests for atomic request reservation and race-condition prevention in SQLite telemetry."""

from __future__ import annotations

import concurrent.futures
from pathlib import Path
import pytest

from utils.usage_tracker import (
    DailyLimitExceededError,
    cancel_reservation,
    complete_api_request,
    get_requests_today_count,
    reserve_api_request,
    reset_api_usage_metrics,
    set_storage_db_path,
)


@pytest.fixture(autouse=True)
def setup_test_db(tmp_path: Path):
    test_db = tmp_path / "test_atomic_usage.db"
    set_storage_db_path(test_db)
    reset_api_usage_metrics()
    yield
    reset_api_usage_metrics()


def test_concurrent_reservations_enforce_exact_daily_limit():
    """Verify that concurrent threads attempting to reserve slots adhere strictly to limit."""
    limit = 5
    concurrency = 20

    success_ids = []
    exceeded_count = 0

    def worker(idx: int):
        try:
            res_id = reserve_api_request("chat", "models/gemini-3.5-flash-lite", limit=limit)
            return ("success", res_id)
        except DailyLimitExceededError:
            return ("exceeded", None)

    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
        futures = [executor.submit(worker, i) for i in range(concurrency)]
        for fut in concurrent.futures.as_completed(futures):
            status, res_id = fut.result()
            if status == "success":
                success_ids.append(res_id)
            else:
                exceeded_count += 1

    assert len(success_ids) == limit
    assert exceeded_count == concurrency - limit
    assert get_requests_today_count() == limit


def test_cancel_reservation_releases_slot():
    """Verify that cancel_reservation releases a reserved slot for another request."""
    limit = 2
    res_id_1 = reserve_api_request("chat", "models/gemini-3.5-flash-lite", limit=limit)
    res_id_2 = reserve_api_request("chat", "models/gemini-3.5-flash-lite", limit=limit)

    # Limit reached
    with pytest.raises(DailyLimitExceededError):
        reserve_api_request("chat", "models/gemini-3.5-flash-lite", limit=limit)

    # Cancel reservation 1
    cancel_reservation(res_id_1, reason="Network client closed before transmission")

    # Now a new reservation should succeed
    res_id_3 = reserve_api_request("chat", "models/gemini-3.5-flash-lite", limit=limit)
    assert res_id_3 > 0
    assert get_requests_today_count() == 2


def test_failed_request_does_not_release_slot():
    """Verify that completed error or 429 requests are not released because they reached the API."""
    limit = 2
    res_id_1 = reserve_api_request("chat", "models/gemini-3.5-flash-lite", limit=limit)
    complete_api_request(res_id_1, status="error", latency=0.2, error_message="API error")

    res_id_2 = reserve_api_request("chat", "models/gemini-3.5-flash-lite", limit=limit)
    complete_api_request(res_id_2, status="429_quota_exceeded", latency=0.1, is_429=True)

    # Both slots remain consumed
    assert get_requests_today_count() == 2
    with pytest.raises(DailyLimitExceededError):
        reserve_api_request("chat", "models/gemini-3.5-flash-lite", limit=limit)
