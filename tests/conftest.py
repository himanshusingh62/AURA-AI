"""Pytest global configuration and fixtures for AURA.AI.

Ensures every test runs against an isolated temporary SQLite telemetry database
and restores global configuration on teardown.
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

import pytest

from config.settings import USAGE_DB_PATH
from utils.usage_tracker import reset_api_usage_metrics, set_storage_db_path


@pytest.fixture(autouse=True)
def global_isolated_test_db():
    """Isolate SQLite telemetry database for all tests to ensure complete test independence."""
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as tmp_f:
        tmp_path = Path(tmp_f.name)

    set_storage_db_path(tmp_path)
    reset_api_usage_metrics()

    # Allow unit and mocked tests to execute freely without hitting low default limits
    old_limit = os.environ.get("INTERNAL_DAILY_REQUEST_LIMIT")
    os.environ["INTERNAL_DAILY_REQUEST_LIMIT"] = "5000"

    yield

    if old_limit is not None:
        os.environ["INTERNAL_DAILY_REQUEST_LIMIT"] = old_limit
    else:
        os.environ.pop("INTERNAL_DAILY_REQUEST_LIMIT", None)

    reset_api_usage_metrics()
    set_storage_db_path(USAGE_DB_PATH)
    try:
        tmp_path.unlink(missing_ok=True)
    except OSError:
        pass
