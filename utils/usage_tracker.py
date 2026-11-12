"""Centralized API Usage and Quota Telemetry Service for AURA.AI.

Persists Gemini API request events into a SQLite database (data/usage_history.db),
tracks requests by calendar date, model name, and category (PDF/embedding, image, chat, verification),
and enforces atomic daily request limits to prevent race conditions during concurrent use.

Real-time Quota Status:
  - GREEN: Healthy, no recent quota errors, usage well within internal limits.
  - YELLOW: Approaching configured internal request limit.
  - RED: HTTP 429 RESOURCE_EXHAUSTED returned by Gemini API today.

PERSISTENCE ARCHITECTURE:
  - Local / Single-Node: Lightweight SQLite with WAL (Write-Ahead Logging) and busy timeout.
  - Streamlit Community Cloud / Multi-Node Notice: SQLite on ephemeral containers does not
    survive container reboots or synchronize across instances. For permanent multi-instance
    production retention, configure DATABASE_URL (PostgreSQL / Supabase).
"""

from __future__ import annotations

import logging
import os
import sqlite3
import threading
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

from config.settings import (
    INTERNAL_DAILY_REQUEST_LIMIT,
    INTERNAL_WARNING_THRESHOLD_PERCENT,
    USAGE_DB_PATH,
)
from utils.helpers import sanitize_error_message

logger = logging.getLogger(__name__)

_LOCK = threading.Lock()
_ACTIVE_DB_PATH: Path | str = USAGE_DB_PATH
RECENT_429_WINDOW_MINUTES: int = 60


class DailyLimitExceededError(RuntimeError):
    """Raised when internal application daily request limit is reached."""
    pass


# In-memory fast cache and fallback
_GLOBAL_USAGE: dict[str, Any] = {
    "date": datetime.now().strftime("%Y-%m-%d"),
    "total_requests": 0,
    "successful_requests": 0,
    "failed_requests": 0,
    "chat_requests": 0,
    "multimodal_requests": 0,
    "embedding_requests": 0,
    "verification_requests": 0,
    "quota_errors_429": 0,
    "other_errors": 0,
    "requests_today": 0,
    "last_request_time": None,
    "last_latency_seconds": 0.0,
    "last_status": "Idle",
    "last_error_message": None,
}


def _check_or_reset_daily_fallback(today_str: str) -> None:
    """Ensure in-memory fallback daily counter resets when calendar date changes."""
    if _GLOBAL_USAGE.get("date") != today_str:
        _GLOBAL_USAGE["date"] = today_str
        _GLOBAL_USAGE["requests_today"] = 0


def get_configured_daily_limit() -> int:
    """Retrieve the authoritative configured daily request limit from settings or environment."""
    return INTERNAL_DAILY_REQUEST_LIMIT


def get_requests_today_count() -> int:
    """Return the exact number of active requests made today according to the database."""
    today_str = datetime.now().strftime("%Y-%m-%d")
    conn = _get_db_connection()
    if conn is None:
        with _LOCK:
            _check_or_reset_daily_fallback(today_str)
            return _GLOBAL_USAGE.get("requests_today", 0)
    try:
        cur = conn.cursor()
        cur.execute(
            "SELECT COUNT(*) FROM api_requests WHERE date = ? AND status != 'cancelled'",
            (today_str,),
        )
        row = cur.fetchone()
        return row[0] if row else 0
    except Exception as exc:
        logger.error("Error querying requests_today_count from SQLite: %s", exc)
        with _LOCK:
            _check_or_reset_daily_fallback(today_str)
            return _GLOBAL_USAGE.get("requests_today", 0)
    finally:
        conn.close()


def check_server_side_daily_limit(limit: int | None = None) -> None:
    """Authoritative server-side pre-check enforcing INTERNAL_DAILY_REQUEST_LIMIT.
    Raises DailyLimitExceededError if the internal application threshold is exceeded.
    Clearly distinguishes internal limits from Google's official API quotas.
    """
    max_allowed = limit if limit is not None else get_configured_daily_limit()
    count_today = get_requests_today_count()
    if count_today >= max_allowed:
        raise DailyLimitExceededError(
            f"Internal daily application limit reached: {count_today}/{max_allowed} requests today. "
            f"This is an internal application safety limit to prevent accidental quota exhaustion, "
            f"distinct from Google's official API quotas. Administrators can configure "
            f"INTERNAL_DAILY_REQUEST_LIMIT to adjust this threshold."
        )


def reserve_api_request(
    request_type: str = "chat",
    model_name: str = "unknown",
    limit: int | None = None,
) -> int:
    """Atomically reserve a daily request slot in SQLite using an immediate transaction.

    Prevents TOCTOU race conditions when concurrent requests arrive simultaneously.
    Counts reserved and completed requests consistently.

    Raises:
        DailyLimitExceededError: if limit has been reached.

    Returns:
        reservation_id (int): Database record ID for the reservation.
    """
    max_allowed = limit if limit is not None else get_configured_daily_limit()
    today_str = datetime.now().strftime("%Y-%m-%d")
    now_ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    conn = _get_db_connection()
    if conn is None:
        with _LOCK:
            _check_or_reset_daily_fallback(today_str)
            if _GLOBAL_USAGE["requests_today"] >= max_allowed:
                raise DailyLimitExceededError(
                    f"Internal daily application limit reached: {_GLOBAL_USAGE['requests_today']}/{max_allowed} requests today. "
                    f"This is an internal application safety limit to prevent accidental quota exhaustion, "
                    f"distinct from Google's official API quotas."
                )
            _GLOBAL_USAGE["requests_today"] += 1
            _GLOBAL_USAGE["total_requests"] += 1
            return -1

    try:
        cur = conn.cursor()
        cur.execute("BEGIN IMMEDIATE")
        cur.execute(
            "SELECT COUNT(*) FROM api_requests WHERE date = ? AND status != 'cancelled'",
            (today_str,),
        )
        current_count = cur.fetchone()[0] or 0
        if current_count >= max_allowed:
            conn.rollback()
            raise DailyLimitExceededError(
                f"Internal daily application limit reached: {current_count}/{max_allowed} requests today. "
                f"This is an internal application safety limit to prevent accidental quota exhaustion, "
                f"distinct from Google's official API quotas."
            )

        cur.execute(
            """
            INSERT INTO api_requests (
                timestamp, date, request_type, model_name, status, is_429, latency_seconds, error_message
            ) VALUES (?, ?, ?, ?, 'reserved', 0, 0.0, NULL)
            """,
            (now_ts, today_str, request_type, model_name),
        )
        res_id = cur.lastrowid
        conn.commit()

        with _LOCK:
            _check_or_reset_daily_fallback(today_str)
            _GLOBAL_USAGE["total_requests"] += 1
            _GLOBAL_USAGE["requests_today"] = current_count + 1

        return res_id
    except DailyLimitExceededError:
        raise
    except Exception as exc:
        if conn:
            try:
                conn.rollback()
            except Exception:
                pass
        logger.error("Failed to atomically reserve request in SQLite: %s", exc)
        raise RuntimeError(f"Database error during request reservation: {exc}") from exc
    finally:
        conn.close()


def complete_api_request(
    reservation_id: int,
    status: str = "success",
    latency: float = 0.0,
    is_429: bool = False,
    error_message: str | None = None,
    model_name: str | None = None,
    request_type: str | None = None,
) -> None:
    """Complete a reserved request, persisting final status, latency, and error details."""
    now = datetime.now()
    now_ts = now.strftime("%Y-%m-%d %H:%M:%S")
    sanitized_err = sanitize_error_message(error_message) if error_message else None
    effective_is_429 = 1 if (is_429 or status == "429_quota_exceeded") else 0

    with _LOCK:
        _GLOBAL_USAGE["last_request_time"] = now_ts
        _GLOBAL_USAGE["last_latency_seconds"] = round(latency, 2)

        if request_type == "embedding":
            _GLOBAL_USAGE["embedding_requests"] += 1
        elif request_type == "multimodal":
            _GLOBAL_USAGE["multimodal_requests"] += 1
        elif request_type == "verification":
            _GLOBAL_USAGE["verification_requests"] += 1
        elif request_type == "chat":
            _GLOBAL_USAGE["chat_requests"] += 1

        if effective_is_429:
            _GLOBAL_USAGE["quota_errors_429"] += 1
            _GLOBAL_USAGE["failed_requests"] += 1
            _GLOBAL_USAGE["last_status"] = "429 Rate Limit Exceeded"
            _GLOBAL_USAGE["last_error_message"] = sanitized_err
        elif status == "success":
            _GLOBAL_USAGE["successful_requests"] += 1
            _GLOBAL_USAGE["last_status"] = "Success"
            _GLOBAL_USAGE["last_error_message"] = None
        else:
            _GLOBAL_USAGE["other_errors"] += 1
            _GLOBAL_USAGE["failed_requests"] += 1
            _GLOBAL_USAGE["last_status"] = f"Error: {status}"
            _GLOBAL_USAGE["last_error_message"] = sanitized_err

        current_snapshot = dict(_GLOBAL_USAGE)

    if reservation_id > 0:
        conn = _get_db_connection()
        if conn is not None:
            try:
                with conn:
                    conn.execute(
                        """
                        UPDATE api_requests
                        SET status = ?, is_429 = ?, latency_seconds = ?, error_message = ?
                        WHERE id = ?
                        """,
                        (status, effective_is_429, round(latency, 3), sanitized_err, reservation_id),
                    )
            except Exception as exc:
                logger.error("Failed to complete reservation %s in SQLite: %s", reservation_id, exc)
            finally:
                conn.close()

    _sync_to_streamlit_session(current_snapshot)


def cancel_reservation(reservation_id: int, reason: str = "Cancelled before dispatch") -> None:
    """Release a reserved slot ONLY when request was confirmed NOT sent to the external API."""
    if reservation_id <= 0:
        with _LOCK:
            if _GLOBAL_USAGE["requests_today"] > 0:
                _GLOBAL_USAGE["requests_today"] -= 1
            if _GLOBAL_USAGE["total_requests"] > 0:
                _GLOBAL_USAGE["total_requests"] -= 1
        return

    conn = _get_db_connection()
    if conn is not None:
        try:
            with conn:
                conn.execute(
                    "UPDATE api_requests SET status = 'cancelled', error_message = ? WHERE id = ?",
                    (reason, reservation_id),
                )
        except Exception as exc:
            logger.error("Failed to cancel reservation %s in SQLite: %s", reservation_id, exc)
        finally:
            conn.close()

    with _LOCK:
        if _GLOBAL_USAGE["requests_today"] > 0:
            _GLOBAL_USAGE["requests_today"] -= 1
        if _GLOBAL_USAGE["total_requests"] > 0:
            _GLOBAL_USAGE["total_requests"] -= 1


def set_storage_db_path(db_path: Path | str) -> None:
    """Set custom SQLite database path (useful for testing or in-memory runs)."""
    global _ACTIVE_DB_PATH
    with _LOCK:
        _ACTIVE_DB_PATH = db_path
        _init_sqlite_db()


def get_storage_db_path() -> Path | str:
    """Return the currently configured SQLite database path."""
    return _ACTIVE_DB_PATH


def _get_db_connection() -> sqlite3.Connection | None:
    """Open a SQLite connection with timeout, WAL mode, and thread safety."""
    global _ACTIVE_DB_PATH
    try:
        if isinstance(_ACTIVE_DB_PATH, Path):
            _ACTIVE_DB_PATH.parent.mkdir(parents=True, exist_ok=True)
            db_target = str(_ACTIVE_DB_PATH)
        else:
            if _ACTIVE_DB_PATH != ":memory:":
                Path(_ACTIVE_DB_PATH).parent.mkdir(parents=True, exist_ok=True)
            db_target = str(_ACTIVE_DB_PATH)

        conn = sqlite3.connect(db_target, timeout=15.0, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        if _ACTIVE_DB_PATH != ":memory:":
            try:
                conn.execute("PRAGMA journal_mode=WAL")
                conn.execute("PRAGMA busy_timeout=15000")
            except Exception:
                pass
        return conn
    except Exception as exc:
        logger.error("Failed to open SQLite connection to %s: %s", _ACTIVE_DB_PATH, exc)
        return None


def _init_sqlite_db() -> None:
    """Create the api_requests table and indexes if they do not exist."""
    conn = _get_db_connection()
    if conn is None:
        return
    try:
        with conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS api_requests (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    date TEXT NOT NULL,
                    request_type TEXT NOT NULL,
                    model_name TEXT NOT NULL,
                    status TEXT NOT NULL,
                    is_429 INTEGER NOT NULL DEFAULT 0,
                    latency_seconds REAL NOT NULL DEFAULT 0.0,
                    error_message TEXT
                )
                """
            )
            conn.execute("CREATE INDEX IF NOT EXISTS idx_req_date ON api_requests(date)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_req_status ON api_requests(status)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_req_is_429 ON api_requests(is_429)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_req_model ON api_requests(model_name)")
    except Exception as exc:
        logger.error("Failed to initialize usage database schema: %s", exc)
    finally:
        conn.close()


# Initialize database schema upon import
_init_sqlite_db()


def _sync_to_streamlit_session(metrics: dict[str, Any]) -> None:
    """Safely synchronize in-memory metrics to Streamlit session_state if running in Streamlit."""
    try:
        from streamlit.runtime import exists
        if not exists():
            return
        import streamlit as st
        if "api_usage" not in st.session_state:
            st.session_state["api_usage"] = dict(metrics)
        else:
            st.session_state["api_usage"].update(metrics)
    except Exception:
        pass


def record_api_request(
    request_type: str = "chat",
    status: str = "success",
    latency: float = 0.0,
    is_429: bool = False,
    error_message: str | None = None,
    model_name: str = "unknown",
    reservation_id: int | None = None,
) -> None:
    """Record an API request event in SQLite and update in-memory telemetry.
    If reservation_id is provided, updates that reservation record; otherwise inserts a completed record.
    """
    if reservation_id is not None and reservation_id > 0:
        complete_api_request(
            reservation_id=reservation_id,
            status=status,
            latency=latency,
            is_429=is_429,
            error_message=error_message,
            model_name=model_name,
            request_type=request_type,
        )
        return

    now = datetime.now()
    now_ts = now.strftime("%Y-%m-%d %H:%M:%S")
    now_date = now.strftime("%Y-%m-%d")
    sanitized_err = sanitize_error_message(error_message) if error_message else None

    effective_is_429 = 1 if (is_429 or status == "429_quota_exceeded") else 0
    clean_model = model_name or "unknown"

    with _LOCK:
        _check_or_reset_daily_fallback(now_date)
        _GLOBAL_USAGE["total_requests"] += 1
        _GLOBAL_USAGE["requests_today"] += 1
        _GLOBAL_USAGE["last_request_time"] = now_ts
        _GLOBAL_USAGE["last_latency_seconds"] = round(latency, 2)

        if request_type == "embedding":
            _GLOBAL_USAGE["embedding_requests"] += 1
        elif request_type == "multimodal":
            _GLOBAL_USAGE["multimodal_requests"] += 1
        elif request_type == "verification":
            _GLOBAL_USAGE["verification_requests"] += 1
        else:
            _GLOBAL_USAGE["chat_requests"] += 1

        if effective_is_429:
            _GLOBAL_USAGE["quota_errors_429"] += 1
            _GLOBAL_USAGE["failed_requests"] += 1
            _GLOBAL_USAGE["last_status"] = "429 Rate Limit Exceeded"
            _GLOBAL_USAGE["last_error_message"] = sanitized_err
        elif status == "success":
            _GLOBAL_USAGE["successful_requests"] += 1
            _GLOBAL_USAGE["last_status"] = "Success"
            _GLOBAL_USAGE["last_error_message"] = None
        else:
            _GLOBAL_USAGE["other_errors"] += 1
            _GLOBAL_USAGE["failed_requests"] += 1
            _GLOBAL_USAGE["last_status"] = f"Error: {status}"
            _GLOBAL_USAGE["last_error_message"] = sanitized_err

        current_snapshot = dict(_GLOBAL_USAGE)

    conn = _get_db_connection()
    if conn is not None:
        try:
            with conn:
                conn.execute(
                    """
                    INSERT INTO api_requests (
                        timestamp, date, request_type, model_name, status, is_429, latency_seconds, error_message
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        now_ts,
                        now_date,
                        request_type,
                        clean_model,
                        status,
                        effective_is_429,
                        round(latency, 3),
                        sanitized_err,
                    ),
                )
        except Exception as exc:
            logger.error("Failed to insert api_request record to SQLite: %s", exc)
        finally:
            conn.close()

    _sync_to_streamlit_session(current_snapshot)


def get_api_usage_metrics() -> dict[str, Any]:
    """Retrieve fast snapshot of active API usage telemetry."""
    try:
        from streamlit.runtime import exists
        if exists():
            import streamlit as st
            if "api_usage" in st.session_state:
                return dict(st.session_state["api_usage"])
    except Exception:
        pass

    with _LOCK:
        return dict(_GLOBAL_USAGE)


def reset_api_usage_metrics() -> None:
    """Reset session and SQLite telemetry counters."""
    with _LOCK:
        _GLOBAL_USAGE["total_requests"] = 0
        _GLOBAL_USAGE["successful_requests"] = 0
        _GLOBAL_USAGE["failed_requests"] = 0
        _GLOBAL_USAGE["chat_requests"] = 0
        _GLOBAL_USAGE["multimodal_requests"] = 0
        _GLOBAL_USAGE["embedding_requests"] = 0
        _GLOBAL_USAGE["verification_requests"] = 0
        _GLOBAL_USAGE["quota_errors_429"] = 0
        _GLOBAL_USAGE["other_errors"] = 0
        _GLOBAL_USAGE["requests_today"] = 0
        _GLOBAL_USAGE["last_request_time"] = None
        _GLOBAL_USAGE["last_latency_seconds"] = 0.0
        _GLOBAL_USAGE["last_status"] = "Reset / Idle"
        _GLOBAL_USAGE["last_error_message"] = None

        current_snapshot = dict(_GLOBAL_USAGE)

    conn = _get_db_connection()
    if conn is not None:
        try:
            with conn:
                conn.execute("DELETE FROM api_requests")
        except Exception as exc:
            logger.error("Failed to clear api_requests table: %s", exc)
        finally:
            conn.close()

    _sync_to_streamlit_session(current_snapshot)


def get_quota_status() -> dict[str, str]:
    """Evaluate overall quota health:
    - GREEN: No recent quota errors, daily usage well within limit.
    - YELLOW: Approaching internal limit (> 80% threshold).
    - RED: Recent HTTP 429 / RESOURCE_EXHAUSTED error recorded today.
    """
    today_str = datetime.now().strftime("%Y-%m-%d")
    requests_today = get_requests_today_count()
    limit = get_configured_daily_limit()
    warn_threshold = limit * (INTERNAL_WARNING_THRESHOLD_PERCENT / 100.0)

    # Check for recent 429 errors today (not lifetime historical 429s)
    recent_429_count = 0
    conn = _get_db_connection()
    if conn is not None:
        try:
            cur = conn.cursor()
            cur.execute("SELECT COUNT(*) FROM api_requests WHERE is_429 = 1 AND date = ?", (today_str,))
            row = cur.fetchone()
            if row and row[0] is not None:
                recent_429_count = row[0]
        except Exception as exc:
            logger.error("Error querying 429 quota status from SQLite: %s", exc)
        finally:
            conn.close()
    else:
        with _LOCK:
            _check_or_reset_daily_fallback(today_str)
            if _GLOBAL_USAGE.get("last_status") == "429 Rate Limit Exceeded":
                recent_429_count = 1

    if recent_429_count > 0:
        return {
            "status": "RED",
            "badge": "🔴 RESOURCE_EXHAUSTED (429)",
            "color": "#ef4444",
            "description": "Google Gemini API returned a 429 Quota Exceeded error today. Temporary per-minute wait or daily quota refresh required.",
        }

    if requests_today >= warn_threshold:
        return {
            "status": "YELLOW",
            "badge": "🟡 Approaching Internal Limit",
            "color": "#f59e0b",
            "description": f"Today's application volume ({requests_today:,} reqs) is approaching the configured warning threshold ({int(warn_threshold):,} of {limit:,} reqs).",
        }

    return {
        "status": "GREEN",
        "badge": "🟢 Normal Quota Health",
        "color": "#10b981",
        "description": "No quota exhaustion or rate limits detected. Application operating normally within configured limits.",
    }


def get_comprehensive_dashboard_metrics(is_admin: bool | None = None) -> dict[str, Any]:
    """Query database for analytics, latest 14 days historical trends, and model breakdowns.
    Falls back gracefully to in-memory stats if SQLite is temporarily inaccessible.
    Enforces admin-only privacy on detailed error messages.
    """
    admin_authorized = False
    if is_admin is True:
        admin_authorized = True
    elif is_admin is False:
        admin_authorized = False
    else:
        try:
            from streamlit.runtime import exists

            if exists():
                import streamlit as st

                if hasattr(st, "session_state") and st.session_state.get("admin_authenticated") is True:
                    admin_authorized = True
            else:
                admin_authorized = True
        except Exception:
            admin_authorized = True

    today_str = datetime.now().strftime("%Y-%m-%d")
    result: dict[str, Any] = {
        "total_requests": 0,
        "successful_requests": 0,
        "failed_requests": 0,
        "quota_errors_429": 0,
        "requests_today": 0,
        "requests_by_model": {},
        "requests_by_category": {"chat": 0, "multimodal": 0, "embedding": 0, "verification": 0},
        "last_request_time": None,
        "last_latency_seconds": 0.0,
        "last_status": "Idle",
        "daily_trends": [],
        "recent_errors": [],
        "estimated_usage_summary": "0 total requests processed",
    }

    conn = _get_db_connection()
    if conn is None:
        with _LOCK:
            _check_or_reset_daily_fallback(today_str)
            result["total_requests"] = _GLOBAL_USAGE["total_requests"]
            result["successful_requests"] = _GLOBAL_USAGE["successful_requests"]
            result["failed_requests"] = _GLOBAL_USAGE["failed_requests"]
            result["quota_errors_429"] = _GLOBAL_USAGE["quota_errors_429"]
            result["requests_today"] = _GLOBAL_USAGE["requests_today"]
            result["last_request_time"] = _GLOBAL_USAGE["last_request_time"]
            result["last_latency_seconds"] = _GLOBAL_USAGE["last_latency_seconds"]
            result["last_status"] = _GLOBAL_USAGE["last_status"]
            result["requests_by_category"] = {
                "chat": _GLOBAL_USAGE["chat_requests"],
                "multimodal": _GLOBAL_USAGE["multimodal_requests"],
                "embedding": _GLOBAL_USAGE["embedding_requests"],
                "verification": _GLOBAL_USAGE["verification_requests"],
            }
            return result

    try:
        cur = conn.cursor()
        # 1. Total counts across active requests
        cur.execute(
            """
            SELECT COUNT(*),
                   SUM(CASE WHEN status = 'success' THEN 1 ELSE 0 END),
                   SUM(CASE WHEN status NOT IN ('success', 'reserved', 'cancelled') THEN 1 ELSE 0 END),
                   SUM(is_429)
            FROM api_requests
            WHERE status != 'cancelled'
            """
        )
        row = cur.fetchone()
        if row and row[0] is not None:
            result["total_requests"] = row[0] or 0
            result["successful_requests"] = row[1] or 0
            result["failed_requests"] = row[2] or 0
            result["quota_errors_429"] = row[3] or 0

        # 2. Requests today
        cur.execute(
            "SELECT COUNT(*) FROM api_requests WHERE date = ? AND status != 'cancelled'",
            (today_str,),
        )
        today_row = cur.fetchone()
        result["requests_today"] = today_row[0] if today_row else 0

        # 3. Requests by model
        cur.execute(
            """
            SELECT model_name, COUNT(*)
            FROM api_requests
            WHERE status != 'cancelled'
            GROUP BY model_name
            ORDER BY COUNT(*) DESC
            """
        )
        for m_row in cur.fetchall():
            m_name = m_row[0] or "unknown"
            result["requests_by_model"][m_name] = m_row[1]

        # 4. Requests by category
        cur.execute(
            """
            SELECT request_type, COUNT(*)
            FROM api_requests
            WHERE status != 'cancelled'
            GROUP BY request_type
            """
        )
        for c_row in cur.fetchall():
            c_name = c_row[0]
            if c_name in result["requests_by_category"]:
                result["requests_by_category"][c_name] = c_row[1]

        # 5. Last request details
        cur.execute(
            """
            SELECT timestamp, latency_seconds, status
            FROM api_requests
            WHERE status != 'cancelled'
            ORDER BY id DESC LIMIT 1
            """
        )
        last_row = cur.fetchone()
        if last_row:
            result["last_request_time"] = last_row[0]
            result["last_latency_seconds"] = round(last_row[1], 2)
            result["last_status"] = last_row[2]

        # 6. Daily trends: latest 14 recorded days displayed chronologically
        cur.execute(
            """
            SELECT date, COUNT(*), SUM(is_429), SUM(CASE WHEN status = 'success' THEN 1 ELSE 0 END)
            FROM api_requests
            WHERE status != 'cancelled'
            GROUP BY date
            ORDER BY date DESC
            LIMIT 14
            """
        )
        trends_desc = []
        for t_row in cur.fetchall():
            trends_desc.append(
                {
                    "date": t_row[0],
                    "total": t_row[1],
                    "quota_429": t_row[2] or 0,
                    "success": t_row[3] or 0,
                }
            )
        result["daily_trends"] = list(reversed(trends_desc))

        # 7. Recent errors (last 10) - strictly restricted to authenticated administrators
        if admin_authorized:
            cur.execute(
                """
                SELECT timestamp, request_type, model_name, status, error_message, latency_seconds
                FROM api_requests
                WHERE status NOT IN ('success', 'reserved', 'cancelled')
                ORDER BY id DESC
                LIMIT 10
                """
            )
            for e_row in cur.fetchall():
                result["recent_errors"].append(
                    {
                        "timestamp": e_row[0],
                        "type": e_row[1],
                        "model": e_row[2],
                        "status": e_row[3],
                        "message": e_row[4] or "No error detail",
                        "latency": round(e_row[5], 2),
                    }
                )

        result["estimated_usage_summary"] = (
            f"{result['total_requests']:,} total requests logged ({result['requests_today']:,} today, "
            f"{result['successful_requests']:,} success, {result['quota_errors_429']:,} 429 quota alerts)"
        )
    except Exception as exc:
        logger.error("Error retrieving dashboard telemetry metrics from SQLite: %s", exc)
    finally:
        conn.close()

    return result
