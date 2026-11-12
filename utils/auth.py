"""Enterprise Authentication and Security Utilities for AURA.AI.

Provides hardened constant-time administrative password verification using
hmac.compare_digest, session-aware brute-force protection with failed attempt tracking,
and automated temporary account lockout.

Session-Aware Isolation:
Failed attempts and lockouts are tracked per session identifier so that
an attacker or automated crawler in one browser session cannot lock out
other legitimate users or the administrator.
"""

from __future__ import annotations

import hmac
import threading
import time
import uuid
from typing import Any, Tuple

from config.settings import get_admin_password

# Security Policy Constants
MAX_ADMIN_ATTEMPTS: int = 5
ADMIN_LOCKOUT_SECONDS: int = 300  # 5 minutes
MAX_SESSION_ENTRIES: int = 2000

_AUTH_LOCK = threading.Lock()
# {session_id: {"failed_attempts": int, "lockout_until": float, "last_seen": float}}
_SESSION_FAILURES: dict[str, dict[str, Any]] = {}


def _get_active_session_id() -> str:
    """Retrieve session id from Streamlit session_state if available, else fallback to default."""
    try:
        from streamlit.runtime import exists

        if exists():
            import streamlit as st

            if "_aura_session_id" not in st.session_state:
                st.session_state["_aura_session_id"] = str(uuid.uuid4())
            return str(st.session_state["_aura_session_id"])
    except Exception:
        pass
    return "global_default"


def _prune_expired_sessions_locked(now: float) -> None:
    """Prune stale session failure records when table exceeds MAX_SESSION_ENTRIES."""
    if len(_SESSION_FAILURES) <= MAX_SESSION_ENTRIES:
        return
    stale_keys = [
        sid
        for sid, rec in _SESSION_FAILURES.items()
        if now > rec.get("lockout_until", 0.0) and (now - rec.get("last_seen", now)) > 3600
    ]
    for sid in stale_keys:
        _SESSION_FAILURES.pop(sid, None)


def is_admin_locked_out(session_id: str | None = None) -> Tuple[bool, int]:
    """Check if administrator login is currently locked due to failed attempts for the given session.

    Returns:
        (is_locked, remaining_lockout_seconds)
    """
    sid = session_id or _get_active_session_id()
    with _AUTH_LOCK:
        now = time.time()
        rec = _SESSION_FAILURES.get(sid)
        if rec:
            lockout_until = rec.get("lockout_until", 0.0)
            if now < lockout_until:
                remaining = int(lockout_until - now)
                return True, max(remaining, 1)
        return False, 0


def reset_admin_lockout(session_id: str | None = None) -> None:
    """Reset failed attempts and clear lockout state.

    If session_id is None, clears all session tracking records (useful in test teardown).
    If session_id is specified, clears only that session's state.
    """
    with _AUTH_LOCK:
        if session_id is None:
            _SESSION_FAILURES.clear()
        else:
            _SESSION_FAILURES.pop(session_id, None)


def verify_admin_password(candidate_password: str, session_id: str | None = None) -> Tuple[bool, str]:
    """Verify administrator password against configured password with timing-attack prevention
    and session-aware brute-force lockout protection.

    Returns:
        (is_valid, status_message)
    """
    sid = session_id or _get_active_session_id()
    configured_password = get_admin_password()
    if not configured_password:
        return False, "Administrator access is disabled because ADMIN_PASSWORD is not configured."

    with _AUTH_LOCK:
        now = time.time()
        _prune_expired_sessions_locked(now)

        rec = _SESSION_FAILURES.setdefault(sid, {"failed_attempts": 0, "lockout_until": 0.0, "last_seen": now})
        rec["last_seen"] = now

        # 1. Enforce active lockout window
        if now < rec["lockout_until"]:
            remaining = int(rec["lockout_until"] - now)
            return (
                False,
                f"Too many failed attempts. Admin access locked. Try again in {max(remaining, 1)} seconds.",
            )

        if not candidate_password:
            return False, "Please enter an administrator passphrase."

        # 2. Timing-safe constant-time password comparison
        candidate_bytes = candidate_password.strip().encode("utf-8")
        configured_bytes = configured_password.strip().encode("utf-8")
        is_correct = hmac.compare_digest(candidate_bytes, configured_bytes)

        if is_correct:
            rec["failed_attempts"] = 0
            rec["lockout_until"] = 0.0
            return True, "Administrator authentication successful."

        # 3. Handle failed attempt
        rec["failed_attempts"] += 1
        if rec["failed_attempts"] >= MAX_ADMIN_ATTEMPTS:
            rec["lockout_until"] = now + ADMIN_LOCKOUT_SECONDS
            return (
                False,
                f"Invalid passphrase. Maximum attempts ({MAX_ADMIN_ATTEMPTS}) reached. "
                f"Admin access locked for {ADMIN_LOCKOUT_SECONDS // 60} minutes.",
            )

        remaining_tries = MAX_ADMIN_ATTEMPTS - rec["failed_attempts"]
        return (
            False,
            f"Invalid administrator passphrase. {remaining_tries} attempt{'s' if remaining_tries != 1 else ''} remaining.",
        )
