"""Tests for session-aware admin lockout isolation and authentication security."""

from __future__ import annotations

import pytest

from utils.auth import (
    is_admin_locked_out,
    reset_admin_lockout,
    verify_admin_password,
)


@pytest.fixture(autouse=True)
def setup_auth_env(monkeypatch: pytest.MonkeyPatch):
    reset_admin_lockout()
    monkeypatch.setenv("ADMIN_PASSWORD", "SuperSecurePassphrase!2026")
    yield
    reset_admin_lockout()


def test_session_lockout_isolation():
    """Verify that failed attempts in Session A do NOT lock out Session B or admin."""
    session_a = "user_session_attacker"
    session_b = "legitimate_admin_session"

    # Attacker triggers 5 failed attempts in Session A
    for i in range(5):
        ok, msg = verify_admin_password(f"wrong_pass_{i}", session_id=session_a)
        assert ok is False

    # Session A is now locked out
    locked_a, rem_a = is_admin_locked_out(session_id=session_a)
    assert locked_a is True
    assert rem_a > 0

    # Session B is NOT locked out!
    locked_b, rem_b = is_admin_locked_out(session_id=session_b)
    assert locked_b is False
    assert rem_b == 0

    # Session B can authenticate successfully
    ok_b, msg_b = verify_admin_password("SuperSecurePassphrase!2026", session_id=session_b)
    assert ok_b is True
    assert "successful" in msg_b

    # Meanwhile Session A remains locked out even with correct password
    ok_a, msg_a = verify_admin_password("SuperSecurePassphrase!2026", session_id=session_a)
    assert ok_a is False
    assert "locked" in msg_a


def test_reset_specific_session_leaves_others_intact():
    """Verify resetting session A only resets session A."""
    session_a = "session_alpha"
    session_b = "session_beta"

    for _ in range(5):
        verify_admin_password("wrong", session_id=session_a)
        verify_admin_password("wrong", session_id=session_b)

    assert is_admin_locked_out(session_id=session_a)[0] is True
    assert is_admin_locked_out(session_id=session_b)[0] is True

    # Reset only Session A
    reset_admin_lockout(session_id=session_a)
    assert is_admin_locked_out(session_id=session_a)[0] is False
    assert is_admin_locked_out(session_id=session_b)[0] is True
