"""Unit tests for enterprise authentication, constant-time verification, and admin brute-force lockout.
"""

import time
import pytest
from unittest.mock import patch

from utils.auth import (
    MAX_ADMIN_ATTEMPTS,
    is_admin_locked_out,
    reset_admin_lockout,
    verify_admin_password,
)


@pytest.fixture(autouse=True)
def clean_auth_state():
    reset_admin_lockout()
    yield
    reset_admin_lockout()


def test_verify_admin_password_success():
    with patch("utils.auth.get_admin_password", return_value="SuperSecretAdminPass123!"):
        ok, msg = verify_admin_password("SuperSecretAdminPass123!")
        assert ok is True
        assert "successful" in msg.lower()


def test_verify_admin_password_missing_config():
    with patch("utils.auth.get_admin_password", return_value=None):
        ok, msg = verify_admin_password("any_password")
        assert ok is False
        assert "disabled" in msg.lower()


def test_verify_admin_password_lockout_after_max_attempts():
    configured = "CorrectPass123"
    with patch("utils.auth.get_admin_password", return_value=configured):
        # First 4 failed attempts should warn with remaining tries
        for attempt in range(1, MAX_ADMIN_ATTEMPTS):
            ok, msg = verify_admin_password(f"wrong_pass_{attempt}")
            assert ok is False
            assert "remaining" in msg.lower()
            is_locked, _ = is_admin_locked_out()
            assert is_locked is False

        # 5th failed attempt should trigger lockout
        ok, msg = verify_admin_password("wrong_pass_5")
        assert ok is False
        assert "locked" in msg.lower()

        # Account is now locked
        is_locked, rem_sec = is_admin_locked_out()
        assert is_locked is True
        assert rem_sec > 0

        # Even correct password should now be rejected during lockout
        ok, msg = verify_admin_password(configured)
        assert ok is False
        assert "locked" in msg.lower()

        # Resetting lockout restores access
        reset_admin_lockout()
        is_locked, _ = is_admin_locked_out()
        assert is_locked is False

        ok, msg = verify_admin_password(configured)
        assert ok is True
