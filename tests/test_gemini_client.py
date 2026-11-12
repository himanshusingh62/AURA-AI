"""
Unit tests for Gemini Client Factory, key validation, and error translation.
"""

from __future__ import annotations

import os
import pytest
from google.genai import errors

from backend.gemini_client import (
    resolve_api_key,
    translate_genai_error,
    validate_api_key_format,
)


def test_validate_api_key_format_valid():
    is_valid, msg = validate_api_key_format("AIzaSyD7XYZ9876543210FEDCBA0123456789")
    assert is_valid is True
    assert msg == ""

    # Test modern AQ-format keys
    is_valid_aq, msg_aq = validate_api_key_format("AQ.AbCdEfGhIjKlMnOpQrStUvWxYz1234567890_test_key_sample")
    assert is_valid_aq is True
    assert msg_aq == ""


def test_validate_api_key_format_missing_or_empty():
    is_valid, msg = validate_api_key_format("")
    assert is_valid is False
    assert "missing" in msg.lower()


def test_validate_api_key_format_too_short():
    is_valid, msg = validate_api_key_format("AIzaShort")
    assert is_valid is False
    assert "too short" in msg.lower()


def test_validate_api_key_format_invalid_prefix():
    is_valid, msg = validate_api_key_format("sk-abcdefghijklmnopqrstuvwxyz123456")
    assert is_valid is False
    assert "AIza" in msg


def test_resolve_api_key_precedence(monkeypatch):
    monkeypatch.setenv("GOOGLE_API_KEY", "AIzaSyFromEnvironmentVariable123456")

    # When no session key is passed, falls back to env
    assert resolve_api_key(None) == "AIzaSyFromEnvironmentVariable123456"

    # When session key is passed, session key overrides without mutating os.environ
    session_key = "AIzaSyFromSessionOverride78901234"
    assert resolve_api_key(session_key) == session_key
    assert os.environ.get("GOOGLE_API_KEY") == "AIzaSyFromEnvironmentVariable123456"


def test_translate_genai_error_rate_limit():
    exc = errors.ClientError(429, {"error": {"message": "Resource has been exhausted (e.g. check quota)."}}, None)
    translated = translate_genai_error(exc)
    assert "rate limit" in translated.lower() and "429" in translated


def test_translate_genai_error_invalid_key():
    exc = errors.ClientError(400, {"error": {"message": "API_KEY_INVALID: Key not found"}}, None)
    translated = translate_genai_error(exc)
    assert "invalid google gemini api key" in translated.lower()


def test_translate_genai_error_forbidden():
    exc = errors.ClientError(403, {"error": {"message": "Permission denied"}}, None)
    translated = translate_genai_error(exc)
    assert "access denied" in translated.lower()


def test_translate_genai_error_timeout():
    exc = TimeoutError("The connection timed out after 30 seconds")
    translated = translate_genai_error(exc)
    assert "network timeout" in translated.lower()
