"""
Unit tests for enterprise security, API key sanitization, path traversal prevention, and XSS defenses.
"""

from __future__ import annotations

import pytest

from frontend.components import render_citation_card, render_image_citation_card
from utils.helpers import (
    mask_api_key,
    safe_filename,
    sanitize_error_message,
)


def test_sanitize_error_message_redacts_google_api_keys():
    raw_error = "Request failed: https://generativelanguage.googleapis.com/v1beta/models/gemini-3.5-flash:generateContent?key=AIzaSyA1234567890abcdefghijklmnopqrstuv"
    sanitized = sanitize_error_message(raw_error)
    assert "AIzaSyA1234567890abcdefghijklmnopqrstuv" not in sanitized
    assert "[REDACTED_API_KEY]" in sanitized


def test_sanitize_error_message_redacts_tokens_and_params():
    raw_error = "ConnectError: host/api?token=secretToken123456&api_key=privToken987654"
    sanitized = sanitize_error_message(raw_error)
    assert "secretToken123456" not in sanitized
    assert "privToken987654" not in sanitized
    assert "token=[REDACTED]" in sanitized
    assert "api_key=[REDACTED]" in sanitized


def test_sanitize_error_message_redacts_bearer_headers():
    raw_error = "401 Unauthorized: Authorization Bearer ya29.a0ARrda8-secret_oauth_token"
    sanitized = sanitize_error_message(raw_error)
    assert "ya29.a0ARrda8" not in sanitized
    assert "Bearer [REDACTED]" in sanitized


def test_sanitize_error_message_redacts_windows_and_posix_paths():
    # Windows path redaction
    win_err = r"Internal error in C:\Users\bob\secret_project\api_client.py at line 42"
    sanitized_win = sanitize_error_message(win_err)
    assert r"C:\Users\bob" not in sanitized_win
    assert "[REDACTED_PATH]" in sanitized_win

    # Linux /home path redaction
    linux_err = "Internal crash in /home/alice/dev/backend/server.py:99"
    sanitized_linux = sanitize_error_message(linux_err)
    assert "/home/alice" not in sanitized_linux
    assert "[REDACTED_PATH]" in sanitized_linux

    # macOS /Users path redaction
    mac_err = "Traceback at /Users/alice/projects/ml_app/main.py:12"
    sanitized_mac = sanitize_error_message(mac_err)
    assert "/Users/alice" not in sanitized_mac
    assert "[REDACTED_PATH]" in sanitized_mac


def test_safe_filename_cross_platform_separators():
    # Windows-style backslashes on POSIX / cross-platform
    assert safe_filename(r"..\..\windows\system32\calc.exe") == "calc.exe"
    assert safe_filename(r"folder\subfolder\deep\data.pdf") == "data.pdf"
    assert safe_filename("folder/subfolder/deep/data.pdf") == "data.pdf"
    assert safe_filename(r"C:\Windows\System32\drivers\etc\hosts") == "hosts"


def test_mask_api_key():
    assert mask_api_key("") == ""
    assert mask_api_key("12345") == "••••••••"
    assert mask_api_key("AIzaSyD7XYZ9876543210FEDCBA0123456789") == "AIza••••••••6789"


def test_safe_filename_path_traversal_prevention():
    assert safe_filename("../../etc/passwd") == "passwd"
    assert safe_filename("..\\..\\windows\\system32\\cmd.exe") == "cmd.exe"
    assert safe_filename("C:\\Users\\admin\\secret.pdf") == "secret.pdf"
    assert safe_filename("/var/www/uploads/test.png") == "test.png"
    assert safe_filename(".hidden_doc.pdf") == "hidden_doc.pdf"
    assert safe_filename("") == "unnamed_file"


def test_citation_card_escapes_xss():
    malicious_src = {
        "page_number": 2,
        "source": "<script>alert('XSS')</script>.pdf",
        "text_snippet": "<b onmouseover='alert(1)'>Dangerous snippet</b>",
        "relevance_score": 0.88,
    }
    card_html = render_citation_card(malicious_src)
    assert "<script>" not in card_html
    assert "<b onmouseover" not in card_html
    assert "&lt;script&gt;" in card_html
    assert "&lt;b onmouseover" in card_html


def test_image_citation_card_escapes_xss_and_blocks_bad_uri():
    malicious_img = {
        "filename": "<svg onload=alert(1)>.png",
        "format": "PNG",
        "dimensions": "800x600",
        "thumbnail_base64": "javascript:alert(document.domain)",
    }
    card_html = render_image_citation_card(malicious_img)
    assert "<svg" not in card_html
    assert "&lt;svg onload=alert(1)&gt;.png" in card_html
    assert "javascript:" not in card_html
