"""
Unit tests for application configuration and settings.
"""

from __future__ import annotations

import os
from config.settings import (
    ALLOWED_EXTENSIONS,
    ALLOWED_IMAGE_EXTENSIONS,
    ALLOWED_PDF_EXTENSIONS,
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    GEMINI_CHAT_MODEL,
    GEMINI_EMBEDDING_MODEL,
    MAX_IMAGE_SIZE_MB,
    MAX_UPLOAD_SIZE_MB,
    TOP_K_RESULTS,
    get_configured_api_key,
    get_google_api_key,
)


def test_chunking_and_retrieval_constants():
    assert CHUNK_SIZE > 0
    assert CHUNK_OVERLAP > 0
    assert CHUNK_OVERLAP < CHUNK_SIZE
    assert TOP_K_RESULTS > 0
    assert MAX_UPLOAD_SIZE_MB >= 10
    assert MAX_IMAGE_SIZE_MB >= 5


def test_model_names_are_modern_and_valid():
    assert "gemini-3.5-flash" in GEMINI_CHAT_MODEL
    assert "gemini-embedding-001" in GEMINI_EMBEDDING_MODEL


def test_supported_file_extensions():
    assert ".pdf" in ALLOWED_PDF_EXTENSIONS
    for ext in [".png", ".jpg", ".jpeg", ".webp"]:
        assert ext in ALLOWED_IMAGE_EXTENSIONS
    assert len(ALLOWED_EXTENSIONS) == len(ALLOWED_PDF_EXTENSIONS) + len(ALLOWED_IMAGE_EXTENSIONS)


def test_get_configured_api_key(monkeypatch):
    monkeypatch.setenv("GOOGLE_API_KEY", "AIzaSyTestEnvKey123456789012345678")
    assert get_configured_api_key() == "AIzaSyTestEnvKey123456789012345678"
    assert get_google_api_key() == "AIzaSyTestEnvKey123456789012345678"
