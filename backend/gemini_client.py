"""
Gemini Client Factory and Error Handler using the modern official google-genai SDK.
Handles client instantiation, explicit API key resolution, credential validation,
and user-friendly error translation.
"""

from __future__ import annotations

import os
import time
from typing import Any

from google import genai
from google.genai import errors

from config.settings import GEMINI_CHAT_MODEL, get_configured_api_key
from utils.helpers import sanitize_error_message
from utils.usage_tracker import record_api_request
from backend.rate_limit import parse_quota_error


def resolve_api_key(session_key: str | None = None) -> str:
    """Resolve the active Google Gemini API key following secure precedence:
    1. Explicitly passed in session key (from Streamlit session_state).
    2. Streamlit secrets (if running in Streamlit Cloud / local secrets.toml).
    3. Environment variable (from .env or OS environment).

    NOTE: Never mutates os.environ at runtime.
    """
    if session_key and session_key.strip():
        return session_key.strip()

    # Check Streamlit secrets safely without throwing if streamlit is headless or has no secrets
    try:
        import streamlit as st
        if hasattr(st, "secrets") and "GOOGLE_API_KEY" in st.secrets:
            key = st.secrets["GOOGLE_API_KEY"]
            if key and str(key).strip():
                return str(key).strip()
    except Exception:
        pass

    return get_configured_api_key()


def validate_api_key_format(api_key: str) -> tuple[bool, str]:
    """Validate API key format without making network requests.
    Supports both traditional 'AIza...' keys and modern AI Studio 'AQ...' keys.
    """
    if not api_key or not api_key.strip():
        return False, "API key is missing. Please configure your key in .env or settings."

    key = api_key.strip()
    if len(key) < 20:
        return False, "API key is too short to be a valid Google Gemini key."

    if not (key.startswith("AIza") or key.startswith("AQ")):
        return False, "Invalid key format: Google Gemini API keys typically start with 'AIza' or 'AQ'."

    return True, ""


def get_genai_client(api_key: str | None = None) -> genai.Client:
    """Instantiate the official google-genai Client with an explicit API key."""
    resolved_key = resolve_api_key(api_key)
    is_valid, err = validate_api_key_format(resolved_key)
    if not is_valid:
        raise ValueError(f"Authentication error: {err}")

    return genai.Client(api_key=resolved_key)


def verify_api_key_live(api_key: str, model_name: str | None = None) -> tuple[bool, str]:
    """Verify an API key by making a minimal live request to Google Gen AI.

    Returns:
        (is_valid, message)
    """
    is_valid, err = validate_api_key_format(api_key)
    if not is_valid:
        return False, err

    target_model = model_name or GEMINI_CHAT_MODEL
    start_t = time.time()

    try:
        client = genai.Client(api_key=api_key.strip())
        # Make a quick, low-cost count_tokens request to verify auth
        client.models.count_tokens(
            model=target_model,
            contents="ping",
        )
        latency = time.time() - start_t
        record_api_request("verification", "success", latency=latency, model_name=target_model)
        return True, "API Key successfully validated with Google Gen AI."
    except errors.ClientError as exc:
        latency = time.time() - start_t
        quota_info = parse_quota_error(exc, model_name=target_model)
        if quota_info.is_429:
            record_api_request(
                "verification",
                "429_quota_exceeded",
                latency=latency,
                is_429=True,
                error_message=quota_info.friendly_message,
                model_name=target_model,
            )
            return False, quota_info.friendly_message
        record_api_request("verification", "error", latency=latency, error_message=str(exc), model_name=target_model)
        if exc.code == 400 and "API_KEY_INVALID" in str(exc):
            return False, "Google rejected this API key (Invalid API Key)."
        if exc.code == 403:
            return False, "Permission denied for this API key. Ensure Gemini API is enabled in your Google Cloud / AI Studio project."
        return False, f"Google API error: {sanitize_error_message(str(exc))}"
    except Exception as exc:
        latency = time.time() - start_t
        quota_info = parse_quota_error(exc, model_name=target_model)
        if quota_info.is_429:
            record_api_request(
                "verification",
                "429_quota_exceeded",
                latency=latency,
                is_429=True,
                error_message=quota_info.friendly_message,
                model_name=target_model,
            )
            return False, quota_info.friendly_message
        record_api_request("verification", "error", latency=latency, error_message=str(exc), model_name=target_model)
        return False, f"Verification failed: {sanitize_error_message(str(exc))}"


def translate_genai_error(exc: Exception, model_name: str = "") -> str:
    """Translate technical or Google Gen AI exceptions into clean, user-friendly messages."""
    err_str = str(exc)

    quota_info = parse_quota_error(exc, model_name=model_name)
    if quota_info.is_429:
        return quota_info.friendly_message

    if isinstance(exc, errors.ClientError):
        if exc.code == 400:
            if "API_KEY_INVALID" in err_str:
                return "Invalid Google Gemini API key. Please verify your API key in settings or .env."
            return f"Invalid request parameters sent to Gemini: {sanitize_error_message(err_str)}"
        elif exc.code == 403:
            return "Access denied (403): Gemini API service is either disabled or restricted for this API key."
        elif exc.code == 404:
            return f"Requested model was not found: {sanitize_error_message(err_str)}"

    if isinstance(exc, errors.ServerError):
        return "Google Gemini service is temporarily unavailable. Please retry in a few moments."

    if isinstance(exc, TimeoutError) or "timeout" in err_str.lower() or "timed out" in err_str.lower():
        return "Network timeout connecting to Google Gemini API. Please check your internet connection."

    if "connection" in err_str.lower() or "nameresolutionerror" in err_str.lower():
        return "Failed to establish a network connection to Google Gemini API."

    return f"AI processing error: {sanitize_error_message(err_str)}"
