"""
Application configuration and settings.

Loads sensitive values from environment variables and defines
application-wide constants for chunking, embeddings, and UI.
"""

from __future__ import annotations

import os
from pathlib import Path
from dotenv import load_dotenv

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent.parent
UPLOAD_DIR = BASE_DIR / "data" / "uploads"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

# Load environment variables from .env if present
ENV_PATH = BASE_DIR / ".env"
load_dotenv(dotenv_path=ENV_PATH)


# ---------------------------------------------------------------------------
# Google Gemini API
# ---------------------------------------------------------------------------
def get_configured_api_key() -> str:
    """Retrieve the Google Gemini API key from environment variables (.env or OS).

    Checks GOOGLE_API_KEY first, followed by GEMINI_API_KEY.
    """
    key = os.environ.get("GOOGLE_API_KEY", "") or os.environ.get("GEMINI_API_KEY", "")
    return key.strip()


def get_google_api_key() -> str:
    """Legacy alias for backward compatibility."""
    return get_configured_api_key()


GOOGLE_API_KEY: str = get_configured_api_key()

# Model used for chat / Q&A (fast, high quota, vision-enabled)
GEMINI_CHAT_MODEL: str = os.environ.get("GEMINI_CHAT_MODEL", "models/gemini-3.5-flash-lite")

# Model used for generating embeddings
GEMINI_EMBEDDING_MODEL: str = os.environ.get("GEMINI_EMBEDDING_MODEL", "models/gemini-embedding-001")

# Supported models for runtime selection and validation
SUPPORTED_CHAT_MODELS: list[str] = [
    "models/gemini-3.5-flash-lite",
    "models/gemini-3.1-flash-lite",
    "models/gemini-2.5-flash-lite",
]

SUPPORTED_EMBEDDING_MODELS: list[str] = [
    "models/gemini-embedding-001",
]

# ---------------------------------------------------------------------------
# Document Chunking
# ---------------------------------------------------------------------------
CHUNK_SIZE: int = 1024          # tokens per chunk
CHUNK_OVERLAP: int = 128       # overlapping tokens between chunks

# ---------------------------------------------------------------------------
# Vector Store
# ---------------------------------------------------------------------------
TOP_K_RESULTS: int = 5         # number of chunks to retrieve per query

# ---------------------------------------------------------------------------
# UI Constants & File Formats
# ---------------------------------------------------------------------------
APP_TITLE: str = "✨ Multimodal AI Research Assistant"
APP_ICON: str = "✨"
MAX_UPLOAD_SIZE_MB: int = 50
MAX_IMAGE_SIZE_MB: int = 20

ALLOWED_PDF_EXTENSIONS: list[str] = [".pdf"]
ALLOWED_IMAGE_EXTENSIONS: list[str] = [".png", ".jpg", ".jpeg", ".webp"]
ALLOWED_EXTENSIONS: list[str] = ALLOWED_PDF_EXTENSIONS + ALLOWED_IMAGE_EXTENSIONS

# ---------------------------------------------------------------------------
# API Usage Monitoring & Quota Guard Settings
# ---------------------------------------------------------------------------
INTERNAL_DAILY_REQUEST_LIMIT: int = int(os.environ.get("INTERNAL_DAILY_REQUEST_LIMIT", "15"))
INTERNAL_WARNING_THRESHOLD_PERCENT: float = float(os.environ.get("INTERNAL_WARNING_THRESHOLD_PERCENT", "80.0"))
USAGE_DB_PATH: Path = BASE_DIR / "data" / "usage_history.db"

# ---------------------------------------------------------------------------
# Embedding Cache Settings
# ---------------------------------------------------------------------------
MAX_EMBEDDING_CACHE_SIZE: int = int(os.environ.get("MAX_EMBEDDING_CACHE_SIZE", "10000"))
EMBEDDING_CACHE_TTL_SECONDS: float | None = (
    float(os.environ["EMBEDDING_CACHE_TTL_SECONDS"])
    if os.environ.get("EMBEDDING_CACHE_TTL_SECONDS")
    else None
)

def get_admin_password() -> str | None:
    """Retrieve administrator password following strict precedence:
    1. Streamlit secrets (st.secrets["ADMIN_PASSWORD"])
    2. Environment variable (ADMIN_PASSWORD)

    Returns None if unconfigured. Never falls back to a hardcoded default password.
    """
    try:
        import streamlit as st
        if hasattr(st, "secrets") and "ADMIN_PASSWORD" in st.secrets:
            key = str(st.secrets["ADMIN_PASSWORD"]).strip()
            if key:
                return key
    except Exception:
        pass
    env_pass = os.environ.get("ADMIN_PASSWORD", "").strip()
    return env_pass if env_pass else None

