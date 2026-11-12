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
def get_google_api_key() -> str:
    """Retrieve the Google Gemini API key from environment variables."""
    return os.environ.get("GOOGLE_API_KEY", "").strip()


GOOGLE_API_KEY: str = get_google_api_key()

# Model used for chat / Q&A
GEMINI_CHAT_MODEL: str = "models/gemini-3.5-flash"

# Model used for generating embeddings
GEMINI_EMBEDDING_MODEL: str = "models/gemini-embedding-001"

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
# UI Constants
# ---------------------------------------------------------------------------
APP_TITLE: str = "📄 AI PDF Research Assistant"
APP_ICON: str = "📄"
MAX_UPLOAD_SIZE_MB: int = 50
ALLOWED_EXTENSIONS: list[str] = [".pdf"]
