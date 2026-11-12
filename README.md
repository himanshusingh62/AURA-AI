# 🌌 AURA.AI — AI-Powered Multimodal PDF & Image Research Assistant

[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.45%2B-FF4B4B?logo=streamlit&logoColor=white)](https://streamlit.io)
[![Google GenAI SDK](https://img.shields.io/badge/Google%20GenAI%20SDK-v3.0.0%2B-4285F4?logo=google&logoColor=white)](https://pypi.org/project/google-genai/)
[![Gemini](https://img.shields.io/badge/Gemini-3.5--Flash--Lite-orange)](https://aistudio.google.com/)
[![Embeddings](https://img.shields.io/badge/Embeddings-3072--dim%20(gemini--embedding--001)-blueviolet)](https://ai.google.dev/)
[![LlamaIndex](https://img.shields.io/badge/LlamaIndex-Core%200.12%2B-181717)](https://llamaindex.ai)
[![Security Hardened](https://img.shields.io/badge/Security-XSS%20%26%20Key%20Hardened-brightgreen)](https://github.com)
[![License](https://img.shields.io/badge/License-Decision%20Pending-lightgrey)](#-license)

**AURA.AI** is an enterprise-grade, security-hardened **Multimodal Retrieval-Augmented Generation (RAG)** SaaS platform built with Python, Streamlit, LlamaIndex Core, and the official modern **Google Gen AI SDK (`google-genai`)**. 

It enables researchers, data scientists, and engineers to upload multi-page research PDFs alongside technical visual diagrams, workflow flowcharts, benchmark graphs, and screenshots for cross-modal synthesis with exact page citations and visual source references.

---

## 🌟 Key Capabilities

| Capability | Technical Details |
|---|---|
| 📑 **Multi-PDF RAG Pipeline** | PyMuPDF page-by-page extraction, unified vector store indexing across multiple uploaded PDFs, and document scoping filters. |
| 🗑️ **Dynamic PDF Deletion** | Safe node and vector removal via `remove_document_from_index`. Deleted document content is purged from both the docstore and vector store, preventing it from appearing in subsequent query answers. |
| 🔍 **Scanned PDF Detection** | Automated character density analysis detecting scanned image-only PDFs. Flags non-extractable documents via `ScannedPDFError` and provides actionable guidance (clarification: detects scanned documents; does not run local OCR on PDFs). |
| 🖼️ **Native Vision OCR & Visual Analysis** | High-fidelity image understanding for `.png`, `.jpg`, `.jpeg`, and `.webp` using Gemini multimodal vision. Reads embedded text, interprets system diagrams, extracts chart percentages, and inspects visual structures. |
| 🔄 **Cross-Modal Reasoning** | Synthesize information across uploaded PDFs and visual imagery in a single prompt (e.g. comparing architectural diagrams with PDF specifications). |
| 📊 **API Usage & Quota Dashboard** | Centralized SQLite telemetry (`data/usage_history.db`) tracking daily requests, model utilization, HTTP 429 alerts, with GREEN / YELLOW / RED status indicators and admin password authentication. |
| 🛡️ **HTTP 429 Quota Guard & Cache** | Intelligent rate-limit parsing extracting Google's `retry_delay`, zero-retry on daily quota exhaustion, short backoff on per-minute limits, and chunk embedding caching to prevent duplicate API calls. |
| 📚 **Dual Source Attribution** | Exact page numbers and cosine similarity scores for PDF chunks; visual thumbnail preview badges for cited images. |
| ⚡ **100% Modern GenAI SDK** | Migrated fully to official `google-genai` (v3.0.0+); zero legacy `google-generativeai` or deprecated LlamaIndex wrappers. |
| 🛡️ **Enterprise Security** | XSS protection via `html.escape()`, path traversal neutralization (`safe_filename`), image decompression bomb defenses (`MAX_IMAGE_PIXELS`), magic-byte inspection, and credential sanitization. |
| 🔑 **Flexible API Key Management** | Precedence-based key resolution: UI Session Key -> Streamlit Secrets (`st.secrets`) -> `.env` file. Never mutates `os.environ` at runtime. |
| 🎨 **Dark SaaS Design System** | Modern Linear / Perplexity-inspired aesthetic with glassmorphism cards, responsive chat layout, and real-time status telemetry. |

---

## 🏗️ Architecture

```
                                      ┌────────────────────────────────────────────────────────┐
                                      │                      User Request                      │
                                      │              (Prompt + Documents + Images)             │
                                      └───────────────────────────┬────────────────────────────┘
                                                                  │
                                                                  ▼
                                      ┌────────────────────────────────────────────────────────┐
                                      │                 Streamlit Web Platform                 │
                                      │             (Dark SaaS Theme / Chat / Hub)             │
                                      └───────────────────────────┬────────────────────────────┘
                                                                  │
                                 ┌────────────────────────────────┴────────────────────────────────┐
                                 ▼                                                                 ▼
                    ┌─────────────────────────┐                                       ┌─────────────────────────┐
                    │      PDF Processor      │                                       │     Image Processor     │
                    │   (PyMuPDF Extraction)  │                                       │ (Pillow / Decompress)   │
                    └────────────┬────────────┘                                       └────────────┬────────────┘
                                 │                                                                 │
                                 ▼                                                                 │
                    ┌─────────────────────────┐                                                    │
                    │   GoogleGenAIEmbedding  │                                                    │
                    │  (3,072-dim Embeddings) │                                                    │
                    └────────────┬────────────┘                                                    │
                                 │                                                                 │
                                 ▼                                                                 │
                    ┌─────────────────────────┐                                                    │
                    │    VectorStoreIndex     │                                                    │
                    │  (In-Memory Similarity) │                                                    │
                    └────────────┬────────────┘                                                    │
                                 │                                                                 │
                                 ▼                                                                 ▼
                    ┌──────────────────────────────────────────────────────────────────────────────────┐
                    │                      Multimodal Routing & Grounding Engine                       │
                    │               (PDF-only  |  Image-only  |  Combined Cross-Modal)                 │
                    └─────────────────────────────────────────┬────────────────────────────────────────┘
                                                              │
                                                              ▼
                                            ┌───────────────────────────────────┐
                                            │        Google Gen AI Client       │
                                            │  (models/gemini-3.5-flash-lite)   │
                                            └─────────────────┬─────────────────┘
                                                              │
                                                              ▼
                                            ┌───────────────────────────────────┐
                                            │      Grounded Answer Output       │
                                            │   + Page Citations & Image Badges │
                                            └───────────────────────────────────┘
```

---

## 📁 Repository Structure

```
API_PROJECT_ML/
├── config/
│   ├── __init__.py
│   └── settings.py              # Application settings, model parameters, API key loader
├── backend/
│   ├── __init__.py
│   ├── gemini_client.py         # Google Gen AI Client factory, key validation, error translation
│   ├── embeddings.py            # Custom GoogleGenAIEmbedding with embedding cache, retry backoff & node deletion
│   ├── query_engine.py          # Custom GoogleGenAILLM (CustomLLM) and single-pass RAG retriever
│   ├── multimodal_engine.py     # Cross-modal router (PDF, Vision & Combined synthesis)
│   ├── rate_limit.py            # 429 quota exception parser, retry-delay extractor, and exponential backoff
│   ├── pdf_processor.py         # PyMuPDF extraction, ScannedPDFError detection, traversal protection
│   └── image_processor.py       # Pillow processing, EXIF normalization, thumbnail generator
├── frontend/
│   ├── __init__.py
│   ├── app.py                   # Main Streamlit application dashboard (Linear/ChatGPT style)
│   ├── styles.py                # Dark SaaS design system CSS & responsive grid layouts
│   └── components.py            # Reusable UI components (header, hero, source cards, media badges)
├── utils/
│   ├── __init__.py
│   ├── helpers.py               # Validation, safe_filename, sanitize_error_message, XSS defense
│   └── usage_tracker.py         # Centralized SQLite usage telemetry & real-time quota status engine
├── data/
│   └── usage_history.db         # Persistent SQLite database for API request events and error telemetry
├── tests/
│   ├── __init__.py
│   ├── test_config.py           # Configuration unit tests (offline)
│   ├── test_gemini_client.py    # Key validation and error translation unit tests (offline)
│   ├── test_security.py         # XSS, path traversal, decompression bomb tests (offline)
│   ├── test_pdf_processor.py    # PyMuPDF and ScannedPDFError tests (offline)
│   ├── test_image_processor.py  # Image validation and thumbnail tests (offline)
│   ├── test_rag_pipeline_mocked.py # Mocked GenAI client RAG tests (offline)
│   ├── test_multi_pdf_deletion.py  # 3-PDF upload, deletion & retrieval isolation tests (offline)
│   ├── test_components.py       # Hardened component unit test suite (offline)
│   ├── test_rate_limit_and_caching.py # 429 quota detection, delay parsing, backoff & embedding cache tests (offline)
│   ├── test_usage_tracker.py    # SQLite telemetry, quota states, and admin authentication tests (offline)
│   ├── test_ingestion_pipeline.py  # Live PDF ingestion and embedding test
│   └── test_multimodal_pipeline.py # Live cross-modal visual reasoning test
├── pytest.ini                   # Pytest testpaths, pythonpath, and marker configuration
├── main.py                      # Application launcher
├── sample_report.pdf            # 19-page non-sensitive AI & ML test PDF document
├── sample_architecture.png      # Architectural workflow diagram test image
├── sample_chart.jpg             # Benchmark accuracy chart test image
├── requirements.txt             # Clean dependencies (google-genai, llama-index-core, etc.)
├── .env.example                 # Safe environment template
├── .gitignore                   # Excludes secrets, pytest cache, data uploads, and venvs
└── README.md
```

---

## 🚀 Getting Started

### 1. Prerequisites

- Python 3.10 to 3.14
- Google Gemini API Key (Get a free key from [Google AI Studio](https://aistudio.google.com/apikey))

### 2. Clone and Setup

```bash
cd c:\API_PROJECT_ML
```

Create and activate a virtual environment (optional but recommended):

```bash
python -m venv venv
# On Windows:
.\venv\Scripts\activate
# On Linux/macOS:
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

> **Note**: AURA.AI uses the modern official `google-genai` SDK. All legacy packages (`google-generativeai`, `llama-index-llms-gemini`, `llama-index-embeddings-gemini`) have been completely removed.

### 4. Configure Your API Key

Copy the `.env.example` file to `.env`:

```bash
cp .env.example .env
```

Open `.env` and insert your API key:

```env
GOOGLE_API_KEY=your_gemini_api_key_here
```

*(Alternatively, you can provide your API key dynamically in the Streamlit web sidebar or via Streamlit Cloud Secrets!)*

---

## 🧪 Running the Test Suite

AURA.AI provides automated test coverage with strict separation between offline unit/mock tests and live API integration tests:

### Run Offline Unit & Security Tests (No API Key Required)

```bash
python -m pytest -m "not live" -v
```

This executes **132 automated tests** across 20 offline test suites:
- Configuration and path resolution (`test_config.py`)
- API key format validation and error translation (`test_gemini_client.py`)
- Constant-time password verification and admin brute-force lockout (`test_auth.py`)
- Session-aware admin lockout isolation (`test_auth_session.py`)
- Cross-platform path redaction, XSS defenses, path traversal & decompression bomb protections (`test_security.py`)
- Scanned PDF detection, password-protected PDF handling, and text extraction (`test_pdf_processor.py`, `test_pdf_password.py`)
- Image processing, EXIF normalization, and thumbnails (`test_image_processor.py`)
- Multi-PDF upload, node deletion, and direct vector store purging (`test_multi_pdf_deletion.py`)
- Multi-PDF same-name hash collision isolation (`test_same_name_pdfs.py`)
- LlamaIndex retrieval-time metadata filtering and cross-document isolation (`test_metadata_filtering.py`)
- Failed upload state management and manual retry flow (`test_failed_upload_retry.py`)
- Scoped multi-document RAG query retrieval (`test_query_scope.py`)
- Mocked RAG retrieval and citation accuracy (`test_rag_pipeline_mocked.py`)
- UI components, cards, and theme rendering (`test_components.py`)
- Rate limiting, strict 429 detection, delay extraction, exponential backoff, and embedding caching (`test_rate_limit_and_caching.py`)
- Bounded LRU embedding cache, TTL eviction, and tenant isolation (`test_embedding_cache_lru.py`)
- Atomic SQLite daily request limit reservation and race condition prevention (`test_atomic_reservation.py`)
- SQLite telemetry persistence, quota states (GREEN/YELLOW/RED), 14-day chronological trends, and admin authentication (`test_usage_tracker.py`, `test_usage_trends.py`)

### Run Live End-to-End Integration Tests (Requires `GOOGLE_API_KEY`)

Live tests make genuine requests to the Google Gen AI API and should only be executed when a valid key is configured:

```bash
# Test 1: Full PDF ingestion, 3,072-dim embeddings, and grounded RAG Q&A
python tests/test_ingestion_pipeline.py

# Test 2: Full multimodal suite (PDF + PNG architecture diagram + JPG chart benchmark)
python tests/test_multimodal_pipeline.py
```

---

## 💻 Running the Application

Launch the Streamlit web interface:

```bash
python -m streamlit run main.py
```

The application will be available at `http://localhost:8501`.

---

## ☁️ Streamlit Community Cloud Deployment

To deploy AURA.AI to **Streamlit Community Cloud**:

1. **Push to GitHub**:
   Ensure `.env`, `data/uploads/`, and `.pytest_cache` are ignored by `.gitignore` (verified).
   ```bash
   git add .
   git commit -m "Deploy AURA.AI Multimodal Research Assistant"
   git push origin main
   ```

2. **Create New App on Streamlit Cloud**:
   - Go to [share.streamlit.io](https://share.streamlit.io)
   - Select your repository and branch (`main`)
   - Set **Main file path** to `main.py`

3. **Configure Secrets**:
   - In the Streamlit Cloud dashboard, navigate to **App Settings** > **Secrets**.
   - Add your Gemini API key in TOML format:
     ```toml
     GOOGLE_API_KEY = "your_actual_gemini_api_key_here"
     ```
   - Save secrets. AURA.AI will automatically detect `st.secrets["GOOGLE_API_KEY"]` on startup!

---

## ⚙️ Configuration Reference

All settings can be customized in [`config/settings.py`](config/settings.py):

| Variable | Default Value | Description |
|---|---|---|
| `GEMINI_CHAT_MODEL` | `models/gemini-3.5-flash-lite` | Primary reasoning and vision model |
| `GEMINI_EMBEDDING_MODEL` | `models/gemini-embedding-001` | 3,072-dimensional vector embedding model |
| `CHUNK_SIZE` | `1024` | Passage chunk size (tokens) |
| `CHUNK_OVERLAP` | `128` | Overlap between consecutive chunks (tokens) |
| `TOP_K_RESULTS` | `5` | Retrieved passages per user query |
| `MAX_UPLOAD_SIZE_MB` | `50` | Maximum PDF file upload size |
| `MAX_IMAGE_SIZE_MB` | `20` | Maximum image file upload size |
| `MAX_IMAGE_PIXELS` | `40,000,000` | Decompression bomb pixel threshold |
| `ALLOWED_IMAGE_EXTENSIONS` | `['.png', '.jpg', '.jpeg', '.webp']` | Whitelisted image file extensions |
| `INTERNAL_DAILY_REQUEST_LIMIT` | `15` | Internal application daily limit (triggers YELLOW alert at 80% / 12 reqs) |
| `INTERNAL_WARNING_THRESHOLD_PERCENT` | `80.0` | Warning percentage threshold before alert |
| `ADMIN_PASSWORD` | *None* | Required administrator passphrase (configured via Streamlit secrets or `.env`) |
| `MAX_EMBEDDING_CACHE_SIZE` | `10000` | Maximum number of text chunk embeddings cached in memory (LRU eviction) |
| `EMBEDDING_CACHE_TTL_SECONDS` | *None* (no expiry) | Optional TTL in seconds for embedding cache entries |

---

## 📊 Gemini API Usage & Quota Monitoring

AURA.AI features a centralized, persistent API usage monitoring system accessible via the **Settings & Specs** tab:

### 1. Quota Health Status Indicators
- **🟢 GREEN**: Normal operations; no recent quota errors, and application request volume is within internal limits.
- **🟡 YELLOW**: Approaching limit; daily application request volume has reached >= 80% of `INTERNAL_DAILY_REQUEST_LIMIT`.
- **🔴 RED**: Quota exceeded; Gemini API returned an HTTP 429 `RESOURCE_EXHAUSTED` error.

### 2. Administrator Security Gate
- The detailed monitoring telemetry is protected behind an administrative passphrase gate (`ADMIN_PASSWORD` or `st.secrets["ADMIN_PASSWORD"]`).
- Public users see only high-level quota health without accessing sensitive request logs, database tables, or model metrics.
- API keys are redacted and excluded from database telemetry records and logging.

### 3. Transparency & Quota Distinction
- **Application Request Count**: Exact count of requests recorded by AURA.AI in SQLite (`data/usage_history.db`).
- **Official Google API Quota**: Server-side quota managed by Google Cloud / AI Studio; Google provides no client endpoint to query real-time remaining free-tier tokens or request balances.
- **Estimated Usage**: Tracked through client-side session telemetry and HTTP 429 response interception.
- **Ephemeral Cloud Storage Notice**: On Streamlit Community Cloud (ephemeral containers), local SQLite database files reset when instances reboot. Connect an external database (e.g. PostgreSQL) for permanent production retention.

---

## 🛡️ HTTP 429 Rate Limit Guard & Embedding Cache

To mitigate and resolve Gemini API 429 quota errors:

1. **Intelligent 429 Detection & Delay Extraction**:
   - `backend/rate_limit.py` parses Google's `retry_delay` from RPC details (`RetryInfo`), HTTP response headers (`Retry-After`), and error message regex.
   - Computes user-friendly messages with the exact estimated retry wait time (e.g., `~15 seconds`).

2. **Zero-Retry on Daily Quota Exhaustion**:
   - Detects whether a 429 is a temporary per-minute rate limit or daily quota exhaustion (`GenerateContent requests per day`, `generate_content_free_tier_requests`).
   - If daily quota is spent, AURA.AI avoids repeated retry requests, terminating early with clear user notification.

3. **Exponential Backoff for Temporary Limits**:
   - For temporary per-minute limits with short retry delays (<= 6 seconds), applies exponential backoff and retries.
   - For long delays (> 6 seconds), immediately surfaces the wait time to the user interface.

4. **Bounded Chunk Embedding Caching**:
   - `backend/embeddings.py` maintains a thread-safe bounded LRU cache (`EmbeddingLRUCache`) keyed by deterministic SHA-256 hashes of `model_name + tenant_fingerprint + chunk_text`.
   - Repeated uploads of identical PDFs or re-indexing operations avoid redundant calls to Google's embedding API.

5. **Single-Pass Inference**:
   - Avoids redundant Gemini calls: prompts, multi-document RAG retrieval, and image vision are executed in a single structured invocation.

---

## 🛡️ Security & Hardening Architecture

1. **Credential Isolation**:
   - `os.environ` is not mutated at runtime with user-supplied session keys.
   - API keys are resolved with deterministic precedence: Session Key -> Streamlit Secrets -> Environment Variables.
   - Keys are masked in the UI (`AQ.A•••••••••••••••••••••••••••••••abcd`).

2. **HTML Injection & Cross-Site Scripting (XSS) Hardening**:
   - All dynamic strings interpolated into `st.markdown(..., unsafe_allow_html=True)` pass through `html.escape()`.
   - File names, document titles, search queries, text snippets, and error messages are strictly escaped.
   - Image thumbnail URIs are verified to start strictly with `data:image/`, blocking `javascript:` execution.

3. **Path Traversal & File Sanitization**:
   - Filenames are sanitized via `safe_filename()` which strips path separators (`/`, `\`), `..`, and null bytes.
   - File uploads are validated via magic bytes (`%PDF-`, PNG, JPEG, WebP) rather than relying solely on user-supplied file extensions.

4. **Resource Exhaustion Defense**:
   - Pillow decompression bomb defense (`MAX_IMAGE_PIXELS = 40_000_000`) prevents Denial-of-Service via zip bombs or hyper-dense images.
   - Scanned PDFs lacking text layers are detected and flagged via `ScannedPDFError` before wasting embedding tokens.

5. **Centralized Error Sanitization**:
   - `translate_genai_error()` maps raw HTTP 400, 403, 404, 429, timeouts, and network errors into clean, professional user messages.
   - `sanitize_error_message()` applies pattern-based redaction to mask recognized Google API keys, Bearer tokens, query parameters, and filesystem path patterns before messages are logged or displayed in UI alerts.

---

## 📄 License

**Status: Pending Owner Determination**

No open-source license (such as MIT or Apache) has been granted. All rights are reserved by the project owner pending an explicit licensing decision.
