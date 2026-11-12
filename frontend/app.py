"""
AURA.AI — AI-Powered Multimodal Research Platform.
Sleek, responsive dark SaaS interface inspired by Linear, Perplexity, Notion AI, and ChatGPT.
Provides concurrent multi-document PDF semantic retrieval, visual diagram comprehension, and OCR intelligence.
"""

from __future__ import annotations

import hashlib
import html
import time
from collections import Counter
from datetime import datetime
from pathlib import Path

import pandas as pd
import streamlit as st

from llama_index.core.node_parser import SentenceSplitter

from backend.embeddings import build_index, remove_document_from_index
from backend.gemini_client import (
    resolve_api_key,
    translate_genai_error,
    validate_api_key_format,
    verify_api_key_live,
)
from backend.image_processor import ProcessedImage, process_uploaded_image
from backend.multimodal_engine import MultimodalQueryResult, query_multimodal
from backend.rate_limit import parse_quota_error
from backend.pdf_processor import (
    CorruptedPDFError,
    EmptyPDFError,
    PasswordProtectedPDFError,
    ScannedPDFError,
    inspect_and_extract_pdf,
)
from config.settings import (
    ALLOWED_EXTENSIONS,
    ALLOWED_IMAGE_EXTENSIONS,
    ALLOWED_PDF_EXTENSIONS,
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    GEMINI_CHAT_MODEL,
    GEMINI_EMBEDDING_MODEL,
    INTERNAL_DAILY_REQUEST_LIMIT,
    SUPPORTED_CHAT_MODELS,
    TOP_K_RESULTS,
    get_admin_password,
)
from utils.auth import is_admin_locked_out, verify_admin_password
from utils.usage_tracker import (
    DailyLimitExceededError,
    get_api_usage_metrics,
    get_comprehensive_dashboard_metrics,
    get_quota_status,
    reset_api_usage_metrics,
)
from frontend.components import (
    render_brand_header,
    render_citation_card,
    render_hero,
    render_image_citation_card,
    render_sidebar_profile,
)
from frontend.styles import get_saas_css
from utils.helpers import (
    file_hash,
    is_image,
    is_pdf,
    mask_api_key,
    safe_filename,
    sanitize_error_message,
    validate_image,
    validate_pdf,
)

# ── Page Configuration ─────────────────────────────────────────────────────
st.set_page_config(
    page_title="AURA.AI · Multimodal Research Platform",
    page_icon="✨",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ── Session State Initialization ───────────────────────────────────────────
def init_session_state() -> None:
    defaults = {
        "messages": [],           # List of chat messages
        "history_threads": [],     # Previous saved threads
        "index": None,             # Active VectorStoreIndex (Shared multi-doc)
        "doc_name": None,          # Display name of active document context
        "doc_size_mb": 0.0,        # Aggregate PDF size in MB
        "doc_pages": 0,            # Aggregate PDF page count
        "doc_chunks": 0,           # Total embedded chunks in vector store
        "uploaded_docs": {},       # Registered PDFs cache {doc_hash: dict}
        "processed_upload_hashes": set(),  # Set of file hashes successfully processed or permanently rejected
        "processing_hashes": set(),        # Set of file hashes currently undergoing ingestion
        "failed_uploads": {},      # Cache of failed uploads {hash: {"status": ..., "filename": ..., "error": ..., "file_bytes": ...}}
        "active_doc_filter": None, # Optional filter to scope queries to a single PDF
        "active_images": [],       # Currently loaded ProcessedImage objects
        "uploaded_images": {},     # Registered Images cache {filename: ProcessedImage}
        "prompt_to_submit": None,  # Pre-filled prompt from inquiry cards
        "top_k_val": TOP_K_RESULTS,
        "custom_api_key": "",      # User session API key (never written to os.environ)
        "chat_model": GEMINI_CHAT_MODEL,  # Runtime active Gemini reasoning model
        "embedding_model": GEMINI_EMBEDDING_MODEL,  # Runtime active Gemini embedding model
        "admin_authenticated": False,  # Administrator access for API telemetry dashboard
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val


# ── Document & Media Handlers ──────────────────────────────────────────────
def handle_pdf_upload(file_bytes: bytes, filename: str) -> bool:
    """Validate, parse, chunk, embed, and index an uploaded PDF into the vector store.
    Distinguishes temporary failures from permanent rejections and supports retry.
    """
    clean_name = safe_filename(filename)
    full_hash = file_hash(file_bytes)
    sha_hash = full_hash[:12]

    # Check if this exact PDF is already indexed to prevent duplicate chunks in vector store
    if full_hash in st.session_state.uploaded_docs:
        st.session_state.processed_upload_hashes.add(full_hash)
        st.toast(f"PDF '{clean_name}' is already indexed in your workspace.", icon="ℹ️")
        return True

    # Handle same-name PDFs by establishing distinct source identifiers if needed
    existing_names = [d.get("name") for d in st.session_state.uploaded_docs.values()]
    if clean_name in existing_names:
        stem = Path(clean_name).stem
        ext = Path(clean_name).suffix
        source_name = f"{stem}_{sha_hash[:6]}{ext}"
    else:
        source_name = clean_name

    api_key = resolve_api_key(st.session_state.custom_api_key)

    if not api_key:
        err_msg = "No active API key found. Please enter your Gemini API key in Settings & Specs."
        st.toast("⚠️ Google Gemini API Key required. Configure in Settings or .env.", icon="🔑")
        st.warning(f"⚠️ {err_msg}")
        # Temporary failure: do NOT mark as processed so user can configure key and retry
        st.session_state.processed_upload_hashes.discard(full_hash)
        st.session_state.failed_uploads[full_hash] = {
            "status": "temporary_error",
            "filename": clean_name,
            "error": err_msg,
            "file_bytes": file_bytes,
            "is_pdf": True,
        }
        return False

    is_valid, error_msg = validate_pdf(file_bytes, clean_name)
    if not is_valid:
        st.error(f"❌ {error_msg}")
        # Permanent failure: file structure invalid
        st.session_state.processed_upload_hashes.add(full_hash)
        st.session_state.failed_uploads[full_hash] = {
            "status": "permanent_error",
            "filename": clean_name,
            "error": error_msg,
            "file_bytes": None,
            "is_pdf": True,
        }
        return False

    size_mb = len(file_bytes) / (1024 * 1024)

    with st.status(f"⚡ Ingesting {clean_name}…", expanded=True) as status:
        st.write("📖 **Step 1:** Parsing document pages with PyMuPDF…")
        start_t = time.time()
        try:
            documents, meta = inspect_and_extract_pdf(file_bytes, source_name=source_name, file_id=full_hash)
        except ScannedPDFError as exc:
            status.update(label="Scanned PDF detected", state="error")
            st.warning(f"⚠️ {exc}")
            st.session_state.processed_upload_hashes.add(full_hash)
            st.session_state.failed_uploads[full_hash] = {
                "status": "permanent_error",
                "filename": clean_name,
                "error": str(exc),
                "file_bytes": None,
                "is_pdf": True,
            }
            return False
        except PasswordProtectedPDFError as exc:
            status.update(label="Password-protected PDF", state="error")
            st.error(f"🔒 {exc}")
            st.session_state.processed_upload_hashes.add(full_hash)
            st.session_state.failed_uploads[full_hash] = {
                "status": "permanent_error",
                "filename": clean_name,
                "error": str(exc),
                "file_bytes": None,
                "is_pdf": True,
            }
            return False
        except (CorruptedPDFError, EmptyPDFError) as exc:
            status.update(label="Document parsing error", state="error")
            st.error(f"❌ {exc}")
            st.session_state.processed_upload_hashes.add(full_hash)
            st.session_state.failed_uploads[full_hash] = {
                "status": "permanent_error",
                "filename": clean_name,
                "error": str(exc),
                "file_bytes": None,
                "is_pdf": True,
            }
            return False
        except Exception as exc:
            status.update(label="Extraction failed", state="error")
            sanitized = sanitize_error_message(exc)
            st.error(f"❌ Extraction error: {sanitized}")
            # Extraction failure on valid bytes is permanent if file format issue, or temporary otherwise
            st.session_state.processed_upload_hashes.discard(full_hash)
            st.session_state.failed_uploads[full_hash] = {
                "status": "temporary_error",
                "filename": clean_name,
                "error": sanitized,
                "file_bytes": file_bytes,
                "is_pdf": True,
            }
            return False

        extract_t = time.time() - start_t
        st.write(f"✓ Extracted {len(documents)} text pages ({meta.char_count:,} characters) in {extract_t:.2f}s")

        st.write("🔢 **Step 2:** Generating 3,072-dim embeddings via Google Gen AI…")
        start_t = time.time()

        # Compute accurate sentence chunk count
        splitter = SentenceSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
        nodes = splitter.get_nodes_from_documents(documents, show_progress=False)
        real_chunk_count = len(nodes)

        try:
            active_emb_model = st.session_state.get("embedding_model", GEMINI_EMBEDDING_MODEL)
            index = build_index(
                documents=documents,
                api_key=api_key,
                existing_index=st.session_state.index,
                embedding_model=active_emb_model,
            )
        except DailyLimitExceededError as exc:
            status.update(label="Daily Limit Exceeded", state="error")
            st.error(f"🛑 **Application Limit:** {exc}")
            # Temporary error: allow retry once quota resets or limit increases
            st.session_state.processed_upload_hashes.discard(full_hash)
            st.session_state.failed_uploads[full_hash] = {
                "status": "temporary_error",
                "filename": clean_name,
                "error": str(exc),
                "file_bytes": file_bytes,
                "is_pdf": True,
            }
            return False
        except Exception as exc:
            status.update(label="Embedding failed", state="error")
            quota_info = parse_quota_error(exc, model_name=active_emb_model)
            if quota_info.is_429:
                st.error(f"⚠️ **Google Gemini Quota Alert:** {quota_info.friendly_message}")
                err_text = quota_info.friendly_message
            else:
                translated = translate_genai_error(exc, model_name=active_emb_model)
                st.error(f"❌ Vector embedding error: {translated}")
                err_text = translated
            # Temporary embedding/network/quota error: allow retry
            st.session_state.processed_upload_hashes.discard(full_hash)
            st.session_state.failed_uploads[full_hash] = {
                "status": "temporary_error",
                "filename": clean_name,
                "error": err_text,
                "file_bytes": file_bytes,
                "is_pdf": True,
            }
            return False

        embed_t = time.time() - start_t
        total_chunks = len(index.docstore.docs) if hasattr(index, "docstore") else real_chunk_count
        st.write(f"✓ Embedded into vector store in {embed_t:.2f}s (Total chunks: {total_chunks})")
        status.update(label="✅ Document successfully indexed and active!", state="complete")

    # Update session state ONLY upon successful completion
    st.session_state.index = index
    st.session_state.uploaded_docs[full_hash] = {
        "id": full_hash,
        "name": clean_name,
        "source": source_name,
        "pages": len(documents),
        "size_mb": round(size_mb, 2),
        "chunks": real_chunk_count,
        "hash": sha_hash,
        "full_hash": full_hash,
        "indexed_at": datetime.now().strftime("%H:%M:%S"),
        "documents": documents,
        "doc_ids": [d.doc_id for d in documents],
    }
    st.session_state.processed_upload_hashes.add(full_hash)
    st.session_state.failed_uploads.pop(full_hash, None)

    doc_count = len(st.session_state.uploaded_docs)
    if doc_count == 1:
        st.session_state.doc_name = clean_name
    else:
        st.session_state.doc_name = f"{doc_count} Documents Active"

    st.session_state.doc_size_mb = round(sum(d["size_mb"] for d in st.session_state.uploaded_docs.values()), 2)
    st.session_state.doc_pages = sum(d["pages"] for d in st.session_state.uploaded_docs.values())
    st.session_state.doc_chunks = sum(d["chunks"] for d in st.session_state.uploaded_docs.values())

    st.toast(f"PDF '{clean_name}' indexed successfully!", icon="📄")
    return True


def handle_image_upload(file_bytes: bytes, filename: str) -> bool:
    """Validate, inspect, and register an uploaded image.
    Distinguishes temporary failures from permanent rejections and supports retry.
    """
    clean_name = safe_filename(filename)
    full_hash = file_hash(file_bytes)

    # Check if this exact image is already active in workspace
    if any(file_hash(i.raw_bytes) == full_hash for i in st.session_state.active_images if getattr(i, "raw_bytes", None)):
        st.session_state.processed_upload_hashes.add(full_hash)
        st.toast(f"Image '{clean_name}' is already active in your workspace.", icon="ℹ️")
        return True

    is_valid, val_err = validate_image(file_bytes, clean_name)
    if not is_valid:
        st.error(f"❌ {val_err}")
        st.session_state.processed_upload_hashes.add(full_hash)
        st.session_state.failed_uploads[full_hash] = {
            "status": "permanent_error",
            "filename": clean_name,
            "error": val_err,
            "file_bytes": None,
            "is_pdf": False,
        }
        return False

    try:
        proc_img = process_uploaded_image(file_bytes, clean_name)
    except Exception as exc:
        from PIL import UnidentifiedImageError
        from PIL.Image import DecompressionBombError

        is_permanent = (
            isinstance(exc, (UnidentifiedImageError, DecompressionBombError))
            or "corrupt" in str(exc).lower()
            or "unsupported" in str(exc).lower()
        )
        sanitized = sanitize_error_message(exc)
        st.error(f"❌ Image processing error: {sanitized}")

        if is_permanent:
            st.session_state.processed_upload_hashes.add(full_hash)
            st.session_state.failed_uploads[full_hash] = {
                "status": "permanent_error",
                "filename": clean_name,
                "error": sanitized,
                "file_bytes": None,
                "is_pdf": False,
            }
        else:
            st.session_state.processed_upload_hashes.discard(full_hash)
            st.session_state.failed_uploads[full_hash] = {
                "status": "temporary_error",
                "filename": clean_name,
                "error": sanitized,
                "file_bytes": file_bytes,
                "is_pdf": False,
            }
        return False

    st.session_state.uploaded_images[clean_name] = proc_img
    if not any(img.filename == clean_name for img in st.session_state.active_images):
        st.session_state.active_images.append(proc_img)

    st.session_state.processed_upload_hashes.add(full_hash)
    st.session_state.failed_uploads.pop(full_hash, None)
    st.toast(f"Image '{clean_name}' loaded for visual analysis!", icon="🖼️")
    return True


def retry_failed_upload(full_hash: str) -> bool:
    """Retry ingesting a previously failed upload using its preserved byte stream."""
    rec = st.session_state.failed_uploads.get(full_hash)
    if not rec or not rec.get("file_bytes"):
        return False

    file_bytes = rec["file_bytes"]
    filename = rec["filename"]
    is_pdf_file = rec.get("is_pdf", True)

    st.session_state.failed_uploads.pop(full_hash, None)
    st.session_state.processed_upload_hashes.discard(full_hash)

    if is_pdf_file:
        return handle_pdf_upload(file_bytes, filename)
    else:
        return handle_image_upload(file_bytes, filename)


def dismiss_failed_upload(full_hash: str) -> None:
    """Dismiss a failed upload notification card."""
    st.session_state.failed_uploads.pop(full_hash, None)


def render_failed_uploads() -> None:
    """Render notification cards and retry buttons for any failed uploads."""
    if not st.session_state.get("failed_uploads"):
        return

    st.markdown("<div style='height: 6px;'></div>", unsafe_allow_html=True)
    for h, rec in list(st.session_state.failed_uploads.items()):
        fname = rec.get("filename", "Uploaded File")
        err = rec.get("error", "Processing failed")
        status_type = rec.get("status", "temporary_error")
        has_bytes = rec.get("file_bytes") is not None

        if status_type == "temporary_error" and has_bytes:
            st.markdown(
                f"""
                <div style="background: rgba(245, 158, 11, 0.08); border: 1px solid rgba(245, 158, 11, 0.3); border-radius: 8px; padding: 10px 12px; margin-bottom: 8px;">
                    <div style="font-size: 0.82rem; font-weight: 700; color: #f59e0b; margin-bottom: 3px;">
                        ⚠️ Ingestion Paused: {html.escape(fname)}
                    </div>
                    <div style="font-size: 0.76rem; color: #cbd5e1; margin-bottom: 8px;">
                        {html.escape(err)}
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            col_ret, col_dism = st.columns([2, 1])
            with col_ret:
                if st.button(f"🔄 Retry Ingestion", key=f"retry_btn_{h}", use_container_width=True, type="primary"):
                    ok = retry_failed_upload(h)
                    if ok:
                        st.toast(f"'{fname}' successfully indexed on retry!", icon="✅")
                    st.rerun()
            with col_dism:
                if st.button(f"Dismiss", key=f"dism_btn_{h}", use_container_width=True):
                    dismiss_failed_upload(h)
                    st.rerun()
        else:
            st.markdown(
                f"""
                <div style="background: rgba(239, 68, 68, 0.08); border: 1px solid rgba(239, 68, 68, 0.3); border-radius: 8px; padding: 10px 12px; margin-bottom: 8px;">
                    <div style="font-size: 0.82rem; font-weight: 700; color: #ef4444; margin-bottom: 3px;">
                        ❌ File Rejected: {html.escape(fname)}
                    </div>
                    <div style="font-size: 0.76rem; color: #cbd5e1; margin-bottom: 8px;">
                        {html.escape(err)}
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            if st.button(f"Dismiss Error", key=f"perm_dism_{h}", use_container_width=True):
                dismiss_failed_upload(h)
                st.rerun()


def process_incoming_files(uploaded_files: list) -> bool:
    """Process a batch of uploaded files (PDFs and/or Images).
    Returns True if any new file was successfully processed (indicating state change).
    """
    has_state_changed = False
    for uploaded_file in uploaded_files:
        file_bytes = uploaded_file.read()
        uploaded_file.seek(0)
        h = file_hash(file_bytes)

        if h in st.session_state.processed_upload_hashes:
            continue
        if h in st.session_state.processing_hashes:
            continue

        st.session_state.processing_hashes.add(h)
        try:
            name = safe_filename(uploaded_file.name)
            if is_pdf(name):
                ok = handle_pdf_upload(file_bytes, name)
                if ok:
                    has_state_changed = True
            elif is_image(name):
                ok = handle_image_upload(file_bytes, name)
                if ok:
                    has_state_changed = True
            else:
                st.error(f"Unsupported file type: {html.escape(name)}")
                st.session_state.processed_upload_hashes.add(h)
                st.session_state.failed_uploads[h] = {
                    "status": "permanent_error",
                    "filename": name,
                    "error": f"Unsupported file type '{name}'. Supported formats: PDF, PNG, JPG, JPEG, WEBP.",
                    "file_bytes": None,
                    "is_pdf": False,
                }
        finally:
            st.session_state.processing_hashes.discard(h)

    return has_state_changed


# ── Sidebar Navigation & Management ────────────────────────────────────────
def render_sidebar() -> None:
    with st.sidebar:
        render_brand_header()

        # 1. New Thread Button
        col1, col2 = st.columns([4, 1])
        with col1:
            if st.button("＋  New Thread", use_container_width=True, type="primary"):
                if st.session_state.messages:
                    thread_title = st.session_state.messages[0]["content"][:28] + "…"
                    st.session_state.history_threads.insert(
                        0,
                        {
                            "id": str(time.time()),
                            "title": thread_title,
                            "time": datetime.now().strftime("%b %d, %H:%M"),
                            "messages": list(st.session_state.messages),
                        },
                    )
                st.session_state.messages = []
                st.session_state.prompt_to_submit = None
                st.toast("Started a new multimodal thread.", icon="✨")
                st.rerun()
        with col2:
            st.markdown(
                '<div style="text-align: center; padding-top: 8px; font-size: 0.75rem; color: #64748b; font-family: monospace;">⌘K</div>',
                unsafe_allow_html=True,
            )

        st.markdown("<div style='height: 10px;'></div>", unsafe_allow_html=True)

        # 2. Unified Upload Section in Sidebar
        with st.expander("📤  Upload Documents & Images", expanded=(st.session_state.index is None and not st.session_state.active_images)):
            sidebar_files = st.file_uploader(
                "Upload PDF or Images",
                type=["pdf", "png", "jpg", "jpeg", "webp"],
                accept_multiple_files=True,
                key="sidebar_uploader",
                help="PDF up to 50MB, Images (PNG/JPG/WEBP) up to 20MB",
                label_visibility="collapsed",
            )
            if sidebar_files:
                changed = process_incoming_files(sidebar_files)
                if changed:
                    st.rerun()

            render_failed_uploads()

        # 3. Active Knowledge Context Overview
        has_pdf = st.session_state.index is not None
        has_imgs = len(st.session_state.active_images) > 0

        if has_pdf or has_imgs:
            st.markdown(
                """
                <div style="font-size: 0.72rem; font-weight: 700; color: #64748b; text-transform: uppercase; letter-spacing: 0.06em; margin-bottom: 6px;">
                    Active Knowledge Context
                </div>
                """,
                unsafe_allow_html=True,
            )

            if has_pdf:
                for doc_key, dinfo in st.session_state.uploaded_docs.items():
                    safe_doc_name = html.escape(str(dinfo["name"]))
                    st.markdown(
                        f"""
                        <div style="background: rgba(99, 102, 241, 0.08); border: 1px solid rgba(99, 102, 241, 0.25); border-radius: 8px; padding: 7px 10px; margin-bottom: 5px;">
                            <div style="display: flex; justify-content: space-between; align-items: center;">
                                <span style="font-size: 0.8rem; font-weight: 600; color: #f8fafc; white-space: nowrap; overflow: hidden; text-overflow: ellipsis;">
                                    📄 {safe_doc_name}
                                </span>
                                <span class="media-badge-pdf" style="font-size: 0.62rem;">PDF</span>
                            </div>
                            <div style="font-size: 0.7rem; color: #94a3b8; margin-top: 2px;">
                                {dinfo['pages']} pages · {dinfo['size_mb']} MB
                            </div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

            if has_imgs:
                for idx, img_item in enumerate(st.session_state.active_images):
                    safe_img_name = html.escape(str(img_item.filename))
                    safe_img_format = html.escape(str(img_item.format))
                    img_uid = file_hash(img_item.raw_bytes)[:10] if getattr(img_item, "raw_bytes", None) else str(idx)
                    col_im1, col_im2 = st.columns([5, 1])
                    with col_im1:
                        st.markdown(
                            f"""
                            <div style="background: rgba(236, 72, 153, 0.08); border: 1px solid rgba(236, 72, 153, 0.25); border-radius: 8px; padding: 6px 10px; margin-bottom: 4px;">
                                <div style="display: flex; justify-content: space-between; align-items: center;">
                                    <span style="font-size: 0.8rem; font-weight: 600; color: #f8fafc; white-space: nowrap; overflow: hidden; text-overflow: ellipsis;">
                                        🖼️ {safe_img_name}
                                    </span>
                                    <span class="media-badge-image" style="font-size: 0.62rem;">{safe_img_format}</span>
                                </div>
                                <div style="font-size: 0.7rem; color: #94a3b8;">
                                    {img_item.width}x{img_item.height} · {img_item.size_mb} MB
                                </div>
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )
                    with col_im2:
                        if st.button("×", key=f"rm_sb_img_{idx}_{img_uid}", help="Remove from active context"):
                            st.session_state.active_images = [i for j, i in enumerate(st.session_state.active_images) if j != idx]
                            st.rerun()

        # 4. Scope Filter for Multi-PDF queries
        if len(st.session_state.uploaded_docs) > 1:
            st.markdown("<div style='height: 6px;'></div>", unsafe_allow_html=True)
            filter_map = {"All Indexed Documents": None}
            for d in st.session_state.uploaded_docs.values():
                source_lbl = d.get("source", d["name"])
                filter_map[source_lbl] = source_lbl
            filter_options = list(filter_map.keys())
            current_choice = st.session_state.active_doc_filter if st.session_state.active_doc_filter in filter_options else "All Indexed Documents"
            chosen_filter = st.selectbox(
                "Query Scope",
                options=filter_options,
                index=filter_options.index(current_choice),
                help="Scope retrieval to a specific PDF or across the global multi-document index.",
            )
            st.session_state.active_doc_filter = filter_map[chosen_filter]

        # 5. Recent Conversations History
        st.markdown("<div style='height: 8px;'></div>", unsafe_allow_html=True)
        st.markdown(
            '<div style="font-size: 0.72rem; font-weight: 700; color: #64748b; text-transform: uppercase; letter-spacing: 0.06em; margin-bottom: 8px;">Recent Threads</div>',
            unsafe_allow_html=True,
        )

        if not st.session_state.history_threads:
            st.markdown(
                '<div style="font-size: 0.78rem; color: #475569; padding: 4px; font-style: italic;">No saved threads yet</div>',
                unsafe_allow_html=True,
            )
        else:
            for idx, thread in enumerate(st.session_state.history_threads[:5]):
                col_t1, col_t2 = st.columns([5, 1])
                with col_t1:
                    if st.button(f"💬 {thread['title']}", key=f"th_{thread['id']}", use_container_width=True):
                        st.session_state.messages = list(thread["messages"])
                        st.toast(f"Loaded thread: {thread['title']}", icon="📂")
                        st.rerun()
                with col_t2:
                    if st.button("×", key=f"del_th_{thread['id']}", help="Delete thread"):
                        st.session_state.history_threads.pop(idx)
                        st.rerun()

        # 6. Clear History Action
        if st.session_state.messages or st.session_state.history_threads:
            st.markdown("<div style='height: 8px;'></div>", unsafe_allow_html=True)
            if st.button("🗑️  Clear All Threads", use_container_width=True):
                st.session_state.messages = []
                st.session_state.history_threads = []
                st.toast("Conversation history cleared.", icon="🧹")
                st.rerun()

        # 7. Sidebar Footer User Profile
        st.markdown("<div style='height: 20px;'></div>", unsafe_allow_html=True)
        render_sidebar_profile(
            chat_model=st.session_state.get("chat_model", GEMINI_CHAT_MODEL),
            embedding_model=st.session_state.get("embedding_model", GEMINI_EMBEDDING_MODEL),
        )


# ── Suggested Questions Grid ───────────────────────────────────────────────
def render_suggested_prompts(has_pdf: bool, has_images: bool) -> None:
    """Render smart context-aware starter question cards."""
    st.markdown(
        """
        <div style="margin-top: 1.5rem; margin-bottom: 0.8rem; font-size: 0.85rem; font-weight: 600; color: #94a3b8; text-transform: uppercase; letter-spacing: 0.05em;">
            Suggested Inquiries
        </div>
        """,
        unsafe_allow_html=True,
    )

    col1, col2 = st.columns(2)

    if has_pdf and has_images:
        with col1:
            if st.button("🔄  Cross-Modal Comparison\nCompare the visual diagram/chart with facts stated in the PDF.", key="p_comb1", use_container_width=True):
                st.session_state.prompt_to_submit = "Compare the uploaded visual image/diagram with the relevant sections in the PDF document. What are the key alignments, differences, or complementary insights?"
                st.rerun()
            if st.button("📈  Correlate Chart with Metrics\nExplain the chart in the image using evidence from the document.", key="p_comb2", use_container_width=True):
                st.session_state.prompt_to_submit = "Explain the chart or visual data shown in the uploaded image. Correlate its findings with any corresponding metrics or text in the PDF document."
                st.rerun()
        with col2:
            if st.button("📝  Visual & Textual Synthesis\nProvide an integrated summary combining both image and text.", key="p_comb3", use_container_width=True):
                st.session_state.prompt_to_submit = "Provide a comprehensive synthesis combining insights from both the uploaded image and the PDF document."
                st.rerun()
            if st.button("⚠️  Discrepancy Check\nCheck for contradictions between the visual image and PDF claims.", key="p_comb4", use_container_width=True):
                st.session_state.prompt_to_submit = "Identify any potential discrepancies, contradictions, or unverified claims between the image and the PDF text."
                st.rerun()

    elif has_images:
        with col1:
            if st.button("🔍  Visual Understanding\nWhat is shown in this image? Describe objects, scene, and structure.", key="p_img1", use_container_width=True):
                st.session_state.prompt_to_submit = "Describe what is shown in this image in detail, including all key objects, diagrams, charts, and structural elements."
                st.rerun()
            if st.button("📝  Optical Character Recognition (OCR)\nExtract all readable text, tables, and labels from this image.", key="p_img2", use_container_width=True):
                st.session_state.prompt_to_submit = "Extract all readable text, labels, numbers, and data points present in this image."
                st.rerun()
        with col2:
            if st.button("📊  Diagram & Architecture Analysis\nExplain the technical workflow, data flow, or diagram steps.", key="p_img3", use_container_width=True):
                st.session_state.prompt_to_submit = "Explain the technical workflow, architectural steps, or process diagram shown in this image."
                st.rerun()
            if st.button("💡  Key Visual Insights\nSummarize the main takeaway or findings represented in this visual.", key="p_img4", use_container_width=True):
                st.session_state.prompt_to_submit = "Summarize the primary conclusions, trends, or insights that can be drawn from this image."
                st.rerun()

    else:
        with col1:
            if st.button("⚡  Executive Summary\nSummarize key objectives, methodologies, and conclusions.", key="p_pdf1", use_container_width=True):
                st.session_state.prompt_to_submit = "Please provide a comprehensive executive summary of this document, covering key objectives, core methodologies, and major findings."
                st.rerun()
            if st.button("📊  Quantitative Benchmarks\nExtract all reported metrics, benchmark comparisons, and statistical results.", key="p_pdf2", use_container_width=True):
                st.session_state.prompt_to_submit = "Extract and tabulate all key metrics, quantitative benchmarks, and statistical results reported throughout this document."
                st.rerun()
        with col2:
            if st.button("🔬  Methodology & Architecture\nDetail the experimental setup, system architecture, and data pipeline.", key="p_pdf3", use_container_width=True):
                st.session_state.prompt_to_submit = "What is the technical methodology, architectural pipeline, and implementation details described in this document?"
                st.rerun()
            if st.button("⚠️  Limitations & Risks\nIdentify critical assumptions, failure modes, limitations, and future work.", key="p_pdf4", use_container_width=True):
                st.session_state.prompt_to_submit = "Identify the key limitations, potential failure modes, critical assumptions, and future directions highlighted in this document."
                st.rerun()


# ── Quick-Load Workspace Files ─────────────────────────────────────────────
def render_workspace_shelf() -> None:
    """Show available sample files found in workspace for 1-click loading.
    Uses genuinely unique, stable document identifiers and handles identical filenames correctly.
    """
    workspace_dir = Path(__file__).resolve().parent.parent

    # Collect sample PDFs with deduplication by resolved filesystem path
    seen_pdf_paths: set[Path] = set()
    candidate_pdfs = list(workspace_dir.glob("*.pdf")) + [
        p for p in (workspace_dir / "tests").glob("*.pdf")
        if "corrupt" not in p.name.lower() and "empty" not in p.name.lower()
    ]
    sample_pdfs: list[Path] = []
    for p in candidate_pdfs:
        res = p.resolve()
        if res not in seen_pdf_paths and p.is_file():
            seen_pdf_paths.add(res)
            sample_pdfs.append(p)

    # Collect sample images with deduplication by resolved filesystem path
    seen_img_paths: set[Path] = set()
    candidate_imgs = [p for p in workspace_dir.glob("*.*") if p.suffix.lower() in ALLOWED_IMAGE_EXTENSIONS] + [
        p for p in (workspace_dir / "tests").glob("*.*") if p.suffix.lower() in ALLOWED_IMAGE_EXTENSIONS
    ]
    sample_imgs: list[Path] = []
    for p in candidate_imgs:
        res = p.resolve()
        if res not in seen_img_paths and p.is_file():
            seen_img_paths.add(res)
            sample_imgs.append(p)

    if not sample_pdfs and not sample_imgs:
        return

    st.markdown(
        """
        <div style="margin-top: 2rem; margin-bottom: 0.8rem; font-size: 0.85rem; font-weight: 600; color: #94a3b8; text-transform: uppercase; letter-spacing: 0.05em;">
            Quick Start · Workspace Sample Assets
        </div>
        """,
        unsafe_allow_html=True,
    )

    pdf_name_counts = Counter(p.name for p in sample_pdfs)

    for pdf in sample_pdfs[:4]:
        size_kb = pdf.stat().st_size / 1024
        safe_pdf_name = html.escape(pdf.name)

        # Genuinely unique, stable document identifier based on resolved canonical path
        doc_uid = hashlib.sha256(str(pdf.resolve()).encode("utf-8")).hexdigest()[:12]

        # Disambiguate identical filenames by displaying parent directory if needed
        if pdf_name_counts[pdf.name] > 1:
            try:
                rel_parent = pdf.relative_to(workspace_dir).parent.as_posix()
                type_desc = f"PDF Document ({rel_parent}/)" if rel_parent != "." else "PDF Document (root/)"
            except Exception:
                type_desc = f"PDF Document ({pdf.parent.name}/)"
        else:
            type_desc = "PDF Document"

        # Check if already indexed in active session
        file_bytes = pdf.read_bytes()
        full_hash = file_hash(file_bytes)
        is_already_indexed = full_hash in st.session_state.uploaded_docs

        col_f1, col_f2 = st.columns([4, 1.2])
        with col_f1:
            st.markdown(
                f"""
                <div style="background: rgba(255, 255, 255, 0.02); border: 1px solid rgba(255, 255, 255, 0.07); border-radius: 10px; padding: 10px 14px; display: flex; align-items: center; gap: 12px;">
                    <div style="font-size: 1.3rem;">📄</div>
                    <div>
                        <div style="font-size: 0.88rem; font-weight: 600; color: #f8fafc;">{safe_pdf_name}</div>
                        <div style="font-size: 0.74rem; color: #64748b;">{type_desc} · {size_kb:.1f} KB</div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with col_f2:
            st.markdown("<div style='height: 4px;'></div>", unsafe_allow_html=True)
            btn_key = f"load_ws_pdf_{doc_uid}"
            if is_already_indexed:
                st.button("✓ Active", key=btn_key, use_container_width=True, disabled=True)
            else:
                if st.button("Analyze", key=btn_key, use_container_width=True):
                    handle_pdf_upload(file_bytes, pdf.name)
                    st.rerun()

    img_name_counts = Counter(p.name for p in sample_imgs)

    for img in sample_imgs[:4]:
        size_kb = img.stat().st_size / 1024
        safe_img_name = html.escape(img.name)
        safe_ext_name = html.escape(img.suffix.upper()[1:])

        img_uid = hashlib.sha256(str(img.resolve()).encode("utf-8")).hexdigest()[:12]

        if img_name_counts[img.name] > 1:
            try:
                rel_parent = img.relative_to(workspace_dir).parent.as_posix()
                type_desc = f"Visual Image ({safe_ext_name} · {rel_parent}/)" if rel_parent != "." else f"Visual Image ({safe_ext_name} · root/)"
            except Exception:
                type_desc = f"Visual Image ({safe_ext_name} · {img.parent.name}/)"
        else:
            type_desc = f"Visual Image ({safe_ext_name})"

        img_bytes = img.read_bytes()
        full_hash = file_hash(img_bytes)
        is_already_active = any(
            file_hash(i.raw_bytes) == full_hash for i in st.session_state.active_images if getattr(i, "raw_bytes", None)
        )

        col_f1, col_f2 = st.columns([4, 1.2])
        with col_f1:
            st.markdown(
                f"""
                <div style="background: rgba(255, 255, 255, 0.02); border: 1px solid rgba(255, 255, 255, 0.07); border-radius: 10px; padding: 10px 14px; display: flex; align-items: center; gap: 12px;">
                    <div style="font-size: 1.3rem;">🖼️</div>
                    <div>
                        <div style="font-size: 0.88rem; font-weight: 600; color: #f8fafc;">{safe_img_name}</div>
                        <div style="font-size: 0.74rem; color: #64748b;">{type_desc} · {size_kb:.1f} KB</div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with col_f2:
            st.markdown("<div style='height: 4px;'></div>", unsafe_allow_html=True)
            btn_key = f"load_ws_img_{img_uid}"
            if is_already_active:
                st.button("✓ Active", key=btn_key, use_container_width=True, disabled=True)
            else:
                if st.button("Load Image", key=btn_key, use_container_width=True):
                    handle_image_upload(img_bytes, img.name)
                    st.rerun()


# ── Chat Tab View ──────────────────────────────────────────────────────────
def render_chat_tab() -> None:
    """Render the primary Multimodal conversation interface."""
    has_pdf = st.session_state.index is not None
    has_images = len(st.session_state.active_images) > 0

    render_hero(
        doc_loaded=has_pdf,
        doc_name=st.session_state.doc_name or "",
        image_count=len(st.session_state.active_images),
    )

    # Empty State: No PDF and No Images loaded
    if not has_pdf and not has_images:
        st.markdown(
            """
            <div style="max-width: 680px; margin: 0 auto; text-align: center; padding: 1.5rem 1rem;">
                <div style="font-size: 0.95rem; color: #94a3b8; line-height: 1.6; margin-bottom: 1.5rem;">
                    Upload PDF documents and/or images using the unified drag-and-drop zone below to begin multimodal research.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        main_uploads = st.file_uploader(
            "Upload PDF documents or images (.png, .jpg, .jpeg, .webp)",
            type=["pdf", "png", "jpg", "jpeg", "webp"],
            accept_multiple_files=True,
            key="main_landing_uploader",
            help="Upload PDF (up to 50MB) and/or Images (up to 20MB)",
        )
        if main_uploads:
            changed = process_incoming_files(main_uploads)
            if changed:
                st.rerun()

        render_failed_uploads()
        render_workspace_shelf()
        return

    render_failed_uploads()

    # Visual Thumbnail Strip for currently active images
    if has_images:
        st.markdown(
            f"""
            <div style="display: flex; align-items: center; justify-content: space-between; margin-bottom: 8px;">
                <span style="font-size: 0.76rem; font-weight: 700; color: #ec4899; text-transform: uppercase; letter-spacing: 0.05em;">
                    Active Visual Context ({len(st.session_state.active_images)})
                </span>
            </div>
            """,
            unsafe_allow_html=True,
        )
        img_cols = st.columns(min(len(st.session_state.active_images), 4))
        for idx, proc_img in enumerate(st.session_state.active_images):
            col_target = img_cols[idx % 4]
            with col_target:
                st.image(
                    proc_img.thumbnail_base64 or proc_img.pil_image,
                    caption=f"{proc_img.filename} ({proc_img.width}x{proc_img.height})",
                    use_container_width=True,
                )

    # Render Chat History
    for msg in st.session_state.messages:
        role = msg["role"]
        with st.chat_message(role):
            st.markdown(msg["content"])

            # Render PDF Sources
            if msg.get("pdf_sources"):
                with st.expander(f"📚 Cited PDF Sources ({len(msg['pdf_sources'])} passages)", expanded=False):
                    for src in msg["pdf_sources"]:
                        st.markdown(render_citation_card(src), unsafe_allow_html=True)

            # Render Image Sources
            if msg.get("image_sources"):
                with st.expander(f"🖼️ Cited Visual Sources ({len(msg['image_sources'])} images)", expanded=False):
                    for img_src in msg["image_sources"]:
                        st.markdown(render_image_citation_card(img_src), unsafe_allow_html=True)

    # Handle Pre-filled or Direct User Input
    user_prompt = st.session_state.prompt_to_submit
    st.session_state.prompt_to_submit = None

    if not user_prompt:
        placeholder_text = "Ask a question about the document and images…" if (has_pdf and has_images) else ("Ask about the uploaded image(s)…" if has_images else "Ask a question about the document…")
        user_prompt = st.chat_input(placeholder_text)

    if user_prompt:
        st.session_state.messages.append(
            {
                "role": "user",
                "content": user_prompt,
                "pdf_sources": [],
                "image_sources": [],
                "timestamp": datetime.now().strftime("%H:%M"),
            }
        )
        with st.chat_message("user"):
            st.markdown(user_prompt)

        with st.chat_message("assistant"):
            active_key = resolve_api_key(st.session_state.custom_api_key)
            runtime_chat_model = st.session_state.get("chat_model", GEMINI_CHAT_MODEL)
            model_clean = runtime_chat_model.replace("models/", "")
            with st.spinner(f"Synthesizing multimodal insights with {model_clean}…"):
                try:
                    result: MultimodalQueryResult = query_multimodal(
                        question=user_prompt,
                        pdf_index=st.session_state.index,
                        images=st.session_state.active_images,
                        top_k=st.session_state.top_k_val,
                        api_key=active_key,
                        doc_filter=st.session_state.active_doc_filter,
                        model_name=runtime_chat_model,
                    )
                except DailyLimitExceededError as exc:
                    if st.session_state.messages and st.session_state.messages[-1].get("role") == "user":
                        st.session_state.messages.pop()
                    st.error(f"🛑 **Application Limit:** {exc}")
                    return
                except Exception as exc:
                    if st.session_state.messages and st.session_state.messages[-1].get("role") == "user":
                        st.session_state.messages.pop()
                    quota_info = parse_quota_error(exc, model_name=runtime_chat_model)
                    if quota_info.is_429:
                        st.error(f"⚠️ **Google Gemini Quota Alert:** {quota_info.friendly_message}")
                    else:
                        err_msg = translate_genai_error(exc, model_name=runtime_chat_model)
                        st.error(f"❌ {err_msg}")
                    return

            st.markdown(result.answer)

            if result.pdf_sources:
                with st.expander(f"📚 Cited PDF Sources ({len(result.pdf_sources)} passages · {result.latency_seconds}s)", expanded=True):
                    for src in result.pdf_sources:
                        st.markdown(render_citation_card(src), unsafe_allow_html=True)

            if result.image_sources:
                with st.expander(f"🖼️ Cited Visual Sources ({len(result.image_sources)} images)", expanded=True):
                    for img_src in result.image_sources:
                        st.markdown(render_image_citation_card(img_src), unsafe_allow_html=True)

        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": result.answer,
                "pdf_sources": result.pdf_sources,
                "image_sources": result.image_sources,
                "query_mode": result.query_mode,
                "timestamp": datetime.now().strftime("%H:%M"),
            }
        )
        st.rerun()

    # If no messages in current thread, render suggested inquiry cards
    if not st.session_state.messages and (has_pdf or has_images):
        render_suggested_prompts(has_pdf=has_pdf, has_images=has_images)


# ── Document & Media Hub Tab View ──────────────────────────────────────────
def render_document_hub_tab() -> None:
    """Render Document & Image Library Management Hub."""
    st.markdown(
        """
        <div style="margin-bottom: 1.5rem;">
            <h2 style="font-size: 1.8rem; font-weight: 700; color: #f8fafc; letter-spacing: -0.02em; margin-bottom: 6px;">
                📁 Document & Media Library
            </h2>
            <p style="color: #94a3b8; font-size: 0.95rem;">
                Manage indexed multi-document PDF libraries, inspect visual image assets, monitor chunk counts, and scope knowledge context.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Metrics Overview Row
    m_col1, m_col2, m_col3, m_col4 = st.columns(4)
    with m_col1:
        st.metric("Indexed Documents", len(st.session_state.uploaded_docs))
    with m_col2:
        st.metric("Active Images", len(st.session_state.active_images))
    with m_col3:
        st.metric("Vector Chunks", st.session_state.doc_chunks)
    with m_col4:
        st.metric("Dense Vectors", "3,072 Dims")

    st.markdown("<div style='height: 16px;'></div>", unsafe_allow_html=True)
    render_failed_uploads()

    tab_pdf_lib, tab_img_lib = st.tabs(["📑  PDF Documents", "🖼️  Visual Images"])

    # 1. PDF Documents Library
    with tab_pdf_lib:
        if not st.session_state.uploaded_docs:
            st.info("No PDF documents currently indexed. Upload one or more PDFs from the sidebar or main chat tab.", icon="📄")
        else:
            for doc_key, dinfo in list(st.session_state.uploaded_docs.items()):
                safe_dname = html.escape(str(dinfo["name"]))
                safe_dhash = html.escape(str(dinfo["hash"]))
                doc_source = dinfo.get("source", dinfo["name"])
                is_filtered = (st.session_state.active_doc_filter == doc_source)

                st.markdown(
                    f"""
                    <div style="background: rgba(19, 27, 43, 0.7); border: 1px solid {'rgba(99, 102, 241, 0.5)' if is_filtered else 'rgba(255, 255, 255, 0.08)'}; border-radius: 14px; padding: 18px 22px; margin-bottom: 0.8rem;">
                        <div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 10px;">
                            <div style="display: flex; align-items: center; gap: 14px;">
                                <div style="width: 44px; height: 44px; border-radius: 10px; background: rgba(99, 102, 241, 0.15); display: flex; align-items: center; justify-content: center; font-size: 1.3rem;">
                                    📑
                                </div>
                                <div>
                                    <div style="font-size: 1.05rem; font-weight: 700; color: #f8fafc;">{safe_dname}</div>
                                    <div style="font-size: 0.78rem; color: #94a3b8; display: flex; gap: 14px; margin-top: 4px; flex-wrap: wrap;">
                                        <span>📄 <strong>{dinfo['pages']}</strong> pages</span>
                                        <span>💾 <strong>{dinfo['size_mb']}</strong> MB</span>
                                        <span>🔢 <strong>{dinfo['chunks']}</strong> chunks</span>
                                        <span>🔒 SHA-256: <code>{safe_dhash}</code></span>
                                    </div>
                                </div>
                            </div>
                            <div>
                                <span class="{'status-pill-ready' if is_filtered else 'status-pill-waiting'}">
                                    {'● Scoped Query Target' if is_filtered else '● In Global Index'}
                                </span>
                            </div>
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
                col_act1, col_act2, col_act3 = st.columns([2.5, 2.5, 5])
                with col_act1:
                    if is_filtered:
                        if st.button("Unscope (Search All)", key=f"unscope_{doc_key}", use_container_width=True):
                            st.session_state.active_doc_filter = None
                            st.toast("Queries will now search all indexed documents.", icon="🌐")
                            st.rerun()
                    else:
                        if st.button("Scope Queries to This PDF", key=f"scope_{doc_key}", use_container_width=True):
                            st.session_state.active_doc_filter = doc_source
                            st.toast(f"Scoped queries to '{safe_dname}'", icon="🎯")
                            st.rerun()
                with col_act2:
                    if st.button("🗑️ Remove PDF", key=f"del_{doc_key}", use_container_width=True):
                        doc_info = st.session_state.uploaded_docs.get(doc_key, {})
                        doc_ids = doc_info.get("doc_ids", [])
                        del st.session_state.uploaded_docs[doc_key]
                        st.session_state.processed_upload_hashes.discard(doc_key)

                        # Collect remaining documents from active uploaded_docs
                        remaining_docs = [
                            doc
                            for d in st.session_state.uploaded_docs.values()
                            for doc in d.get("documents", [])
                        ]

                        active_key = resolve_api_key(st.session_state.get("custom_api_key"))
                        new_index = remove_document_from_index(
                            index=st.session_state.index,
                            filename=doc_source,
                            doc_ids=doc_ids,
                            remaining_documents=remaining_docs,
                            api_key=active_key,
                        )
                        st.session_state.index = new_index

                        if st.session_state.active_doc_filter == doc_source:
                            st.session_state.active_doc_filter = None

                        if not st.session_state.uploaded_docs or st.session_state.index is None:
                            st.session_state.index = None
                            st.session_state.doc_name = None
                            st.session_state.doc_pages = 0
                            st.session_state.doc_chunks = 0
                            st.session_state.doc_size_mb = 0.0
                        else:
                            st.session_state.doc_pages = sum(d["pages"] for d in st.session_state.uploaded_docs.values())
                            doc_count = len(st.session_state.uploaded_docs)
                            st.session_state.doc_name = (
                                list(st.session_state.uploaded_docs.values())[0]["name"]
                                if doc_count == 1
                                else f"{doc_count} Documents Active"
                            )
                            st.session_state.doc_chunks = sum(d["chunks"] for d in st.session_state.uploaded_docs.values())
                            st.session_state.doc_size_mb = round(sum(d["size_mb"] for d in st.session_state.uploaded_docs.values()), 2)
                        st.toast(f"Document '{safe_dname}' removed and vector index updated.", icon="🗑️")
                        st.rerun()

    # 2. Visual Images Library
    with tab_img_lib:
        if not st.session_state.active_images and not st.session_state.uploaded_images:
            st.info("No visual images currently loaded. Upload images (.png, .jpg, .jpeg, .webp) from the sidebar.", icon="🖼️")
        else:
            all_imgs: dict[str, ProcessedImage] = {}
            for img in st.session_state.active_images:
                uid = file_hash(img.raw_bytes) if getattr(img, "raw_bytes", None) else img.filename
                all_imgs[uid] = img
            for name, img in st.session_state.uploaded_images.items():
                if isinstance(img, ProcessedImage):
                    uid = file_hash(img.raw_bytes) if getattr(img, "raw_bytes", None) else name
                    all_imgs[uid] = img

            for img_uid, pimg in all_imgs.items():
                is_in_active = any(
                    (file_hash(i.raw_bytes) == img_uid if getattr(i, "raw_bytes", None) else i.filename == pimg.filename)
                    for i in st.session_state.active_images
                )
                safe_short_uid = img_uid[:12]
                col_i1, col_i2 = st.columns([1, 4])
                safe_ifname = html.escape(str(pimg.filename))
                safe_iformat = html.escape(str(pimg.format))
                with col_i1:
                    st.image(pimg.thumbnail_base64 or pimg.pil_image, use_container_width=True)
                with col_i2:
                    st.markdown(
                        f"""
                        <div style="font-size: 1.05rem; font-weight: 700; color: #f8fafc; margin-bottom: 4px;">
                            🖼️ {safe_ifname}
                        </div>
                        <div style="font-size: 0.8rem; color: #94a3b8; display: flex; gap: 16px; margin-bottom: 10px;">
                            <span>Format: <strong>{safe_iformat}</strong></span>
                            <span>Dimensions: <strong>{pimg.width}x{pimg.height} px</strong></span>
                            <span>Size: <strong>{pimg.size_mb} MB</strong></span>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
                    col_b1, col_b2, col_b3 = st.columns([2.5, 2.5, 5])
                    with col_b1:
                        if is_in_active:
                            if st.button("Exclude from Context", key=f"excl_img_{safe_short_uid}", use_container_width=True):
                                st.session_state.active_images = [
                                    i for i in st.session_state.active_images
                                    if (file_hash(i.raw_bytes) != img_uid if getattr(i, "raw_bytes", None) else i.filename != pimg.filename)
                                ]
                                st.rerun()
                        else:
                            if st.button("Include in Context", key=f"incl_img_{safe_short_uid}", use_container_width=True):
                                st.session_state.active_images.append(pimg)
                                st.rerun()
                    with col_b2:
                        if st.button("🗑️ Delete Image", key=f"del_img_{safe_short_uid}", use_container_width=True):
                            st.session_state.active_images = [
                                i for i in st.session_state.active_images
                                if (file_hash(i.raw_bytes) != img_uid if getattr(i, "raw_bytes", None) else i.filename != pimg.filename)
                            ]
                            st.session_state.uploaded_images.pop(pimg.filename, None)
                            st.session_state.uploaded_images.pop(img_uid, None)
                            st.toast(f"Image '{safe_ifname}' deleted.", icon="🗑️")
                            st.rerun()


# ── Settings Tab View ──────────────────────────────────────────────────────
def render_settings_tab() -> None:
    """Render Settings, Hyperparameters, and System Specs."""
    st.markdown(
        """
        <div style="margin-bottom: 1.5rem;">
            <h2 style="font-size: 1.8rem; font-weight: 700; color: #f8fafc; letter-spacing: -0.02em; margin-bottom: 6px;">
                ⚙️ Platform Configuration & Settings
            </h2>
            <p style="color: #94a3b8; font-size: 0.95rem;">
                Inspect multimodal AI models, tune RAG retrieval depth, configure API credentials securely, and audit specifications.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    col1, col2 = st.columns(2)

    with col1:
        current_chat = st.session_state.get("chat_model", GEMINI_CHAT_MODEL)
        current_emb = st.session_state.get("embedding_model", GEMINI_EMBEDDING_MODEL)
        safe_chat_model = html.escape(str(current_chat))
        safe_emb_model = html.escape(str(current_emb))

        st.markdown("### 🤖 Modern Google Gen AI Specifications")
        st.markdown(
            f"""
            <div style="background: rgba(19, 27, 43, 0.7); border: 1px solid rgba(255, 255, 255, 0.08); border-radius: 12px; padding: 18px 20px;">
                <div style="margin-bottom: 12px;">
                    <div style="font-size: 0.75rem; color: #ec4899; font-weight: 700; text-transform: uppercase;">Official SDK</div>
                    <div style="font-size: 1.05rem; font-weight: 700; color: #f8fafc;">google-genai (v3.0.0+)</div>
                    <div style="font-size: 0.8rem; color: #94a3b8;">Direct API bindings without deprecated legacy wrappers</div>
                </div>
                <hr style="border: none; border-top: 1px solid rgba(255, 255, 255, 0.08); margin: 12px 0;">
                <div style="margin-bottom: 12px;">
                    <div style="font-size: 0.75rem; color: #818cf8; font-weight: 700; text-transform: uppercase;">Active Runtime Reasoning LLM</div>
                    <div style="font-size: 1.05rem; font-weight: 700; color: #f8fafc;">{safe_chat_model}</div>
                    <div style="font-size: 0.8rem; color: #94a3b8;">High-speed native vision, diagram comprehension, OCR & document synthesis</div>
                </div>
                <hr style="border: none; border-top: 1px solid rgba(255, 255, 255, 0.08); margin: 12px 0;">
                <div>
                    <div style="font-size: 0.75rem; color: #34d399; font-weight: 700; text-transform: uppercase;">Active Embedding Engine</div>
                    <div style="font-size: 1.05rem; font-weight: 700; color: #f8fafc;">{safe_emb_model}</div>
                    <div style="font-size: 0.8rem; color: #94a3b8;">3,072-dimensional dense semantic vectors</div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        chosen_idx = (
            SUPPORTED_CHAT_MODELS.index(current_chat)
            if current_chat in SUPPORTED_CHAT_MODELS
            else 0
        )
        selected_chat_model = st.selectbox(
            "Configure Active Reasoning Model",
            options=SUPPORTED_CHAT_MODELS,
            index=chosen_idx,
            help="Select the exact Google Gemini model invoked at runtime for chat, RAG, and image vision.",
            key="select_active_chat_model",
        )
        if selected_chat_model != current_chat:
            st.session_state.chat_model = selected_chat_model
            st.toast(f"Active runtime model updated to: {selected_chat_model}", icon="🤖")
            st.rerun()

        st.markdown("### 🔑 API Key Credentials & Security")
        active_key = resolve_api_key(st.session_state.custom_api_key)

        if active_key:
            masked = mask_api_key(active_key)
            source_note = "Session Override" if st.session_state.custom_api_key else "Environment / Secrets"
            st.success(f"Google Gemini API Key Configured (`{masked}`) — *{source_note}*")
            if st.session_state.custom_api_key:
                if st.button("Clear Session API Key", key="clear_session_key"):
                    st.session_state.custom_api_key = ""
                    st.toast("Session API Key cleared.", icon="🧹")
                    st.rerun()
        else:
            st.warning("No API Key detected in `.env` or Streamlit Secrets. Provide a key for this session:")

        with st.form("session_key_form"):
            new_key = st.text_input(
                "Configure Session API Key",
                type="password",
                placeholder="AIzaSy...",
                help="Stored securely in your browser session only. Never mutates global os.environ.",
            )
            submit_key = st.form_submit_button("Validate & Apply Key")
            if submit_key and new_key:
                clean_k = new_key.strip()
                is_valid, msg = validate_api_key_format(clean_k)
                if not is_valid:
                    st.error(f"❌ {msg}")
                else:
                    with st.spinner("Validating key with Google Gen AI…"):
                        live_ok, live_msg = verify_api_key_live(clean_k, model_name=st.session_state.get("chat_model", GEMINI_CHAT_MODEL))
                    if live_ok:
                        st.session_state.custom_api_key = clean_k
                        st.success(f"✅ {live_msg}")
                        st.rerun()
                    else:
                        st.error(f"❌ {live_msg}")

    with col2:
        st.markdown("### 🔍 RAG Retrieval Tuning")
        st.markdown(
            f"""
            <div style="background: rgba(19, 27, 43, 0.7); border: 1px solid rgba(255, 255, 255, 0.08); border-radius: 12px; padding: 18px 20px; margin-bottom: 12px;">
                <div style="display: flex; justify-content: space-between; margin-bottom: 10px;">
                    <span style="color: #94a3b8; font-size: 0.9rem;">Chunk Token Size:</span>
                    <strong style="color: #f8fafc; font-family: monospace;">{CHUNK_SIZE} tokens</strong>
                </div>
                <div style="display: flex; justify-content: space-between; margin-bottom: 10px;">
                    <span style="color: #94a3b8; font-size: 0.9rem;">Chunk Overlap:</span>
                    <strong style="color: #f8fafc; font-family: monospace;">{CHUNK_OVERLAP} tokens</strong>
                </div>
                <div style="display: flex; justify-content: space-between;">
                    <span style="color: #94a3b8; font-size: 0.9rem;">Distance Metric:</span>
                    <strong style="color: #f8fafc; font-family: monospace;">Cosine Similarity</strong>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.session_state.top_k_val = st.slider(
            "Top-K Passage Retrieval Count",
            min_value=1,
            max_value=10,
            value=st.session_state.top_k_val,
            help="Number of most relevant PDF passages retrieved to answer each question.",
        )

    # ── Gemini API Usage & Quota Monitoring Section ───────────────────────────
    st.markdown("<hr style='border: none; border-top: 1px solid rgba(255, 255, 255, 0.08); margin: 2.2rem 0 1.5rem 0;'>", unsafe_allow_html=True)
    st.markdown("### 📊 Gemini API Usage & Quota Monitoring")
    st.markdown(
        """
        <p style="color: #94a3b8; font-size: 0.92rem; margin-bottom: 1.25rem;">
            Centralized telemetry tracking API request volumes, model utilization, SQLite historical trends, and real-time HTTP 429 quota alerts.
        </p>
        """,
        unsafe_allow_html=True,
    )

    # 1. Quota Status Assessment Banner (Accessible to all visitors)
    quota_state = get_quota_status()
    st.markdown(
        f"""
        <div style="background: rgba(19, 27, 43, 0.7); border: 1px solid {quota_state['color']}; border-radius: 12px; padding: 14px 18px; margin-bottom: 1.25rem; display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 10px;">
            <div style="display: flex; align-items: center; gap: 12px;">
                <span style="font-size: 1.4rem;">{quota_state['badge'].split()[0]}</span>
                <div>
                    <div style="font-size: 1.0rem; font-weight: 700; color: #f8fafc;">
                        Quota Health: <span style="color: {quota_state['color']};">{quota_state['status']}</span>
                    </div>
                    <div style="font-size: 0.8rem; color: #94a3b8; margin-top: 2px;">
                        {html.escape(quota_state['description'])}
                    </div>
                </div>
            </div>
            <div style="font-size: 0.75rem; color: #64748b; font-family: monospace; background: rgba(15, 23, 42, 0.5); padding: 4px 10px; border-radius: 6px;">
                Internal Limit: {INTERNAL_DAILY_REQUEST_LIMIT:,} reqs/day
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # 2. Admin Security Verification
    is_admin = st.session_state.get("admin_authenticated", False)

    if not is_admin:
        st.markdown(
            """
            <div style="background: rgba(15, 23, 42, 0.5); border: 1px solid rgba(255, 255, 255, 0.08); border-radius: 12px; padding: 20px; text-align: center; margin-bottom: 1rem;">
                <div style="font-size: 1.5rem; margin-bottom: 6px;">🔒</div>
                <div style="font-size: 1.05rem; font-weight: 700; color: #f8fafc; margin-bottom: 6px;">Administrator Access Required</div>
                <div style="font-size: 0.85rem; color: #94a3b8; max-width: 500px; margin: 0 auto 14px auto;">
                    Detailed API telemetry, daily SQLite request histories, model breakdown analytics, and error logs are restricted to authorized administrators.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        configured_admin_pass = get_admin_password()
        if not configured_admin_pass:
            st.warning(
                "⚠️ **Administrator Access Disabled:** `ADMIN_PASSWORD` is not configured in your environment "
                "or Streamlit secrets (`secrets.toml`). To access detailed telemetry, configure a secure password.",
                icon="🔒",
            )
        else:
            col_ad1, col_ad2, col_ad3 = st.columns([1, 2, 1])
            with col_ad2:
                is_locked, rem_sec = is_admin_locked_out()
                if is_locked:
                    st.error(f"🔒 Admin login temporarily locked due to failed attempts. Try again in {rem_sec}s.")
                else:
                    with st.form("admin_login_form"):
                        pass_input = st.text_input(
                            "Administrator Passphrase",
                            type="password",
                            placeholder="Enter admin password…",
                            help="Configured in Streamlit secrets or ADMIN_PASSWORD environment variable.",
                        )
                        btn_login = st.form_submit_button("Unlock Admin Telemetry Dashboard", use_container_width=True)
                        if btn_login:
                            auth_ok, auth_msg = verify_admin_password(pass_input)
                            if auth_ok:
                                st.session_state.admin_authenticated = True
                                st.toast("Administrator access granted.", icon="🔓")
                                st.rerun()
                            else:
                                st.error(f"❌ {auth_msg}")
    else:
        # Administrator Dashboard Content
        col_adm_header1, col_adm_header2 = st.columns([7, 3])
        with col_adm_header1:
            st.markdown(
                """
                <div style="display: flex; align-items: center; gap: 8px; margin-bottom: 8px;">
                    <span style="font-size: 0.75rem; background: rgba(16, 185, 129, 0.15); color: #10b981; border: 1px solid rgba(16, 185, 129, 0.3); padding: 2px 8px; border-radius: 9999px; font-weight: 700; text-transform: uppercase;">
                        Admin Mode Active
                    </span>
                    <span style="font-size: 0.8rem; color: #94a3b8;">Full telemetry database view</span>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with col_adm_header2:
            if st.button("🔒 Lock Dashboard (Sign Out)", key="btn_admin_signout", use_container_width=True):
                st.session_state.admin_authenticated = False
                st.toast("Signed out of administrator mode.", icon="🔒")
                st.rerun()

        # Quota and Storage Transparency Notice
        st.info(
            "ℹ️ **Quota & Storage Transparency:**\n"
            "• **Application Request Count:** Local request volume tracked centrally in SQLite (`data/usage_history.db`).\n"
            "• **Official Google API Quota:** Managed by Google Cloud Console / AI Studio; Google provides no client API to query remaining balances.\n"
            "• **Estimated Usage:** Inferred from local invocation logs and intercepted HTTP 429 RESOURCE_EXHAUSTED responses.\n"
            "• **Cloud Durability Note:** On ephemeral platforms like Streamlit Community Cloud, local SQLite files reset upon container restart. For persistent production analytics, connect an external database (e.g., PostgreSQL/Supabase).",
            icon="ℹ️",
        )

        metrics = get_comprehensive_dashboard_metrics(is_admin=st.session_state.get("admin_authenticated", False))
        tot = metrics["total_requests"]
        succ = metrics["successful_requests"]
        failed = metrics["failed_requests"]
        q429 = metrics["quota_errors_429"]
        today_req = metrics["requests_today"]
        last_t = metrics["last_request_time"] or "No requests recorded"
        last_lat = f"{metrics['last_latency_seconds']}s"

        # Metric Cards Row 1
        mcol1, mcol2, mcol3, mcol4 = st.columns(4)
        with mcol1:
            st.metric("Total API Requests", f"{tot:,}")
        with mcol2:
            rate = f"{(succ / tot * 100):.1f}%" if tot > 0 else "100%"
            st.metric("Successful Requests", f"{succ:,}", delta=rate)
        with mcol3:
            st.metric(
                "Failed Requests",
                f"{failed:,}",
                delta=f"{failed} errors" if failed > 0 else None,
                delta_color="inverse",
            )
        with mcol4:
            st.metric(
                "HTTP 429 Quota Alerts",
                f"{q429:,}",
                delta="RESOURCE_EXHAUSTED" if q429 > 0 else "No alerts",
                delta_color="inverse",
            )

        # Metric Cards Row 2
        m2col1, m2col2, m2col3 = st.columns(3)
        with m2col1:
            st.metric(
                "Requests Made Today",
                f"{today_req:,}",
                delta=f"{today_req} / {INTERNAL_DAILY_REQUEST_LIMIT} internal limit",
            )
        with m2col2:
            st.metric(
                "Last API Request",
                last_t.split()[-1] if " " in str(last_t) else str(last_t),
                delta=f"Latency: {last_lat}",
            )
        with m2col3:
            st.metric(
                "Estimated App Usage",
                f"{tot:,} calls",
                delta=f"{metrics['requests_by_category'].get('embedding', 0)} emb · {metrics['requests_by_category'].get('chat', 0) + metrics['requests_by_category'].get('multimodal', 0)} chat",
            )

        st.markdown("<div style='height: 14px;'></div>", unsafe_allow_html=True)

        # Charts Row
        chart_col1, chart_col2 = st.columns(2)

        with chart_col1:
            st.markdown("#### 📅 Daily Request Usage (Last 14 Days)")
            trends = metrics.get("daily_trends", [])
            if trends:
                df_trends = pd.DataFrame(trends)
                df_trends = df_trends.rename(
                    columns={
                        "date": "Date",
                        "total": "Total Requests",
                        "quota_429": "429 Quota Errors",
                    }
                )
                st.bar_chart(
                    df_trends,
                    x="Date",
                    y=["Total Requests", "429 Quota Errors"],
                    color=["#6366f1", "#ef4444"],
                    use_container_width=True,
                )
            else:
                st.info("No historical daily request records yet in SQLite.", icon="📊")

        with chart_col2:
            st.markdown("#### 🤖 Model-Wise Usage Breakdown")
            models_dict = metrics.get("requests_by_model", {})
            if models_dict:
                clean_model_items = [
                    {"Model": k.replace("models/", ""), "Requests": v}
                    for k, v in models_dict.items()
                ]
                df_models = pd.DataFrame(clean_model_items).set_index("Model")
                st.bar_chart(
                    df_models,
                    color="#ec4899",
                    use_container_width=True,
                )
            else:
                st.info("No model usage data recorded in SQLite yet.", icon="🤖")

        # Recent Errors Table
        st.markdown("#### ⚠️ Recent API Errors & Rate Limit Alerts")
        errors_list = metrics.get("recent_errors", [])
        if errors_list:
            df_errs = pd.DataFrame(errors_list)
            df_errs = df_errs.rename(
                columns={
                    "timestamp": "Timestamp",
                    "type": "Type",
                    "model": "Model",
                    "status": "Status",
                    "message": "Error Message",
                    "latency": "Latency (s)",
                }
            )
            st.dataframe(df_errs, use_container_width=True, hide_index=True)
        else:
            st.success("✅ No recent API errors or HTTP 429 quota exceptions recorded in SQLite.", icon="✅")

        # Action Buttons Row
        now_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        st.markdown(
            f"""
            <div style="font-size: 0.75rem; color: #64748b; margin-top: 8px; margin-bottom: 8px;">
                Dashboard Last Updated: <strong>{html.escape(now_time)}</strong> · SQLite Storage: <code>{html.escape(str(USAGE_DB_PATH.name))}</code>
            </div>
            """,
            unsafe_allow_html=True,
        )

        btn_c1, btn_c2, btn_c3 = st.columns([2.5, 2.5, 5])
        with btn_c1:
            if st.button("🔄 Refresh Dashboard", key="btn_refresh_dashboard", use_container_width=True):
                st.rerun()
        with btn_c2:
            if st.button("🧹 Reset Telemetry DB", key="btn_reset_db", use_container_width=True, help="Clears all SQLite usage history and session metrics."):
                reset_api_usage_metrics()
                st.toast("SQLite usage history and telemetry counters reset.", icon="🧹")
                st.rerun()


# ── Main Application Router ────────────────────────────────────────────────
def main() -> None:
    st.markdown(get_saas_css(), unsafe_allow_html=True)
    init_session_state()
    render_sidebar()

    tab_chat, tab_docs, tab_settings = st.tabs(
        [
            "💬  Multimodal Chat",
            "📁  Document & Media Hub",
            "⚙️  Settings & Specs",
        ]
    )

    with tab_chat:
        render_chat_tab()

    with tab_docs:
        render_document_hub_tab()

    with tab_settings:
        render_settings_tab()


if __name__ == "__main__":
    main()
