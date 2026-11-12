"""
Premium AI SaaS Frontend for AI PDF Research Assistant.
Redesigned with inspiration from ChatGPT, Linear, Notion AI, and Perplexity.
"""

from __future__ import annotations

import os
import time
from datetime import datetime
from pathlib import Path

import streamlit as st

from backend.embeddings import build_index
from backend.pdf_processor import extract_documents_from_pdf
from backend.query_engine import QueryResult, query_index
from config.settings import (
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    GEMINI_CHAT_MODEL,
    GEMINI_EMBEDDING_MODEL,
    TOP_K_RESULTS,
    get_google_api_key,
)
from frontend.components import (
    render_brand_header,
    render_citation_card,
    render_hero,
    render_sidebar_profile,
)
from frontend.styles import get_saas_css
from utils.helpers import file_hash, validate_pdf


# ── Page Configuration ─────────────────────────────────────────────────────
st.set_page_config(
    page_title="RESEARCH.AI · Document Intelligence",
    page_icon="✨",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ── Session State Initialization ───────────────────────────────────────────
def init_session_state() -> None:
    defaults = {
        "messages": [],           # Current conversation [{"role": ..., "content": ..., "sources": [...]}]
        "history_threads": [],     # Previous saved threads [{"id": ..., "title": ..., "messages": [...]}]
        "index": None,             # Active VectorStoreIndex
        "doc_name": None,          # Active PDF filename
        "doc_size_mb": 0.0,        # Active PDF size in MB
        "doc_pages": 0,            # Page count
        "doc_chunks": 0,           # Number of embedded chunks
        "doc_hash": "",            # SHA-256 preview
        "uploaded_docs": {},       # Registered documents cache {filename: {"pages": ..., "size": ..., "index": ...}}
        "prompt_to_submit": None,  # For suggested prompt card clicks
        "theme_accent": "Linear Violet",
        "custom_api_key": "",
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val


# ── Document Ingestion Handler ─────────────────────────────────────────────
def process_pdf_file(file_bytes: bytes, filename: str) -> bool:
    """Validate, parse, chunk, embed and index a PDF file."""
    if not get_google_api_key():
        st.toast("⚠️ Google API Key is required before analyzing documents.", icon="🔑")
        return False

    is_valid, error_msg = validate_pdf(file_bytes, filename)
    if not is_valid:
        st.error(f"❌ {error_msg}")
        return False

    size_mb = len(file_bytes) / (1024 * 1024)
    sha_hash = file_hash(file_bytes)[:12]

    # Visual status container
    with st.status(f"⚡ Ingesting {filename}…", expanded=True) as status:
        st.write("📖 **Step 1:** Extracting page structures with PyMuPDF…")
        start_t = time.time()
        try:
            documents = extract_documents_from_pdf(file_bytes, filename)
        except Exception as exc:
            status.update(label="Extraction failed", state="error")
            st.error(f"❌ Extraction error: {exc}")
            return False

        extract_t = time.time() - start_t
        st.write(f"✓ Extracted {len(documents)} pages in {extract_t:.2f}s")

        st.write("🔢 **Step 2:** Generating 3,072-dim embeddings via Google Gemini…")
        start_t = time.time()
        try:
            index = build_index(documents)
        except Exception as exc:
            status.update(label="Embedding failed", state="error")
            st.error(f"❌ Vector embedding error: {exc}")
            return False

        embed_t = time.time() - start_t
        chunk_count = len(index.docstore.docs)
        st.write(f"✓ Stored {chunk_count} vector chunks in {embed_t:.2f}s")

        status.update(label="✅ Document successfully indexed and active!", state="complete")

    # Update session state
    st.session_state.index = index
    st.session_state.doc_name = filename
    st.session_state.doc_size_mb = round(size_mb, 2)
    st.session_state.doc_pages = len(documents)
    st.session_state.doc_chunks = chunk_count
    st.session_state.doc_hash = sha_hash

    # Save to uploaded docs registry
    st.session_state.uploaded_docs[filename] = {
        "name": filename,
        "pages": len(documents),
        "size_mb": round(size_mb, 2),
        "chunks": chunk_count,
        "hash": sha_hash,
        "index": index,
        "indexed_at": datetime.now().strftime("%H:%M:%S"),
    }

    # If previous chat had messages, archive it to thread history
    if st.session_state.messages:
        thread_title = st.session_state.messages[0]["content"][:32] + "…"
        st.session_state.history_threads.insert(
            0,
            {
                "id": str(time.time()),
                "title": thread_title,
                "doc": filename,
                "time": datetime.now().strftime("%b %d, %H:%M"),
                "messages": list(st.session_state.messages),
            },
        )
        st.session_state.messages = []

    st.toast(f"Document '{filename}' is ready for research!", icon="🚀")
    return True


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
                            "doc": st.session_state.doc_name or "General",
                            "time": datetime.now().strftime("%b %d, %H:%M"),
                            "messages": list(st.session_state.messages),
                        },
                    )
                st.session_state.messages = []
                st.session_state.prompt_to_submit = None
                st.toast("Started a new research thread.", icon="✨")
                st.rerun()
        with col2:
            st.markdown(
                '<div style="text-align: center; padding-top: 8px; font-size: 0.75rem; color: #64748b; font-family: monospace;">⌘K</div>',
                unsafe_allow_html=True,
            )

        st.markdown("<div style='height: 12px;'></div>", unsafe_allow_html=True)

        # 2. Quick Upload in Sidebar
        with st.expander("📤  Upload New Document", expanded=(st.session_state.index is None)):
            uploaded_file = st.file_uploader(
                "Drop PDF file here",
                type=["pdf"],
                help="PDF up to 50 MB with text layer",
                label_visibility="collapsed",
            )
            if uploaded_file is not None:
                if st.session_state.doc_name != uploaded_file.name:
                    file_bytes = uploaded_file.read()
                    if process_pdf_file(file_bytes, uploaded_file.name):
                        st.rerun()

        # 3. Active Document Badge
        if st.session_state.index is not None:
            st.markdown(
                f"""
                <div style="background: rgba(99, 102, 241, 0.08); border: 1px solid rgba(99, 102, 241, 0.25); border-radius: 10px; padding: 10px 12px; margin-bottom: 1rem;">
                    <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 4px;">
                        <span style="font-size: 0.72rem; font-weight: 700; color: #a5b4fc; text-transform: uppercase;">Active Document</span>
                        <span class="status-pill-ready" style="font-size: 0.68rem; padding: 2px 7px;">Indexed</span>
                    </div>
                    <div style="font-size: 0.88rem; font-weight: 600; color: #f8fafc; white-space: nowrap; overflow: hidden; text-overflow: ellipsis;">
                        📄 {st.session_state.doc_name}
                    </div>
                    <div style="font-size: 0.75rem; color: #94a3b8; margin-top: 4px;">
                        {st.session_state.doc_pages} pages · {st.session_state.doc_chunks} vector chunks
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        # 4. Recent Conversations History
        st.markdown(
            '<div style="font-size: 0.75rem; font-weight: 700; color: #64748b; text-transform: uppercase; letter-spacing: 0.06em; margin-bottom: 8px;">Recent Threads</div>',
            unsafe_allow_html=True,
        )

        if not st.session_state.history_threads:
            st.markdown(
                '<div style="font-size: 0.78rem; color: #475569; padding: 6px 4px; font-style: italic;">No saved threads yet</div>',
                unsafe_allow_html=True,
            )
        else:
            for idx, thread in enumerate(st.session_state.history_threads[:5]):
                col_t1, col_t2 = st.columns([5, 1])
                with col_t1:
                    if st.button(
                        f"💬 {thread['title']}",
                        key=f"th_{thread['id']}",
                        use_container_width=True,
                    ):
                        st.session_state.messages = list(thread["messages"])
                        st.toast(f"Loaded thread: {thread['title']}", icon="📂")
                        st.rerun()
                with col_t2:
                    if st.button("×", key=f"del_th_{thread['id']}", help="Delete thread"):
                        st.session_state.history_threads.pop(idx)
                        st.rerun()

        # 5. Clear History Action
        if st.session_state.messages or st.session_state.history_threads:
            st.markdown("<div style='height: 8px;'></div>", unsafe_allow_html=True)
            if st.button("🗑️  Clear All Threads", use_container_width=True):
                st.session_state.messages = []
                st.session_state.history_threads = []
                st.toast("Conversation history cleared.", icon="🧹")
                st.rerun()

        # 6. Sidebar Footer User Profile
        st.markdown("<div style='height: 24px;'></div>", unsafe_allow_html=True)
        render_sidebar_profile(
            chat_model=GEMINI_CHAT_MODEL,
            embedding_model=GEMINI_EMBEDDING_MODEL,
        )


# ── Suggested Questions Grid ───────────────────────────────────────────────
def render_suggested_prompts() -> None:
    """Render 4 interactive starter question cards."""
    st.markdown(
        """
        <div style="margin-top: 1.5rem; margin-bottom: 0.8rem; font-size: 0.85rem; font-weight: 600; color: #94a3b8; text-transform: uppercase; letter-spacing: 0.05em;">
            Suggested Inquiries
        </div>
        """,
        unsafe_allow_html=True,
    )

    col1, col2 = st.columns(2)
    with col1:
        if st.button(
            "⚡  Executive Summary\nSummarize the key objectives, findings, and executive conclusions.",
            key="p1",
            use_container_width=True,
        ):
            st.session_state.prompt_to_submit = "Please provide a comprehensive executive summary of this document, covering key objectives, core methodologies, and major findings."
            st.rerun()

        if st.button(
            "📊  Quantitative Benchmarks\nExtract all reported metrics, benchmark comparisons, and statistical results.",
            key="p2",
            use_container_width=True,
        ):
            st.session_state.prompt_to_submit = "Extract and tabulate all key metrics, quantitative benchmarks, and statistical results reported throughout this document."
            st.rerun()

    with col2:
        if st.button(
            "🔬  Methodology & Architecture\nDetail the experimental setup, system architecture, and data pipeline.",
            key="p3",
            use_container_width=True,
        ):
            st.session_state.prompt_to_submit = "What is the technical methodology, architectural pipeline, and implementation details described in this document?"
            st.rerun()

        if st.button(
            "⚠️  Limitations & Risks\nIdentify critical assumptions, failure modes, limitations, and future work.",
            key="p4",
            use_container_width=True,
        ):
            st.session_state.prompt_to_submit = "Identify the key limitations, potential failure modes, critical assumptions, and future directions highlighted in this document."
            st.rerun()


# ── Quick-Load Workspace Documents ─────────────────────────────────────────
def render_workspace_documents_shelf() -> None:
    """Show available PDF files found in workspace for instant 1-click loading."""
    workspace_dir = Path(__file__).resolve().parent.parent
    sample_files = list(workspace_dir.glob("*.pdf"))

    if not sample_files:
        return

    st.markdown(
        """
        <div style="margin-top: 2rem; margin-bottom: 0.8rem; font-size: 0.85rem; font-weight: 600; color: #94a3b8; text-transform: uppercase; letter-spacing: 0.05em;">
            Quick Start · Workspace Sample Documents
        </div>
        """,
        unsafe_allow_html=True,
    )

    for pdf in sample_files[:3]:
        size_kb = pdf.stat().st_size / 1024
        col_f1, col_f2 = st.columns([4, 1.2])
        with col_f1:
            st.markdown(
                f"""
                <div style="background: rgba(255, 255, 255, 0.02); border: 1px solid rgba(255, 255, 255, 0.07); border-radius: 10px; padding: 10px 14px; display: flex; align-items: center; gap: 12px;">
                    <div style="font-size: 1.3rem;">📄</div>
                    <div>
                        <div style="font-size: 0.88rem; font-weight: 600; color: #f8fafc;">{pdf.name}</div>
                        <div style="font-size: 0.74rem; color: #64748b;">Ready on disk · {size_kb:.1f} KB</div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with col_f2:
            st.markdown("<div style='height: 4px;'></div>", unsafe_allow_html=True)
            if st.button(f"Analyze", key=f"load_{pdf.name}", use_container_width=True):
                file_bytes = pdf.read_bytes()
                if process_pdf_file(file_bytes, pdf.name):
                    st.rerun()


# ── Chat Tab View ──────────────────────────────────────────────────────────
def render_chat_tab() -> None:
    """Render the primary ChatGPT/Linear-style conversation interface."""
    render_hero(
        doc_loaded=(st.session_state.index is not None),
        doc_name=st.session_state.doc_name or "",
    )

    # Empty state / Hero when no document is active
    if st.session_state.index is None:
        st.markdown(
            """
            <div style="max-width: 680px; margin: 0 auto; text-align: center; padding: 2rem 1rem;">
                <div style="font-size: 0.95rem; color: #94a3b8; line-height: 1.6; margin-bottom: 1.5rem;">
                    Upload any research document using the drag-and-drop zone below or in the sidebar to activate the semantic search engine.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # Main landing upload zone
        main_upload = st.file_uploader(
            "Upload your PDF document to begin research",
            type=["pdf"],
            key="main_landing_uploader",
            help="Supported: PDF up to 50MB (Text layer)",
        )
        if main_upload is not None:
            if st.session_state.doc_name != main_upload.name:
                file_bytes = main_upload.read()
                if process_pdf_file(file_bytes, main_upload.name):
                    st.rerun()

        # Render available workspace PDFs shelf
        render_workspace_documents_shelf()
        return

    # Render Existing Messages
    for msg_idx, msg in enumerate(st.session_state.messages):
        role = msg["role"]
        with st.chat_message(role):
            st.markdown(msg["content"])

            # Render Expandable Citations
            if msg.get("sources"):
                with st.expander(f"📚 Cited Sources ({len(msg['sources'])} passages retrieved)", expanded=False):
                    for src in msg["sources"]:
                        st.markdown(render_citation_card(src), unsafe_allow_html=True)

    # Handle suggested prompt card clicks or direct user chat input
    user_prompt = st.session_state.prompt_to_submit
    st.session_state.prompt_to_submit = None  # Consume pre-filled prompt

    if not user_prompt:
        user_prompt = st.chat_input("Ask a research question about this document…")

    if user_prompt:
        # 1. Append & Display User Message
        st.session_state.messages.append(
            {
                "role": "user",
                "content": user_prompt,
                "sources": [],
                "timestamp": datetime.now().strftime("%H:%M"),
            }
        )
        with st.chat_message("user"):
            st.markdown(user_prompt)

        # 2. Generate Assistant Response with Progress Indicator
        with st.chat_message("assistant"):
            with st.spinner("Analyzing semantic vectors & formulating answer…"):
                start_t = time.time()
                try:
                    result: QueryResult = query_index(st.session_state.index, user_prompt)
                except Exception as exc:
                    st.error(f"❌ Error generating response: {exc}")
                    return
                query_duration = time.time() - start_t

            # Display markdown response
            st.markdown(result.answer)

            # Display citations
            if result.sources:
                with st.expander(f"📚 Cited Sources ({len(result.sources)} passages · {query_duration:.2f}s)", expanded=True):
                    for src in result.sources:
                        st.markdown(render_citation_card(src), unsafe_allow_html=True)

        # 3. Save assistant message with structured sources
        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": result.answer,
                "sources": result.sources,
                "timestamp": datetime.now().strftime("%H:%M"),
            }
        )
        st.rerun()

    # If no messages yet on active document, show suggested question cards
    if not st.session_state.messages and st.session_state.index is not None:
        render_suggested_prompts()


# ── Document Hub Tab View ──────────────────────────────────────────────────
def render_document_hub_tab() -> None:
    """Render Document Management, Ingestion Stats, and Upload Dashboard."""
    st.markdown(
        """
        <div style="margin-bottom: 1.5rem;">
            <h2 style="font-size: 1.8rem; font-weight: 700; color: #f8fafc; letter-spacing: -0.02em; margin-bottom: 6px;">
                📁 Document Management Hub
            </h2>
            <p style="color: #94a3b8; font-size: 0.95rem;">
                Inspect indexed documents, chunk distributions, vector metrics, and manage your research library.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if not st.session_state.uploaded_docs and not st.session_state.doc_name:
        st.info("No documents are currently indexed in this session. Upload a document from the sidebar to view metrics.", icon="📄")
        return

    # Metrics Overview Row
    m_col1, m_col2, m_col3, m_col4 = st.columns(4)
    with m_col1:
        st.metric("Total Indexed Docs", len(st.session_state.uploaded_docs) or (1 if st.session_state.doc_name else 0))
    with m_col2:
        st.metric("Active Pages", st.session_state.doc_pages)
    with m_col3:
        st.metric("Vector Chunks", st.session_state.doc_chunks)
    with m_col4:
        st.metric("Embedding Dim", "3,072 Dims")

    st.markdown("<div style='height: 20px;'></div>", unsafe_allow_html=True)

    # Document Cards
    docs_to_display = dict(st.session_state.uploaded_docs)
    if st.session_state.doc_name and st.session_state.doc_name not in docs_to_display:
        docs_to_display[st.session_state.doc_name] = {
            "name": st.session_state.doc_name,
            "pages": st.session_state.doc_pages,
            "size_mb": st.session_state.doc_size_mb,
            "chunks": st.session_state.doc_chunks,
            "hash": st.session_state.doc_hash,
            "index": st.session_state.index,
            "indexed_at": "Active",
        }

    for fname, dinfo in docs_to_display.items():
        is_active = (st.session_state.doc_name == fname)
        st.markdown(
            f"""
            <div style="background: rgba(19, 27, 43, 0.7); border: 1px solid {'rgba(99, 102, 241, 0.4)' if is_active else 'rgba(255, 255, 255, 0.08)'}; border-radius: 14px; padding: 18px 22px; margin-bottom: 1rem;">
                <div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 10px;">
                    <div style="display: flex; align-items: center; gap: 14px;">
                        <div style="width: 44px; height: 44px; border-radius: 10px; background: rgba(99, 102, 241, 0.15); display: flex; align-items: center; justify-content: center; font-size: 1.3rem;">
                            📑
                        </div>
                        <div>
                            <div style="font-size: 1.05rem; font-weight: 700; color: #f8fafc;">{dinfo['name']}</div>
                            <div style="font-size: 0.78rem; color: #94a3b8; display: flex; gap: 14px; margin-top: 4px;">
                                <span>📄 <strong>{dinfo['pages']}</strong> pages</span>
                                <span>💾 <strong>{dinfo['size_mb']}</strong> MB</span>
                                <span>🔢 <strong>{dinfo['chunks']}</strong> vector chunks</span>
                                <span>🔒 SHA-256: <code>{dinfo['hash']}</code></span>
                            </div>
                        </div>
                    </div>
                    <div>
                        <span class="{'status-pill-ready' if is_active else 'status-pill-waiting'}">
                            {'● Active' if is_active else '● In Cache'}
                        </span>
                    </div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        col_act1, col_act2, col_act3 = st.columns([2, 2, 6])
        with col_act1:
            if not is_active:
                if st.button("Set Active", key=f"set_act_{fname}", use_container_width=True):
                    st.session_state.index = dinfo["index"]
                    st.session_state.doc_name = dinfo["name"]
                    st.session_state.doc_pages = dinfo["pages"]
                    st.session_state.doc_chunks = dinfo["chunks"]
                    st.session_state.doc_size_mb = dinfo["size_mb"]
                    st.session_state.doc_hash = dinfo["hash"]
                    st.toast(f"Switched active document to {fname}", icon="🔄")
                    st.rerun()
        with col_act2:
            if st.button("🗑️ Delete Document", key=f"del_{fname}", use_container_width=True):
                if fname in st.session_state.uploaded_docs:
                    del st.session_state.uploaded_docs[fname]
                if st.session_state.doc_name == fname:
                    st.session_state.index = None
                    st.session_state.doc_name = None
                    st.session_state.doc_pages = 0
                    st.session_state.doc_chunks = 0
                    st.session_state.messages = []
                st.toast(f"Document '{fname}' deleted.", icon="🗑️")
                st.rerun()


# ── Settings Tab View ──────────────────────────────────────────────────────
def render_settings_tab() -> None:
    """Render SaaS Preferences, Model Specs, and API Configuration."""
    st.markdown(
        """
        <div style="margin-bottom: 1.5rem;">
            <h2 style="font-size: 1.8rem; font-weight: 700; color: #f8fafc; letter-spacing: -0.02em; margin-bottom: 6px;">
                ⚙️ Platform Configuration & Settings
            </h2>
            <p style="color: #94a3b8; font-size: 0.95rem;">
                Manage AI models, vector search parameters, API security, and UI preferences.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### 🤖 Model Specifications")
        st.markdown(
            f"""
            <div style="background: rgba(19, 27, 43, 0.7); border: 1px solid rgba(255, 255, 255, 0.08); border-radius: 12px; padding: 18px 20px;">
                <div style="margin-bottom: 12px;">
                    <div style="font-size: 0.75rem; color: #818cf8; font-weight: 700; text-transform: uppercase;">Generation LLM</div>
                    <div style="font-size: 1.1rem; font-weight: 700; color: #f8fafc;">{GEMINI_CHAT_MODEL}</div>
                    <div style="font-size: 0.8rem; color: #94a3b8;">High-speed reasoning with grounded RAG context</div>
                </div>
                <hr style="border: none; border-top: 1px solid rgba(255, 255, 255, 0.08); margin: 12px 0;">
                <div>
                    <div style="font-size: 0.75rem; color: #34d399; font-weight: 700; text-transform: uppercase;">Embedding Engine</div>
                    <div style="font-size: 1.1rem; font-weight: 700; color: #f8fafc;">{GEMINI_EMBEDDING_MODEL}</div>
                    <div style="font-size: 0.8rem; color: #94a3b8;">3,072-dimensional semantic dense vectors</div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown("### 🔑 API Key Status")
        active_key = get_google_api_key()
        if active_key:
            masked = active_key[:6] + "..." + active_key[-4:]
            st.success(f"Google Gemini API Key Configured (`{masked}`)")
        else:
            st.warning("No API Key detected in `.env`. Enter a key below for this session:")
            user_key = st.text_input("Gemini API Key", type="password", key="settings_key_input")
            if user_key:
                os.environ["GOOGLE_API_KEY"] = user_key.strip()
                st.session_state.custom_api_key = user_key.strip()
                st.toast("API Key stored for session!", icon="✅")
                st.rerun()

    with col2:
        st.markdown("### 🔍 RAG Hyperparameters")
        st.markdown(
            f"""
            <div style="background: rgba(19, 27, 43, 0.7); border: 1px solid rgba(255, 255, 255, 0.08); border-radius: 12px; padding: 18px 20px;">
                <div style="display: flex; justify-content: space-between; margin-bottom: 10px;">
                    <span style="color: #94a3b8; font-size: 0.9rem;">Chunk Token Size:</span>
                    <strong style="color: #f8fafc; font-family: monospace;">{CHUNK_SIZE} tokens</strong>
                </div>
                <div style="display: flex; justify-content: space-between; margin-bottom: 10px;">
                    <span style="color: #94a3b8; font-size: 0.9rem;">Chunk Overlap:</span>
                    <strong style="color: #f8fafc; font-family: monospace;">{CHUNK_OVERLAP} tokens</strong>
                </div>
                <div style="display: flex; justify-content: space-between; margin-bottom: 10px;">
                    <span style="color: #94a3b8; font-size: 0.9rem;">Similarity Top-K:</span>
                    <strong style="color: #f8fafc; font-family: monospace;">{TOP_K_RESULTS} passages</strong>
                </div>
                <div style="display: flex; justify-content: space-between;">
                    <span style="color: #94a3b8; font-size: 0.9rem;">Distance Metric:</span>
                    <strong style="color: #f8fafc; font-family: monospace;">Cosine Similarity</strong>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown("### 🎨 Theme Accents")
        accent_choice = st.selectbox(
            "Accent Palette",
            ["Linear Violet (Default)", "Midnight Sapphire", "Emerald Mint"],
            index=0,
        )
        if accent_choice:
            st.session_state.theme_accent = accent_choice


# ── Main Application Router ────────────────────────────────────────────────
def main() -> None:
    # Inject Custom CSS Design System
    st.markdown(get_saas_css(), unsafe_allow_html=True)

    # Initialize State
    init_session_state()

    # Render Sidebar
    render_sidebar()

    # Top Navigation Tabs
    tab_chat, tab_docs, tab_settings = st.tabs(
        [
            "💬  Research Chat",
            "📁  Document Hub",
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
