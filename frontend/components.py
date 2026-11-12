"""
Reusable UI Components for the Multimodal AI Research Assistant.
Provides sleek SaaS widgets, PDF citations, image source cards, hero sections, and dashboard cards.
"""

from __future__ import annotations

import html
import streamlit as st


def render_brand_header() -> None:
    """Render the top SaaS branding in the sidebar."""
    st.markdown(
        """
        <div class="brand-container">
            <div class="brand-icon-wrapper">
                <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="white" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round">
                    <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"></path>
                    <polyline points="14 2 14 8 20 8"></polyline>
                    <line x1="16" y1="13" x2="8" y2="13"></line>
                    <line x1="16" y1="17" x2="8" y2="17"></line>
                    <polyline points="10 9 9 9 8 9"></polyline>
                </svg>
            </div>
            <div>
                <div style="display: flex; align-items: center;">
                    <span class="brand-title">AURA.AI</span>
                    <span class="brand-badge">MULTIMODAL</span>
                </div>
                <div style="font-size: 0.72rem; color: #64748b; font-weight: 500;">
                    Document & Visual Intelligence
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_hero(doc_loaded: bool = False, doc_name: str = "", image_count: int = 0) -> None:
    """Render the main landing page hero section."""
    if not doc_loaded and image_count == 0:
        st.markdown(
            """
            <div class="hero-container">
                <div class="hero-badge">
                    <span>✨</span> NEXT-GEN MULTIMODAL RAG · GEMINI 3.5 FLASH & LLAMAINDEX
                </div>
                <h1 class="hero-title">Your Research, Reimagined.</h1>
                <p class="hero-subtitle">
                    Upload multi-page PDF documents and high-resolution images together.
                    Extract text, understand complex architectural diagrams, analyze financial charts,
                    and cross-reference visual evidence with textual context.
                </p>
                <div class="metrics-bar">
                    <div class="metric-pill">📄 <strong>PDF</strong> Semantic RAG</div>
                    <div class="metric-pill">🖼️ <strong>Vision</strong> OCR & Diagrams</div>
                    <div class="metric-pill">🔢 <strong>3,072-dim</strong> Dense Vectors</div>
                    <div class="metric-pill">⚡ <strong>Real-Time</strong> Synthesis</div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        escaped_doc = html.escape(str(doc_name))
        doc_pill = f'<span class="media-badge-pdf">📄 {escaped_doc}</span>' if doc_loaded else ''
        img_pill = f'<span class="media-badge-image">🖼️ {image_count} Image{"s" if image_count != 1 else ""}</span>' if image_count > 0 else ''
        st.markdown(
            f"""
            <div style="margin-bottom: 1.5rem; padding: 1.2rem 1.4rem; background: rgba(19, 27, 43, 0.6); border: 1px solid rgba(255, 255, 255, 0.08); border-radius: 14px; display: flex; align-items: center; justify-content: space-between; flex-wrap: wrap; gap: 12px;">
                <div style="display: flex; align-items: center; gap: 14px;">
                    <div style="width: 40px; height: 40px; border-radius: 10px; background: rgba(99, 102, 241, 0.15); border: 1px solid rgba(99, 102, 241, 0.3); display: flex; align-items: center; justify-content: center; font-size: 1.2rem;">
                        ⚡
                    </div>
                    <div>
                        <div style="font-size: 0.76rem; text-transform: uppercase; letter-spacing: 0.05em; color: #818cf8; font-weight: 700;">Active Knowledge Context</div>
                        <div style="display: flex; align-items: center; gap: 8px; margin-top: 4px; flex-wrap: wrap;">
                            {doc_pill}
                            {img_pill}
                        </div>
                    </div>
                </div>
                <div style="display: flex; align-items: center; gap: 10px;">
                    <span class="status-pill-ready">● Multimodal Engine Ready</span>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )


def render_citation_card(src: dict) -> str:
    """Return HTML string for a PDF citation reference card with strict XSS sanitization."""
    score = src.get("relevance_score")
    if score is not None:
        percentage = int(score * 100) if score <= 1.0 else int(score)
        score_pill = f'<span class="citation-relevance-pill">{percentage}% Match</span>'
    else:
        score_pill = ''

    # Sanitize user/document inputs against HTML injection
    page_num = int(src.get("page_number", 0))
    source_name = html.escape(str(src.get("source", "document.pdf")))
    text_snippet = html.escape(str(src.get("text_snippet", "")))

    return f"""
    <div class="citation-card">
        <div class="citation-header">
            <span class="citation-page-badge">
                <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5">
                    <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"></path>
                    <polyline points="14 2 14 8 20 8"></polyline>
                </svg>
                Page {page_num}
            </span>
            <div style="display: flex; align-items: center; gap: 8px;">
                <span style="color: #64748b; font-size: 0.75rem;">{source_name}</span>
                {score_pill}
            </div>
        </div>
        <div class="citation-snippet">
            "{text_snippet}"
        </div>
    </div>
    """


def render_image_citation_card(img_src: dict) -> str:
    """Return HTML string for an image citation card with thumbnail preview and XSS sanitization."""
    raw_thumb = img_src.get("thumbnail_base64", "")
    # Ensure thumbnail is a valid data URI before embedding in <img> tag
    if raw_thumb and raw_thumb.startswith("data:image/"):
        safe_src = html.escape(raw_thumb, quote=True)
        thumb_html = f'<img src="{safe_src}" class="image-citation-thumb" alt="Citation Thumbnail" />'
    else:
        thumb_html = '<div class="image-citation-thumb" style="display:flex;align-items:center;justify-content:center;background:#1e293b;">🖼️</div>'

    safe_filename = html.escape(str(img_src.get("filename", "image")))
    safe_format = html.escape(str(img_src.get("format", "IMAGE")))
    safe_dims = html.escape(str(img_src.get("dimensions", "N/A")))

    return f"""
    <div class="image-citation-card">
        {thumb_html}
        <div style="flex: 1; min-width: 0;">
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 4px;">
                <span style="font-weight: 700; color: #f472b6; font-size: 0.88rem; display: flex; align-items: center; gap: 6px;">
                    <span>🖼️</span> {safe_filename}
                </span>
                <span class="media-badge-image" style="font-size: 0.68rem;">Visual Source</span>
            </div>
            <div style="font-size: 0.78rem; color: #94a3b8; display: flex; gap: 12px;">
                <span>Format: <strong>{safe_format}</strong></span>
                <span>Dimensions: <strong>{safe_dims}</strong></span>
            </div>
        </div>
    </div>
    """


def render_sidebar_profile(chat_model: str, embedding_model: str) -> None:
    """Render modern user profile and system status card at bottom of sidebar."""
    model_name_clean = html.escape(str(chat_model).replace("models/", ""))
    st.markdown(
        f"""
        <div class="user-profile-box">
            <div class="user-avatar">AI</div>
            <div style="flex: 1; min-width: 0;">
                <div style="display: flex; align-items: center; gap: 6px;">
                    <div style="font-size: 0.85rem; font-weight: 600; color: #f8fafc; white-space: nowrap; overflow: hidden; text-overflow: ellipsis;">
                        AI Research Assistant
                    </div>
                </div>
                <div style="font-size: 0.72rem; color: #10b981; font-weight: 500; display: flex; align-items: center; gap: 4px;">
                    <span style="display: inline-block; width: 6px; height: 6px; border-radius: 50%; background: #10b981;"></span>
                    Online · {model_name_clean}
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
