"""
Reusable UI Components for the AI PDF Research Assistant.
Provides sleek SaaS widgets, citation cards, hero sections, and dashboard cards.
"""

from __future__ import annotations

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
                    <span class="brand-title">RESEARCH.AI</span>
                    <span class="brand-badge">SaaS PRO</span>
                </div>
                <div style="font-size: 0.72rem; color: #64748b; font-weight: 500;">
                    Document Intelligence Platform
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_hero(doc_loaded: bool = False, doc_name: str = "") -> None:
    """Render the main landing page hero section."""
    if not doc_loaded:
        st.markdown(
            """
            <div class="hero-container">
                <div class="hero-badge">
                    <span>✨</span> NEXT-GEN MULTI-PAGE RAG ENGINE · GEMINI 3.5 & LLAMAINDEX
                </div>
                <h1 class="hero-title">Your Research, Reimagined.</h1>
                <p class="hero-subtitle">
                    Upload complex research papers, financial reports, or technical manuals.
                    Ask questions in natural language and receive grounded synthesis with exact page citations.
                </p>
                <div class="metrics-bar">
                    <div class="metric-pill">⚡ <strong>0.01s</strong> Extraction</div>
                    <div class="metric-pill">🔢 <strong>3,072-dim</strong> Gemini Embeddings</div>
                    <div class="metric-pill">📄 <strong>Page-Level</strong> Source Attributions</div>
                    <div class="metric-pill">🛡️ <strong>Zero</strong> Data Retention</div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            f"""
            <div style="margin-bottom: 1.5rem; padding: 1.2rem 1.4rem; background: rgba(19, 27, 43, 0.6); border: 1px solid rgba(255, 255, 255, 0.08); border-radius: 14px; display: flex; align-items: center; justify-content: space-between; flex-wrap: wrap; gap: 12px;">
                <div style="display: flex; align-items: center; gap: 14px;">
                    <div style="width: 40px; height: 40px; border-radius: 10px; background: rgba(99, 102, 241, 0.15); border: 1px solid rgba(99, 102, 241, 0.3); display: flex; align-items: center; justify-content: center; font-size: 1.2rem;">
                        📑
                    </div>
                    <div>
                        <div style="font-size: 0.76rem; text-transform: uppercase; letter-spacing: 0.05em; color: #818cf8; font-weight: 700;">Active Document</div>
                        <div style="font-size: 1.05rem; font-weight: 700; color: #f8fafc;">{doc_name}</div>
                    </div>
                </div>
                <div style="display: flex; align-items: center; gap: 10px;">
                    <span class="status-pill-ready">● Vector Index Active</span>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )


def render_citation_card(src: dict) -> str:
    """Return HTML string for a citation reference card."""
    score = src.get("relevance_score")
    if score is not None:
        percentage = int(score * 100) if score <= 1.0 else int(score)
        score_pill = f'<span class="citation-relevance-pill">{percentage}% Match</span>'
    else:
        score_pill = ''

    return f"""
    <div class="citation-card">
        <div class="citation-header">
            <span class="citation-page-badge">
                <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5">
                    <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"></path>
                    <polyline points="14 2 14 8 20 8"></polyline>
                </svg>
                Page {src['page_number']}
            </span>
            <div style="display: flex; align-items: center; gap: 8px;">
                <span style="color: #64748b; font-size: 0.75rem;">{src.get('source', 'document.pdf')}</span>
                {score_pill}
            </div>
        </div>
        <div class="citation-snippet">
            "{src.get('text_snippet', '')}"
        </div>
    </div>
    """


def render_sidebar_profile(chat_model: str, embedding_model: str) -> None:
    """Render modern user profile and system status card at bottom of sidebar."""
    st.markdown(
        f"""
        <div class="user-profile-box">
            <div class="user-avatar">AI</div>
            <div style="flex: 1; min-width: 0;">
                <div style="display: flex; align-items: center; gap: 6px;">
                    <div style="font-size: 0.85rem; font-weight: 600; color: #f8fafc; white-space: nowrap; overflow: hidden; text-overflow: ellipsis;">
                        Research Engineer
                    </div>
                </div>
                <div style="font-size: 0.72rem; color: #10b981; font-weight: 500; display: flex; align-items: center; gap: 4px;">
                    <span style="display: inline-block; width: 6px; height: 6px; border-radius: 50%; background: #10b981;"></span>
                    Online · {chat_model.replace("models/", "")}
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
