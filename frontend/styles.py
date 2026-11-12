"""
Modern Dark SaaS Theme and CSS Design System for AI PDF Research Assistant.
Inspired by Linear, Perplexity, Notion AI, and ChatGPT.
"""

def get_saas_css() -> str:
    """Return comprehensive custom CSS for the modern SaaS dark theme."""
    return """
    <style>
    /* =========================================================================
       1. CORE DESIGN TOKENS & RESET
       ========================================================================= */
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600&display=swap');

    :root {
        --bg-main: #0a0e17;
        --bg-surface: #0f1624;
        --bg-card: rgba(19, 27, 43, 0.7);
        --bg-card-hover: rgba(26, 38, 62, 0.85);
        --bg-sidebar: #0b111e;
        --border-subtle: rgba(255, 255, 255, 0.07);
        --border-glow: rgba(99, 102, 241, 0.35);
        --accent-primary: #6366f1;
        --accent-gradient: linear-gradient(135deg, #6366f1 0%, #8b5cf6 50%, #ec4899 100%);
        --accent-gradient-subtle: linear-gradient(135deg, rgba(99, 102, 241, 0.15) 0%, rgba(139, 92, 246, 0.15) 100%);
        --text-primary: #f8fafc;
        --text-secondary: #94a3b8;
        --text-muted: #64748b;
        --badge-emerald: #10b981;
        --badge-amber: #f59e0b;
        --radius-sm: 8px;
        --radius-md: 12px;
        --radius-lg: 18px;
        --radius-xl: 24px;
        --font-sans: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif;
        --font-mono: 'JetBrains Mono', monospace;
    }

    /* Overall background & typography */
    .stApp {
        background-color: var(--bg-main) !important;
        font-family: var(--font-sans) !important;
        color: var(--text-primary) !important;
    }

    /* Hide default Streamlit top decoration */
    header[data-testid="stHeader"] {
        background: transparent !important;
        backdrop-filter: blur(12px) !important;
    }

    /* Main container constraint */
    .main .block-container {
        max-width: 1100px !important;
        padding-top: 1.5rem !important;
        padding-bottom: 5rem !important;
    }

    /* Custom smooth scrollbars */
    ::-webkit-scrollbar {
        width: 6px;
        height: 6px;
    }
    ::-webkit-scrollbar-track {
        background: rgba(11, 17, 30, 0.6);
    }
    ::-webkit-scrollbar-thumb {
        background: rgba(255, 255, 255, 0.15);
        border-radius: 999px;
    }
    ::-webkit-scrollbar-thumb:hover {
        background: rgba(255, 255, 255, 0.3);
    }

    /* =========================================================================
       2. SIDEBAR STYLING
       ========================================================================= */
    [data-testid="stSidebar"] {
        background-color: var(--bg-sidebar) !important;
        border-right: 1px solid var(--border-subtle) !important;
        padding-top: 1rem !important;
    }

    [data-testid="stSidebar"] .block-container {
        padding: 1.25rem 1rem !important;
    }

    .brand-container {
        display: flex;
        align-items: center;
        gap: 12px;
        padding: 0.5rem 0.25rem 1.2rem 0.25rem;
        border-bottom: 1px solid var(--border-subtle);
        margin-bottom: 1.2rem;
    }

    .brand-icon-wrapper {
        width: 38px;
        height: 38px;
        border-radius: var(--radius-md);
        background: var(--accent-gradient);
        display: flex;
        align-items: center;
        justify-content: center;
        box-shadow: 0 4px 16px rgba(99, 102, 241, 0.4);
    }

    .brand-title {
        font-size: 1.15rem;
        font-weight: 700;
        letter-spacing: -0.02em;
        color: var(--text-primary);
        line-height: 1.2;
    }

    .brand-badge {
        font-size: 0.65rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        background: rgba(99, 102, 241, 0.2);
        color: #a5b4fc;
        padding: 2px 7px;
        border-radius: 999px;
        border: 1px solid rgba(99, 102, 241, 0.4);
        margin-left: 6px;
    }

    /* User Profile Card in Sidebar */
    .user-profile-box {
        display: flex;
        align-items: center;
        gap: 12px;
        background: rgba(255, 255, 255, 0.03);
        border: 1px solid var(--border-subtle);
        border-radius: var(--radius-md);
        padding: 10px 12px;
        margin-top: 1rem;
        transition: all 0.2s ease;
    }

    .user-profile-box:hover {
        background: rgba(255, 255, 255, 0.05);
        border-color: rgba(255, 255, 255, 0.12);
    }

    .user-avatar {
        width: 32px;
        height: 32px;
        border-radius: 50%;
        background: linear-gradient(135deg, #3b82f6, #10b981);
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 0.85rem;
        font-weight: 700;
        color: white;
    }

    /* =========================================================================
       3. HERO & LANDING DASHBOARD
       ========================================================================= */
    .hero-container {
        text-align: center;
        padding: 3.5rem 1.5rem 2.5rem 1.5rem;
        position: relative;
    }

    .hero-badge {
        display: inline-flex;
        align-items: center;
        gap: 8px;
        font-size: 0.78rem;
        font-weight: 600;
        letter-spacing: 0.04em;
        color: #c7d2fe;
        background: rgba(99, 102, 241, 0.12);
        border: 1px solid rgba(99, 102, 241, 0.3);
        padding: 6px 14px;
        border-radius: 999px;
        margin-bottom: 1.5rem;
        box-shadow: 0 2px 12px rgba(99, 102, 241, 0.15);
    }

    .hero-title {
        font-size: 3.2rem;
        font-weight: 800;
        letter-spacing: -0.04em;
        line-height: 1.15;
        margin-bottom: 1rem;
        background: linear-gradient(180deg, #ffffff 30%, #94a3b8 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }

    .hero-subtitle {
        font-size: 1.15rem;
        font-weight: 400;
        color: var(--text-secondary);
        max-width: 620px;
        margin: 0 auto 2rem auto;
        line-height: 1.6;
    }

    /* Feature & Metric Highlights */
    .metrics-bar {
        display: flex;
        justify-content: center;
        gap: 24px;
        margin: 1.5rem 0 2rem 0;
        flex-wrap: wrap;
    }

    .metric-pill {
        display: flex;
        align-items: center;
        gap: 8px;
        background: rgba(255, 255, 255, 0.02);
        border: 1px solid var(--border-subtle);
        border-radius: var(--radius-sm);
        padding: 6px 14px;
        font-size: 0.8rem;
        color: var(--text-secondary);
    }

    .metric-pill strong {
        color: var(--text-primary);
    }

    /* Suggested Questions Cards Grid */
    .prompt-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(240px, 1fr));
        gap: 14px;
        margin-top: 1.8rem;
        margin-bottom: 2rem;
    }

    .prompt-card {
        background: var(--bg-card);
        border: 1px solid var(--border-subtle);
        border-radius: var(--radius-md);
        padding: 16px 18px;
        text-align: left;
        cursor: pointer;
        transition: all 0.25s cubic-bezier(0.16, 1, 0.3, 1);
        position: relative;
        overflow: hidden;
    }

    .prompt-card:hover {
        background: var(--bg-card-hover);
        border-color: rgba(99, 102, 241, 0.4);
        transform: translateY(-2px);
        box-shadow: 0 10px 24px -10px rgba(99, 102, 241, 0.3);
    }

    .prompt-card-icon {
        font-size: 1.25rem;
        margin-bottom: 8px;
    }

    .prompt-card-title {
        font-size: 0.92rem;
        font-weight: 600;
        color: var(--text-primary);
        margin-bottom: 4px;
    }

    .prompt-card-desc {
        font-size: 0.8rem;
        color: var(--text-muted);
        line-height: 1.45;
    }

    /* =========================================================================
       4. CHAT INTERFACE & MESSAGES
       ========================================================================= */
    /* Streamlit Chat Messages Dark SaaS Overhaul */
    div[data-testid="stChatMessage"] {
        background: transparent !important;
        border: none !important;
        padding: 1rem 0.5rem !important;
        margin-bottom: 0.5rem !important;
    }

    /* User Message Styling */
    div[data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-user"]) {
        flex-direction: row-reverse !important;
    }

    div[data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-user"]) [data-testid="stChatMessageContent"] {
        background: linear-gradient(135deg, rgba(99, 102, 241, 0.25) 0%, rgba(79, 70, 229, 0.35) 100%) !important;
        border: 1px solid rgba(129, 140, 248, 0.35) !important;
        border-radius: 18px 18px 4px 18px !important;
        padding: 14px 18px !important;
        color: #f1f5f9 !important;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.2) !important;
        max-width: 82% !important;
        margin-left: auto !important;
    }

    /* Assistant Message Styling */
    div[data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-assistant"]) [data-testid="stChatMessageContent"] {
        background: rgba(19, 27, 43, 0.75) !important;
        border: 1px solid var(--border-subtle) !important;
        border-radius: 18px 18px 18px 4px !important;
        padding: 16px 20px !important;
        color: #f8fafc !important;
        backdrop-filter: blur(12px) !important;
        box-shadow: 0 4px 24px rgba(0, 0, 0, 0.25) !important;
        max-width: 90% !important;
        margin-right: auto !important;
        line-height: 1.65 !important;
    }

    /* Code blocks inside chat */
    pre, code {
        font-family: var(--font-mono) !important;
    }
    pre {
        background: #060910 !important;
        border: 1px solid var(--border-subtle) !important;
        border-radius: var(--radius-sm) !important;
        padding: 12px 16px !important;
    }

    /* =========================================================================
       5. SOURCE REFERENCES & CITATIONS
       ========================================================================= */
    .citation-container {
        margin-top: 14px;
        padding-top: 12px;
        border-top: 1px solid rgba(255, 255, 255, 0.08);
    }

    .citation-card {
        background: rgba(11, 16, 26, 0.85);
        border: 1px solid rgba(255, 255, 255, 0.07);
        border-left: 3px solid var(--accent-primary);
        border-radius: var(--radius-sm);
        padding: 10px 14px;
        margin: 8px 0;
        transition: all 0.2s ease;
    }

    .citation-card:hover {
        background: rgba(15, 23, 38, 0.95);
        border-color: rgba(99, 102, 241, 0.4);
        transform: translateX(2px);
    }

    .citation-header {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 6px;
        font-size: 0.82rem;
    }

    .citation-page-badge {
        font-weight: 700;
        color: #a5b4fc;
        display: inline-flex;
        align-items: center;
        gap: 5px;
    }

    .citation-relevance-pill {
        background: rgba(16, 185, 129, 0.15);
        color: #34d399;
        border: 1px solid rgba(16, 185, 129, 0.3);
        padding: 2px 8px;
        border-radius: 999px;
        font-size: 0.72rem;
        font-family: var(--font-mono);
        font-weight: 600;
    }

    .citation-snippet {
        font-size: 0.8rem;
        color: var(--text-secondary);
        font-style: italic;
        line-height: 1.5;
    }

    /* =========================================================================
       6. BUTTONS & INPUTS
       ========================================================================= */
    /* Streamlit Chat Input Bar */
    div[data-testid="stChatInput"] {
        border-radius: var(--radius-xl) !important;
        background: rgba(15, 23, 42, 0.8) !important;
        backdrop-filter: blur(16px) !important;
        border: 1px solid rgba(255, 255, 255, 0.12) !important;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.4) !important;
    }

    div[data-testid="stChatInput"]:focus-within {
        border-color: var(--accent-primary) !important;
        box-shadow: 0 0 0 2px rgba(99, 102, 241, 0.3), 0 8px 32px rgba(0, 0, 0, 0.5) !important;
    }

    /* SaaS Secondary & Primary Buttons */
    .stButton > button {
        border-radius: var(--radius-md) !important;
        font-weight: 600 !important;
        font-size: 0.88rem !important;
        letter-spacing: -0.01em !important;
        transition: all 0.2s cubic-bezier(0.16, 1, 0.3, 1) !important;
        border: 1px solid var(--border-subtle) !important;
        background: rgba(255, 255, 255, 0.04) !important;
        color: var(--text-primary) !important;
    }

    .stButton > button:hover {
        background: rgba(255, 255, 255, 0.08) !important;
        border-color: rgba(255, 255, 255, 0.2) !important;
        transform: translateY(-1px) !important;
    }

    /* Primary CTA Buttons */
    .stButton > button[kind="primary"] {
        background: var(--accent-gradient) !important;
        border: none !important;
        color: white !important;
        box-shadow: 0 4px 14px rgba(99, 102, 241, 0.4) !important;
    }

    .stButton > button[kind="primary"]:hover {
        box-shadow: 0 6px 20px rgba(99, 102, 241, 0.6) !important;
        transform: translateY(-2px) !important;
    }

    /* File Uploader Container */
    div[data-testid="stFileUploader"] {
        border-radius: var(--radius-lg) !important;
    }

    div[data-testid="stFileUploader"] section {
        background: rgba(255, 255, 255, 0.02) !important;
        border: 2px dashed rgba(99, 102, 241, 0.35) !important;
        border-radius: var(--radius-lg) !important;
        padding: 1.5rem !important;
        transition: all 0.25s ease !important;
    }

    div[data-testid="stFileUploader"] section:hover {
        background: rgba(99, 102, 241, 0.04) !important;
        border-color: var(--accent-primary) !important;
    }

    /* Tabs Styling (Linear Style) */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px !important;
        background-color: transparent !important;
        border-bottom: 1px solid var(--border-subtle) !important;
        padding-bottom: 4px !important;
    }

    .stTabs [data-baseweb="tab"] {
        background-color: transparent !important;
        border-radius: var(--radius-sm) !important;
        color: var(--text-secondary) !important;
        font-weight: 500 !important;
        font-size: 0.9rem !important;
        padding: 8px 16px !important;
        border: none !important;
        transition: all 0.2s ease !important;
    }

    .stTabs [data-baseweb="tab"]:hover {
        color: var(--text-primary) !important;
        background: rgba(255, 255, 255, 0.04) !important;
    }

    .stTabs [aria-selected="true"] {
        color: #ffffff !important;
        background: rgba(99, 102, 241, 0.15) !important;
        border-bottom: 2px solid var(--accent-primary) !important;
    }

    /* Status Pills */
    .status-pill-ready {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        background: rgba(16, 185, 129, 0.12);
        color: #34d399;
        border: 1px solid rgba(16, 185, 129, 0.25);
        padding: 4px 10px;
        border-radius: 999px;
        font-size: 0.78rem;
        font-weight: 600;
    }

    .status-pill-waiting {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        background: rgba(245, 158, 11, 0.12);
        color: #fbbf24;
        border: 1px solid rgba(245, 158, 11, 0.25);
        padding: 4px 10px;
        border-radius: 999px;
        font-size: 0.78rem;
        font-weight: 600;
    }

    /* =========================================================================
       7. MULTIMODAL & IMAGE ENHANCEMENTS
       ========================================================================= */
    .image-citation-card {
        background: rgba(15, 23, 42, 0.85);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-left: 3px solid #ec4899;
        border-radius: var(--radius-sm);
        padding: 12px 14px;
        margin: 8px 0;
        display: flex;
        gap: 14px;
        align-items: center;
        transition: all 0.2s ease;
    }

    .image-citation-card:hover {
        background: rgba(26, 38, 62, 0.95);
        border-color: rgba(236, 72, 153, 0.4);
        transform: translateX(2px);
    }

    .image-citation-thumb {
        width: 64px;
        height: 64px;
        border-radius: 8px;
        object-fit: cover;
        border: 1px solid rgba(255, 255, 255, 0.12);
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.3);
    }

    .media-badge-pdf {
        background: rgba(99, 102, 241, 0.15);
        color: #a5b4fc;
        border: 1px solid rgba(99, 102, 241, 0.3);
        padding: 2px 8px;
        border-radius: 999px;
        font-size: 0.72rem;
        font-weight: 600;
    }

    .media-badge-image {
        background: rgba(236, 72, 153, 0.15);
        color: #f472b6;
        border: 1px solid rgba(236, 72, 153, 0.3);
        padding: 2px 8px;
        border-radius: 999px;
        font-size: 0.72rem;
        font-weight: 600;
    }

    .image-strip-container {
        display: flex;
        gap: 12px;
        overflow-x: auto;
        padding: 10px 4px;
        margin-bottom: 1rem;
    }

    .image-strip-card {
        background: rgba(19, 27, 43, 0.7);
        border: 1px solid var(--border-subtle);
        border-radius: var(--radius-md);
        padding: 8px;
        min-width: 140px;
        text-align: center;
        transition: all 0.2s ease;
    }

    .image-strip-card:hover {
        border-color: rgba(236, 72, 153, 0.5);
        transform: translateY(-2px);
    }

    /* Responsive adjustments */
    @media (max-width: 768px) {
        .hero-title {
            font-size: 2.2rem !important;
        }
        .hero-subtitle {
            font-size: 1rem !important;
        }
        .prompt-grid {
            grid-template-columns: 1fr !important;
        }
        .metrics-bar {
            gap: 10px !important;
        }
    }
    </style>
    """
