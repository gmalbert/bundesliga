import streamlit as st

# Nordic Ice — hardcoded permanent theme
# (config.toml sets base bg/text/primary; CSS handles sidebar + accents)

_T = {
    "bg": "#f0f4ff", "text": "#1e3a5f", "card_bg": "#e0eaff",
    "sidebar_bg": "#0f172a", "sidebar_text": "#bfdbfe",
    "accent": "#1d4ed8", "df_header_bg": "#0f172a", "df_header_text": "#bfdbfe",
}

# ── Keep THEMES dict so old imports don't break ────────────────────────────
THEMES: dict[str, dict[str, str]] = {
    "Bundesliga Classic": {
        "bg": "#ffffff", "text": "#111111", "card_bg": "#f7f7f7",
        "sidebar_bg": "#1a1a1a", "sidebar_text": "#f5f5f5",
        "accent": "#D20515", "df_header_bg": "#1a1a1a", "df_header_text": "#ffffff",
    },
    "Bundesliga White": {
        "bg": "#ffffff", "text": "#111111", "card_bg": "#f5f5f5",
        "sidebar_bg": "#e8e8e8", "sidebar_text": "#1a1a1a",
        "accent": "#D20515", "df_header_bg": "#D20515", "df_header_text": "#ffffff",
    },
    "BVB Dortmund": {
        "bg": "#ffffff", "text": "#111111", "card_bg": "#f5f5f0",
        "sidebar_bg": "#0d0d0d", "sidebar_text": "#FDE100",
        "accent": "#c8ad00", "df_header_bg": "#0d0d0d", "df_header_text": "#FDE100",
    },
    "Signal Iduna": {
        "bg": "#fffff8", "text": "#0d0d0d", "card_bg": "#fffce0",
        "sidebar_bg": "#FDE100", "sidebar_text": "#0d0d0d",
        "accent": "#0d0d0d", "df_header_bg": "#FDE100", "df_header_text": "#0d0d0d",
    },
    "Bayern Crimson": {
        "bg": "#fff8f8", "text": "#1a0000", "card_bg": "#fdecea",
        "sidebar_bg": "#5e0010", "sidebar_text": "#fde8ec",
        "accent": "#8B0000", "df_header_bg": "#5e0010", "df_header_text": "#ffffff",
    },
    "Nordic Ice": {
        "bg": "#f0f4ff", "text": "#1e3a5f", "card_bg": "#e0eaff",
        "sidebar_bg": "#0f172a", "sidebar_text": "#bfdbfe",
        "accent": "#1d4ed8", "df_header_bg": "#0f172a", "df_header_text": "#bfdbfe",
    },
    "Monochrome Pro": {
        "bg": "#ffffff", "text": "#111827", "card_bg": "#f3f4f6",
        "sidebar_bg": "#1f2937", "sidebar_text": "#f9fafb",
        "accent": "#374151", "df_header_bg": "#1f2937", "df_header_text": "#f9fafb",
    },
    "Pitch Green": {
        "bg": "#f0fdf4", "text": "#052e16", "card_bg": "#dcfce7",
        "sidebar_bg": "#052e16", "sidebar_text": "#bbf7d0",
        "accent": "#166534", "df_header_bg": "#052e16", "df_header_text": "#bbf7d0",
    },
    "Ocean Wave": {
        "bg": "#eff6ff", "text": "#0c2a4a", "card_bg": "#dbeafe",
        "sidebar_bg": "#0c1a2e", "sidebar_text": "#bae6fd",
        "accent": "#0369a1", "df_header_bg": "#0c1a2e", "df_header_text": "#bae6fd",
    },
    "Stadium Lights": {
        "bg": "#fffdf7", "text": "#1a1200", "card_bg": "#fef9e7",
        "sidebar_bg": "#111111", "sidebar_text": "#fde68a",
        "accent": "#d97706", "df_header_bg": "#111111", "df_header_text": "#fde68a",
    },
    "Amber Arena": {
        "bg": "#fffbeb", "text": "#1c0a00", "card_bg": "#fef3c7",
        "sidebar_bg": "#1c0a00", "sidebar_text": "#fde68a",
        "accent": "#b45309", "df_header_bg": "#1c0a00", "df_header_text": "#fde68a",
    },
    "Golden Cup": {
        "bg": "#fefce8", "text": "#1a1000", "card_bg": "#fef9c3",
        "sidebar_bg": "#1a1400", "sidebar_text": "#fef08a",
        "accent": "#ca8a04", "df_header_bg": "#1a1400", "df_header_text": "#fef08a",
    },
    "Rose Champion": {
        "bg": "#fff1f2", "text": "#3d0012", "card_bg": "#ffe4e6",
        "sidebar_bg": "#4c0519", "sidebar_text": "#fecdd3",
        "accent": "#be123c", "df_header_bg": "#4c0519", "df_header_text": "#fecdd3",
    },
    "Slate Modern": {
        "bg": "#f8fafc", "text": "#1e293b", "card_bg": "#e2e8f0",
        "sidebar_bg": "#0f172a", "sidebar_text": "#cbd5e1",
        "accent": "#475569", "df_header_bg": "#0f172a", "df_header_text": "#cbd5e1",
    },
    "Emerald Elite": {
        "bg": "#f0fdf8", "text": "#064e3b", "card_bg": "#d1fae5",
        "sidebar_bg": "#064e3b", "sidebar_text": "#a7f3d0",
        "accent": "#059669", "df_header_bg": "#064e3b", "df_header_text": "#a7f3d0",
    },
    "Violet Kick": {
        "bg": "#faf5ff", "text": "#1e0a3c", "card_bg": "#ede9fe",
        "sidebar_bg": "#2e1065", "sidebar_text": "#ddd6fe",
        "accent": "#7c3aed", "df_header_bg": "#2e1065", "df_header_text": "#ddd6fe",
    },
}

THEME_NAMES = list(THEMES.keys())
DEFAULT_THEME = "Nordic Ice"


def _build_css(t: dict[str, str]) -> str:
    return f"""
        <style>
        /* ── Page & card backgrounds ── */
        .stApp,
        [data-testid="stAppViewContainer"],
        [data-testid="stMain"] {{
            background-color: {t["bg"]} !important;
        }}
        [data-testid="stMain"] section > div,
        [data-testid="stVerticalBlock"] > div > div[data-testid="element-container"] > div,
        [data-testid="stForm"],
        .stTabs [data-baseweb="tab-panel"] {{
            background-color: {t["card_bg"]} !important;
        }}

        /* ── Main text ── */
        [data-testid="stMain"] p,
        [data-testid="stMain"] h1,
        [data-testid="stMain"] h2,
        [data-testid="stMain"] h3,
        [data-testid="stMain"] h4,
        [data-testid="stMain"] label,
        [data-testid="stMarkdownContainer"] * {{
            color: {t["text"]} !important;
        }}

        /* ── Tabs ── */
        .stTabs [data-baseweb="tab-list"] {{
            gap: 6px;
            background-color: {t["bg"]} !important;
        }}
        .stTabs [data-baseweb="tab"] {{
            border-radius: 4px 4px 0 0;
            padding: 8px 14px;
            color: {t["text"]} !important;
            background-color: {t["card_bg"]} !important;
        }}
        .stTabs [aria-selected="true"] {{
            border-bottom: 2px solid {t["accent"]} !important;
            color: {t["accent"]} !important;
        }}

        /* ── Metric & progress ── */
        [data-testid="stMetricValue"] {{ color: {t["accent"]} !important; font-weight: 700; }}
        [data-testid="stProgressBar"] > div > div {{ background-color: {t["accent"]}; }}

        /* ── Sidebar ── */
        [data-testid="stSidebar"],
        [data-testid="stSidebar"] > div {{
            background-color: {t["sidebar_bg"]} !important;
        }}
        /* Blanket: catch every element Streamlit renders in the sidebar */
        [data-testid="stSidebar"] *,
        [data-testid="stSidebar"] *::before,
        [data-testid="stSidebar"] *::after {{
            color: {t["sidebar_text"]} !important;
        }}
        /* Nav item active highlight — keep accent bg, override text to match */
        [data-testid="stSidebar"] [aria-selected="true"],
        [data-testid="stSidebar"] [aria-current="page"] {{
            background-color: {t["accent"]}33 !important;
            color: {t["sidebar_text"]} !important;
        }}
        /* Selectbox control */
        [data-testid="stSidebar"] [data-baseweb="select"] > div {{
            background-color: {t["sidebar_bg"]} !important;
            border-color: {t["sidebar_text"]}55 !important;
        }}
        [data-testid="stSidebar"] [data-baseweb="select"] svg {{
            fill: {t["sidebar_text"]} !important;
        }}

        /* ── All dropdown popups ── */
        [data-baseweb="popover"] [role="listbox"],
        [data-baseweb="menu"] {{
            background-color: {t["sidebar_bg"]} !important;
        }}
        [data-baseweb="popover"] [role="option"],
        [data-baseweb="menu"] li {{
            background-color: {t["sidebar_bg"]} !important;
            color: {t["sidebar_text"]} !important;
        }}
        [data-baseweb="popover"] [role="option"]:hover,
        [data-baseweb="menu"] li:hover {{
            background-color: {t["accent"]}33 !important;
        }}
        [data-baseweb="popover"] [aria-selected="true"] {{
            background-color: {t["accent"]}55 !important;
        }}
        </style>
    """


def apply_theme(theme_name: str = DEFAULT_THEME) -> None:
    """Inject CSS for the chosen theme."""
    t = THEMES.get(theme_name, _T)
    st.markdown(_build_css(t), unsafe_allow_html=True)
