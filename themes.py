import streamlit as st


def apply_theme() -> None:
    """Inject La Liga brand CSS overrides."""
    st.markdown(
        """
        <style>
        /* La Liga red accent on metric values */
        [data-testid="stMetricValue"] { color: #EF0000; font-weight: 700; }

        /* Dark sidebar */
        [data-testid="stSidebar"] { background-color: #1a1a2e; }

        /* Tab strip spacing */
        .stTabs [data-baseweb="tab-list"] { gap: 6px; }
        .stTabs [data-baseweb="tab"] {
            border-radius: 4px 4px 0 0;
            padding: 8px 14px;
        }

        /* Tighten dataframe header */
        [data-testid="stDataFrame"] thead th {
            background-color: #1a1a2e;
            color: #fafafa;
        }

        /* Progress bar fill color */
        [data-testid="stProgressBar"] > div > div {
            background-color: #EF0000;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )
