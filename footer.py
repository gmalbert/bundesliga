import streamlit as st


def add_betting_oracle_footer() -> None:
    """Render the Betting Oracle disclaimer footer on every page."""
    st.divider()
    st.markdown(
        """
        <div style='text-align:center;color:#888;font-size:0.78em;padding-bottom:8px;'>
        <strong>Bet Bundesliga</strong> — part of the
        <strong>Betting Oracle</strong> suite ·
        Predictions are for informational purposes only ·
        Past performance does not guarantee future results ·
        <em>Please gamble responsibly.</em>
        </div>
        """,
        unsafe_allow_html=True,
    )
