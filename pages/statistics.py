"""Statistics page — xG rankings, team form, H2H, Copa del Rey congestion."""

from os import path

import pandas as pd
import streamlit as st

from utils import get_dataframe_height, load_historical_data

HIST_PATH = "data_files/combined_historical_data.csv"

st.title("📊 Statistics")

if not path.exists(HIST_PATH):
    st.info("Run `python fetch_historical_csvs.py` to unlock statistics.")
    st.stop()

hist_df = load_historical_data(HIST_PATH)


# ── xG Rankings ───────────────────────────────────────────────────────────
st.subheader("⚽ Team xG Rankings (FBref)")
fbref_path = "data_files/raw/fbref_team_xg.csv"
if path.exists(fbref_path):
    xg_df = pd.read_csv(fbref_path)
    xg_df = xg_df.sort_values("xG", ascending=False).reset_index(drop=True)
    xg_df.insert(0, "#", xg_df.index + 1)
    st.dataframe(xg_df, hide_index=True, use_container_width=True,
                 height=get_dataframe_height(xg_df, max_height=500))
else:
    st.info("xG data not loaded. Run `python fetch_fbref_xg.py`.")

st.divider()


# ── Team Form ─────────────────────────────────────────────────────────────
st.subheader("📈 Recent Team Form (Last 5 Matches)")

all_teams = sorted(
    set(hist_df["HomeTeam"].dropna()) | set(hist_df["AwayTeam"].dropna())
)

form_rows = []
icons = {"W": "🟢", "D": "🟡", "L": "🔴"}

for team in all_teams:
    home_m = hist_df[hist_df["HomeTeam"] == team][["MatchDate", "FullTimeResult"]].assign(
        Won=lambda d: (d["FullTimeResult"] == "H"),
        Drew=lambda d: (d["FullTimeResult"] == "D"),
    )
    away_m = hist_df[hist_df["AwayTeam"] == team][["MatchDate", "FullTimeResult"]].assign(
        Won=lambda d: (d["FullTimeResult"] == "A"),
        Drew=lambda d: (d["FullTimeResult"] == "D"),
    )
    all_m = (
        pd.concat([home_m, away_m])
        .sort_values("MatchDate")
        .tail(5)
    )
    form_str  = "".join("W" if r["Won"] else ("D" if r["Drew"] else "L") for _, r in all_m.iterrows())
    form_disp = " ".join(icons.get(c, c) for c in form_str)
    pts_l5    = sum(3 if c == "W" else (1 if c == "D" else 0) for c in form_str)

    form_rows.append({"Team": team, "Form": form_disp, "Pts (L5)": pts_l5, "_form_str": form_str})

form_df = (
    pd.DataFrame(form_rows)
    .sort_values("Pts (L5)", ascending=False)
    .reset_index(drop=True)
)
form_df.insert(0, "#", form_df.index + 1)
st.dataframe(
    form_df[["#", "Team", "Form", "Pts (L5)"]],
    hide_index=True,
    use_container_width=True,
    height=get_dataframe_height(form_df, max_height=680),
)

st.divider()


# ── Head-to-Head Analyzer ─────────────────────────────────────────────────
st.subheader("🏆 Head-to-Head Analyzer")
hc1, hc2 = st.columns(2)
with hc1:
    t1 = st.selectbox("Team 1", all_teams, key="h2h_t1")
with hc2:
    t2 = st.selectbox("Team 2", [t for t in all_teams if t != t1], key="h2h_t2")

if st.button("🔍 Analyse H2H", use_container_width=False):
    mask = (
        ((hist_df["HomeTeam"] == t1) & (hist_df["AwayTeam"] == t2)) |
        ((hist_df["HomeTeam"] == t2) & (hist_df["AwayTeam"] == t1))
    )
    h2h = hist_df[mask].sort_values("MatchDate", ascending=False).head(10)
    if h2h.empty:
        st.info(f"No recorded meetings between {t1} and {t2}.")
    else:
        t1_wins = (
            ((h2h["HomeTeam"] == t1) & (h2h["FullTimeResult"] == "H")).sum() +
            ((h2h["AwayTeam"] == t1) & (h2h["FullTimeResult"] == "A")).sum()
        )
        t2_wins = (
            ((h2h["HomeTeam"] == t2) & (h2h["FullTimeResult"] == "H")).sum() +
            ((h2h["AwayTeam"] == t2) & (h2h["FullTimeResult"] == "A")).sum()
        )
        draws = (h2h["FullTimeResult"] == "D").sum()

        hc1r, hc2r, hc3r = st.columns(3)
        hc1r.metric(f"{t1} Wins",  int(t1_wins))
        hc2r.metric("Draws",       int(draws))
        hc3r.metric(f"{t2} Wins",  int(t2_wins))

        show_cols = ["MatchDate", "HomeTeam", "FullTimeHomeGoals",
                     "FullTimeAwayGoals", "AwayTeam", "FullTimeResult"]
        show_cols = [c for c in show_cols if c in h2h.columns]
        st.dataframe(h2h[show_cols].rename(columns={
            "MatchDate": "Date", "FullTimeHomeGoals": "HG",
            "FullTimeAwayGoals": "AG", "FullTimeResult": "FTR",
        }), hide_index=True, use_container_width=True)

st.divider()


# ── Copa del Rey Congestion ────────────────────────────────────────────────
st.subheader("🏆 Copa del Rey Congestion Flag")
copa_path = "data_files/raw/copa_fixtures.csv"
if path.exists(copa_path):
    copa_df = pd.read_csv(copa_path)
    copa_df["MatchDate"] = pd.to_datetime(copa_df["MatchDate"], errors="coerce")
    recent_copa = copa_df[
        copa_df["MatchDate"] >= (pd.Timestamp.now() - pd.Timedelta(days=7))
    ]
    if recent_copa.empty:
        st.success("No teams played Copa del Rey in the last 7 days.")
    else:
        flagged = recent_copa["TeamName"].nunique() if "TeamName" in recent_copa.columns else "?"
        st.warning(f"⚠️ {flagged} team(s) played Copa del Rey in the last 7 days.")
        st.dataframe(recent_copa, hide_index=True, use_container_width=True)
else:
    st.info("Copa del Rey data not loaded. Run `python fetch_copa_fixtures.py`.")
