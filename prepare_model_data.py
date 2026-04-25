"""Feature engineering pipeline for La Liga Linea — offline / standalone use.

This module is imported by train_models.py and the nightly pipeline.
It deliberately does NOT import streamlit so it runs cleanly in CI / GitHub Actions.

Key function:
    load_and_engineer_features(df) → pd.DataFrame
        Accepts a raw combined_historical_data.csv DataFrame and returns one
        with all model features computed.

Standalone usage:
    python prepare_model_data.py
    → reads  data_files/combined_historical_data.csv
    → writes data_files/model_ready_data.csv
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

# ── Constants (must stay in sync with utils.py) ───────────────────────────

LA_LIGA_AVG_HOME_GOALS = 1.45
LA_LIGA_AVG_AWAY_GOALS = 1.12

FEATURE_COLS: list[str] = [
    "HomeGoals_Avg_L5",
    "HomeConceded_Avg_L5",
    "HomeWinRate_L10",
    "HomeMomentum_L3",
    "HomeRestDays",
    "AwayGoals_Avg_L5",
    "AwayConceded_Avg_L5",
    "AwayWinRate_L10",
    "AwayMomentum_L3",
    "AwayRestDays",
    "ImpliedProb_HomeWin",
    "ImpliedProb_Draw",
    "ImpliedProb_AwayWin",
]

COPA_CONGESTION_WINDOW_DAYS = 4  # flag = 1 if team played Copa ≤ 4 days before


# ── Core Feature Engineering ──────────────────────────────────────────────

def _rolling_team_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute rolling per-team features using shift(1) to prevent data leakage.
    Returns df with Home*/Away* feature columns appended.
    """
    df = df.copy().sort_values("MatchDate").reset_index(drop=True)

    home_side = df[
        ["MatchDate", "HomeTeam", "FullTimeHomeGoals", "FullTimeAwayGoals", "FullTimeResult"]
    ].copy()
    home_side.columns = ["MatchDate", "Team", "GF", "GA", "FTR"]
    home_side["Pts"] = home_side["FTR"].map({"H": 3, "D": 1, "A": 0}).fillna(0)
    home_side["Won"] = (home_side["FTR"] == "H").astype(float)

    away_side = df[
        ["MatchDate", "AwayTeam", "FullTimeAwayGoals", "FullTimeHomeGoals", "FullTimeResult"]
    ].copy()
    away_side.columns = ["MatchDate", "Team", "GF", "GA", "FTR"]
    away_side["Pts"] = away_side["FTR"].map({"A": 3, "D": 1, "H": 0}).fillna(0)
    away_side["Won"] = (away_side["FTR"] == "A").astype(float)

    long = pd.concat([home_side, away_side], ignore_index=True)
    long = long.sort_values(["Team", "MatchDate"]).reset_index(drop=True)

    grp = long.groupby("Team")
    long["GF_L5"]   = grp["GF"].transform(lambda x: x.shift(1).rolling(5,  min_periods=1).mean())
    long["GA_L5"]   = grp["GA"].transform(lambda x: x.shift(1).rolling(5,  min_periods=1).mean())
    long["Won_L10"] = grp["Won"].transform(lambda x: x.shift(1).rolling(10, min_periods=1).mean())
    long["Pts_L3"]  = grp["Pts"].transform(lambda x: x.shift(1).rolling(3,  min_periods=1).sum())

    long["PrevDate"] = grp["MatchDate"].shift(1)
    long["RestDays"] = (
        (long["MatchDate"] - long["PrevDate"]).dt.days
        .clip(1, 30)
        .fillna(7)
        .astype(float)
    )

    feat_cols = ["MatchDate", "Team", "GF_L5", "GA_L5", "Won_L10", "Pts_L3", "RestDays"]

    home_feats = long[feat_cols].rename(columns={
        "Team":     "HomeTeam",
        "GF_L5":    "HomeGoals_Avg_L5",
        "GA_L5":    "HomeConceded_Avg_L5",
        "Won_L10":  "HomeWinRate_L10",
        "Pts_L3":   "HomeMomentum_L3",
        "RestDays": "HomeRestDays",
    })
    away_feats = long[feat_cols].rename(columns={
        "Team":     "AwayTeam",
        "GF_L5":    "AwayGoals_Avg_L5",
        "GA_L5":    "AwayConceded_Avg_L5",
        "Won_L10":  "AwayWinRate_L10",
        "Pts_L3":   "AwayMomentum_L3",
        "RestDays": "AwayRestDays",
    })

    df = df.merge(home_feats, on=["MatchDate", "HomeTeam"], how="left")
    df = df.merge(away_feats, on=["MatchDate", "AwayTeam"], how="left")
    return df


def _implied_probability_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute vig-removed implied probabilities from historical odds columns."""
    # Use Pinnacle if available, fall back to Bet365, then BW
    def _pick_odds_col(candidates: list[str]) -> str | None:
        return next((c for c in candidates if c in df.columns), None)

    home_col = _pick_odds_col(["Pinnacle_HomeWinOdds", "Bet365_HomeWinOdds", "BW_HomeWinOdds"])
    draw_col = _pick_odds_col(["Pinnacle_DrawOdds",    "Bet365_DrawOdds",    "BW_DrawOdds"])
    away_col = _pick_odds_col(["Pinnacle_AwayWinOdds", "Bet365_AwayWinOdds", "BW_AwayWinOdds"])

    if not all([home_col, draw_col, away_col]):
        df["ImpliedProb_HomeWin"] = 0.45
        df["ImpliedProb_Draw"]    = 0.27
        df["ImpliedProb_AwayWin"] = 0.28
        return df

    df = df.copy()
    for col in [home_col, draw_col, away_col]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    valid = df[home_col].notna() & df[draw_col].notna() & df[away_col].notna()
    vig = 1 / df.loc[valid, home_col] + 1 / df.loc[valid, draw_col] + 1 / df.loc[valid, away_col]
    df.loc[valid, "ImpliedProb_HomeWin"] = ((1 / df.loc[valid, home_col]) / vig).round(4)
    df.loc[valid, "ImpliedProb_Draw"]    = ((1 / df.loc[valid, draw_col]) / vig).round(4)
    df.loc[valid, "ImpliedProb_AwayWin"] = ((1 / df.loc[valid, away_col]) / vig).round(4)
    df.loc[valid, "BookmakerMargin"]     = ((vig - 1) * 100).round(2)

    df["ImpliedProb_HomeWin"] = df["ImpliedProb_HomeWin"].fillna(0.45)
    df["ImpliedProb_Draw"]    = df["ImpliedProb_Draw"].fillna(0.27)
    df["ImpliedProb_AwayWin"] = df["ImpliedProb_AwayWin"].fillna(0.28)
    return df


def _copa_congestion_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add HomeCopaCongestion and AwayCopaCongestion binary flags.
    Reads data_files/raw/copa_fixtures.csv if it exists; silently skips if not.
    """
    copa_path = Path("data_files/raw/copa_fixtures.csv")
    if not copa_path.exists():
        df["HomeCopaCongestion"] = 0
        df["AwayCopaCongestion"] = 0
        return df

    copa = pd.read_csv(copa_path)
    copa["MatchDate"] = pd.to_datetime(copa["MatchDate"], errors="coerce")

    copa_dict: dict[str, list] = {}
    for _, row in copa.iterrows():
        copa_dict.setdefault(row["TeamName"], []).append(row["MatchDate"])

    def _has_copa_congestion(team: str, match_date: pd.Timestamp) -> int:
        dates = copa_dict.get(team, [])
        return int(
            any(
                0 < (match_date - d).days <= COPA_CONGESTION_WINDOW_DAYS
                for d in dates
                if pd.notna(d)
            )
        )

    df = df.copy()
    df["HomeCopaCongestion"] = df.apply(
        lambda r: _has_copa_congestion(r["HomeTeam"], r["MatchDate"]), axis=1
    )
    df["AwayCopaCongestion"] = df.apply(
        lambda r: _has_copa_congestion(r["AwayTeam"], r["MatchDate"]), axis=1
    )
    return df


def load_and_engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Full feature engineering pipeline.
    Input:  raw combined_historical_data.csv loaded into a DataFrame
    Output: DataFrame with all FEATURE_COLS present + target-ready columns
    """
    df = df.copy()

    # Standardise key column names
    rename = {
        "Date":   "MatchDate",
        "FTHG":   "FullTimeHomeGoals",
        "FTAG":   "FullTimeAwayGoals",
        "FTR":    "FullTimeResult",
        "B365H":  "Bet365_HomeWinOdds",
        "B365D":  "Bet365_DrawOdds",
        "B365A":  "Bet365_AwayWinOdds",
    }
    for old, new in rename.items():
        if old in df.columns and new not in df.columns:
            df.rename(columns={old: new}, inplace=True)

    df["MatchDate"] = pd.to_datetime(df.get("MatchDate"), errors="coerce")
    df["FullTimeHomeGoals"] = pd.to_numeric(df.get("FullTimeHomeGoals"), errors="coerce").fillna(0)
    df["FullTimeAwayGoals"] = pd.to_numeric(df.get("FullTimeAwayGoals"), errors="coerce").fillna(0)

    # Keep only rows with a valid result
    df = df[df["FullTimeResult"].isin(["H", "D", "A"])].copy()
    df = df.dropna(subset=["MatchDate", "HomeTeam", "AwayTeam"])
    df = df.sort_values("MatchDate").reset_index(drop=True)

    df = _rolling_team_features(df)
    df = _implied_probability_features(df)
    df = _copa_congestion_features(df)

    # Fill any remaining NaN in feature columns with La Liga averages / sensible defaults
    fill_values = {
        "HomeGoals_Avg_L5":    LA_LIGA_AVG_HOME_GOALS,
        "HomeConceded_Avg_L5": LA_LIGA_AVG_AWAY_GOALS,
        "HomeWinRate_L10":     0.33,
        "HomeMomentum_L3":     3.0,
        "HomeRestDays":        7.0,
        "AwayGoals_Avg_L5":    LA_LIGA_AVG_AWAY_GOALS,
        "AwayConceded_Avg_L5": LA_LIGA_AVG_HOME_GOALS,
        "AwayWinRate_L10":     0.33,
        "AwayMomentum_L3":     3.0,
        "AwayRestDays":        7.0,
        "HomeCopaCongestion":  0,
        "AwayCopaCongestion":  0,
    }
    for col, val in fill_values.items():
        if col in df.columns:
            df[col] = df[col].fillna(val)

    return df


# ── Standalone script ─────────────────────────────────────────────────────

if __name__ == "__main__":
    hist_path = "data_files/combined_historical_data.csv"
    out_path  = "data_files/model_ready_data.csv"

    if not Path(hist_path).exists():
        print(f"✗ {hist_path} not found. Run fetch_historical_csvs.py first.")
        raise SystemExit(1)

    print(f"Loading {hist_path}…")
    raw = pd.read_csv(hist_path, low_memory=False)
    print(f"  Raw rows: {len(raw)}")

    engineered = load_and_engineer_features(raw)
    print(f"  After engineering: {len(engineered)} rows, {len(engineered.columns)} columns")

    engineered.to_csv(out_path, index=False)
    print(f"✓ Saved model-ready data → {out_path}")

    # Quick sanity check: all FEATURE_COLS present?
    missing = [c for c in FEATURE_COLS if c not in engineered.columns]
    if missing:
        print(f"  ⚠ Missing feature columns: {missing}")
    else:
        print(f"  All {len(FEATURE_COLS)} FEATURE_COLS present.")
