"""Download SP1.csv files for every La Liga season from football-data.co.uk
and merge them into data_files/combined_historical_data.csv.

Usage:
    python fetch_historical_csvs.py
"""

from __future__ import annotations

import io
from pathlib import Path

import pandas as pd
import requests

# ── Seasons to download ────────────────────────────────────────────────────

SEASONS: dict[str, str] = {
    "1516": "2015-16",
    "1617": "2016-17",
    "1718": "2017-18",
    "1819": "2018-19",
    "1920": "2019-20",
    "2021": "2020-21",
    "2122": "2021-22",
    "2223": "2022-23",
    "2324": "2023-24",
    "2425": "2024-25",
    "2526": "2025-26",
}

BASE_URL = "https://www.football-data.co.uk/mmz4281/{code}/SP1.csv"

COLUMN_MAP: dict[str, str] = {
    "Date":   "MatchDate",
    "HomeTeam": "HomeTeam",
    "AwayTeam": "AwayTeam",
    "FTHG":   "FullTimeHomeGoals",
    "FTAG":   "FullTimeAwayGoals",
    "FTR":    "FullTimeResult",
    "HTHG":   "HalfTimeHomeGoals",
    "HTAG":   "HalfTimeAwayGoals",
    "HTR":    "HalfTimeResult",
    "Referee": "Referee",
    "HS":     "HomeShots",
    "AS":     "AwayShots",
    "HST":    "HomeShotsOnTarget",
    "AST":    "AwayShotsOnTarget",
    "HF":     "HomeFouls",
    "AF":     "AwayFouls",
    "HC":     "HomeCorners",
    "AC":     "AwayCorners",
    "HY":     "HomeYellowCards",
    "AY":     "AwayYellowCards",
    "HR":     "HomeRedCards",
    "AR":     "AwayRedCards",
    # Bet365 odds
    "B365H":  "Bet365_HomeWinOdds",
    "B365D":  "Bet365_DrawOdds",
    "B365A":  "Bet365_AwayWinOdds",
    # BetWin
    "BWH":    "BW_HomeWinOdds",
    "BWD":    "BW_DrawOdds",
    "BWA":    "BW_AwayWinOdds",
    # Pinnacle
    "PSH":    "Pinnacle_HomeWinOdds",
    "PSD":    "Pinnacle_DrawOdds",
    "PSA":    "Pinnacle_AwayWinOdds",
}


def download_season(season_code: str, season_label: str) -> pd.DataFrame:
    """Download one season CSV and return a normalised DataFrame."""
    url = BASE_URL.format(code=season_code)
    try:
        resp = requests.get(url, timeout=20)
        resp.raise_for_status()
        df = pd.read_csv(
            io.StringIO(resp.text),
            encoding="latin-1",
            on_bad_lines="skip",
        )
        df = df.rename(columns={k: v for k, v in COLUMN_MAP.items() if k in df.columns}).copy()
        df["Season"] = season_label
        df["MatchDate"] = pd.to_datetime(df["MatchDate"], format="mixed", dayfirst=True, errors="coerce")
        # Drop completely empty rows
        df = df.dropna(subset=["HomeTeam", "AwayTeam"])
        print(f"  ✓ {season_label}: {len(df)} matches")
        return df
    except Exception as exc:
        print(f"  ✗ {season_label}: {exc}")
        return pd.DataFrame()


def build_historical_dataset() -> pd.DataFrame:
    """Download all seasons, combine, and save to CSV."""
    Path("data_files/raw").mkdir(parents=True, exist_ok=True)

    frames: list[pd.DataFrame] = []
    for code, label in SEASONS.items():
        print(f"Downloading {label}…")
        df = download_season(code, label)
        if not df.empty:
            raw_path = f"data_files/raw/SP1_{code}.csv"
            df.to_csv(raw_path, index=False)
            frames.append(df)

    if not frames:
        print("No data downloaded. Check your internet connection.")
        return pd.DataFrame()

    combined = pd.concat(frames, ignore_index=True)
    combined = combined.sort_values("MatchDate").reset_index(drop=True)

    out_path = "data_files/combined_historical_data.csv"
    combined.to_csv(out_path, index=False)
    print(f"\n✓ Combined: {len(combined)} matches across {len(frames)} seasons → {out_path}")
    return combined


if __name__ == "__main__":
    build_historical_dataset()
