"""Fetch upcoming Bundesliga fixtures from football-data.org (competition BL1).

Saves: data_files/upcoming_fixtures.csv

Usage:
    python fetch_upcoming_fixtures.py

Requires:
    FOOTBALL_DATA_KEY in .env (free tier covers BL1 at 10 req/min)
"""

from __future__ import annotations

import os
from datetime import datetime
from pathlib import Path

import pandas as pd
import pytz
import requests
from dotenv import load_dotenv

load_dotenv()

FOOTBALL_DATA_KEY = os.getenv("FOOTBALL_DATA_KEY", "")
BASE_URL = "https://api.football-data.org/v4"
HEADERS = {"X-Auth-Token": FOOTBALL_DATA_KEY}

OUT_PATH = "data_files/upcoming_fixtures.csv"


def fetch_upcoming_bl1_fixtures(season: int | None = None) -> pd.DataFrame:
    """
    Fetch SCHEDULED Bundesliga matches from football-data.org.
    Returns a DataFrame and saves to upcoming_fixtures.csv.
    """
    if not FOOTBALL_DATA_KEY:
        raise EnvironmentError(
            "FOOTBALL_DATA_KEY not set. Copy .env.example to .env and add your key."
        )

    params: dict = {"status": "SCHEDULED"}
    if season:
        params["season"] = season

    resp = requests.get(
        f"{BASE_URL}/competitions/BL1/matches",
        headers=HEADERS,
        params=params,
        timeout=15,
    )
    resp.raise_for_status()
    matches = resp.json().get("matches", [])

    et = pytz.timezone("America/New_York")
    rows = []
    for m in matches:
        utc_dt = datetime.fromisoformat(m["utcDate"].replace("Z", "+00:00"))
        et_dt = utc_dt.astimezone(et)
        rows.append({
            "Date":      et_dt.strftime("%Y-%m-%d"),
            "Time":      et_dt.strftime("%I:%M %p ET"),
            "Matchday":  m.get("matchday"),
            "HomeTeam":  m["homeTeam"]["name"],
            "AwayTeam":  m["awayTeam"]["name"],
            "Status":    m["status"],
        })

    if not rows:
        print("No upcoming fixtures found (off-season?).")
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    df = df.sort_values("Date").reset_index(drop=True)

    Path("data_files").mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_PATH, index=False)
    print(f"✓ Saved {len(df)} upcoming fixtures → {OUT_PATH}")
    return df


def fetch_recent_results(n_matchdays: int = 3) -> pd.DataFrame:
    """
    Fetch the most recently FINISHED Bundesliga matches (for predictions log enrichment).
    """
    if not FOOTBALL_DATA_KEY:
        return pd.DataFrame()

    resp = requests.get(
        f"{BASE_URL}/competitions/BL1/matches",
        headers=HEADERS,
        params={"status": "FINISHED"},
        timeout=15,
    )
    resp.raise_for_status()
    matches = resp.json().get("matches", [])

    rows = []
    for m in matches:
        score = m["score"]["fullTime"]
        h_goals = score.get("home")
        a_goals = score.get("away")
        if h_goals is None or a_goals is None:
            continue
        result = "H" if h_goals > a_goals else ("A" if a_goals > h_goals else "D")
        rows.append({
            "MatchDate": m["utcDate"][:10],
            "Matchday":  m.get("matchday"),
            "HomeTeam":  m["homeTeam"]["name"],
            "AwayTeam":  m["awayTeam"]["name"],
            "FullTimeHomeGoals": h_goals,
            "FullTimeAwayGoals": a_goals,
            "FullTimeResult": result,
        })

    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values("MatchDate", ascending=False).reset_index(drop=True)
        # Keep only last n matchdays
        if "Matchday" in df.columns:
            latest = df["Matchday"].max()
            df = df[df["Matchday"] >= latest - n_matchdays + 1]

    return df


if __name__ == "__main__":
    fetch_upcoming_bl1_fixtures()
