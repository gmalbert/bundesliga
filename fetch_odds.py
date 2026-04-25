"""Fetch La Liga betting odds from The Odds API.

Saves: data_files/raw/odds.csv

Usage:
    python fetch_odds.py

Requires:
    ODDS_API_KEY in .env (free tier: 500 req/month)
    Sign up at https://the-odds-api.com/
"""

from __future__ import annotations

import os
from pathlib import Path

import pandas as pd
import requests
from dotenv import load_dotenv

load_dotenv()

ODDS_API_KEY  = os.getenv("ODDS_API_KEY", "")
ODDS_API_BASE = "https://api.the-odds-api.com/v4"
SPORT_KEY     = "soccer_spain_la_liga"

OUT_PATH = "data_files/raw/odds.csv"

Path("data_files/raw").mkdir(parents=True, exist_ok=True)


def fetch_upcoming_odds(
    regions: str = "us,eu",
    markets: str = "h2h",
    bookmakers: str = "draftkings,betmgm,pinnacle,bet365",
) -> pd.DataFrame:
    """
    Fetch upcoming La Liga 1X2 (h2h) odds.
    Returns a DataFrame with one row per game per bookmaker.
    """
    if not ODDS_API_KEY:
        raise EnvironmentError(
            "ODDS_API_KEY not set. Copy .env.example to .env and add your key."
        )

    resp = requests.get(
        f"{ODDS_API_BASE}/sports/{SPORT_KEY}/odds",
        params={
            "apiKey":       ODDS_API_KEY,
            "regions":      regions,
            "markets":      markets,
            "oddsFormat":   "decimal",
            "bookmakers":   bookmakers,
        },
        timeout=20,
    )
    resp.raise_for_status()

    # Log remaining quota from response headers
    remaining = resp.headers.get("x-requests-remaining", "?")
    used = resp.headers.get("x-requests-used", "?")
    print(f"  Odds API quota — used: {used}, remaining: {remaining}")

    games = resp.json()
    if not games:
        print("  No upcoming odds returned. Off-season or no active markets.")
        return pd.DataFrame()

    rows = []
    for game in games:
        home = game["home_team"]
        away = game["away_team"]
        date = game["commence_time"][:10]

        for bm in game.get("bookmakers", []):
            for market in bm.get("markets", []):
                if market["key"] != "h2h":
                    continue
                prices = {o["name"]: o["price"] for o in market["outcomes"]}
                rows.append({
                    "Date":         date,
                    "HomeTeam":     home,
                    "AwayTeam":     away,
                    "Bookmaker":    bm["key"],
                    "HomeWinOdds":  prices.get(home),
                    "DrawOdds":     prices.get("Draw"),
                    "AwayWinOdds":  prices.get(away),
                })

    df = pd.DataFrame(rows)
    df = _add_implied_probabilities(df)
    df.to_csv(OUT_PATH, index=False)
    print(f"  ✓ Odds for {len(games)} games, {len(df)} bookmaker rows → {OUT_PATH}")
    return df


def _add_implied_probabilities(df: pd.DataFrame) -> pd.DataFrame:
    """Vig-removed implied probabilities from decimal odds."""
    df = df.copy()
    for col in ["HomeWinOdds", "DrawOdds", "AwayWinOdds"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    valid = df["HomeWinOdds"].notna() & df["DrawOdds"].notna() & df["AwayWinOdds"].notna()
    df.loc[valid, "_vig"] = (
        1 / df.loc[valid, "HomeWinOdds"]
        + 1 / df.loc[valid, "DrawOdds"]
        + 1 / df.loc[valid, "AwayWinOdds"]
    )
    df.loc[valid, "ImpliedProb_HomeWin"] = (
        (1 / df.loc[valid, "HomeWinOdds"]) / df.loc[valid, "_vig"]
    ).round(4)
    df.loc[valid, "ImpliedProb_Draw"] = (
        (1 / df.loc[valid, "DrawOdds"]) / df.loc[valid, "_vig"]
    ).round(4)
    df.loc[valid, "ImpliedProb_AwayWin"] = (
        (1 / df.loc[valid, "AwayWinOdds"]) / df.loc[valid, "_vig"]
    ).round(4)
    df.loc[valid, "BookmakerMargin"] = (
        (df.loc[valid, "_vig"] - 1) * 100
    ).round(2)
    df.drop(columns=["_vig"], inplace=True, errors="ignore")
    return df


if __name__ == "__main__":
    fetch_upcoming_odds()
