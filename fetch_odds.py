"""Fetch Bundesliga betting odds from The Odds API.

Saves: data_files/raw/odds.csv

Usage:
    python fetch_odds.py            # skips if cache is fresh or no fixtures soon
    python fetch_odds.py --force    # always hits the API

Requires:
    ODDS_API_KEY in .env (free tier: 500 req/month)
    Sign up at https://the-odds-api.com/

Credit-saving behaviour (applied unless --force):
  1. If odds.csv is less than CACHE_TTL_HOURS old, return the cached file.
  2. If upcoming_fixtures.csv has no games within FIXTURE_HORIZON_DAYS, skip.
"""

from __future__ import annotations

import argparse
import os
from datetime import datetime
from pathlib import Path

import pandas as pd
import requests
from dotenv import load_dotenv

load_dotenv()

ODDS_API_KEY        = os.getenv("ODDS_API_KEY", "")
ODDS_API_BASE       = "https://api.the-odds-api.com/v4"
SPORT_KEY           = "soccer_germany_bundesliga"

OUT_PATH            = "data_files/raw/odds.csv"
FIXTURES_PATH       = "data_files/upcoming_fixtures.csv"
CACHE_TTL_HOURS     = 12   # skip re-fetch if cache is younger than this
FIXTURE_HORIZON_DAYS = 10  # skip fetch if no fixtures within this many days

Path("data_files/raw").mkdir(parents=True, exist_ok=True)


def _cache_is_fresh() -> bool:
    """Return True if odds.csv exists and is younger than CACHE_TTL_HOURS."""
    p = Path(OUT_PATH)
    if not p.exists():
        return False
    age_hours = (datetime.now().timestamp() - p.stat().st_mtime) / 3600
    return age_hours < CACHE_TTL_HOURS


def _has_upcoming_fixtures() -> bool:
    """Return True if upcoming_fixtures.csv has at least one game within FIXTURE_HORIZON_DAYS."""
    fp = Path(FIXTURES_PATH)
    if not fp.exists():
        return True  # can't tell — proceed with fetch
    try:
        df = pd.read_csv(fp)
        date_col = next(
            (c for c in df.columns if "date" in c.lower() or c.lower() == "matchdate"),
            None,
        )
        if date_col is None:
            return True
        dates = pd.to_datetime(df[date_col], errors="coerce").dropna()
        now = pd.Timestamp.now()
        cutoff = now + pd.Timedelta(days=FIXTURE_HORIZON_DAYS)
        return bool(((dates >= now) & (dates <= cutoff)).any())
    except Exception:
        return True  # if anything goes wrong, don't suppress the fetch


def fetch_upcoming_odds(
    regions: str = "us,eu",
    markets: str = "h2h",
    bookmakers: str = "draftkings,betmgm,pinnacle,bet365",
    force: bool = False,
) -> pd.DataFrame:
    """
    Fetch upcoming Bundesliga 1X2 (h2h) odds.
    Returns a DataFrame with one row per game per bookmaker.

    Skips the API call (and returns cached data or empty DataFrame) when:
      - force=False and odds.csv is younger than CACHE_TTL_HOURS, OR
      - force=False and no fixtures are scheduled within FIXTURE_HORIZON_DAYS.
    """
    if not force and _cache_is_fresh():
        age_h = (datetime.now().timestamp() - Path(OUT_PATH).stat().st_mtime) / 3600
        print(
            f"  Odds cache is {age_h:.1f}h old (TTL={CACHE_TTL_HOURS}h). "
            "Skipping API call. Use --force to override."
        )
        return pd.read_csv(OUT_PATH)

    if not force and not _has_upcoming_fixtures():
        print(
            f"  No upcoming fixtures within {FIXTURE_HORIZON_DAYS} days. "
            "Skipping odds fetch (off-season). Use --force to override."
        )
        return pd.DataFrame()

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
    parser = argparse.ArgumentParser(description="Fetch Bundesliga odds from The Odds API")
    parser.add_argument(
        "--force",
        action="store_true",
        help="Ignore cache TTL and upcoming-fixtures check; always call the API",
    )
    args = parser.parse_args()
    fetch_upcoming_odds(force=args.force)
