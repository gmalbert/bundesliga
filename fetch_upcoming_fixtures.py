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
from requests.adapters import HTTPAdapter
from requests.exceptions import RequestException
from urllib3.util.retry import Retry

load_dotenv()

FOOTBALL_DATA_KEY = os.getenv("FOOTBALL_DATA_KEY", "")
BASE_URL = "https://api.football-data.org/v4"
ESPN_BASE = "https://site.api.espn.com/apis/site/v2/sports/soccer/ger.1/scoreboard"
OUT_PATH = "data_files/upcoming_fixtures.csv"


def _build_session() -> requests.Session:
    """Create a requests session with retry handling for flaky TLS/network errors."""
    session = requests.Session()
    retry = Retry(
        total=4,
        connect=4,
        read=4,
        backoff_factor=1.5,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=("GET",),
        raise_on_status=False,
    )
    adapter = HTTPAdapter(max_retries=retry)
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    session.headers.update({
        "User-Agent": "bundesliga-fetcher/1.0",
        "Accept": "application/json",
    })
    if FOOTBALL_DATA_KEY:
        session.headers.update({"X-Auth-Token": FOOTBALL_DATA_KEY})
    return session


def _request_matches(status: str, season: int | None = None) -> list[dict]:
    """Fetch raw matches from football-data.org with retry support."""
    params = {"status": status}
    if season:
        params["season"] = season

    session = _build_session()
    resp = session.get(
        f"{BASE_URL}/competitions/BL1/matches",
        params=params,
        timeout=(10, 30),
    )
    resp.raise_for_status()
    return resp.json().get("matches", [])


def _fetch_from_espn() -> list[dict]:
    """Fallback fetch for scheduled Bundesliga fixtures using ESPN's public scoreboard."""
    cursor = datetime.now().replace(day=1)
    month_codes = [cursor.strftime("%Y%m")]
    for _ in range(2):
        if cursor.month == 12:
            cursor = cursor.replace(year=cursor.year + 1, month=1)
        else:
            cursor = cursor.replace(month=cursor.month + 1)
        month_codes.append(cursor.strftime("%Y%m"))
    seen: set[str] = set()
    events: list[dict] = []
    for date_str in month_codes:
        if date_str in seen:
            continue
        seen.add(date_str)
        try:
            resp = requests.get(ESPN_BASE, params={"dates": date_str}, timeout=15)
            resp.raise_for_status()
            events.extend(resp.json().get("events", []))
        except RequestException as exc:
            print(f"  Warning: ESPN fallback failed for {date_str}: {exc}")
    return events


def fetch_upcoming_bl1_fixtures(season: int | None = None) -> pd.DataFrame:
    """
    Fetch SCHEDULED Bundesliga matches from football-data.org.
    Returns a DataFrame and saves to upcoming_fixtures.csv.
    """
    matches: list[dict]
    if not FOOTBALL_DATA_KEY:
        print("FOOTBALL_DATA_KEY not set; using ESPN fallback for upcoming fixtures.")
        matches = _fetch_from_espn()
    else:
        try:
            matches = _request_matches("SCHEDULED", season)
        except RequestException as exc:
            print(f"football-data.org request failed: {exc}. Falling back to ESPN.")
            matches = _fetch_from_espn()

    et = pytz.timezone("America/New_York")
    rows = []
    for m in matches:
        if "utcDate" in m:
            utc_dt = datetime.fromisoformat(m["utcDate"].replace("Z", "+00:00"))
            et_dt = utc_dt.astimezone(et)
            date_value = et_dt.strftime("%Y-%m-%d")
            time_value = et_dt.strftime("%I:%M %p ET")
            home_team = m["homeTeam"]["name"]
            away_team = m["awayTeam"]["name"]
            status = m.get("status", "SCHEDULED")
            matchday = m.get("matchday")
        else:
            event_date = m.get("date", "")
            utc_dt = datetime.fromisoformat(event_date.replace("Z", "+00:00")) if event_date else datetime.now()
            et_dt = utc_dt.astimezone(et)
            date_value = et_dt.strftime("%Y-%m-%d")
            time_value = et_dt.strftime("%I:%M %p ET")
            competition = (m.get("competitions") or [{}])[0]
            status = (competition.get("status") or {}).get("type", {}).get("state", "SCHEDULED")
            matchday = competition.get("matchday") or competition.get("round")
            competitors = competition.get("competitors", [])
            home_team = next((c["team"]["displayName"] for c in competitors if c.get("homeAway") == "home"), "")
            away_team = next((c["team"]["displayName"] for c in competitors if c.get("homeAway") == "away"), "")

        rows.append({
            "Date": date_value,
            "Time": time_value,
            "Matchday": matchday,
            "HomeTeam": home_team,
            "AwayTeam": away_team,
            "Status": status,
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

    try:
        matches = _request_matches("FINISHED")
    except RequestException as exc:
        print(f"Warning: recent results fetch failed: {exc}")
        return pd.DataFrame()

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
