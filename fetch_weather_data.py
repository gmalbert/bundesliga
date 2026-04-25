"""Fetch weather data for upcoming Bundesliga fixtures.

Uses the Open-Meteo API (free, no API key required).
For each upcoming fixture, looks up the stadium coordinates and fetches
the forecast for that match date.

Outputs:
    data_files/raw/match_weather.csv

Usage:
    python fetch_weather_data.py

Called by: automation/nightly_pipeline.py
"""

from __future__ import annotations

import time
from pathlib import Path

import pandas as pd
import requests

# ── Stadium Coordinates ────────────────────────────────────────────────────
# (latitude, longitude) for each Bundesliga club's home stadium
STADIUM_COORDS: dict[str, tuple[float, float]] = {
    "Bayern Munich":  (48.2188,  11.6247),   # Allianz Arena
    "Dortmund":       (51.4926,   7.4518),   # Signal Iduna Park
    "Leverkusen":     (51.0383,   7.0023),   # BayArena
    "RB Leipzig":     (51.3455,  12.3484),   # Red Bull Arena
    "Ein Frankfurt":  (50.0687,   8.6454),   # Deutsche Bank Park
    "Wolfsburg":      (52.4327,  10.8027),   # Volkswagen Arena
    "M'gladbach":     (51.1743,   6.3852),   # Borussia-Park
    "Hoffenheim":     (49.2384,   8.8883),   # PreZero Arena
    "Freiburg":       (47.9875,   7.8943),   # Europa-Park Stadion
    "Union Berlin":   (52.4573,  13.5672),   # An der Alten Försterei
    "Stuttgart":      (48.7924,   9.2320),   # MHPArena
    "Augsburg":       (48.3237,  10.8864),   # WWK Arena
    "Mainz":          (49.9845,   8.2249),   # Mewa Arena
    "Werder Bremen":  (53.0664,   8.8377),   # Wohninvest Weserstadion
    "FC Koln":        (50.9333,   6.8750),   # RheinEnergieStadion
    "Hertha":         (52.5145,  13.2394),   # Olympiastadion Berlin
    "Hamburg":        (53.5872,   9.8980),   # Volksparkstadion
    "Schalke 04":     (51.5541,   7.0677),   # Veltins-Arena
    "Bochum":         (51.4906,   7.2148),   # Vonovia Ruhrstadion
    "Heidenheim":     (48.6764,  10.1469),   # Voith-Arena
    "Darmstadt":      (49.8640,   8.6490),   # Merck-Stadion
    "St Pauli":       (53.5560,   9.9682),   # Millerntor-Stadion
    "Holstein Kiel":  (54.3384,  10.1318),   # Holstein-Stadion
    "Greuther Furth": (49.4766,  10.9897),   # Sportpark Ronhof
    "Ingolstadt":     (48.7744,  11.4333),   # Audi Sportpark
}

# ── Helpers ────────────────────────────────────────────────────────────────

FORECAST_URL = "https://api.open-meteo.com/v1/forecast"
DAILY_PARAMS  = "temperature_2m_max,precipitation_sum,windspeed_10m_max,weathercode"


def _get_coords(team: str) -> tuple[float, float] | None:
    """Return (lat, lon) for a team, or None if not found."""
    return STADIUM_COORDS.get(team) or STADIUM_COORDS.get(team.split()[0])


def _weather_description(code: int) -> str:
    """Map WMO weather code → human-readable label."""
    if code == 0:
        return "Clear"
    if code in (1, 2, 3):
        return "Partly cloudy"
    if code in (45, 48):
        return "Foggy"
    if code in (51, 53, 55, 56, 57, 61, 63, 65, 66, 67):
        return "Rainy"
    if code in (71, 73, 75, 77):
        return "Snowy"
    if code in (80, 81, 82, 85, 86):
        return "Showers"
    if code in (95, 96, 99):
        return "Thunderstorm"
    return "Unknown"


def fetch_fixture_weather(home_team: str, match_date: str) -> dict:
    """Fetch single-day forecast for a fixture. Returns a weather dict."""
    coords = _get_coords(home_team)
    if coords is None:
        return {
            "WeatherDesc": "N/A",
            "TempMaxC": None,
            "PrecipMM": None,
            "WindKmh": None,
        }

    lat, lon = coords
    params = {
        "latitude":         lat,
        "longitude":        lon,
        "daily":            DAILY_PARAMS,
        "timezone":         "Europe/Madrid",
        "start_date":       match_date,
        "end_date":         match_date,
    }
    try:
        resp = requests.get(FORECAST_URL, params=params, timeout=10)
        resp.raise_for_status()
        data = resp.json().get("daily", {})
        code = data.get("weathercode", [None])[0]
        return {
            "WeatherDesc": _weather_description(int(code)) if code is not None else "N/A",
            "TempMaxC":    data.get("temperature_2m_max", [None])[0],
            "PrecipMM":    data.get("precipitation_sum", [None])[0],
            "WindKmh":     data.get("windspeed_10m_max", [None])[0],
        }
    except Exception as exc:  # noqa: BLE001
        print(f"  ⚠ Weather fetch failed for {home_team} on {match_date}: {exc}")
        return {"WeatherDesc": "N/A", "TempMaxC": None, "PrecipMM": None, "WindKmh": None}


def fetch_all_weather(
    fixtures_path: str = "data_files/upcoming_fixtures.csv",
    out_path: str = "data_files/raw/match_weather.csv",
) -> None:
    """Fetch forecast weather for all upcoming fixtures and write CSV."""
    fixtures_p = Path(fixtures_path)
    if not fixtures_p.exists():
        print(f"✗ Fixtures file not found: {fixtures_path}")
        return

    df = pd.read_csv(fixtures_path)
    if df.empty:
        print("  No upcoming fixtures to fetch weather for.")
        return

    date_col = next((c for c in ["Date", "MatchDate", "date"] if c in df.columns), None)
    home_col = next((c for c in ["HomeTeam", "home_team", "home"] if c in df.columns), None)
    if date_col is None or home_col is None:
        print("  ✗ Could not find Date/HomeTeam columns in fixtures.")
        return

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []
    print(f"  Fetching weather for {len(df)} fixtures…")

    for _, row in df.iterrows():
        match_date = str(row[date_col])[:10]          # YYYY-MM-DD
        home_team  = str(row[home_col])
        weather    = fetch_fixture_weather(home_team, match_date)
        rows.append({
            "Date":      match_date,
            "HomeTeam":  home_team,
            **weather,
        })
        time.sleep(0.05)   # polite rate limiting (Open-Meteo allows 10k/day free)

    out_df = pd.DataFrame(rows)
    out_df.to_csv(out_path, index=False)
    print(f"  Saved {len(out_df)} rows → {out_path}")


if __name__ == "__main__":
    fetch_all_weather()
    print("Weather data fetch complete.")
