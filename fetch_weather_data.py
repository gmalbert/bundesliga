"""Fetch weather data for upcoming La Liga fixtures.

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
# (latitude, longitude) for each La Liga club's home stadium
STADIUM_COORDS: dict[str, tuple[float, float]] = {
    "Alaves":         (42.8466,  -2.6838),   # Mendizorrotza
    "Almeria":        (36.8376,  -2.4593),   # Power Horse Stadium
    "Ath Bilbao":     (43.2644,  -2.9494),   # San Mamés
    "Ath Madrid":     (40.4361,  -3.5995),   # Estadio Metropolitano
    "Atletico":       (40.4361,  -3.5995),   # alias
    "Barcelona":      (41.3809,   2.1228),   # Spotify Camp Nou (Estadi Olímpic during works)
    "Betis":          (37.3564,  -5.9823),   # Estadio Benito Villamarín
    "Cadiz":          (36.5303,  -6.2970),   # Estadio Ramón de Carranza
    "Celta":          (42.2116,  -8.7392),   # Abanca-Balaídos
    "Espanol":        (41.3471,   2.0751),   # RCDE Stadium
    "Getafe":         (40.3257,  -3.7184),   # Coliseum Alfonso Pérez
    "Girona":         (41.9649,   2.8250),   # Estadi Montilivi
    "Granada":        (37.1524,  -3.6075),   # Estadio Nuevo Los Cármenes
    "Las Palmas":     (28.1000, -15.4200),   # Gran Canaria Stadium
    "Leganes":        (40.3570,  -3.7694),   # Estadio Municipal de Butarque
    "Mallorca":       (39.5896,   2.6537),   # Visit Mallorca Estadi
    "Osasuna":        (42.7969,  -1.6370),   # El Sadar
    "Oviedo":         (43.3532,  -5.8615),   # Carlos Tartiere
    "Rayo Vallecano": (40.3914,  -3.6540),   # Estadio de Vallecas
    "Rayo":           (40.3914,  -3.6540),   # alias
    "Real Madrid":    (40.4531,  -3.6883),   # Santiago Bernabéu
    "Sevilla":        (37.3840,  -5.9706),   # Ramón Sánchez-Pizjuán
    "Sociedad":       (43.2998,  -1.9735),   # Reale Arena
    "Vallecano":      (40.3914,  -3.6540),   # alias
    "Valencia":       (39.4748,  -0.3584),   # Estadio Mestalla
    "Valladolid":     (41.6408,  -4.7420),   # Nuevo Estadio José Zorrilla
    "Villarreal":     (39.9445,  -0.1034),   # Estadio de la Cerámica
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
