"""Normalize team names from football-data.co.uk, football-data.org, and FBref
to the canonical names used in the historical data (football-data.co.uk short names).

The canonical form is the football-data.co.uk short name because that is what
combined_historical_data.csv contains and what the model was trained on.
All external sources (football-data.org API full names, FBref, ESPN) are
mapped back to those short names so team stats lookups succeed.
"""

# Maps source variant → football-data.co.uk canonical short name
LA_LIGA_TEAM_MAP: dict[str, str] = {
    # ── football-data.org full names → co.uk short names ─────────────────
    "FC Barcelona":                  "Barcelona",
    "Real Madrid CF":                "Real Madrid",
    "Club Atlético de Madrid":       "Ath Madrid",
    "Athletic Club":                 "Ath Bilbao",
    "Real Betis Balompié":           "Betis",
    "RC Celta de Vigo":              "Celta",
    "RCD Espanyol de Barcelona":     "Espanol",
    "Getafe CF":                     "Getafe",
    "Girona FC":                     "Girona",
    "Levante UD":                    "Levante",
    "RCD Mallorca":                  "Mallorca",
    "CA Osasuna":                    "Osasuna",
    "Sevilla FC":                    "Sevilla",
    "Real Sociedad de Fútbol":       "Sociedad",
    "Real Sociedad de Futbol":       "Sociedad",
    "Valencia CF":                   "Valencia",
    "Rayo Vallecano de Madrid":      "Vallecano",
    "Villarreal CF":                 "Villarreal",
    "Deportivo Alavés":              "Alaves",
    "Deportivo Alaves":              "Alaves",
    "Elche CF":                      "Elche",
    "UD Almería":                    "Almeria",
    "UD Almeria":                    "Almeria",
    "Real Valladolid CF":            "Valladolid",
    "CD Leganés":                    "Leganes",
    "CD Leganes":                    "Leganes",
    "UD Las Palmas":                 "Las Palmas",
    "Real Oviedo":                   "Oviedo",   # may not be in historical data
    # ── FBref variants → co.uk short names ───────────────────────────────
    "Atlético Madrid":               "Ath Madrid",
    "Atlético de Madrid":            "Ath Madrid",
    "Atletico Madrid":               "Ath Madrid",
    "Atletico de Madrid":            "Ath Madrid",
    "Atl. Madrid":                   "Ath Madrid",
    "Athletic Bilbao":               "Ath Bilbao",
    "Ath Bilbao":                    "Ath Bilbao",
    "Ath Madrid":                    "Ath Madrid",
    "Celta Vigo":                    "Celta",
    "Espanyol":                      "Espanol",
    "RCD Espanyol":                  "Espanol",
    "Real Betis":                    "Betis",
    "R. Betis":                      "Betis",
    "Real Sociedad":                 "Sociedad",
    "R. Sociedad":                   "Sociedad",
    "Rayo Vallecano":                "Vallecano",
    "Dep. Alaves":                   "Alaves",
    "Alaves":                        "Alaves",
    "Almeria":                       "Almeria",
    "Valladolid":                    "Valladolid",
    "Leganes":                       "Leganes",
    "Las Palmas":                    "Las Palmas",
    "Mallorca":                      "Mallorca",
    "Osasuna":                       "Osasuna",
    "Getafe":                        "Getafe",
    "Girona":                        "Girona",
    "Sevilla":                       "Sevilla",
    "Valencia":                      "Valencia",
    "Villarreal":                    "Villarreal",
    "Levante":                       "Levante",
    "Elche":                         "Elche",
    "Bilbao":                        "Ath Bilbao",
    "Barcelona":                     "Barcelona",
    "Real Madrid":                   "Real Madrid",
}


def normalize_team_name(name: str) -> str:
    """Return the canonical team name, or the original if not in the map."""
    if not isinstance(name, str):
        return str(name)
    return LA_LIGA_TEAM_MAP.get(name.strip(), name.strip())


def normalize_dataframe_teams(
    df,
    home_col: str = "HomeTeam",
    away_col: str = "AwayTeam",
):
    """Apply normalize_team_name to both team columns in a DataFrame."""
    import pandas as pd
    df = df.copy()
    if home_col in df.columns:
        df[home_col] = df[home_col].map(normalize_team_name)
    if away_col in df.columns:
        df[away_col] = df[away_col].map(normalize_team_name)
    return df
