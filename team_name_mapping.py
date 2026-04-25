"""Normalize team names from football-data.co.uk, football-data.org, and FBref
to the canonical names used in the historical data (football-data.co.uk short names).

The canonical form is the football-data.co.uk short name because that is what
combined_historical_data.csv contains and what the model was trained on.
All external sources (football-data.org API full names, FBref, ESPN) are
mapped back to those short names so team stats lookups succeed.
"""

# Maps source variant → football-data.co.uk canonical short name
BUNDESLIGA_TEAM_MAP: dict[str, str] = {
    # ── football-data.org full names → co.uk short names ─────────────────────
    "FC Bayern München":               "Bayern Munich",
    "FC Bayern Munich":                 "Bayern Munich",
    "Borussia Dortmund":                "Dortmund",
    "Bayer 04 Leverkusen":              "Leverkusen",
    "RB Leipzig":                       "RB Leipzig",
    "Eintracht Frankfurt":              "Ein Frankfurt",
    "VfL Wolfsburg":                    "Wolfsburg",
    "Borussia Mönchengladbach":         "M'gladbach",
    "Borussia Monchengladbach":         "M'gladbach",
    "TSG 1899 Hoffenheim":              "Hoffenheim",
    "TSG Hoffenheim":                   "Hoffenheim",
    "SC Freiburg":                      "Freiburg",
    "1. FC Union Berlin":               "Union Berlin",
    "FC Union Berlin":                  "Union Berlin",
    "VfB Stuttgart":                    "Stuttgart",
    "FC Augsburg":                      "Augsburg",
    "1. FSV Mainz 05":                  "Mainz",
    "FSV Mainz 05":                     "Mainz",
    "SV Werder Bremen":                 "Werder Bremen",
    "Werder Bremen":                    "Werder Bremen",
    "1. FC Köln":                       "FC Koln",
    "1. FC Koeln":                      "FC Koln",
    "FC Koeln":                         "FC Koln",
    "Hertha BSC":                       "Hertha",
    "Hertha Berlin":                    "Hertha",
    "Hamburger SV":                     "Hamburg",
    "HSV Hamburg":                      "Hamburg",
    "FC Schalke 04":                    "Schalke 04",
    "Schalke 04":                       "Schalke 04",
    "VfL Bochum":                       "Bochum",
    "VfL Bochum 1848":                  "Bochum",
    "1. FC Heidenheim":                 "Heidenheim",
    "1. FC Heidenheim 1846":            "Heidenheim",
    "SV Darmstadt 98":                  "Darmstadt",
    "Darmstadt 98":                     "Darmstadt",
    "SpVgg Greuther Fürth":            "Greuther Furth",
    "Greuther Furth":                   "Greuther Furth",
    "FC Ingolstadt 04":                 "Ingolstadt",
    "Holstein Kiel":                    "Holstein Kiel",
    "1. FC Holstein Kiel":              "Holstein Kiel",
    "FC St. Pauli":                     "St Pauli",
    "FC St Pauli":                      "St Pauli",
    # ── FBref / Understat variants → co.uk short names ───────────────────────
    "Bayern Munich":                    "Bayern Munich",
    "Dortmund":                         "Dortmund",
    "Bayer Leverkusen":                 "Leverkusen",
    "Leverkusen":                       "Leverkusen",
    "Frankfurt":                        "Ein Frankfurt",
    "Ein Frankfurt":                    "Ein Frankfurt",
    "M'gladbach":                       "M'gladbach",
    "Gladbach":                         "M'gladbach",
    "Monchengladbach":                  "M'gladbach",
    "Hoffenheim":                       "Hoffenheim",
    "Freiburg":                         "Freiburg",
    "Union Berlin":                     "Union Berlin",
    "Stuttgart":                        "Stuttgart",
    "Augsburg":                         "Augsburg",
    "Mainz":                            "Mainz",
    "Mainz 05":                         "Mainz",
    "Bremen":                           "Werder Bremen",
    "FC Koln":                          "FC Koln",
    "Koln":                             "FC Koln",
    "Köln":                             "FC Koln",
    "Hertha":                           "Hertha",
    "Hamburg":                          "Hamburg",
    "Schalke":                          "Schalke 04",
    "Bochum":                           "Bochum",
    "Heidenheim":                       "Heidenheim",
    "Darmstadt":                        "Darmstadt",
    "St Pauli":                         "St Pauli",
    "Leipzig":                          "RB Leipzig",
    "Wolfsburg":                        "Wolfsburg",
}


def normalize_team_name(name: str) -> str:
    """Return the canonical team name, or the original if not in the map."""
    if not isinstance(name, str):
        return str(name)
    return BUNDESLIGA_TEAM_MAP.get(name.strip(), name.strip())


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
