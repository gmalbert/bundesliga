# Bundesliga Predictor — Architecture

## Overview
Streamlit multi-page app predicting Bundesliga match outcomes and surfacing betting market value. Part of the Betting Oracle football suite (sibling to La Liga Linea, Premier League Predictor).

## Data Flow
```
football-data.co.uk (D1.csv)     The Odds API
        ↓                               ↓
fetch_historical_csvs.py        fetch_odds.py
        ↓                               ↓
data_files/combined_historical_data.csv
        ↓
prepare_model_data.py / utils.py
        ↓
Feature Engineering (13 features, shift(1) leakage prevention)
        ↓
VotingClassifier ensemble (XGBoost + RF + GB + LR)
        ↓
models/ensemble_model.pkl
        ↓
Streamlit pages → predictions.py (entry)
```

## ML Model
- **Ensemble**: `VotingClassifier` (soft voting)
  - XGBoost weight=2, RF weight=1.5, GB weight=1, LR weight=0.5
- **Target encoding**: A=0, D=1, H=2 (alphabetical, scikit-learn LabelEncoder)
- **`predict_proba` column order**: [P(Away), P(Draw), P(Home)]
- **Features** (`FEATURE_COLS` in `utils.py`): 13 features using `shift(1)` rolling windows

## API Integrations
| Source | Purpose | Key |
|--------|---------|-----|
| football-data.co.uk | Historical D1.csv per season | None (public download) |
| football-data.org | Upcoming fixtures (BL1 competition) | `FOOTBALL_DATA_API_KEY` |
| FBref | Team xG (comp ID 9) | None (scraped) |
| The Odds API | Live market odds | `ODDS_API_KEY` |
| ESPN API | Scoreboard fallback | None (public) |

## Key Components
- `predictions.py` — entry point, `st.set_page_config`, `st.navigation`
- `utils.py` — ALL shared functions: data loading, feature engineering, model training
- `themes.py` — `apply_theme()`, `plotly_theme()`, day/night CSS
- `pages/*.py` — individual Streamlit pages (no `st.set_page_config`)
- `fetch_historical_csvs.py` — downloads D1.csv seasons
- `fetch_upcoming_fixtures.py` — football-data.org BL1 fixtures
- `fetch_fbref_xg.py` — FBref Bundesliga xG scraper
- `team_name_mapping.py` — normalises team names across data sources

## Storage
- `data_files/combined_historical_data.csv` — 10 seasons historical
- `data_files/upcoming_fixtures.csv` — scheduled fixtures
- `data_files/predictions_log.csv` — rolling predictions log
- `models/ensemble_model.pkl` — trained VotingClassifier (gitignored)
