# La Liga Linea — Roadmap Index

**App name:** La Liga Linea  
**Framework:** Streamlit  
**League:** La Liga (Spain) — competition code `PD` (Primera División)  
**Season:** August – May  
**DraftKings:** Full market coverage

---

## What This App Does

La Liga Linea predicts the likely outcome of upcoming La Liga matches (Home Win, Draw, Away Win) using historical match data, FBref advanced metrics (xG, possession), and a multi-model ensemble. It also surfaces betting market context (implied probabilities, line value), team form, head-to-head history, and Copa del Rey fixture congestion — all presented in a clean Streamlit multi-page interface modeled after the MLS Predictor and Premier League Predictor apps.

---

## Roadmaps

| Roadmap | What it covers |
|---|---|
| [Features Roadmap](roadmap-features.md) | All UI features, tabs, pages, and user-facing functionality |
| [Models Roadmap](roadmap-models.md) | ML model stack: ensemble, Poisson, neural network, LSTM, calibration |
| [Data Roadmap](roadmap-data.md) | Data sources, feature engineering, injury/weather/odds pipelines |
| [Layout Roadmap](roadmap-layout.md) | Streamlit page structure, sidebar, theming, multi-page navigation |
| [Infrastructure Roadmap](roadmap-infrastructure.md) | Automation, GitHub Actions, caching, logging, testing |
| [Quick Wins](roadmap-quick-wins.md) | Easy short-effort improvements with immediate impact |

---

## Build Priority

```
Phase 1 — Foundation (Week 1-2)
  ├── Data pipeline: football-data.org PD fixtures + results
  ├── FBref xG scraper (team-level)
  ├── Feature engineering: form, xG rolling windows, rest days
  ├── Ensemble model (XGBoost + RF + GB + LR)
  └── Core Streamlit app: 5 tabs

Phase 2 — Enrichment (Week 3-4)
  ├── Odds integration (The Odds API or football-data.org odds)
  ├── Copa del Rey congestion flag
  ├── Poisson regression model
  └── Statistics tab: team form, head-to-head, league averages

Phase 3 — Advanced (Month 2)
  ├── Neural network (PyTorch)
  ├── LSTM momentum model
  ├── Markets page (EV engine, best bets)
  └── Prediction tracker (log + validate)

Phase 4 — Production (Month 3)
  ├── GitHub Actions nightly pipeline
  ├── SQLite migration
  ├── PDF report export
  └── Mobile-responsive tuning
```

---

## Project Structure (Target)

```
la-liga/
├── la_liga_linea.py              # Main Streamlit entry point
├── footer.py                     # Shared footer (Betting Oracle branding)
├── themes.py                     # Streamlit theme helpers
├── team_name_mapping.py          # Normalize team names across sources
├── fetch_upcoming_fixtures.py    # Pull upcoming PD fixtures from ESPN/football-data
├── fetch_fbref_xg.py             # Scrape xG data from FBref
├── fetch_odds.py                 # Pull market lines from The Odds API
├── prepare_model_data.py         # Feature engineering pipeline
├── train_models.py               # Offline model training script
├── track_predictions.py          # Log and validate predictions
├── data_files/
│   ├── logo.png
│   ├── combined_historical_data.csv
│   ├── upcoming_fixtures.csv
│   └── raw/
│       ├── fbref_team_xg.csv
│       └── odds.csv
├── models/
│   ├── ensemble_model.pkl
│   ├── poisson_predictor.py
│   └── lstm_predictor.py
├── pages/
│   ├── 6_Markets.py
│   └── 7_Best_Bets.py
├── automation/
│   └── nightly_pipeline.py
├── docs/
│   ├── README.md            ← you are here
│   ├── la-liga.md
│   ├── roadmap-features.md
│   ├── roadmap-models.md
│   ├── roadmap-data.md
│   ├── roadmap-layout.md
│   ├── roadmap-infrastructure.md
│   └── roadmap-quick-wins.md
├── .github/workflows/
│   └── nightly.yml
├── requirements.txt
└── .gitignore
```

---

## Key La Liga Differences from EPL/MLS

| Factor | EPL | MLS | La Liga |
|---|---|---|---|
| API code (football-data.org) | `PL` | N/A | `PD` |
| xG source | FBref / API-Football | American Soccer Analysis | FBref (identical structure to EPL) |
| Referee data (English) | Playmaker Stats | N/A | Limited — simplify or drop |
| Fixture weeks | ~38 rounds | ~34 rounds + playoffs | ~38 rounds |
| Cup competition | FA Cup / EFL Cup | US Open Cup | Copa del Rey |
| Average goals/game | ~2.8 | ~3.0 | ~2.6 |
| Dominant clubs | Man City, Liverpool | None (salary cap) | Real Madrid, Barcelona, Atlético |
| Stadium surface | All grass | ~25% turf | All grass |
| Travel fatigue | Low | Very high | Low-Medium (domestic flights for Canary Islands) |
