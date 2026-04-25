# La Liga Linea — GitHub Copilot Instructions

## Project Overview

**App name:** La Liga Linea
**Purpose:** Streamlit multi-page app predicting La Liga (Spain) match outcomes and surfacing betting market value.
**Entry point:** `streamlit run predictions.py`
**Part of:** Betting Oracle suite (sibling apps: MLS Predictor, Premier League Predictor)

---

## Tech Stack

| Layer | Technology |
|---|---|
| UI | Streamlit ≥ 1.36 (`st.navigation`, `st.Page`) |
| ML | XGBoost, scikit-learn (VotingClassifier), scipy |
| Data | pandas, numpy |
| Scraping | requests, BeautifulSoup4, lxml |
| Visualization | Plotly Express |
| Config | python-dotenv (`.env` file) |
| PDF export | fpdf2 |
| Python | 3.9+ |

---

## File Conventions

### Key files
- `predictions.py` — entry point; sets `st.set_page_config`, sidebar, and `st.navigation`. **No model code here.**
- `utils.py` — ALL shared functions: data loading, feature engineering, model training, display helpers. Import from here, don't duplicate.
- `pages/*.py` — individual Streamlit pages. No `st.set_page_config` calls here.
- `footer.py` — `add_betting_oracle_footer()` must be called in `predictions.py` after `pg.run()`.
- `themes.py` — `apply_theme()` must be called in `predictions.py` before `pg.run()`.

### Data files
- `data_files/combined_historical_data.csv` — 10 seasons SP1.csv from football-data.co.uk
- `data_files/upcoming_fixtures.csv` — upcoming PD fixtures from football-data.org or ESPN
- `data_files/predictions_log.csv` — rolling predictions log (auto-generated)
- `data_files/raw/` — raw scraped data (odds, FBref xG, Copa fixtures)
- `models/ensemble_model.pkl` — trained VotingClassifier (auto-generated)

### Fetch scripts (not yet created — see roadmaps)
- `fetch_historical_csvs.py` — downloads SP1.csv for seasons 2015-16 → present
- `fetch_upcoming_fixtures.py` — football-data.org PD competition, status=SCHEDULED
- `fetch_fbref_xg.py` — scrapes FBref La Liga team xG (comp ID 12)
- `fetch_odds.py` — The Odds API, sport key `soccer_spain_la_liga`
- `fetch_copa_fixtures.py` — Copa del Rey fixtures from football-data.org

---

## La Liga Domain Knowledge

### API / data source specifics
- **football-data.org competition code:** `PD` (Primera División)
- **football-data.co.uk file:** `SP1.csv` for each season
- **FBref La Liga competition ID:** `12`
- **The Odds API sport key:** `soccer_spain_la_liga`
- **ESPN API URL:** `https://site.api.espn.com/apis/site/v2/sports/soccer/esp.1/scoreboard`

### football-data.co.uk column names (raw CSV)
- `Date` → `MatchDate`, `FTHG` → `FullTimeHomeGoals`, `FTAG` → `FullTimeAwayGoals`
- `FTR` → `FullTimeResult` (H/D/A)
- `B365H` / `B365D` / `B365A` → `OddsHome` / `OddsDraw` / `OddsAway`

### Model internals
- **Target encoding:** A=0, D=1, H=2 (alphabetical — matches scikit-learn LabelEncoder)
- **`predict_proba` column order:** [P(Away), P(Draw), P(Home)]
- **Feature set:** `FEATURE_COLS` in `utils.py` — 13 features, all shift(1) to prevent leakage
- **Ensemble weights:** XGBoost=2, RF=1.5, GB=1, LR=0.5 (soft voting)
- **La Liga avg goals:** home=1.45, away=1.12 (used as defaults when data is missing)

### La Liga-specific features
- **Copa del Rey congestion flag** — binary: 1 if team played Copa del Rey ≤ 4 days before the match
- **No referee feature** — referee assignment data is sparse in English for La Liga (unlike EPL)
- **No turf/surface flag** — all 20 La Liga stadiums use natural grass
- **No travel-distance feature** — not needed (unlike MLS)

---

## Coding Conventions

### Streamlit patterns
```python
# Cache DataFrames (serializable)
@st.cache_data(ttl=3600)
def load_something(path: str) -> pd.DataFrame: ...

# Cache models / non-serializable objects
@st.cache_resource
def load_model() -> VotingClassifier: ...

# Never call st.set_page_config() in pages/*.py — only in predictions.py
# Always use st.session_state["selected_season"] for cross-page state
```

### Feature engineering rules
- Always `sort_values(["Team", "MatchDate"])` before groupby rolling
- Always use `shift(1)` before `.rolling(n).mean()` to prevent data leakage
- Fill NaN features with La Liga averages, not 0

### Team name normalization
```python
from team_name_mapping import normalize_team_name, normalize_dataframe_teams
```
Always normalize team names when merging data from different sources.

### Error handling
- Use `path.exists(csv_path)` before loading; show `st.info()` with the fix command
- Use `st.stop()` after blocking warnings (don't render partial pages)
- Wrap external API calls in try/except; return empty DataFrame on failure

### Security
- API keys via `python-dotenv`: `from dotenv import load_dotenv; load_dotenv()`
- Never hardcode keys; never log keys
- `.env` is gitignored; `.streamlit/secrets.toml` is gitignored

---

## Adding a New Page

1. Create `pages/new_page.py`
2. Add `st.Page("pages/new_page.py", title="...", icon="...")` to the appropriate group in `predictions.py`
3. No `st.set_page_config` in the page file
4. Import helpers from `utils.py`; do not re-implement

## Adding a New Feature Column

1. Add the column name to `FEATURE_COLS` in `utils.py`
2. Implement the computation in `calculate_la_liga_features()` using vectorized groupby
3. Delete `models/ensemble_model.pkl` to force retraining
4. Update `_team_stats_for_upcoming()` to compute the same feature for upcoming fixtures

---

## Roadmaps

Full implementation details with code are in `docs/`:
- `docs/roadmap-features.md` — UI features
- `docs/roadmap-models.md` — ML model stack
- `docs/roadmap-data.md` — data sources and scrapers
- `docs/roadmap-layout.md` — Streamlit layout
- `docs/roadmap-infrastructure.md` — GitHub Actions, caching, logging
- `docs/roadmap-quick-wins.md` — quick improvements
