# Bundesliga Predictor — Next 5 Features to Implement

> **Based on:** Codebase gap analysis as of July 2025

---

## Feature 1: xG-Based Expected Goals Integration

**Why:** `fetch_fbref_xg.py` exists and targets FBref (adapt comp ID to Bundesliga: `20`), but xGF/xGA are not yet in `FEATURE_COLS` in `utils.py`. xG is the strongest rolling predictor of future match outcomes.

**How:**
1. Verify FBref Bundesliga comp ID (typically `20`) in `fetch_fbref_xg.py`
2. Add `home_xg_l5`, `home_xga_l5`, `away_xg_l5`, `away_xga_l5` to `FEATURE_COLS` in `utils.py`
3. Compute rolling 5-match averages in `calculate_la_liga_features()` using `shift(1)` to prevent leakage
4. Update `_team_stats_for_upcoming()` to include the same features
5. Delete `models/ensemble_model.pkl` to force retraining

**Complexity:** Low

---

## Feature 2: DFB-Pokal Congestion Feature

**Why:** The existing `fetch_copa_fixtures.py` is designed for Copa del Rey. Adapting it for the DFB-Pokal (German Cup) would add a meaningful fatigue binary flag. Top Bundesliga clubs often play midweek DFB-Pokal ties that affect their weekend league form.

**How:**
1. Copy `fetch_copa_fixtures.py` → `fetch_dfb_pokal_fixtures.py`
2. Update to use football-data.org competition code `DFB` (DFB-Pokal)
3. Create `home_dfb_fatigue` / `away_dfb_fatigue`: 1 if team played a Pokal match ≤4 days before this Bundesliga game
4. Add to `FEATURE_COLS` and update `_team_stats_for_upcoming()`

**Complexity:** Low

---

## Feature 3: Model Calibration Reliability Diagram

**Why:** The reliability diagram is the most diagnostic single chart for any probabilistic model — it reveals systematic over- or under-confidence. `data_files/predictions_log.csv` already accumulates resolved predictions.

**How:**
1. Load `predictions_log.csv` and filter to completed matches (result known)
2. Bin predictions into 10 probability deciles (0–10%, ..., 90–100%)
3. Compute mean predicted probability and actual outcome rate per bin
4. Plot as Plotly scatter with reference diagonal (perfect calibration line)
5. Add to `pages/model_performance.py` as a new tab or expander section

**Complexity:** Low

---

## Feature 4: Automated Season Transition Handling

**Why:** At the start of each new Bundesliga season, rolling stats from `shift(1)` windows are NaN for the first 5 matches. Currently these are filled with `0` which biases early-season predictions. Handling this gracefully improves the first 5 matchday predictions significantly.

**How:**
1. In `calculate_la_liga_features()`, replace `fillna(0)` with `fillna(La Liga team average goals)` for rolling stats
2. Add a startup check: if `data_files/combined_historical_data.csv` has no matches from the current season, show an `st.info()` banner explaining early-season data is limited
3. Optionally: use prior season's last-10 form as the initial rolling window for each team

**Complexity:** Low

---

## Feature 5: Bundesliga 2 Relegation Playoff Prediction

**Why:** Each season, the 16th-place Bundesliga team plays a 2-leg playoff against the 3rd-place Bundesliga 2 side for the final promotion/relegation spot. This is a high-interest, limited-information matchup — the model can fill a gap that sportsbooks price inefficiently.

**How:**
1. Fetch Bundesliga 2 data from football-data.co.uk (D2.csv) to compute the 3rd-place team's form features
2. Add a dedicated page `pages/relegation_playoff.py` that presents both teams' form, xG, and the model's predicted win probability for each leg
3. Use the existing ensemble model with home/away team stats — no new training required
4. Show market odds from The Odds API for edge calculation

**Complexity:** High
