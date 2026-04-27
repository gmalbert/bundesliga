# External Research: Bundesliga Predictor (kontainer-sh) & OpenLigaDB

**Sources reviewed:**
- Reddit: r/sportsanalytics — ["I built a Bundesliga predictor that reaches 97.5%…"](https://www.reddit.com/r/sportsanalytics/comments/1sw5moy/)
- GitHub: [kontainer-sh/bundesliga-predictor](https://github.com/kontainer-sh/bundesliga-predictor)
- API: [OpenLigaDB](https://www.openligadb.de/) / [api.openligadb.de](https://api.openligadb.de)

---

## 1. What kontainer-sh Built

A command-line Kicktipp optimizer (not a Streamlit app) for the Bundesliga, focused entirely on **maximizing Kicktipp-scoring-scheme points** (Tendenz 1 / Tordifferenz 2 / Remis 2 / Exakt 3) rather than general match-outcome accuracy.

### Architecture: Dixon-Coles (30%) + Pinnacle Odds (70%)

The core insight is a **two-component blended score matrix**:

1. **Dixon-Coles Poisson model** — estimates `λ_home` and `λ_away` from historical results using Maximum Likelihood with exponential time weighting. Outputs an `(N+1) × (N+1)` score probability matrix.
2. **Pinnacle odds-derived score matrix** — converts H/D/A odds into `λ_home` and `λ_away` by fitting a Poisson distribution to minimize KL divergence from the implied probabilities. Outputs a second score matrix.

The two matrices are linearly blended: `combined = 0.3 × DC_matrix + 0.7 × odds_matrix`.

### Tip Selection via Expected Value

Rather than predicting H/D/A class labels, the model selects the tip `(t_home, t_away)` that maximizes:

```
E[points | tip t] = Σ_{h,a} P(h:a) × KicktippPoints(t, h:a)
```

This is computed analytically with `numpy.einsum` over the score matrix — deterministic and fast (no Monte Carlo).

### Performance on Backtest (Season 2024/25, Spieltage 1–30, 270 matches)

| Approach | Total Points | Avg per match |
|---|---|---|
| Always predict 2:1 (uninformed baseline) | ~192 | 0.711 |
| Dixon-Coles model only | 211 | 0.781 |
| Model + Pinnacle odds | 231 | 0.856 |
| Poisson-ceiling (odds only, optimal) | 232 | 0.859 |
| True ceiling estimate | ~237–252 | 0.878–0.933 |
| Perfect oracle | 810 | 3.000 |

**Key takeaway:** The model captures ~75–85% of the achievable information gap above the naive baseline. Odds alone (70%) dominate the signal; the Dixon-Coles model adds marginal but measurable value.

### Hyperparameters (cross-validated over 3 seasons)

| Parameter | Value | Description |
|---|---|---|
| `HALF_LIFE_DAYS` | 300 | Exponential decay ~10 months |
| `NUM_PREV_SEASONS` | 3 | Training window |
| `ODDS_WEIGHT` | 0.70 | Fraction of odds matrix in blend |
| `MAX_TIP_GOALS` | 2 | Tips capped at 2:2 (avoids long-shot bets) |

### What They Tested and Rejected

This is the most valuable section — **negative knowledge**:

| Feature / Modification | Effect | Reason |
|---|---|---|
| xG (Expected Goals) | ±0 points | Already priced into the odds |
| Negative Binomial distribution | ±0 | Poisson overdispersion (~12–15%) is too small to matter at this sample size |
| L2 regularization | ±0 | Enough training data; not overfit |
| Over/Under odds | −11 points | Degrades tendency accuracy |
| Team-specific home advantage | −13 points | Overfits across seasons |
| Draw boost | −13 points | Model already optimizes correctly |
| Score matrix recalibration | −33 points | Bias patterns are unstable across seasons |

---

## 2. Dixon-Coles Model — Technical Details

### Time Weighting
Each historical match is weighted by: `w = exp(-ln(2) × Δdays / half_life)`

This means a match from 300 days ago counts as half a recent match. Matches from 3+ years ago contribute negligibly.

### Low-Score Correction (ρ parameter)
Dixon-Coles (1997) adds a dependency parameter ρ to correct Poisson's independence assumption at low scores:

```
Correction factor τ(h, a, λ_h, λ_a, ρ):
  h=0, a=0 → 1 − λ_h × λ_a × ρ
  h=1, a=0 → 1 + λ_a × ρ
  h=0, a=1 → 1 + λ_h × ρ
  h=1, a=1 → 1 − ρ
  else     → 1.0
```

ρ is estimated jointly with attack/defense strengths via MLE. Empirically, 0:0, 1:0, 0:1, 1:1 occur at different rates than pure independent Poisson predicts.

### Attack/Defense Parameterization
Each team has two log-scale parameters: attack strength `α_i` and defense weakness `β_i`. A global home advantage `γ` is estimated.

```
λ_home = exp(α_home − β_away + γ)
λ_away = exp(α_away − β_home)
```

Parameters are fitted by MLE on all historical matches, weighted by recency.

---

## 3. OpenLigaDB — Free Results API

**URL:** `https://api.openligadb.de`
**Auth:** None required
**License:** Open Database License (ODbL)
**Updates:** Community-maintained, results entered by registered users

### League Codes
| Competition | Code |
|---|---|
| 1. Bundesliga | `bl1` |
| 2. Bundesliga | `bl2` |
| DFB-Pokal | `dfb` |
| Champions League | `cl` |

### Key Endpoints

```
GET /getmatchdata/bl1/{season}                  → all matches in a season
GET /getmatchdata/bl1/{season}/{matchday}        → specific matchday
GET /getmatchdata/bl1/{season}/{teamNameFilter}  → all matches for a team
GET /getcurrentgroup/bl1                         → current matchday number
GET /getbltable/bl1/{season}                     → league table
GET /getgoalgetters/bl1/{season}                 → top scorers
GET /getavailableteams/bl1/{season}              → team list for a season
GET /getnextmatchbyleagueshortcut/bl1            → next upcoming match
GET /getlastmatchbyleagueshortcut/bl1            → most recently completed match
GET /getmatchdata/{teamId1}/{teamId2}            → head-to-head history
```

Season format: `2024` means the 2024/25 season.

### Match Data Schema
Each match object includes:
- `matchDateTimeUTC` — kickoff time (ISO 8601, UTC)
- `team1` / `team2` — `{ teamId, teamName, shortName, teamIconUrl }`
- `matchResults` — array of result objects; `resultTypeID=2` = final score
- `goals` — array of individual goal events with minute, scorer, penalty/own-goal flags
- `group.groupOrderID` — matchday number
- `location` — stadium name (when available)

### Limitations
- No betting odds
- No referee data
- No xG or advanced stats
- Results entered by community (occasional delays/errors)
- The kontainer-sh project uses a **permanent local cache** for historical data and **6-hour cache for live odds**

---

## 4. Data Source Comparison

| Source | Data | Cost | Notes |
|---|---|---|---|
| OpenLigaDB | Bundesliga results, 1. + 2. + DFB Pokal, real-time | Free, no key | Community maintained |
| football-data.co.uk | Historical D1.csv with Pinnacle H/D/A odds | Free, no key | Static CSVs, updated post-season |
| The Odds API | Live pre-match odds (sport: `soccer_germany_bundesliga`) | Free tier: 500 req/month | Real-time; Pinnacle + others |
| FBref | xG, possession, pressing stats | Free (scraping) | Rate-limited; not needed per research |

---

## 5. Theoretical Ceiling Analysis

The project includes a rigorous ceiling analysis using three modes:

- **Market ceiling** — what the Poisson-reconstructed H/D/A odds can achieve at best
- **Hindsight ceiling** — oracle that picks the optimal tip per match after the fact
- **Bins ceiling** — one fixed tip per "odds bucket" across all matches in that bucket

The finding: without Correct Score odds (not freely available), you cannot close the last ~5–20 points gap to the true ceiling. H/D/A odds are sufficient for tendency, but not for precision within tendencies.

---

## 6. GitHub Actions Automation

The project auto-runs a prediction generation workflow daily at 10:00 CEST:
- Checks if the next matchday is within 3 days
- Fetches live odds from The Odds API (if key available)
- Generates a markdown tip file at `tips/YYYY_YYYY_spieltag_XX.md`
- Deploys an HTML summary to GitHub Pages

This pattern is directly applicable to our automation setup.
