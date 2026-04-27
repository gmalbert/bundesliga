# Recommendations from External Research

**Based on:** [research-external-predictor.md](research-external-predictor.md)
**Applies to:** This project (Bet Bundesliga — Streamlit app)
**Date:** April 2026

---

## Priority Matrix

| Recommendation | Impact | Effort | Priority |
|---|---|---|---|
| Dixon-Coles ρ low-score correction | High | Low | **P0** |
| Exponential time weighting in Poisson model | High | Low | **P0** |
| OpenLigaDB as primary results source | High | Medium | **P1** |
| Odds-blended score matrix (70/30) | High | Medium | **P1** |
| Score matrix → full probability distribution | Medium | Medium | **P1** |
| 2. Bundesliga data for promoted teams | Medium | Low | **P2** |
| OpenLigaDB goal scorer data | Low | Low | **P2** |
| Ceiling analysis tooling | Medium | Medium | **P2** |
| Drop xG from feature set | Neutral | Low | **P2** (cleanup) |
| Do NOT add team-specific home advantage | Negative | — | **Avoid** |
| Do NOT add draw boost | Negative | — | **Avoid** |
| Do NOT add score recalibration | Negative | — | **Avoid** |

---

## P0 — Immediate Improvements to `poisson_predictor.py`

### 1. Add Dixon-Coles ρ Low-Score Correction

The current `poisson_predictor.py` uses independent Poisson without any low-score correction. The Dixon-Coles paper (1997) showed that 0:0, 1:0, 0:1, and 1:1 occur at different rates than pure independence predicts.

**Current state:** `score_matrix[i, j] = poisson(λ_h, i) × poisson(λ_a, j)`

**Recommended:** Apply the ρ correction factor after building the matrix:

```python
def dc_tau(h, a, lh, la, rho):
    if h == 0 and a == 0: return 1 - lh * la * rho
    if h == 1 and a == 0: return 1 + la * rho
    if h == 0 and a == 1: return 1 + lh * rho
    if h == 1 and a == 1: return 1 - rho
    return 1.0

# Apply to score_matrix after building it, then renormalize
for h in range(max_goals + 1):
    for a in range(max_goals + 1):
        score_matrix[h, a] *= dc_tau(h, a, lh, la, rho)
score_matrix /= score_matrix.sum()
```

ρ should be estimated from historical data. The kontainer-sh project estimates it jointly via MLE. A reasonable starting value for Bundesliga data is `rho ≈ 0.03–0.06`.

### 2. Add Exponential Time Weighting

The current `compute_team_strengths()` treats all historical matches equally. The kontainer-sh project uses a 300-day half-life, which empirically outperforms equal-weight baselines.

**Recommended addition to `compute_team_strengths`:**

```python
def time_weight(match_date, ref_date, half_life_days=300):
    delta = max((ref_date - match_date).days, 0)
    return math.exp(-math.log(2) * delta / half_life_days)
```

Pass `ref_date=datetime.today()` when training, and weight each match's goal contribution by its time weight before computing averages.

### 3. Switch to MLE Attack/Defense Parameter Estimation

The current approach computes raw goal ratio averages. The Dixon-Coles approach uses Maximum Likelihood Estimation with log-scale attack/defense parameters and a shared home advantage term:

```
λ_home = exp(α_home − β_away + γ)
λ_away = exp(α_away − β_home)
```

This is more statistically principled and allows joint estimation of the ρ correction. The `scipy.optimize.minimize` approach (used by kontainer-sh) converges reliably on ~1,000–3,000 matches.

---

## P1 — Odds-Blended Score Matrix Architecture

### The Core Insight

The external research conclusively demonstrates that **betting odds contain more information than the statistical model**. The Dixon-Coles model alone achieves 0.781 avg points/game; odds alone achieve ~0.856; blending at 70% odds / 30% model achieves the same 0.856 — confirming the model adds marginal value over noise.

**Implication:** The ensemble classifier's use of `ImpliedProb_HomeWin/Draw/AwayWin` as *features* is suboptimal. A better architecture blends the model's score matrix directly with an odds-derived score matrix.

### Recommended Architecture Change

```
Current:  [features including implied probs] → VotingClassifier → P(H/D/A)
Proposed: [historical results] → Dixon-Coles → score_matrix_dc (30%)
          [live odds P(H), P(D), P(A)] → Poisson fit → score_matrix_odds (70%)
          combined = 0.3 × score_matrix_dc + 0.7 × score_matrix_odds
          combined → P(H/D/A), P(over 2.5), P(BTTS), P(exact score)
```

This provides richer outputs (full score distribution) while respecting the market's superior information.

### Odds-to-Score-Matrix Conversion

To convert H/D/A probabilities from odds into a score matrix:
1. Remove overround: `p_h, p_d, p_a = normalize(1/odds_h, 1/odds_d, 1/odds_a)`
2. Fit `λ_home`, `λ_away` by minimizing KL divergence from the Poisson-implied H/D/A probabilities
3. Build score matrix using those lambdas

This is the approach used by kontainer-sh (`odds_to_score_matrix` function in `kicktipp.py`).

---

## P1 — OpenLigaDB Integration

### Why Switch from football-data.co.uk for Results

| | football-data.co.uk CSVs | OpenLigaDB |
|---|---|---|
| Update frequency | Post-season batch | Real-time (community) |
| Authentication | None | None |
| Current matchday | Manual | `/getcurrentgroup/bl1` |
| 2. Bundesliga data | Separate file (`D2.csv`) | Same API (`bl2`) |
| DFB Pokal | Not available | Available (`dfb`) |
| Goal events | No | Yes |
| License | Unclear | ODbL (open) |

### Recommended `fetch_openligadb.py`

Key endpoints to use:

```python
BASE = "https://api.openligadb.de"

# Full season results
GET {BASE}/getmatchdata/bl1/{season}      # e.g. season=2024 → 2024/25

# Current matchday (for automation)
GET {BASE}/getcurrentgroup/bl1

# League table
GET {BASE}/getbltable/bl1/{season}

# Top scorers (optional, for team_deep_dive page)
GET {BASE}/getgoalgetters/bl1/{season}

# 2. Bundesliga (for promoted team context)
GET {BASE}/getmatchdata/bl2/{season}
```

**Caching strategy** (from kontainer-sh): Cache completed seasons permanently; cache live/current data with 6-hour TTL.

### Match Data Parsing

The critical field is `matchResults`: filter for `resultTypeID == 2` (final score). Matches without this field are unplayed.

```python
final = next((r for r in m["matchResults"] if r["resultTypeID"] == 2), None)
if final is None:
    continue  # Not yet played
home_goals = final["pointsTeam1"]
away_goals = final["pointsTeam2"]
```

---

## P2 — Use 2. Bundesliga Data for Promoted Teams

Newly promoted teams often have sparse data in the top flight. Training the Dixon-Coles model on their 2. Bundesliga matches (with an appropriate strength discount) provides better initial estimates than relying on league averages.

The kontainer-sh project includes 2. Bundesliga data in its training set with the same OpenLigaDB API call (`bl2` instead of `bl1`). This is a low-effort addition since OpenLigaDB exposes both leagues identically.

**Consideration:** Apply a small penalty to 2. Bundesliga-derived strengths (e.g., reduce attack/defense rating by 10–15%) since the league is weaker. Alternatively, let the time-weighting handle this naturally as new Bundesliga results accumulate.

---

## P2 — Goal Scorer Data from OpenLigaDB

The `/getgoalgetters/bl1/{season}` endpoint returns a ranked list of top scorers. This can enrich the **team_deep_dive** page without any model changes:

- Show top scorers for each team
- Compute "attack concentration risk" (what % of goals come from the top 1–2 scorers?)
- Flag matches where a team's top scorer is suspended (if injury data is available)

This is a UI/display improvement, not a model improvement.

---

## P2 — Ceiling Analysis Tooling

The external project includes a sophisticated `ceiling` command that calculates:
- **Market ceiling**: theoretical max points using only Pinnacle H/D/A odds optimally
- **Hindsight ceiling**: oracle max (knows the actual result)
- **Bins ceiling**: fixed tip per odds-bucket, calibrated on the same season

Adding a simplified version to the **performance** page would help evaluate whether model improvements are meaningful vs. approaching the theoretical ceiling.

---

## What to Explicitly Avoid

Based on the kontainer-sh project's empirical testing (3 seasons of cross-validation):

### Do NOT add team-specific home advantage
The kontainer-sh project tested this and lost −13 points. The global home advantage γ is sufficient; team-level home advantage overfits across seasons. **Do not add a per-team home advantage multiplier.**

### Do NOT add a draw-probability boost
Tested and lost −13 points. The Dixon-Coles ρ correction already handles systematic errors in draw prediction.

### Do NOT add score matrix recalibration
Tested and lost −33 points. Bias patterns in the score matrix are not stable across seasons — learning a correction from past seasons introduces more error than it fixes.

### Do NOT use xG as a model feature
The external research confirms xG provides ±0 improvement because it is already priced into the betting odds. The `ImpliedProb_*` features already capture everything xG would add (and more). If xG features are added, they should replace implied-prob features, not supplement them.

---

## Feature Engineering Note

The current `FEATURE_COLS` uses rolling averages (L5, L10) computed with `shift(1)`. This is correct and avoids data leakage. The main improvements are:

1. Replace the basic ratio-based `compute_team_strengths()` with proper Dixon-Coles MLE
2. Add time weighting when computing rolling averages (weight recent matches more)
3. Consider replacing `HomeRestDays` / `AwayRestDays` with a binary "fatigue flag" (played within 4 days), which may be more stable

---

## Implementation Order

```
Phase 1 (1–2 days):
  ├── Add ρ correction to poisson_predictor.py
  ├── Add time weighting to compute_team_strengths()
  └── Backtest: verify improvement on held-out season

Phase 2 (3–5 days):
  ├── Implement full Dixon-Coles MLE in poisson_predictor.py
  ├── Create fetch_openligadb.py with caching
  └── Replace football-data.co.uk results with OpenLigaDB

Phase 3 (1 week):
  ├── Implement odds-to-score-matrix conversion
  ├── Implement blended score matrix (70/30)
  └── Update predictions pages to use full score distribution

Phase 4 (ongoing):
  ├── Add 2. Bundesliga data for promoted teams
  ├── Add goal scorer display to team_deep_dive
  └── Implement ceiling analysis for performance page
```
