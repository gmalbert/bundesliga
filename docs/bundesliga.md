# Bundesliga (Germany)

**Tier: A** | Season: August–May | DraftKings: Full markets

## Data Sources

- `football-data.org` (free tier — fixtures, results, standings)
- `OpenFootball` (free — historical results going back decades)
- FBref (scrapeable — xG, possession, advanced metrics)
- Understat.com (free xG data, Python-scrapeable)
- The Odds API (market lines)

## Overview

The Bundesliga is the highest-scoring of Europe's top five leagues, averaging well over 3 goals per game most seasons. That makes it particularly well-suited for totals modeling — over/under predictions tend to perform better in higher-variance, higher-scoring environments. Like La Liga, the data pipeline is a direct port of your EPL work, and both football-data.org and Understat provide excellent free coverage.

Understat.com is a particularly valuable resource for the Bundesliga specifically — it has been tracking xG data for the German top flight since 2014, with a clean, scrapeable structure that works identically to how you'd pull EPL or La Liga data from the same source.

---

## Pros

- **Same copy-paste architecture as EPL/La Liga.** football-data.org covers Bundesliga under league code BL1. Same API, same schema, same pipeline. Swap the competition code and retrain.
- **Understat is exceptional for Bundesliga xG.** Understat.com covers all five major European leagues with consistent xG methodology going back to 2014. The Python scraping pattern is identical across leagues — it's a single reusable function.
- **High-scoring league improves totals modeling.** With 3+ goals per game on average, the over/under market has more statistical signal and less noise than in a low-scoring league like Serie A. Your totals model should perform well here.
- **Strong DraftKings market coverage.** As one of the five major European leagues, Bundesliga gets full market treatment on DraftKings — 1X2, handicap, totals, BTTS, and futures.
- **OpenFootball provides deep historical data.** The OpenFootball project on GitHub has Bundesliga results going back to the 1960s — useful for very long-horizon feature engineering or historical ELO systems.
- **18-team league with strong mid-table competition.** Unlike La Liga (where the top 3 are dominant) or Ligue 1 (PSG), the Bundesliga has genuine mid-table parity below Bayern Munich and Bayer Leverkusen, making spread and total predictions more interesting.

---

## Cons

- **Bayern Munich dominance creates class imbalance.** For much of the last decade, Bayern has won the title by 10+ points. Matches involving Bayern are heavily skewed in win probability — your model will correctly assign very high win probabilities to Bayern, but those picks may be unattractive for bettors (very negative moneyline). Building compelling picks content around favorites is a UX challenge.
- **English-language data sources are slightly thinner than EPL.** Injury reports, press conferences, and lineup news for mid-table Bundesliga clubs are reported primarily in German. English-language aggregators cover the top clubs well but miss some depth on clubs like Mainz, Bochum, or Augsburg.
- **Relegation playoff format adds a wild card.** The Bundesliga's 16th-place team plays a two-legged playoff against the 3rd-place team from Bundesliga 2. This creates unusual late-season incentive structures for teams on the edge of the bottom 3.
- **International break fixture gaps** are common across all European leagues, but Bundesliga's winter break (usually 4–6 weeks in December/January) is longer than most — your pipeline needs to handle extended gaps in training data gracefully.

---

## Recommended Build Approach

**Primary model targets:** 1X2 match result, over/under 2.5 goals, over/under 3.5 goals (particularly relevant given scoring rates), both teams to score, Asian handicap.

**Key features to engineer:**
- xG and xGA from Understat (last 5 and last 10 games, weighted)
- Goals scored per game rolling average (home and away split)
- Form points (last 5 games)
- Head-to-head results
- Bundesliga-specific: home crowd factor (several clubs have extraordinary home advantages — Dortmund's Signal Iduna Park in particular)
- Days rest / fixture congestion (DFB-Pokal cup involvement)

**Understat data pull (Python):**
```python
import understatapi

client = understatapi.UnderstatClient()

# Pull league-level match data
matches = client.league(league="Bundesliga").get_match_data(season="2023")

# Pull team-level stats
team_stats = client.league(league="Bundesliga").get_team_data(season="2023")
```

**Suggested model stack:** Same EPL ensemble. Totals model deserves extra attention given the high-scoring nature — consider a separate Poisson regression model for goals scored/conceded as a complement to the ML ensemble.

**Backtesting window:** 2014–15 through 2023–24 (10 seasons, ~3,060 games). Strong statistical power for both classification (result) and regression (total goals) tasks.

---

## Build Priority

**High — bundle with La Liga and Serie A as a European leagues expansion.** The marginal cost of adding Bundesliga once you've ported to La Liga is almost zero. Ship all three together and give Betting Oracle a full five-league European soccer section.
