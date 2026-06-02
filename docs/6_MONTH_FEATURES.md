# Bundesliga Predictor — 6-Month Feature Roadmap

## Month 1: Core Predictions

- **Today's matches page** — Dedicated "Matchday" page listing only fixtures happening today or tomorrow with prediction cards.
- **Confidence badge** — ✅ BET / ➡ LEAN / ⛔ PASS badge system for each fixture based on edge tier.
- **DFB-Pokal schedule widget** — Sidebar notice when a team has a midweek cup game that may affect form.

## Month 2: Data Enrichment

- **FBref xG integration** — Pull team xG and xGA from FBref, display on the Fixtures page alongside model probabilities.
- **Referee assignment display** — Show today's referee with their home/away win% and cards-per-game stats.
- **Weather for outdoor stadiums** — Allianz Arena and BayArena have a roof; flag open-air stadiums with forecast data.

## Month 3: Analytics Pages

- **Season standings tracker** — Live table from football-data.org alongside model-implied finishing positions.
- **Form guide page** — Last 5 results for every Bundesliga team with xG totals.
- **Relegation battle spotlight** — Dedicated section when positions 16–18 are within 4 points of each other.

## Month 4: Betting Tools

- **Value bet finder** — Filter by minimum edge; sortable by confidence/edge/odds.
- **Parlay builder** — Select multiple fixtures → compute combined implied probability.
- **Bankroll tracker** — Enter starting bankroll; show simulated unit growth chart.

## Month 5: Historical Analysis

- **Season comparison** — Compare team stats across multiple Bundesliga seasons side by side.
- **Head-to-head history** — Last 10 H2H results for any two teams with goal and xG totals.
- **Model accuracy by matchday** — Chart accuracy trends through the season (early season vs. late season).

## Month 6: Automation

- **Weekly email digest** — Saturday morning email with the weekend's value bets.
- **Nightly data refresh** — GitHub Action runs all fetch scripts after midnight on matchdays.
- **Discord/Slack alert** — Post top pick for each matchday to a webhook channel.
