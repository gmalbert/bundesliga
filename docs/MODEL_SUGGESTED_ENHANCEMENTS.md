# Bundesliga Predictor — Model Suggested Enhancements

## Priority 1: Calibration & Accuracy

### Probability Calibration
- Apply **Platt scaling** or **isotonic regression** to the VotingClassifier's `predict_proba` outputs.
- Run a calibration curve plot on held-out seasons to verify that 60% predictions win ~60% of the time.

### Third Model: Neural Network
- Add a simple 3-layer MLP (scikit-learn `MLPClassifier`) to the ensemble. European football models often benefit from non-linear feature interactions.
- Expected improvement: 0.5–1.5% accuracy gain from ensemble diversity.

### Relegation-Pressure Feature
- Teams in positions 16–18 (relegation zone) show markedly different motivation patterns, especially after matchday 25.
- Encode `games_from_relegation` (negative = in zone) as a numeric feature.

## Priority 2: Feature Engineering

### DFB-Pokal Congestion
- Binary flag: 1 if team played DFB-Pokal ≤ 4 days prior. Mirrors Copa del Rey logic from La Liga.

### Referee Effect
- Bundesliga referee stats are well-documented. Add `ref_home_advantage_rate` (home team wins % when this referee officiates).

### xG Integration
- FBref Bundesliga (competition ID 20) provides team xG and xGA. Replace goals-based rolling features with xG-based versions for better signal.

## Priority 3: Infrastructure

### Automatic Model Retraining Trigger
- Retrain automatically if rolling 10-match accuracy drops > 4% below season average.

### GitHub Actions Data Refresh
- Add a weekly workflow that runs `fetch_upcoming_fixtures.py` and `fetch_fbref_xg.py` to keep data current during the season.

### Prediction Log Cleanup
- Archive predictions more than 6 weeks old into a separate parquet to keep `predictions_log.csv` lean.
