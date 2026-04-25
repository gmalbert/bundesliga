"""Pre-generate and log predictions for all upcoming fixtures.

Called nightly by GitHub Actions and automation/nightly_pipeline.py
so the Streamlit app never has to run the model at request time.
"""

from __future__ import annotations

import pickle
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from utils import predict_for_upcoming, FEATURE_COLS  # noqa: E402
from track_predictions import log_predictions         # noqa: E402

HIST_PATH     = ROOT / "data_files" / "combined_historical_data.csv"
FIXTURES_PATH = ROOT / "data_files" / "upcoming_fixtures.csv"
MODEL_PATH    = ROOT / "models" / "ensemble_model.pkl"


def main() -> None:
    if not HIST_PATH.exists():
        print(f"✗ Missing {HIST_PATH} — run fetch_historical_csvs.py first")
        sys.exit(1)
    if not FIXTURES_PATH.exists():
        print(f"✗ Missing {FIXTURES_PATH} — run fetch_upcoming_fixtures.py first")
        sys.exit(1)
    if not MODEL_PATH.exists():
        print(f"✗ Missing {MODEL_PATH} — run train_models.py first")
        sys.exit(1)

    hist = pd.read_csv(HIST_PATH, low_memory=False)
    hist["MatchDate"] = pd.to_datetime(hist["MatchDate"], errors="coerce")

    fix = pd.read_csv(FIXTURES_PATH)

    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)

    preds = predict_for_upcoming(fix, hist, model, FEATURE_COLS)

    if preds.empty:
        print("No upcoming fixtures to predict.")
        return

    log_df = preds.rename(columns={
        "Date":       "MatchDate",
        "Home Win %": "PredHomeWin",
        "Draw %":     "PredDraw",
        "Away Win %": "PredAwayWin",
    })[["MatchDate", "HomeTeam", "AwayTeam", "PredHomeWin", "PredDraw", "PredAwayWin"]].copy()

    def pred_result(row: pd.Series) -> str:
        m = max(row["PredHomeWin"], row["PredDraw"], row["PredAwayWin"])
        if m == row["PredHomeWin"]:
            return "H"
        if m == row["PredDraw"]:
            return "D"
        return "A"

    log_df["PredictedResult"] = log_df.apply(pred_result, axis=1)

    log_predictions(log_df, model_version="ensemble_v1")
    print(f"✓ Pre-generated {len(log_df)} predictions → data_files/predictions_log.csv")


if __name__ == "__main__":
    main()
