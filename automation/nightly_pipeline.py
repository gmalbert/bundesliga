"""Nightly pipeline orchestrator for La Liga Linea.

Runs all fetch, feature engineering, model training, and prediction
pre-generation steps in order. Can be run locally or invoked from
GitHub Actions.

Usage:
    python automation/nightly_pipeline.py
    python automation/nightly_pipeline.py --skip-odds   # spare API quota
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from datetime import datetime
from pathlib import Path

# Run from repo root regardless of where the script is called from
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))


STEPS: list[tuple[str, list[str], bool]] = [
    # (label, command, requires_odds_api)
    ("Fetch historical CSVs",       ["python", "fetch_historical_csvs.py"],   False),
    ("Fetch upcoming fixtures",     ["python", "fetch_upcoming_fixtures.py"], False),
    ("Fetch xG proxy",              ["python", "fetch_fbref_xg.py"],          False),
    ("Fetch Copa fixtures",         ["python", "fetch_copa_fixtures.py"],     False),
    ("Fetch bookmaker odds",        ["python", "fetch_odds.py"],              True),
    ("Fetch weather forecasts",     ["python", "fetch_weather_data.py"],      False),
    ("Prepare model features",      ["python", "prepare_model_data.py"],      False),
    ("Train models",                ["python", "train_models.py"],            False),
    ("Run historical backtest",     ["python", "backtest.py"],                False),
    ("Pre-generate predictions",    ["python", "automation/generate_predictions.py"], False),
    ("Validate prediction log",     ["python", "track_predictions.py", "--validate"], False),
]


def run_pipeline(skip_odds: bool = False) -> None:
    print(f"\n{'='*60}")
    print("La Liga Linea — Nightly Pipeline")
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    if skip_odds:
        print("⚠ Odds fetch skipped (--skip-odds)")
    print(f"{'='*60}\n")

    failed: list[str] = []
    for label, cmd, needs_odds in STEPS:
        if skip_odds and needs_odds:
            print(f"⏭  {label} (skipped)\n")
            continue

        print(f"▶  {label}...")
        result = subprocess.run(cmd, capture_output=False, text=True, cwd=ROOT)
        if result.returncode == 0:
            print(f"   ✓ Done\n")
        else:
            print(f"   ✗ FAILED (exit {result.returncode})\n")
            failed.append(label)

    print(f"{'='*60}")
    print(f"Finished: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    if failed:
        print(f"Failed steps: {', '.join(failed)}")
        sys.exit(1)
    else:
        print("All steps completed successfully.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="La Liga nightly data pipeline")
    parser.add_argument(
        "--skip-odds",
        action="store_true",
        help="Skip fetch_odds.py to preserve Odds API monthly quota",
    )
    args = parser.parse_args()
    run_pipeline(skip_odds=args.skip_odds)
