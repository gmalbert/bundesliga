"""Offline model training script for La Liga Linea.

Trains the ensemble classifier and computes Poisson team strengths,
then saves artefacts to models/.

Outputs:
    models/ensemble_model.pkl   — trained VotingClassifier
    models/metrics.json         — accuracy, F1, log-loss, set sizes
    models/poisson_strengths.csv — Poisson attack/defense multipliers

Usage:
    python train_models.py [--csv path/to/combined_historical_data.csv]

Called nightly by .github/workflows/nightly.yml after prepare_model_data.py.
"""

from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, log_loss
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from models.ensemble_predictor import create_ensemble_model, save_model
from models.poisson_predictor import compute_team_strengths
from prepare_model_data import FEATURE_COLS, load_and_engineer_features

Path("models").mkdir(exist_ok=True)

# Alphabetical LabelEncoder order: A=0, D=1, H=2
RESULT_MAP = {"A": 0, "D": 1, "H": 2}


def _load_training_arrays(
    csv_path: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load CSV, engineer features, return train/test splits."""
    df = pd.read_csv(csv_path, low_memory=False)
    df = load_and_engineer_features(df)

    df = df[df["FullTimeResult"].isin(RESULT_MAP)].copy()
    df["_target"] = df["FullTimeResult"].map(RESULT_MAP)

    available = [c for c in FEATURE_COLS if c in df.columns]
    missing   = [c for c in FEATURE_COLS if c not in df.columns]
    if missing:
        print(f"  ⚠ Feature columns missing from data (will be skipped): {missing}")

    X = df[available].fillna(0).values
    y = df["_target"].values

    return train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)


def train_ensemble(csv_path: str = "data_files/combined_historical_data.csv") -> dict:
    """Train and save the VotingClassifier ensemble."""
    print("Training ensemble model…")
    X_train, X_test, y_train, y_test = _load_training_arrays(csv_path)

    model = create_ensemble_model()
    model.fit(X_train, y_train)

    y_pred  = model.predict(X_test)
    y_proba = model.predict_proba(X_test)
    # Normalise rows to sum to 1.0 (VotingClassifier can drift slightly due to float ops)
    y_proba = y_proba / y_proba.sum(axis=1, keepdims=True)

    metrics = {
        "accuracy":    round(accuracy_score(y_test, y_pred), 4),
        "f1_macro":    round(f1_score(y_test, y_pred, average="macro"), 4),
        "log_loss":    round(log_loss(y_test, y_proba), 4),
        "n_train":     int(len(X_train)),
        "n_test":      int(len(X_test)),
        "feature_cols": [c for c in FEATURE_COLS if True],  # log which were used
    }

    save_model(model, "models/ensemble_model.pkl")
    with open("models/metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    print(
        f"  Accuracy: {metrics['accuracy']:.1%}  |"
        f"  F1: {metrics['f1_macro']:.3f}  |"
        f"  Log Loss: {metrics['log_loss']:.3f}"
    )
    print(f"  Train: {metrics['n_train']}  |  Test: {metrics['n_test']}")
    print("  Saved: models/ensemble_model.pkl + models/metrics.json")
    return metrics


def train_poisson(csv_path: str = "data_files/combined_historical_data.csv") -> None:
    """Compute and save Poisson team strengths."""
    print("Computing Poisson team strengths…")
    df = pd.read_csv(csv_path, low_memory=False)
    strengths = compute_team_strengths(df)
    out = "models/poisson_strengths.csv"
    strengths.to_csv(out, index=False)
    print(f"  Saved: {out}  ({len(strengths)} teams)")


def main(csv_path: str = "data_files/combined_historical_data.csv") -> None:
    if not Path(csv_path).exists():
        print(f"✗ {csv_path} not found. Run fetch_historical_csvs.py first.")
        raise SystemExit(1)

    train_ensemble(csv_path)
    train_poisson(csv_path)
    print("\nAll models trained successfully.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train La Liga Linea models")
    parser.add_argument(
        "--csv",
        default="data_files/combined_historical_data.csv",
        help="Path to combined historical data CSV",
    )
    args = parser.parse_args()
    main(args.csv)
