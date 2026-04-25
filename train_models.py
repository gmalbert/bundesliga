"""Offline model training script for La Liga Linea.

Trains the ensemble classifier and computes Poisson team strengths,
then saves artefacts to models/.

Outputs:
    models/ensemble_model.pkl    — trained VotingClassifier
    models/metrics.json          — accuracy, F1, log-loss, set sizes
    models/poisson_strengths.csv — Poisson attack/defense multipliers
    models/best_hyperparams.json — best XGBoost params (only with --optimize)
    models/nn_model.pt           — trained LaLigaNet weights (PyTorch)
    models/nn_scaler.pkl         — StandardScaler for NN features

Usage:
    python train_models.py [--csv path] [--optimize] [--no-nn]

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
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn.preprocessing import LabelEncoder

from models.ensemble_predictor import create_ensemble_model, save_model
from models.poisson_predictor import compute_team_strengths
from prepare_model_data import FEATURE_COLS, load_and_engineer_features

Path("models").mkdir(exist_ok=True)

# Alphabetical LabelEncoder order: A=0, D=1, H=2
RESULT_MAP = {"A": 0, "D": 1, "H": 2}

# XGBoost hyperparameter search space
XGB_PARAM_DIST = {
    "n_estimators":      [100, 200, 300, 400],
    "max_depth":         [3, 4, 5, 6, 7],
    "learning_rate":     [0.01, 0.05, 0.1, 0.2],
    "subsample":         [0.6, 0.7, 0.8, 0.9, 1.0],
    "colsample_bytree":  [0.6, 0.7, 0.8, 0.9, 1.0],
    "min_child_weight":  [1, 2, 3, 5],
    "gamma":             [0, 0.1, 0.2, 0.5],
}


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


def optimize_xgboost(
    X_train: np.ndarray,
    y_train: np.ndarray,
    n_iter: int = 15,
    cv: int = 3,
    output_path: str = "models/best_hyperparams.json",
) -> dict:
    """Run RandomizedSearchCV over XGBoost and save best params.

    Returns the best parameter dict.
    """
    from xgboost import XGBClassifier

    print(f"  Optimizing XGBoost hyperparameters ({n_iter} iterations, {cv}-fold CV)…")
    xgb = XGBClassifier(
        objective="multi:softprob",
        num_class=3,
        eval_metric="mlogloss",
        use_label_encoder=False,
        random_state=42,
        n_jobs=-1,
    )
    search = RandomizedSearchCV(
        xgb,
        param_distributions=XGB_PARAM_DIST,
        n_iter=n_iter,
        cv=cv,
        scoring="accuracy",
        n_jobs=-1,
        random_state=42,
        verbose=0,
    )
    search.fit(X_train, y_train)
    best = search.best_params_
    best["cv_accuracy"] = round(search.best_score_, 4)

    with open(output_path, "w") as f:
        json.dump(best, f, indent=2)
    print(f"  Best CV accuracy: {search.best_score_:.1%}  |  params saved → {output_path}")
    return best


def train_ensemble(
    csv_path: str = "data_files/combined_historical_data.csv",
    optimize: bool = False,
) -> dict:
    """Train and save the VotingClassifier ensemble."""
    print("Training ensemble model…")
    X_train, X_test, y_train, y_test = _load_training_arrays(csv_path)

    # Optionally run hyperparameter search first
    xgb_params: dict = {}
    hyperparams_path = Path("models/best_hyperparams.json")
    if optimize:
        xgb_params = optimize_xgboost(X_train, y_train)
    elif hyperparams_path.exists():
        with open(hyperparams_path) as f:
            stored = json.load(f)
        # Strip metadata keys that aren't XGBoost params
        xgb_params = {k: v for k, v in stored.items() if k != "cv_accuracy"}
        print(f"  Loaded best hyperparams from {hyperparams_path}")

    model = create_ensemble_model(xgb_params=xgb_params if xgb_params else None)
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


def train_neural_network(
    csv_path: str = "data_files/combined_historical_data.csv",
) -> dict:
    """Train LaLigaNet and save weights."""
    from models.nn_predictor import TORCH_AVAILABLE, train_nn

    if not TORCH_AVAILABLE:
        print("  ⚠ PyTorch not installed — skipping neural network.")
        return {}

    print("Training neural network (LaLigaNet)…")
    X_train, X_test, y_train, y_test = _load_training_arrays(csv_path)
    return train_nn(X_train, y_train, X_test, y_test)


def train_poisson(csv_path: str = "data_files/combined_historical_data.csv") -> None:
    """Compute and save Poisson team strengths."""
    print("Computing Poisson team strengths…")
    df = pd.read_csv(csv_path, low_memory=False)
    strengths = compute_team_strengths(df)
    out = "models/poisson_strengths.csv"
    strengths.to_csv(out, index=False)
    print(f"  Saved: {out}  ({len(strengths)} teams)")


def main(
    csv_path: str = "data_files/combined_historical_data.csv",
    optimize: bool = False,
    train_nn: bool = True,
) -> None:
    if not Path(csv_path).exists():
        print(f"✗ {csv_path} not found. Run fetch_historical_csvs.py first.")
        raise SystemExit(1)

    train_ensemble(csv_path, optimize=optimize)
    train_poisson(csv_path)
    if train_nn:
        train_neural_network(csv_path)
    print("\nAll models trained successfully.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train La Liga Linea models")
    parser.add_argument(
        "--csv",
        default="data_files/combined_historical_data.csv",
        help="Path to combined historical data CSV",
    )
    parser.add_argument(
        "--optimize",
        action="store_true",
        default=False,
        help="Run RandomizedSearchCV to optimise XGBoost hyperparameters (slow)",
    )
    parser.add_argument(
        "--no-nn",
        dest="no_nn",
        action="store_true",
        default=False,
        help="Skip neural network training",
    )
    args = parser.parse_args()
    main(args.csv, optimize=args.optimize, train_nn=not args.no_nn)
