"""Ensemble model definition for La Liga Linea.

Provides create_ensemble_model() — a soft-voting VotingClassifier with
XGBoost, Random Forest, Gradient Boosting, and Logistic Regression.

Weights: XGB=2, RF=1.5, GB=1, LR=0.5 (higher weight → higher influence).
predict_proba column order (alphabetical LabelEncoder): [A=0, D=1, H=2]
"""

from __future__ import annotations

import pickle
from pathlib import Path

from sklearn.ensemble import (
    GradientBoostingClassifier,
    RandomForestClassifier,
    VotingClassifier,
)
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier


def create_ensemble_model() -> VotingClassifier:
    """Return an unfitted soft-voting ensemble."""
    xgb = XGBClassifier(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric="mlogloss",
        random_state=42,
        verbosity=0,
    )
    rf = RandomForestClassifier(
        n_estimators=200,
        max_depth=8,
        random_state=42,
        n_jobs=-1,
    )
    gb = GradientBoostingClassifier(
        n_estimators=150,
        max_depth=5,
        random_state=42,
    )
    lr = LogisticRegression(
        max_iter=1000,
        random_state=42,
    )

    return VotingClassifier(
        estimators=[("xgb", xgb), ("rf", rf), ("gb", gb), ("lr", lr)],
        voting="soft",
        weights=[2, 1.5, 1, 0.5],
    )


def save_model(model: VotingClassifier, path: str = "models/ensemble_model.pkl") -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(model, f)


def load_model(path: str = "models/ensemble_model.pkl") -> VotingClassifier:
    with open(path, "rb") as f:
        return pickle.load(f)
