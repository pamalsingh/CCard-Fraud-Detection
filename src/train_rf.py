# src/train_rf.py
"""Train a RandomForestClassifier with optional tuned parameters.

This module exposes `train_random_forest` which accepts training data
and an optional params dict (from a tuner). If params is None, sensible
defaults are used. Tuned hyperparameters override defaults.
"""

from typing import Optional, Dict, Any
from sklearn.ensemble import RandomForestClassifier


def train_random_forest(
    X_train,
    y_train,
    params: Optional[Dict[str, Any]] = None
):

    print("=" * 50)
    print("Training Random Forest")
    print("=" * 50)

    if params is None:
        params = {}

    model = RandomForestClassifier(
        n_estimators=params.get("n_estimators", 300),
        max_depth=params.get("max_depth", 25),
        class_weight=params.get("class_weight", "balanced"),
        random_state=42,
        n_jobs=-1
    )

    model.fit(X_train, y_train)

    print("Random Forest Training Completed")

    return model