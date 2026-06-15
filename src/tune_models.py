"""
Unified hyperparameter tuning helpers for RandomForest, XGBoost and DNN.

Usage (examples):
  python -m src.tune_models --model rf
  python -m src.tune_models --model xgb
  python -m src.tune_models --model dnn

Functions return (best_estimator, best_params, best_score)
"""

from typing import Optional, Dict, Any, Tuple
import itertools
import json
import os

import numpy as np
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score

try:
    from xgboost import XGBClassifier
    _HAS_XGB = True
except Exception:
    _HAS_XGB = False

from src.train_dnn import build_dnn


def _subsample(X, y, n_samples: Optional[int]):
    if n_samples is None:
        return X, y

    if n_samples < 1:
        raise ValueError("n_samples must be >= 1")

    if len(y) > n_samples:
        X_sample, _, y_sample, _ = train_test_split(
            X, y, train_size=n_samples, random_state=42, stratify=y
        )
        return X_sample, y_sample
    return X, y


def tune_random_forest(X, y, n_samples: Optional[int] = None, param_grid: Optional[Dict[str, Any]] = None, cv: int = 5, scoring: str = "f1") -> Tuple[Any, Dict[str, Any], float]:
    if param_grid is None:
        param_grid = {
            "n_estimators": [100,200, 300, 400],
            "max_depth": [15, 20, 25],
            "class_weight": ["balanced"],
        }

    X_s, y_s = _subsample(X, y, n_samples)

    base_rf = RandomForestClassifier(random_state=42)

    grid = GridSearchCV(
        estimator=base_rf,
        param_grid=param_grid,
        cv=cv,
        n_jobs=-1,
        verbose=2,
        scoring=scoring,
    )

    print(f"Tuning RandomForest on {len(y_s)} samples...")
    grid.fit(X_s, y_s)

    return grid.best_estimator_, grid.best_params_, grid.best_score_


def tune_xgboost(X, y, n_samples: Optional[int] = None, param_grid: Optional[Dict[str, Any]] = None, cv: int = 5, scoring: str = "f1") -> Tuple[Any, Dict[str, Any], float]:
    if not _HAS_XGB:
        raise RuntimeError("xgboost is not available in the environment")

    if param_grid is None:
        param_grid = {
            "n_estimators": [200, 300, 400, 500],
            "max_depth": [6, 7, 8],
            "learning_rate": [0.1, 0.3]
        }

    X_s, y_s = _subsample(X, y, n_samples)

    base = XGBClassifier(use_label_encoder=False, eval_metric="logloss", verbosity=0, n_jobs=-1)

    grid = GridSearchCV(
        estimator=base,
        param_grid=param_grid,
        cv=cv,
        n_jobs=-1,
        verbose=2,
        scoring=scoring,
    )

    print(f"Tuning XGBoost on {len(y_s)} samples...")
    grid.fit(X_s, y_s)

    return grid.best_estimator_, grid.best_params_, grid.best_score_


def tune_dnn(X, y, n_samples: Optional[int] = None, param_grid: Optional[Dict[str, list]] = None, epochs: int = 30, verbose: int = 1) -> Tuple[Any, Dict[str, Any], float]:
    """
    Simple manual grid search for DNN hyperparameters. Does not rely on scikeras.
    Returns best Keras model (trained on full training set with best params), best_params, best_score (val f1).
    """
    if param_grid is None:
        param_grid = {
            "learning_rate": [0.001, 0.01],
            "dropout_rate": [0.3, 0.5],
            "batch_size": [128, 256],
            "epochs": [epochs]
        }

    X_s, y_s = _subsample(X, y, n_samples)

    # split into train/val for selecting hyperparameters
    X_tr, X_val, y_tr, y_val = train_test_split(X_s, y_s, test_size=0.2, random_state=42, stratify=y_s)

    best_score = -1.0
    best_params = None
    best_model = None

    keys = list(param_grid.keys())
    for vals in itertools.product(*[param_grid[k] for k in keys]):
        params = dict(zip(keys, vals))

        print(f"Training DNN with params: {params}")
        model = build_dnn(input_dim=X_tr.shape[1], dropout_rate=params.get("dropout_rate", 0.3), learning_rate=params.get("learning_rate", 0.001))

        hist = model.fit(X_tr, y_tr, epochs=int(params.get("epochs", epochs)), batch_size=int(params.get("batch_size", 256)), validation_data=(X_val, y_val), verbose=verbose)

        # evaluate on validation set using threshold 0.5
        preds = model.predict(X_val, batch_size=int(params.get("batch_size", 256)))
        preds_bin = (preds.ravel() >= 0.5).astype(int)
        score = f1_score(y_val, preds_bin)
        print(f"Validation f1: {score:.4f}")

        if score > best_score:
            best_score = score
            best_params = params
            best_model = model

    # retrain best model on full X_s if desired (here we return model trained on train portion)
    return best_model, best_params, best_score


def _save_results(out_path: str, model_name: str, best_params: Dict[str, Any], best_score: float):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    payload = {"model": model_name, "best_params": best_params, "best_score": float(best_score)}
    with open(out_path, "w") as fh:
        json.dump(payload, fh, indent=2)


if __name__ == "__main__":
    import argparse
    from src.data_loader import load_data
    from src.preprocessing import preprocess_data

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["rf", "xgb", "dnn", "all"], default="all")
    parser.add_argument("--data", default="data/raw/creditcard.csv")
    parser.add_argument("--n-samples", type=int, default=None, help="Optional subsample size to speed up tuning")
    parser.add_argument("--out", default="reports/tuning_results.json")
    args = parser.parse_args()

    df = load_data(args.data)
    X_train, X_test, y_train, y_test, *_ = preprocess_data(df)

    results = {}
    if args.model in ("rf", "all"):
        best_est, best_params, best_score = tune_random_forest(X_train, y_train, n_samples=args.n_samples)
        results["rf"] = {"best_params": best_params, "best_score": best_score}

    if args.model in ("xgb", "all"):
        if not _HAS_XGB:
            print("Skipping XGBoost tuning: xgboost not installed")
        else:
            best_est, best_params, best_score = tune_xgboost(X_train, y_train, n_samples=args.n_samples)
            results["xgb"] = {"best_params": best_params, "best_score": best_score}

    if args.model in ("dnn", "all"):
        best_model, best_params, best_score = tune_dnn(X_train, y_train, n_samples=args.n_samples)
        results["dnn"] = {"best_params": best_params, "best_score": best_score}

    _save_results(args.out, args.model, results, 0.0)
    print(f"Tuning complete. Results written to {args.out}")
