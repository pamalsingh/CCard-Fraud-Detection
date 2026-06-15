# src/tune_rf.py

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, train_test_split
import numpy as np


def tune_random_forest(X, y, n_samples=None, param_grid=None, cv=5):
    """
    Run GridSearchCV for RandomForest on (optionally subsampled) data.

    Parameters
    ----------
    X : DataFrame or array-like
    y : Series or array-like
    n_samples : int or None
        If provided, randomly subsample the training data (stratified) to this
        many rows before running grid search. Useful to keep GridSearch small.
    param_grid : dict or None
        Parameter grid for GridSearch. Defaults to a small grid.
    cv : int
        Number of CV folds for GridSearch.

    Returns
    -------
    best_estimator, best_params, best_score
    """

    if param_grid is None:
        param_grid = {
            "n_estimators": [100, 200, 300],
            "max_depth": [10, 15, 20]
        }

    # Optionally subsample while preserving class proportions
    if n_samples is not None:
        if n_samples < 1:
            raise ValueError("n_samples must be >= 1")
        # If X is a pandas DataFrame/Series, convert to numpy for sklearn split
        try:
            import pandas as pd
            is_pd = isinstance(X, (pd.DataFrame, pd.Series))
        except Exception:
            is_pd = False

        if is_pd:
            X_vals = X
            y_vals = y
        else:
            X_vals = X
            y_vals = y

        # If requested samples >= available, skip subsampling
        if len(y_vals) > n_samples:
            stratify = y_vals
            X_sample, _, y_sample, _ = train_test_split(
                X_vals,
                y_vals,
                train_size=n_samples,
                random_state=42,
                stratify=stratify
            )
        else:
            X_sample, y_sample = X_vals, y_vals
    else:
        X_sample, y_sample = X, y

    base_rf = RandomForestClassifier(random_state=42, class_weight='balanced')

    grid_search = GridSearchCV(
        estimator=base_rf,
        param_grid=param_grid,
        cv=cv,
        n_jobs=-1,
        verbose=2,
        scoring='f1'
    )

    print("Starting GridSearchCV on {} samples...".format(len(y_sample)))
    grid_search.fit(X_sample, y_sample)
    print("GridSearchCV complete.")

    best_estimator = grid_search.best_estimator_
    best_params = grid_search.best_params_
    best_score = grid_search.best_score_

    return best_estimator, best_params, best_score
