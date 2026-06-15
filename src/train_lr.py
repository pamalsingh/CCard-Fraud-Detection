# src/train_lr.py
from sklearn.linear_model import LogisticRegression


def train_logistic_regression(X_train, y_train, params=None):
    """Train a Logistic Regression model with optional params dict."""
    if params is None:
        params = {}

    default = dict(solver=params.get("solver", "lbfgs"),
                   max_iter=params.get("max_iter", 1000),
                   class_weight=params.get("class_weight", None),
                   n_jobs=params.get("n_jobs", -1))

    model = LogisticRegression(**default)
    model.fit(X_train, y_train)
    return model
