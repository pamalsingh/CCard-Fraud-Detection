"""XGBoost training helper.

This module exposes `train_xgboost` which trains an XGBoost classifier
on the provided training data.
"""

from typing import Optional, Dict, Any
from xgboost import XGBClassifier
import numpy as np


def train_xgboost(X_train, y_train, params: Optional[Dict[str, Any]] = None):
    # sensible defaults
    default_params = dict(
        n_estimators=500,
        max_depth=8,
        learning_rate=0.1,
        use_label_encoder=False,
        eval_metric="logloss",
        verbosity=0,
        n_jobs=-1,
    )

    if params:
        default_params.update(params)

    # handle class imbalance with scale_pos_weight if not provided
    if "scale_pos_weight" not in default_params:
        vals = np.bincount(y_train.astype(int))
        if len(vals) == 1:
            neg, pos = vals[0], 1
        else:
            neg, pos = vals[0], vals[1]
        default_params["scale_pos_weight"] = neg / max(pos, 1)

    model = XGBClassifier(**default_params)
    model.fit(X_train, y_train)
    return model
