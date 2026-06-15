# src/imbalance_handler.py

import pandas as pd
import numpy as np
from imblearn.over_sampling import SMOTE


def apply_smote(X_train, y_train):
    """Apply SMOTE and return resampled X (DataFrame) and y (Series).

    This converts inputs to numpy arrays for resampling then reconstructs
    pandas objects so downstream code keeps using DataFrame/Series types.
    """

    print("=" * 50)
    print("Applying SMOTE")
    print("=" * 50)

    print("Before SMOTE")
    print(y_train.value_counts())

    smote = SMOTE(random_state=42)

    # use numpy arrays for fit_resample to avoid ambiguous typing
    X_res, y_res = smote.fit_resample(X_train.values, y_train.values)

    X_train_resampled = pd.DataFrame(X_res, columns=X_train.columns)
    y_train_resampled = pd.Series(y_res, name=y_train.name)

    print("After SMOTE")
    print(y_train_resampled.value_counts())

    return X_train_resampled, y_train_resampled