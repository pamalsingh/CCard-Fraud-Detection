# src/save_artifacts.py

import joblib
import os


class Preprocessor:
    """Serializable preprocessor wrapper that applies amount and feature scalers

    This simple wrapper keeps references to fitted scaler objects and a list of
    feature columns. It exposes `transform(X)` that accepts a pandas DataFrame
    with the expected feature columns and returns a transformed numpy array.
    """
    def __init__(self, amount_scaler=None, feature_scaler=None, feature_columns=None):
        self.amount_scaler = amount_scaler
        self.feature_scaler = feature_scaler
        self.feature_columns = list(feature_columns) if feature_columns is not None else None

    def transform(self, X):
        # X is expected to be a pandas DataFrame
        import pandas as _pd
        import numpy as _np

        if self.feature_columns is not None:
            Xf = X.loc[:, self.feature_columns].copy()
        else:
            Xf = X.copy()

        # Apply amount scaler if present
        if self.amount_scaler is not None and 'Amount' in Xf.columns:
            Xf['Amount'] = self.amount_scaler.transform(Xf[['Amount']])

        # Apply feature scaler to remaining numeric columns
        if self.feature_scaler is not None:
            cols = [c for c in Xf.columns if c != 'Amount']
            if cols:
                Xf[cols] = self.feature_scaler.transform(Xf[cols])

        # Return numpy array for sklearn compatibility
        return Xf.values if hasattr(Xf, 'values') else _np.asarray(Xf)


def save_preprocessor(amount_scaler, feature_scaler, feature_columns, file_path):
    """Create and save a Preprocessor wrapper to `file_path`."""
    pre = Preprocessor(amount_scaler=amount_scaler, feature_scaler=feature_scaler, feature_columns=feature_columns)
    os.makedirs(os.path.dirname(file_path) or '.', exist_ok=True)
    joblib.dump(pre, file_path)
    print(f"Preprocessor saved: {file_path}")


def save_random_forest(
    model,
    file_path
):

    joblib.dump(
        model,
        file_path
    )

    print(
        f"Model Saved : {file_path}"
    )


def save_scaler(
    scaler,
    file_path
):

    joblib.dump(
        scaler,
        file_path
    )

    print(
        f"Scaler Saved : {file_path}"
    )


def save_model(
    model,
    file_path
):
    """Save a generic sklearn-compatible model with joblib."""
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    joblib.dump(model, file_path)
    print(f"Model Saved : {file_path}")


def save_keras_model(
    model,
    dir_path
):
    """Save a Keras model to a directory or HDF5 file."""
    # If a file extension was provided, save to that path. Otherwise save to a single
    # HDF5 file using the given base path (e.g. 'models/dnn_pre_smote' -> 'models/dnn_pre_smote.h5')
    base, ext = os.path.splitext(dir_path)
    if ext.lower() in (".h5", ".keras", ".hdf5"):
        target = dir_path
    else:
        target = f"{dir_path}.h5"

    parent = os.path.dirname(target) or "."
    os.makedirs(parent, exist_ok=True)

    # Delegate to Keras model.save which supports both HDF5 files and SavedModel dirs
    model.save(target)
    print(f"Keras model saved: {target}")