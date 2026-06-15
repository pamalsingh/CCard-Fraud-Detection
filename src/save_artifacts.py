# src/save_artifacts.py

import joblib
import os


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
    os.makedirs(os.path.dirname(dir_path) if os.path.splitext(dir_path)[1] else dir_path, exist_ok=True)
    # If dir_path ends with .h5 or .keras, save as file; otherwise save as TF SavedModel directory
    if dir_path.endswith(".h5") or dir_path.endswith(".keras"):
        model.save(dir_path)
    else:
        model.save(dir_path)

    print(f"Keras model saved: {dir_path}")