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