# src/model_loader.py

import joblib


def load_model(
    model_path
):

    return joblib.load(
        model_path
    )


def load_scaler(
    scaler_path
):

    return joblib.load(
        scaler_path
    )