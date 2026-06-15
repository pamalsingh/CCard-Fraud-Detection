# src/predict.py

import numpy as np


def predict_transaction(
    model,
    transaction
):

    probability = model.predict_proba(
        transaction
    )[0][1]

    prediction = int(
        probability >= 0.50
    )

    return {
        "fraud_probability": round(
            probability,
            4
        ),
        "prediction": prediction
    }