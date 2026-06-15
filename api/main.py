from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
import pandas as pd
import io
import os
import joblib
import pickle

app = FastAPI(title="Credit Card Fraud Detection API")

MODELS_DIR = os.path.join(os.path.dirname(__file__), '..', 'models')


def load_pickle(path: str):
    with open(path, 'rb') as f:
        return pickle.load(f)


def load_model(path: str):
    try:
        return joblib.load(path)
    except Exception:
        return load_pickle(path)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict_csv")
async def predict_csv(file: UploadFile = File(...), model_file: str = None):
    content = await file.read()
    try:
        if file.filename.lower().endswith('.csv'):
            df = pd.read_csv(io.BytesIO(content))
        else:
            df = pd.read_excel(io.BytesIO(content))
    except Exception as e:
        return JSONResponse(status_code=400, content={"error": str(e)})

    if not model_file:
        return JSONResponse(status_code=400, content={"error": "model_file query parameter required"})

    model_path = os.path.join(os.getcwd(), 'models', model_file)
    if not os.path.exists(model_path):
        return JSONResponse(status_code=404, content={"error": "Model not found"})

    model = load_model(model_path)
    X = df.select_dtypes(include=["number"])  # naive
    try:
        proba = model.predict_proba(X)
        if proba.shape[1] == 2:
            fraud_proba = proba[:, 1].tolist()
        else:
            fraud_proba = proba.max(axis=1).tolist()
        preds = (pd.Series(fraud_proba) >= 0.5).astype(int).tolist()
    except Exception:
        preds = model.predict(X).tolist()
        fraud_proba = [1.0 if p == 1 else 0.0 for p in preds]

    return {"predictions": preds, "probabilities": fraud_proba}
