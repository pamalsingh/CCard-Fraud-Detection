from pydantic import BaseModel
from typing import List, Dict, Any


class PredictionRequest(BaseModel):
    records: List[Dict[str, Any]]


class PredictionResponse(BaseModel):
    predictions: List[int]
    probabilities: List[float]
from pydantic import BaseModel


class Transaction(BaseModel):

    Time: float

    Amount: float

    V1: float
    V2: float
    ...
    V28: float