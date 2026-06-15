# Small-scale runner for testing pipeline on subset of data
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd

from src.preprocessing import preprocess_data
from src.train_rf import train_random_forest
from src.evaluate import evaluate_sklearn_model

NROWS = 10000
DATA_PATH = "data/raw/creditcard.csv"

print("Small-scale pipeline: loading first", NROWS, "rows")
df = pd.read_csv(DATA_PATH, nrows=NROWS)

(
    X_train,
    X_test,
    y_train,
    y_test,
    amount_scaler,
    feature_scaler
) = preprocess_data(df)

# Don't run SMOTE on this quick test; use class weights in RF
rf_model = train_random_forest(X_train, y_train)

evaluate_sklearn_model(rf_model, X_test, y_test)

print("Small-scale run completed")
