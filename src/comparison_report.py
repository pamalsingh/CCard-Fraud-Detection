import pandas as pd
from pandas.errors import ParserError
import os


def generate_comparison_report():
    csv_path = "reports/experiments_summary.csv"

    if not os.path.exists(csv_path):
        print(f"No experiments summary found at {csv_path}")
        return

    try:
        df = pd.read_csv(csv_path)
    except ParserError:
        # fallback: be permissive and skip malformed lines
        df = pd.read_csv(csv_path, engine="python", on_bad_lines="skip")

    if df.empty:
        print(f"No valid rows found in {csv_path}")
        return

    # rewrite the experiments_summary.csv with a normalized, cleaned version
    try:
        df.to_csv(csv_path, index=False)
    except Exception:
        pass

    # prefer sorting by f1 if available, otherwise pick a sensible numeric column
    sort_col = "f1" if "f1" in df.columns else None
    if sort_col is None:
        numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
        sort_col = numeric_cols[0] if numeric_cols else df.columns[0]

    df = df.sort_values(by=sort_col, ascending=False)

    os.makedirs("reports", exist_ok=True)
    df.to_csv("reports/model_comparison.csv", index=False)

    print(df)