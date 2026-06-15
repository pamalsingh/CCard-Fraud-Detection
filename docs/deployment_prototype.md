# Deployment Prototype

## Purpose
Provide a lightweight prototype demonstrating model inference and simple interaction for stakeholders.

## Options
- Streamlit app (`dashboard/app.py`) for an interactive demo.
- FastAPI service (`api/main.py`) exposing a `/predict` endpoint for programmatic inference.

## Requirements
- Include a small README section explaining how to run the prototype locally.

## Expected behavior
- Load saved model artifact(s) from `models/` and accept a single transaction (CSV row or JSON) to return a fraud probability and decision.
