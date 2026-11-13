
from pathlib import Path

import joblib
import pandas as pd

from fastapi import FastAPI
from pydantic import BaseModel, Field

# Chemins
ROOT = Path(__file__).resolve().parents[1]

MODEL_DIR = ROOT / "models" / "synthetic"

# Charger le challenger
model = joblib.load(
    MODEL_DIR / "demo_model.joblib"
)

config = joblib.load if False else None

import json

with open(
    MODEL_DIR / "demo_metrics.json",
    encoding="utf-8"
) as f:
    report = json.load(f)

threshold = report["threshold"]

app = FastAPI(
    title="Fraud Detection Synthetic Demo",
    version="1.0.0"
)

class Transaction(BaseModel):
    features: dict[str, float] = Field(
        description="Variables de la transaction"
    )

@app.get("/health")
def health():
    return {
        "status": "ok",
        "model": "synthetic_challenger"
    }

@app.post("/predict")
def predict(transaction: Transaction):

    expected_features = list(
        model.feature_names_in_
    )

    missing = [
        name for name in expected_features
        if name not in transaction.features
    ]

    extra = [
        name for name in transaction.features
        if name not in expected_features
    ]

    if missing or extra:
        return {
            "error": "Invalid features",
            "missing": missing,
            "extra": extra
        }

    X = pd.DataFrame(
        [transaction.features],
        columns=expected_features
    )

    probability = float(
        model.predict_proba(X)[0, 1]
    )

    prediction = int(
        probability >= threshold
    )

    return {
        "fraud_probability": probability,
        "prediction": prediction,
        "threshold": threshold
    }
