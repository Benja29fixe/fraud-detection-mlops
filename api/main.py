#######################################
# Lancer le serveur :
# python -m uvicorn api.main:app --reload --host 127.0.0.1 --port 8000
#
# Puis tester avec une vrai transaction dans :
# test_api.py
#
#######################################

from pathlib import Path
from contextlib import asynccontextmanager
from typing import Any

import joblib
import numpy as np
import pandas as pd

from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, Field, ConfigDict


# --------------------------------------------------
# 1. Définir les chemins
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent

MODEL_PATH = PROJECT_ROOT / "models" / "fraud_model.joblib"
CONFIG_PATH = PROJECT_ROOT / "models" / "decision_config.joblib"

FEATURE_NAMES = (
    ["Time", "Amount"]
    + [f"V{i}" for i in range(1, 29)]
)


# --------------------------------------------------
# 2. Définir le schéma d'une transaction
# --------------------------------------------------

class Transaction(BaseModel):

    model_config = ConfigDict(extra="forbid")

    Time: float = Field(ge=0)
    Amount: float = Field(ge=0)

    V1: float
    V2: float
    V3: float
    V4: float
    V5: float
    V6: float
    V7: float
    V8: float
    V9: float
    V10: float
    V11: float
    V12: float
    V13: float
    V14: float
    V15: float
    V16: float
    V17: float
    V18: float
    V19: float
    V20: float
    V21: float
    V22: float
    V23: float
    V24: float
    V25: float
    V26: float
    V27: float
    V28: float


class PredictionResponse(BaseModel):
    fraud_probability: float
    predicted_class: int
    threshold: float


# --------------------------------------------------
# 3. Charger le modèle au démarrage
# --------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):

    model = joblib.load(MODEL_PATH)
    config = joblib.load(CONFIG_PATH)

    threshold = float(config["threshold"])

    if not 0 <= threshold <= 1:
        raise ValueError("Seuil invalide")

    app.state.model = model
    app.state.threshold = threshold

    yield


# --------------------------------------------------
# 4. Créer l'application
# --------------------------------------------------

app = FastAPI(
    title="Fraud Detection API",
    description="API MLOps de détection de fraude bancaire",
    version="1.0.0",
    lifespan=lifespan
)


# --------------------------------------------------
# 5. Endpoint de santé
# --------------------------------------------------

@app.get("/health")
def health():
    return {"status": "ok"}


# --------------------------------------------------
# 6. Endpoint de prédiction
# --------------------------------------------------

@app.post("/predict", response_model=PredictionResponse)
def predict(transaction: Transaction, request: Request):

    # Transformer la requête en dictionnaire
    transaction_dict = transaction.model_dump()

    # Construire un DataFrame avec l'ordre attendu
    df = pd.DataFrame(
        [transaction_dict],
        columns=FEATURE_NAMES
    )

    # Refuser les valeurs infinies ou non finies
    if not np.isfinite(df.to_numpy(dtype=float)).all():
        raise HTTPException(
            status_code=422,
            detail="Les variables doivent être finies"
        )

    # Récupérer le modèle chargé au démarrage
    model = request.app.state.model
    threshold = request.app.state.threshold

    # Calculer la probabilité de fraude
    probability = float(
        model.predict_proba(df)[0, 1]
    )

    # Appliquer le seuil sélectionné
    predicted_class = int(
        probability >= threshold
    )

    # Retourner la réponse
    return PredictionResponse(
        fraud_probability=probability,
        predicted_class=predicted_class,
        threshold=threshold
    )