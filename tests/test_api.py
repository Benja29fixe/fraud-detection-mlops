from pathlib import Path

import pandas as pd
import pytest

from fastapi.testclient import TestClient
from api.main import app


# ROOT = Path(__file__).resolve().parents[1]
# print(f"{ROOT=}")

@pytest.fixture(scope="module")
def client():
    # Déclenche le lifespan de FastAPI
    with TestClient(app) as test_client:
        yield test_client


def test_health(client):
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_predict(client):

    transaction = {
        "Time": 120.0,
        "Amount": 250.0,
        **{f"V{i}": 0.0 for i in range(1, 29)}
    }

    response = client.post(
        "/predict",
        json=transaction
    )

    assert response.status_code == 200

    result = response.json()

    assert 0 <= result["fraud_probability"] <= 1
    assert result["predicted_class"] in [0, 1]
    assert 0 <= result["threshold"] <= 1

def test_invalid_transaction(client):
    response = client.post(
        "/predict",
        json={"Amount": -100}
    )

    assert response.status_code == 422