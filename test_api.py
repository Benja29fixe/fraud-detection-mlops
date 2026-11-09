from pathlib import Path

import pandas as pd
import requests

ROOT = Path(__file__).resolve().parent

# Charger le dataset de test
test_df = pd.read_parquet(
    ROOT / "data" / "processed" / "test.parquet"
)

# Taille dataset de test
print(f"Taille test_df : {test_df.shape}")

# Sélectionner une transaction
transaction = test_df.drop(
    columns=["Class"]
).iloc[0].to_dict()

# Envoyer la requête à l'API
response = requests.post(
    "http://127.0.0.1:8000/predict",
    json=transaction,
    timeout=10
)

print("Code HTTP :", response.status_code)

response.raise_for_status()

print("Réponse de l'API :")
print(response.json())

print(
    "Classe réelle :",
    int(test_df.iloc[0]["Class"])
)

# Health
# ======
health_1= requests.get(
     "http://127.0.0.1:8000/health"
)

print(f"{type(health_1) = }")


# Afficher le code HTTP
print("Code HTTP :", health_1.status_code)

# Afficher la réponse JSON
print("Réponse :", health_1.json())
print("Réponse :", health_1.json()['status'])