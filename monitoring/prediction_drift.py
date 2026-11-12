
from pathlib import Path

import joblib
import pandas as pd

from evidently import Report, Dataset, DataDefinition
from evidently.presets import DataDriftPreset


# 1. Chemins du projet
ROOT = Path(__file__).resolve().parents[1]

DATA_PATH = ROOT / "data" / "raw" / "creditcard.csv"
MODEL_PATH = ROOT / "models" / "fraud_model.joblib"

REPORT_DIR = ROOT / "monitoring" / "reports"
REPORT_DIR.mkdir(parents=True, exist_ok=True)


# 2. Charger les données
df = pd.read_csv(DATA_PATH, delimiter=";")

features = ["Time", "Amount"] + [
    f"V{i}" for i in range(1, 29)
]

X = df[features]


# 3. Créer deux ensembles initialement identiques
reference = X.sample(
    n=10000,
    random_state=42
).copy()

current = reference.copy()


# 4. Simuler un changement sur Amount
current["Amount"] += 1000


# 5. Charger le modèle entraîné
model = joblib.load(MODEL_PATH)


# 6. Prédire les probabilités de fraude
reference_scores = model.predict_proba(
    reference
)[:, 1]

current_scores = model.predict_proba(
    current
)[:, 1]


# 7. Construire les datasets de scores
reference_predictions = pd.DataFrame({
    "fraud_score": reference_scores
})

current_predictions = pd.DataFrame({
    "fraud_score": current_scores
})


# 8. Afficher quelques statistiques
print("Score moyen de référence :",
      reference_predictions["fraud_score"].mean())

print("Score moyen actuel :",
      current_predictions["fraud_score"].mean())


##############################

CONFIG_PATH = ROOT / "models" / "decision_config.joblib"

config = joblib.load(CONFIG_PATH)
threshold = float(config["threshold"])

reference_classes = (reference_scores >= threshold).astype(int)
current_classes = (current_scores >= threshold).astype(int)

print(
    "Taux de fraude prédit avant :",
    reference_classes.mean()
)

print(
    "Taux de fraude prédit après :",
    current_classes.mean()
)

##############################




# 9. Définir le schéma Evidently
schema = DataDefinition(
    numerical_columns=["fraud_score"]
)

reference_dataset = Dataset.from_pandas(
    reference_predictions,
    data_definition=schema
)

current_dataset = Dataset.from_pandas(
    current_predictions,
    data_definition=schema
)


# 10. Créer le rapport Prediction Drift
report = Report(
    metrics=[DataDriftPreset()]
)

result = report.run(
    reference_data=reference_dataset,
    current_data=current_dataset
)


# 11. Enregistrer le rapport
output_path = REPORT_DIR / "prediction_drift.html"

result.save_html(str(output_path))

print("Rapport créé :", output_path)
