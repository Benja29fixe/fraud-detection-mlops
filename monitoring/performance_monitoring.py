
from pathlib import Path

import joblib
import pandas as pd

from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    average_precision_score,
    confusion_matrix
)

# 1. Chemins
ROOT = Path(__file__).resolve().parents[1]

DATA_PATH = ROOT / "data" / "processed" / "test.parquet"
MODEL_PATH = ROOT / "models" / "fraud_model.joblib"
CONFIG_PATH = ROOT / "models" / "decision_config.joblib"

# 2. Charger le modèle et sa configuration
model = joblib.load(MODEL_PATH)
config = joblib.load(CONFIG_PATH)

threshold = float(config["threshold"])

# 3. Charger les transactions de test
df = pd.read_parquet(DATA_PATH)

# Trier chronologiquement si Time est disponible
df = df.sort_values("Time").reset_index(drop=True)

# 4. Séparer deux périodes
midpoint = len(df) // 2

period_1 = df.iloc[:midpoint].copy()
period_2 = df.iloc[midpoint:].copy()

# 5. Variables utilisées par le modèle
features = ["Time", "Amount"] + [
    f"V{i}" for i in range(1, 29)
]

# 6. Fonction d'évaluation
def evaluate_period(data, period_name):

    X = data[features]
    y_true = data["Class"]

    probabilities = model.predict_proba(X)[:, 1]

    predictions = (
        probabilities >= threshold
    ).astype(int)

    metrics = {
        "period": period_name,
        "n_transactions": len(data),
        "n_frauds": int(y_true.sum()),
        "precision": precision_score(
            y_true, predictions, zero_division=0
        ),
        "recall": recall_score(
            y_true, predictions, zero_division=0
        ),
        "f1": f1_score(
            y_true, predictions, zero_division=0
        ),
        "average_precision": (
            average_precision_score(y_true, probabilities)
            if y_true.nunique() == 2 else float("nan")
        ),
        "predicted_fraud_rate": predictions.mean()
    }

    print(f"\n--- {period_name} ---")
    print(pd.Series(metrics))

    print("\nMatrice de confusion :")
    print(confusion_matrix(
        y_true, predictions, labels=[0, 1]
    ))

    return metrics


# 7. Évaluer les deux périodes
metrics_1 = evaluate_period(period_1, "Période 1")
metrics_2 = evaluate_period(period_2, "Période 2")

# 8. Comparer les résultats
results = pd.DataFrame([metrics_1, metrics_2])

print("\nComparaison des performances :")
print(results.to_string(index=False))

# 9. Sauvegarder les métriques
output_dir = ROOT / "monitoring" / "reports"
output_dir.mkdir(parents=True, exist_ok=True)

results.to_csv(
    output_dir / "performance_metrics.csv",
    index=False
)

print("\nMétriques enregistrées.")

# 10. Détection d'une dégradation du Recall
recall_1 = metrics_1["recall"]
recall_2 = metrics_2["recall"]

recall_drop = recall_1 - recall_2

print(f"\nBaisse du Recall : {recall_drop:.2%}")

if recall_drop > 0.10:
    print("ALERTE : baisse importante du Recall !")
    print("Investigation du modèle nécessaire.")
else:
    print("Pas d'alerte Recall selon le seuil défini.")