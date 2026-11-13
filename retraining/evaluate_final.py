
from pathlib import Path
import joblib
import pandas as pd

from sklearn.metrics import (
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
    average_precision_score
)

# 1. Définir les chemins
ROOT = Path(__file__).resolve().parents[1]

MODEL_DIR = ROOT / "models" / "retraining"

# 2. Charger le modèle optimisé
model = joblib.load(
    MODEL_DIR / "model_B_validated.joblib"
)

# 3. Charger le seuil optimisé
config = joblib.load(
    MODEL_DIR / "threshold_config.joblib"
)

threshold = config["threshold"]

# 4. Charger le jeu de test final
test = pd.read_parquet(
    ROOT / "data" / "retraining" / "test_data.parquet"
)

X_test = test.drop(columns=["Class"])
y_test = test["Class"]

# 5. Calculer les scores de fraude
probabilities = model.predict_proba(X_test)[:, 1]

# 6. Appliquer le seuil optimisé
predictions = (
    probabilities >= threshold
).astype(int)

# 7. Calculer les métriques
precision = precision_score(
    y_test, predictions, zero_division=0
)

recall = recall_score(
    y_test, predictions, zero_division=0
)

f1 = f1_score(
    y_test, predictions, zero_division=0
)

ap = average_precision_score(
    y_test, probabilities
)

# 8. Matrice de confusion
tn, fp, fn, tp = confusion_matrix(
    y_test, predictions, labels=[0, 1]
).ravel()

# 9. Afficher les résultats
print("\n=== EVALUATION FINALE ===")

print(f"Seuil : {threshold:.2f}")
print(f"Precision : {precision:.4f}")
print(f"Recall : {recall:.4f}")
print(f"F1-score : {f1:.4f}")
print(f"Average Precision : {ap:.4f}")

print("\nMatrice de confusion :")
print(f"Vrais positifs (TP) : {tp}")
print(f"Faux positifs (FP) : {fp}")
print(f"Faux négatifs (FN) : {fn}")
print(f"Vrais négatifs (TN) : {tn}")

import json

metrics = {
    "threshold": float(threshold),
    "precision": float(precision),
    "recall": float(recall),
    "f1": float(f1),
    "average_precision": float(ap),
    "tp": int(tp),
    "fp": int(fp),
    "fn": int(fn),
    "tn": int(tn)
}

output_path = MODEL_DIR / "evaluation_metrics.json"

with open(output_path, "w", encoding="utf-8") as f:
    json.dump(metrics, f, indent=4)

print("Métriques sauvegardées :", output_path)