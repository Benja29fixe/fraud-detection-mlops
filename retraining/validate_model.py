
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

METRICS_PATH = (
    ROOT / "models" / "retraining"
    / "evaluation_metrics.json"
)

# 1. Charger les métriques
with open(METRICS_PATH, encoding="utf-8") as f:
    metrics = json.load(f)

# 2. Définir les critères minimaux
MIN_RECALL = 0.70
MIN_PRECISION = 0.85
MIN_F1 = 0.80

# 3. Vérifier les critères
checks = {
    "recall": metrics["recall"] >= MIN_RECALL,
    "precision": metrics["precision"] >= MIN_PRECISION,
    "f1": metrics["f1"] >= MIN_F1
}

# 4. Afficher les résultats
for metric, passed in checks.items():
    print(
        f"{metric}: "
        f"{'OK' if passed else 'ECHEC'}"
    )

# 5. Décision
if all(checks.values()):
    print("Candidat conforme aux seuils minimaux.")
else:
    print("Candidat non conforme aux seuils minimaux.")
    raise SystemExit(1)
