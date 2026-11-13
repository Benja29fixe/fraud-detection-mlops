
import json
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / "models" / "synthetic"

# 1. Charger les règles YAML
with open(
    ROOT / "retraining" / "promotion_rules.yaml",
    encoding="utf-8"
) as f:
    rules = yaml.safe_load(f)["promotion"]

# 2. Charger les métriques du champion
with open(
    MODEL_DIR / "champion_metrics.json",
    encoding="utf-8"
) as f:
    champion = json.load(f)["metrics"]

# 3. Charger les métriques du challenger
with open(
    MODEL_DIR / "demo_metrics.json",
    encoding="utf-8"
) as f:
    challenger = json.load(f)["metrics"]

# 4. Appliquer les règles
checks = {
    "minimum_recall": (
        challenger["recall"] >= rules["min_recall"]
    ),
    "minimum_precision": (
        challenger["precision"] >= rules["min_precision"]
    ),
    "recall_not_degraded": (
        challenger["recall"] >=
        champion["recall"] - rules["max_recall_drop"]
    ),
    "f1_improvement": (
        challenger["f1"] >= champion["f1"]
        if rules["require_f1_improvement"]
        else True
    )
}

# 5. Décision globale
promote = all(checks.values())

decision = {
    "decision": "PROMOTE" if promote else "REJECT",
    "champion": champion,
    "challenger": challenger,
    "checks": checks
}

# 6. Sauvegarder le rapport
output_path = MODEL_DIR / "promotion_decision.json"

with open(output_path, "w", encoding="utf-8") as f:
    json.dump(decision, f, indent=4)

# 7. Afficher les résultats
print("\n=== DECISION DE PROMOTION ===")

for rule, passed in checks.items():
    print(f"{rule}: {'OK' if passed else 'ECHEC'}")

print("\nDécision :", decision["decision"])
print("Rapport :", output_path)
