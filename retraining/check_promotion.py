import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

DECISION_PATH = (
    ROOT / "models" / "synthetic"
    / "promotion_decision.json"
)

with open(DECISION_PATH, encoding="utf-8") as f:
    report = json.load(f)

decision = report["decision"]

print("Decision :", decision)

if decision == "PROMOTE":
    print("Candidat eligible au deploiement.")
elif decision == "REJECT":
    print("Candidat rejete : modele actuel conserve.")
else:
    raise ValueError(f"Decision inconnue : {decision}")
