
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

ALERT_FILE = (
    ROOT / "monitoring"
    / "reports"
    / "monitoring_alert.json"
)

# 1. Charger le rapport de monitoring
with open(ALERT_FILE, "r", encoding="utf-8") as f:
    report = json.load(f)

# 2. Récupérer les alertes
alerts = report.get("alerts", [])

# 3. Vérifier si les performances se dégradent
performance_metrics = {"recall", "precision"}

performance_alerts = [
    alert
    for alert in alerts
    if alert.get("metric") in performance_metrics
]

# 4. Prendre une décision
if performance_alerts:
    print("Réentraînement à envisager.")

    for alert in performance_alerts:
        print(
            f"Métrique : {alert['metric']} | "
            f"Baisse : {alert['drop']:.2%}"
        )
else:
    print("Aucune dégradation de performance signalée.")
