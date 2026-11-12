
from pathlib import Path
import json

import pandas as pd

import os

ROOT = Path(__file__).resolve().parents[1]

REPORT_DIR = ROOT / "monitoring" / "reports"

METRICS_PATH = Path(
    os.getenv(
        "MONITORING_METRICS_PATH",
        str(REPORT_DIR / "performance_metrics.csv")
    )
)


ALERT_PATH = REPORT_DIR / "monitoring_alert.json"

# 1. Charger les performances
df = pd.read_csv(METRICS_PATH)

if len(df) < 2:
    raise ValueError("Deux périodes sont nécessaires")

# 2. Récupérer les métriques
reference = df.iloc[0]
current = df.iloc[1]

recall_reference = float(reference["recall"])
recall_current = float(current["recall"])

precision_reference = float(reference["precision"])
precision_current = float(current["precision"])

# 3. Définir nos seuils d'alerte
MAX_RECALL_DROP = 0.10
MAX_PRECISION_DROP = 0.10

recall_drop = recall_reference - recall_current
precision_drop = precision_reference - precision_current

# 4. Détecter les alertes
alerts = []

if recall_drop > MAX_RECALL_DROP:
    alerts.append({
        "metric": "recall",
        "reference": recall_reference,
        "current": recall_current,
        "drop": recall_drop
    })

if precision_drop > MAX_PRECISION_DROP:
    alerts.append({
        "metric": "precision",
        "reference": precision_reference,
        "current": precision_current,
        "drop": precision_drop
    })

# 5. Construire le résultat
result = {
    "status": "alert" if alerts else "ok",
    "alerts": alerts
}

# 6. Sauvegarder le résultat
REPORT_DIR.mkdir(parents=True, exist_ok=True)

with open(ALERT_PATH, "w", encoding="utf-8") as f:
    json.dump(result, f, indent=2)

print(json.dumps(result, indent=2))

# 7. Retourner un code d'erreur en cas d'alerte
if alerts:
    raise SystemExit(1)
