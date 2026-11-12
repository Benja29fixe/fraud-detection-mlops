
import json
import os
from pathlib import Path
from datetime import datetime, timezone

from azure.identity import AzureCliCredential
from azure.monitor.ingestion import LogsIngestionClient

ROOT = Path(__file__).resolve().parents[2]

# 1. Charger notre rapport JSON
alert_path = (
    ROOT / "monitoring" / "reports"
    / "monitoring_alert.json"
)

with open(alert_path, encoding="utf-8") as f:
    report = json.load(f)

# 2. Préparer les événements
events = []

for alert in report["alerts"]:
    events.append({
        "TimeGenerated": datetime.now(
            timezone.utc
        ).isoformat(),
        "ModelName": "fraud-xgboost",
        "MetricName": alert["metric"],
        "ReferenceValue": float(alert["reference"]),
        "CurrentValue": float(alert["current"]),
        "DropValue": float(alert["drop"]),
        "Status": "alert"
    })

if not events:
    print("Aucune alerte à envoyer.")
    raise SystemExit(0)

# 3. Paramètres Azure
endpoint = os.environ["AZURE_INGESTION_ENDPOINT"]
dcr_id = os.environ["AZURE_DCR_IMMUTABLE_ID"]
stream_name = "Custom-FraudModelMonitoring"

# 4. Authentification
credential = AzureCliCredential()

# 5. Client Azure Monitor
client = LogsIngestionClient(
    endpoint=endpoint,
    credential=credential
)

# 6. Envoi des événements
client.upload(
    rule_id=dcr_id,
    stream_name=stream_name,
    logs=events
)

print(f"{len(events)} alertes envoyées à Azure.")
