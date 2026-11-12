
from pathlib import Path

import pandas as pd

from evidently import Report, Dataset, DataDefinition
from evidently.presets import DataDriftPreset


# 1. Chemins
ROOT = Path(__file__).resolve().parents[1]

DATA_PATH = ROOT / "data" / "raw" / "creditcard.csv"
REPORT_DIR = ROOT / "monitoring" / "reports"

REPORT_DIR.mkdir(parents=True, exist_ok=True)


# 2. Charger les transactions
df = pd.read_csv(DATA_PATH, delimiter=";")

print("Dimensions :", df.shape)
# print(df.iloc[:5])


# 3. Sélectionner les variables
features = ["Time", "Amount"] + [
    f"V{i}" for i in range(1, 29)
]

X = df[features]
#print(X.shape)

# 4. Séparer les observations
# reference = X.iloc[:10000].copy()
# current = X.iloc[10000:12000].copy()



reference = X.sample(n=10000, random_state=42).copy()
current = reference.copy()

# Modifier artificiellement 18 variables
for i in range(1, 19):
    current[f"V{i}"] += 10


# 5. Simuler un Data Drift
current["Amount"] = current["Amount"]+1000


# 6. Définir le schéma
schema = DataDefinition(
    numerical_columns=features
)

reference_data = Dataset.from_pandas(
    reference,
    data_definition=schema
)

current_data = Dataset.from_pandas(
    current,
    data_definition=schema
)



# 7. Construire le rapport
report = Report(
    metrics=[DataDriftPreset()]
)

result = report.run(
    reference_data=reference_data,
    current_data=current_data
)


# 8. Enregistrer le rapport HTML
output_path = REPORT_DIR / "data_drift_report.html"

result.save_html(str(output_path))

print("Rapport créé :", output_path)
