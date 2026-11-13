
from pathlib import Path
import joblib
import mlflow
import numpy as np
import pandas as pd

from sklearn.base import clone
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    average_precision_score
)

# 1. Chemins
ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "retraining"

# 2. Charger les données
old_data = pd.read_parquet(DATA_DIR / "old_data.parquet")
new_data = pd.read_parquet(DATA_DIR / "new_data.parquet")

# 3. Réunir les 80 % de données disponibles
data = pd.concat(
    [old_data, new_data],
    ignore_index=True
)

# 4. Trier chronologiquement
data = data.sort_values("Time").reset_index(drop=True)

# 5. Séparer entraînement et validation
split = int(len(data) * 0.875)

train = data.iloc[:split].copy()
validation = data.iloc[split:].copy()

# 0.875 x 80 % = 70 % du dataset total

# 6. Variables explicatives et cible
features = [c for c in data.columns if c != "Class"]

X_train = train[features]
y_train = train["Class"]

X_val = validation[features]
y_val = validation["Class"]

# 7. Charger la configuration du modèle initial
initial_model = joblib.load(
    ROOT / "models" / "fraud_model.joblib"
)

model = clone(initial_model)

# 8. Entraîner le nouveau modèle
model.fit(X_train, y_train)

# 9. Calculer les probabilités de fraude
probabilities = model.predict_proba(X_val)[:, 1]

# 10. Tester plusieurs seuils
thresholds = np.arange(0.05, 0.96, 0.05)

results = []

for threshold in thresholds:

    predictions = (
        probabilities >= threshold
    ).astype(int)

    precision = precision_score(
        y_val, predictions, zero_division=0
    )

    recall = recall_score(
        y_val, predictions, zero_division=0
    )

    f1 = f1_score(
        y_val, predictions, zero_division=0
    )

    results.append({
        "threshold": float(threshold),
        "precision": precision,
        "recall": recall,
        "f1": f1
    })

# 11. Créer un tableau de résultats
results_df = pd.DataFrame(results)

# 12. Identifier le meilleur seuil
best_row = results_df.loc[
    results_df["f1"].idxmax()
]

best_threshold = float(best_row["threshold"])

print("\nMeilleur seuil :", best_threshold)
print("Precision :", best_row["precision"])
print("Recall :", best_row["recall"])
print("F1-score :", best_row["f1"])

# 13. Configurer MLflow
mlflow.set_tracking_uri(
    f"sqlite:///{(ROOT / 'mlflow.db').as_posix()}"
)

mlflow.set_experiment(
    "fraud-xgboost-threshold-optimization"
)

# 14. Enregistrer l'expérience
with mlflow.start_run(run_name="threshold_optimization"):

    mlflow.log_param(
        "best_threshold", best_threshold
    )

    mlflow.log_param(
        "training_rows", len(X_train)
    )

    mlflow.log_metric(
        "validation_f1", float(best_row["f1"])
    )

    mlflow.log_metric(
        "validation_precision",
        float(best_row["precision"])
    )

    mlflow.log_metric(
        "validation_recall",
        float(best_row["recall"])
    )

    mlflow.log_metric(
        "validation_average_precision",
        average_precision_score(y_val, probabilities)
    )

    # Sauvegarder les résultats
    output_dir = ROOT / "models" / "retraining"
    output_dir.mkdir(parents=True, exist_ok=True)

    results_path = output_dir / "threshold_results.csv"

    results_df.to_csv(results_path, index=False)

    joblib.dump(
        model,
        output_dir / "model_B_validated.joblib"
    )

    joblib.dump(
        {"threshold": best_threshold},
        output_dir / "threshold_config.joblib"
    )

    mlflow.log_artifact(str(results_path))

print("\nOptimisation terminée.")
