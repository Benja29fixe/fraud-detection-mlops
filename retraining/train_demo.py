
import json
from pathlib import Path

import joblib
import mlflow
import numpy as np
import pandas as pd

from xgboost import XGBClassifier

from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    average_precision_score
)

# 1. Chemins
ROOT = Path(__file__).resolve().parents[1]

DATA_PATH = (
    ROOT / "data" / "synthetic"
    / "transactions.parquet"
)

OUTPUT_DIR = ROOT / "models" / "synthetic"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# 2. Charger les données
df = pd.read_parquet(DATA_PATH)

# 3. Découpage chronologique simulé
n = len(df)

train_end = int(n * 0.70)
val_end = int(n * 0.85)

train = df.iloc[:train_end]
validation = df.iloc[train_end:val_end]
test = df.iloc[val_end:]

# 4. Séparer X et y
features = [c for c in df.columns if c != "Class"]

X_train = train[features]
y_train = train["Class"]

X_val = validation[features]
y_val = validation["Class"]

X_test = test[features]
y_test = test["Class"]

# 5. Créer le modèle XGBoost
model = XGBClassifier(
    n_estimators=100,
    max_depth=4,
    learning_rate=0.05,
    objective="binary:logistic",
    eval_metric="logloss",
    random_state=42,
    n_jobs=2
)

# 6. Entraîner
model.fit(X_train, y_train)

# 7. Probabilités sur validation
val_proba = model.predict_proba(X_val)[:, 1]

# 8. Optimiser le seuil sur validation
thresholds = np.arange(0.05, 0.96, 0.05)

best_threshold = 0.5
best_f1 = -1.0

for threshold in thresholds:
    predictions = (
        val_proba >= threshold
    ).astype(int)

    score = f1_score(
        y_val, predictions, zero_division=0
    )

    if score > best_f1:
        best_f1 = score
        best_threshold = float(threshold)

# 9. Évaluation finale sur test
test_proba = model.predict_proba(X_test)[:, 1]

test_pred = (
    test_proba >= best_threshold
).astype(int)

metrics = {
    "precision": float(
        precision_score(y_test, test_pred, zero_division=0)
    ),
    "recall": float(
        recall_score(y_test, test_pred, zero_division=0)
    ),
    "f1": float(
        f1_score(y_test, test_pred, zero_division=0)
    ),
    "average_precision": float(
        average_precision_score(y_test, test_proba)
    )
}

# 10. Sauvegarder les résultats
model_path = OUTPUT_DIR / "demo_model.joblib"
metrics_path = OUTPUT_DIR / "demo_metrics.json"

joblib.dump(model, model_path)

with open(metrics_path, "w", encoding="utf-8") as f:
    json.dump(
        {
            "threshold": best_threshold,
            "metrics": metrics
        },
        f,
        indent=4
    )

# 11. Configurer MLflow
mlflow.set_tracking_uri(
    f"sqlite:///{(ROOT / 'mlflow.db').as_posix()}"
)

mlflow.set_experiment("fraud-synthetic-demo")

# 12. Enregistrer l'expérience
with mlflow.start_run(run_name="synthetic_xgboost"):

    mlflow.log_params({
        "n_estimators": 100,
        "max_depth": 4,
        "learning_rate": 0.05,
        "best_threshold": best_threshold
    })

    mlflow.log_metrics(metrics)

    mlflow.log_artifact(str(model_path))
    mlflow.log_artifact(str(metrics_path))

# 13. Afficher les résultats
print("\n=== RESULTATS SYNTHETIQUES ===")
print("Meilleur seuil :", best_threshold)

for name, value in metrics.items():
    print(f"{name}: {value:.4f}")

print("\nEntraînement terminé.")
