
from pathlib import Path

import joblib
import mlflow
import pandas as pd

from sklearn.base import clone
from sklearn.metrics import (
    recall_score,
    precision_score,
    f1_score,
    average_precision_score
)

# 1. Définir les chemins
ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = ROOT / "data" / "retraining"
MODEL_PATH = ROOT / "models" / "fraud_model.joblib"

OUTPUT_DIR = ROOT / "models" / "retraining"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# 2. Charger les données
old_data = pd.read_parquet(DATA_DIR / "old_data.parquet")
new_data = pd.read_parquet(DATA_DIR / "new_data.parquet")
test_data = pd.read_parquet(DATA_DIR / "test_data.parquet")

# 3. Préparer les deux jeux d'entraînement
train_A = old_data.copy()

train_B = pd.concat(
    [old_data, new_data],
    ignore_index=True
)

# 4. Séparer les variables et la cible
features = [col for col in old_data.columns if col != "Class"]

X_A = train_A[features]
y_A = train_A["Class"]

X_B = train_B[features]
y_B = train_B["Class"]

X_test = test_data[features]
y_test = test_data["Class"]

# 5. Charger le modèle initial
initial_model = joblib.load(MODEL_PATH)

# 6. Reproduire sa configuration
model_A = clone(initial_model)
model_B = clone(initial_model)

# 7. Configurer MLflow
mlflow.set_tracking_uri(
     f"sqlite:///{(ROOT / 'mlflow.db').as_posix()}"
)

mlflow.set_experiment("fraud-xgboost-retraining")

# 8. Fonction d'entraînement et d'évaluation
def train_and_evaluate(name, model, X_train, y_train):

    with mlflow.start_run(run_name=name):

        # Entraîner
        model.fit(X_train, y_train)

        # Calculer les probabilités de fraude
        probabilities = model.predict_proba(X_test)[:, 1]

        # Appliquer un seuil fixe pour cette comparaison
        threshold = 0.5
        predictions = (probabilities >= threshold).astype(int)

        # Calculer les métriques
        metrics = {
            "recall": recall_score(
                y_test, predictions, zero_division=0
            ),
            "precision": precision_score(
                y_test, predictions, zero_division=0
            ),
            "f1": f1_score(
                y_test, predictions, zero_division=0
            ),
            "average_precision": average_precision_score(
                y_test, probabilities
            )
        }

        # Enregistrer les paramètres de l'expérience
        mlflow.log_param("model_version", name)
        mlflow.log_param("training_rows", len(X_train))
        mlflow.log_param("threshold", threshold)

        # Enregistrer les métriques
        mlflow.log_metrics(metrics)

        # Sauvegarder le modèle
        output_path = OUTPUT_DIR / f"{name}.joblib"
        joblib.dump(model, output_path)

        # Enregistrer le fichier dans MLflow
        mlflow.log_artifact(
            str(output_path),
            artifact_path="model"
        )

        print(f"\nRésultats du modèle {name}")

        for metric, value in metrics.items():
            print(f"{metric}: {value:.4f}")

        return metrics


# 9. Entraîner le modèle A
metrics_A = train_and_evaluate(
    "model_A",
    model_A,
    X_A,
    y_A
)

# 10. Entraîner le modèle B
metrics_B = train_and_evaluate(
    "model_B",
    model_B,
    X_B,
    y_B
)

# 11. Comparer les résultats
comparison = pd.DataFrame(
    [metrics_A, metrics_B],
    index=["model_A", "model_B"]
)

print("\nComparaison des modèles :")
print(comparison)

comparison.to_csv(
    OUTPUT_DIR / "comparison.csv"
)

from sklearn.metrics import confusion_matrix

for name, model in [
    ("model_A", model_A),
    ("model_B", model_B)
]:
    probabilities = model.predict_proba(X_test)[:, 1]
    predictions = (probabilities >= 0.5).astype(int)

    tn, fp, fn, tp = confusion_matrix(
        y_test, predictions, labels=[0, 1]
    ).ravel()

    print(f"\n{name}")
    print("Vrais positifs (TP) :", tp)
    print("Faux positifs (FP) :", fp)
    print("Vrais négatifs (TN) :", tn)
    print("Faux négatifs (FN) :", fn)