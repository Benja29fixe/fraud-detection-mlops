
import json
from pathlib import Path

import joblib
import mlflow
import numpy as np
import pandas as pd

from xgboost import XGBClassifier
from sklearn.base import clone

from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    average_precision_score,
    confusion_matrix
)


# ==========================================================
# 1. DEFINIR LES CHEMINS
# ==========================================================

ROOT = Path(__file__).resolve().parents[1]

DATA_PATH = ROOT / "data" / "synthetic" / "transactions.parquet"

OUTPUT_DIR = ROOT / "models" / "synthetic"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ==========================================================
# 2. CHARGER LES DONNEES SYNTHETIQUES
# ==========================================================

if not DATA_PATH.exists():
    raise FileNotFoundError(
        f"Dataset introuvable : {DATA_PATH}\n"
        "Execute d'abord generate_synthetic_data.py"
    )

df = pd.read_parquet(DATA_PATH)

df = df.sort_values("Time").reset_index(drop=True)

print("Nombre de transactions :", len(df))
print("Nombre de fraudes :", int(df["Class"].sum()))
print("Taux de fraude :", f"{df['Class'].mean():.2%}")


# ==========================================================
# 3. DECOUPER TRAIN / VALIDATION / TEST
# ==========================================================

n = len(df)

train_end = int(n * 0.70)
val_end = int(n * 0.85)

train = df.iloc[:train_end].copy()
validation = df.iloc[train_end:val_end].copy()
test = df.iloc[val_end:].copy()

features = [c for c in df.columns if c != "Class"]

X_train = train[features]
y_train = train["Class"]

X_val = validation[features]
y_val = validation["Class"]

X_test = test[features]
y_test = test["Class"]

print("\nRepartition des donnees :")
print("Train :", len(train))
print("Validation :", len(validation))
print("Test :", len(test))

if y_train.nunique() < 2 or y_val.nunique() < 2 or y_test.nunique() < 2:
    raise ValueError(
        "Au moins une partition ne contient pas les deux classes."
    )


# ==========================================================
# 4. CONFIGURER LE MODELE XGBOOST DE BASE
# ==========================================================

base_model = XGBClassifier(
    n_estimators=100,
    max_depth=4,
    learning_rate=0.05,
    objective="binary:logistic",
    eval_metric="logloss",
    random_state=42,
    n_jobs=2
)


# ==========================================================
# 5. FONCTION D'OPTIMISATION DU SEUIL
# ==========================================================

thresholds = np.arange(0.05, 0.96, 0.05)


def optimize_threshold(model):

    probabilities = model.predict_proba(X_val)[:, 1]

    best_threshold = 0.5
    best_f1 = -1.0

    for threshold in thresholds:

        predictions = (
            probabilities >= threshold
        ).astype(int)

        score = f1_score(
            y_val,
            predictions,
            zero_division=0
        )

        if score > best_f1:
            best_f1 = score
            best_threshold = float(threshold)

    return best_threshold, best_f1


# ==========================================================
# 6. FONCTION D'EVALUATION SUR LE JEU DE TEST
# ==========================================================

def evaluate_model(model, threshold):

    probabilities = model.predict_proba(X_test)[:, 1]

    predictions = (
        probabilities >= threshold
    ).astype(int)

    tn, fp, fn, tp = confusion_matrix(
        y_test,
        predictions,
        labels=[0, 1]
    ).ravel()

    metrics = {
        "precision": float(
            precision_score(
                y_test, predictions, zero_division=0
            )
        ),
        "recall": float(
            recall_score(
                y_test, predictions, zero_division=0
            )
        ),
        "f1": float(
            f1_score(
                y_test, predictions, zero_division=0
            )
        ),
        "average_precision": float(
            average_precision_score(
                y_test, probabilities
            )
        )
    }

    confusion = {
        "tp": int(tp),
        "fp": int(fp),
        "fn": int(fn),
        "tn": int(tn)
    }

    return metrics, confusion


# ==========================================================
# 7. ENTRAINER LE CHAMPION
# ==========================================================

print("\n=== ENTRAINEMENT DU CHAMPION ===")

champion = clone(base_model)

# Le champion utilise 75 % du jeu d'entrainement
old_end = int(len(X_train) * 0.75)

champion.fit(
    X_train.iloc[:old_end],
    y_train.iloc[:old_end]
)

champion_threshold, champion_val_f1 = optimize_threshold(
    champion
)

champion_metrics, champion_confusion = evaluate_model(
    champion,
    champion_threshold
)


# ==========================================================
# 8. ENTRAINER LE CHALLENGER
# ==========================================================

print("\n=== ENTRAINEMENT DU CHALLENGER ===")

challenger = clone(base_model)

# Le challenger utilise tout le jeu d'entrainement
challenger.fit(X_train, y_train)

challenger_threshold, challenger_val_f1 = optimize_threshold(
    challenger
)

challenger_metrics, challenger_confusion = evaluate_model(
    challenger,
    challenger_threshold
)


# ==========================================================
# 9. SAUVEGARDER LES MODELES ET LES METRIQUES
# ==========================================================

def save_results(
    name,
    model,
    threshold,
    validation_f1,
    metrics,
    confusion
):

    model_path = OUTPUT_DIR / f"{name}_model.joblib"
    metrics_path = OUTPUT_DIR / f"{name}_metrics.json"

    joblib.dump(model, model_path)

    report = {
        "threshold": float(threshold),
        "validation_f1": float(validation_f1),
        "metrics": metrics,
        "confusion_matrix": confusion
    }

    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=4)

    print(f"\n=== RESULTATS {name.upper()} ===")
    print(f"Seuil optimal : {threshold:.2f}")

    for metric, value in metrics.items():
        print(f"{metric}: {value:.4f}")

    print("Matrice de confusion :", confusion)

    return model_path, metrics_path


# Champion
champion_model_path, champion_metrics_path = save_results(
    name="champion",
    model=champion,
    threshold=champion_threshold,
    validation_f1=champion_val_f1,
    metrics=champion_metrics,
    confusion=champion_confusion
)

# Challenger
# Le nom "demo" est conserve pour decide_promotion.py
challenger_model_path, challenger_metrics_path = save_results(
    name="demo",
    model=challenger,
    threshold=challenger_threshold,
    validation_f1=challenger_val_f1,
    metrics=challenger_metrics,
    confusion=challenger_confusion
)


# ==========================================================
# 10. CONFIGURER MLFLOW
# ==========================================================

MLFLOW_DB = ROOT / "mlflow.db"

mlflow.set_tracking_uri(
    f"sqlite:///{MLFLOW_DB.as_posix()}"
)

mlflow.set_experiment("fraud-synthetic-demo")


# ==========================================================
# 11. ENREGISTRER LES EXPERIENCES DANS MLFLOW
# ==========================================================

def log_experiment(
    run_name,
    model_path,
    metrics_path,
    metrics,
    threshold,
    validation_f1,
    training_rows
):

    with mlflow.start_run(run_name=run_name):

        mlflow.log_params({
            "model_type": "XGBoost",
            "n_estimators": 100,
            "max_depth": 4,
            "learning_rate": 0.05,
            "threshold": float(threshold),
            "training_rows": int(training_rows)
        })

        mlflow.log_metrics(metrics)

        mlflow.log_metric(
            "validation_f1",
            float(validation_f1)
        )

        mlflow.log_artifact(str(model_path))
        mlflow.log_artifact(str(metrics_path))


log_experiment(
    run_name="champion",
    model_path=champion_model_path,
    metrics_path=champion_metrics_path,
    metrics=champion_metrics,
    threshold=champion_threshold,
    validation_f1=champion_val_f1,
    training_rows=old_end
)

log_experiment(
    run_name="challenger",
    model_path=challenger_model_path,
    metrics_path=challenger_metrics_path,
    metrics=challenger_metrics,
    threshold=challenger_threshold,
    validation_f1=challenger_val_f1,
    training_rows=len(X_train)
)


# ==========================================================
# 12. COMPARER LES PERFORMANCES
# ==========================================================

comparison = pd.DataFrame(
    [champion_metrics, challenger_metrics],
    index=["champion", "challenger"]
)

print("\n=== COMPARAISON DES MODELES ===")
print(comparison)

comparison.to_csv(
    OUTPUT_DIR / "comparison.csv"
)


# ==========================================================
# 13. VERIFIER LES FICHIERS GENERES
# ==========================================================

print("\n=== FICHIERS GENERES ===")

for path in [
    champion_model_path,
    champion_metrics_path,
    challenger_model_path,
    challenger_metrics_path
]:
    print(f"{path.name} : {path.exists()}")

print("\nEntrainement termine avec succes.")
