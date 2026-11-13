
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.datasets import make_classification

# 1. Racine du projet
ROOT = Path(__file__).resolve().parents[1]

OUTPUT_DIR = ROOT / "data" / "synthetic"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# 2. Générer 10 000 transactions fictives
X, y = make_classification(
    n_samples=10000,
    n_features=20,
    n_informative=10,
    n_redundant=5,
    weights=[0.98, 0.02],
    flip_y=0.001,
    class_sep=1.2,
    random_state=42
)

# 3. Construire le DataFrame
columns = [f"V{i}" for i in range(1, 21)]

df = pd.DataFrame(X, columns=columns)

# 4. Ajouter des montants fictifs
rng = np.random.default_rng(42)

df["Amount"] = rng.lognormal(
    mean=3.5,
    sigma=1.0,
    size=len(df)
)

# 5. Ajouter une variable temporelle simulée
df["Time"] = np.arange(len(df))

# 6. Ajouter la cible
df["Class"] = y

# 7. Sauvegarder les données
output_path = OUTPUT_DIR / "transactions.parquet"

df.to_parquet(output_path, index=False)

print("Dataset synthétique créé :", output_path)
print("Nombre de transactions :", len(df))
print("Nombre de fraudes :", int(df["Class"].sum()))
print("Taux de fraude :", f"{df['Class'].mean():.2%}")
