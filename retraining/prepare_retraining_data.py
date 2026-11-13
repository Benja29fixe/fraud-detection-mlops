
from pathlib import Path
import pandas as pd

# 1. Racine du projet
ROOT = Path(__file__).resolve().parents[1]

# 2. Chemin du fichier source
DATA_PATH = ROOT / "data" / "raw" / "creditcard.csv"

# 3. Dossier de sortie
OUTPUT_DIR = ROOT / "data" / "retraining"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# 4. Charger les données
df = pd.read_csv(DATA_PATH, delimiter=";")

print("Dimensions initiales :", df.shape)

# 5. Vérifier les colonnes
expected_columns = (
    ["Time"]
    + [f"V{i}" for i in range(1, 29)]
    + ["Amount", "Class"]
)

missing = set(expected_columns) - set(df.columns)

if missing:
    raise ValueError(f"Colonnes manquantes : {missing}")

# 6. Vérifier les valeurs manquantes
if df[expected_columns].isna().any().any():
    raise ValueError("Le dataset contient des valeurs manquantes.")

# 7. Trier chronologiquement
df = df.sort_values("Time").reset_index(drop=True)

# 8. Calculer les limites
n = len(df)

end_old = int(n * 0.60)
end_new = int(n * 0.80)

# 9. Découper les données
old_data = df.iloc[:end_old].copy()
new_data = df.iloc[end_old:end_new].copy()
test_data = df.iloc[end_new:].copy()

# 10. Sauvegarder les fichiers
old_data.to_parquet(
    OUTPUT_DIR / "old_data.parquet",
    index=False
)

new_data.to_parquet(
    OUTPUT_DIR / "new_data.parquet",
    index=False
)

test_data.to_parquet(
    OUTPUT_DIR / "test_data.parquet",
    index=False
)

# 11. Afficher les résultats
for name, data in [
    ("Anciennes données", old_data),
    ("Nouvelles données", new_data),
    ("Test final", test_data)
]:
    print(f"\n{name}")
    print("Nombre de transactions :", len(data))
    print("Nombre de fraudes :", int(data["Class"].sum()))
    print("Taux de fraude :", f"{data['Class'].mean():.4%}")

print("\nPréparation terminée.")
