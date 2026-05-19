"""
SenSante - Exploration du dataset patients_dakar.csv
Lab 1 : Git, Python et structure de projet
"""
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT / "data" / "patients_dakar.csv"


def load_data(path: Path = DATA_PATH) -> pd.DataFrame:
    return pd.read_csv(path, encoding="latin-1", sep=";")


def print_dataset_summary(df: pd.DataFrame) -> None:
    print("=" * 60)
    print("SenSante - Exploration du dataset")
    print("=" * 60)
    print(f"Nombre de patients : {len(df)}")
    print(f"Nombre de colonnes : {df.shape[1]}")
    print(f"Colonnes : {list(df.columns)}")
    print("\n--- 5 premiers patients ---")
    print(df.head())
    print("\n--- Statistiques descriptives ---")
    print(df.describe().round(2))


def print_distributions(df: pd.DataFrame) -> None:
    print("\n--- Répartition des diagnostics ---")
    diag_counts = df["diagnostic"].value_counts()
    for diag, count in diag_counts.items():
        pct = count / len(df) * 100
        print(f"{diag:12s} : {count:3d} patients ({pct:.1f}%)")

    print("\n--- Répartition par région (top 5) ---")
    region_counts = df["region"].value_counts().head(5)
    for region, count in region_counts.items():
        print(f"{region:15s} : {count:3d} patients")

    print("\n--- Température moyenne par diagnostic ---")
    temp_by_diag = df.groupby("diagnostic")["temperature"].mean()
    for diag, temp in temp_by_diag.items():
        print(f"{diag:12s} : {temp:.1f} °C")

    print("\n--- Nombre de patients par sexe et diagnostic ---")
    sexe_diag = df.groupby(["sexe", "diagnostic"]).size()
    print(sexe_diag)


def main() -> None:
    df = load_data()
    print_dataset_summary(df)
    print_distributions(df)
    print("\nExploration terminée !")


if __name__ == "__main__":
    main()
