import joblib
from pathlib import Path
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT / "data" / "patients_dakar.csv"
MODEL_DIR = ROOT / "models"
FIGURES_DIR = ROOT / "figures"
FEATURE_COLS = [
    "age",
    "sexe_encoded",
    "temperature",
    "tension_sys",
    "toux",
    "fatigue",
    "maux_tete",
    "region_encoded",
]


def load_data(path: Path = DATA_PATH) -> pd.DataFrame:
    return pd.read_csv(path, encoding="latin-1", sep=";")


def encode_features(df: pd.DataFrame):
    df = df.copy()
    le_sexe = LabelEncoder()
    le_region = LabelEncoder()
    df["sexe_encoded"] = le_sexe.fit_transform(df["sexe"])
    df["region_encoded"] = le_region.fit_transform(df["region"])
    return df, le_sexe, le_region


def split_dataset(df: pd.DataFrame):
    X = df[FEATURE_COLS]
    y = df["diagnostic"]
    return train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )


def create_model() -> RandomForestClassifier:
    return RandomForestClassifier(n_estimators=100, random_state=42)


def evaluate_model(model: RandomForestClassifier, X_test: pd.DataFrame, y_test: pd.Series):
    y_pred = model.predict(X_test)
    return (
        y_pred,
        accuracy_score(y_test, y_pred),
        classification_report(y_test, y_pred, zero_division=0),
        confusion_matrix(y_test, y_pred, labels=model.classes_),
    )


def plot_confusion_matrix(cm: np.ndarray, labels: List[str]) -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=labels,
        yticklabels=labels,
    )
    plt.xlabel("Prediction du modele")
    plt.ylabel("Vrai diagnostic")
    plt.title("Matrice de confusion - SenSante")
    plt.tight_layout()
    output_path = FIGURES_DIR / "confusion_matrix.png"
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Figure sauvegardée : {output_path}")


def save_artifacts(model: RandomForestClassifier, le_sexe: LabelEncoder, le_region: LabelEncoder) -> None:
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_DIR / "model.pkl")
    joblib.dump(le_sexe, MODEL_DIR / "encoder_sexe.pkl")
    joblib.dump(le_region, MODEL_DIR / "encoder_region.pkl")
    joblib.dump(FEATURE_COLS, MODEL_DIR / "feature_cols.pkl")
    size = (MODEL_DIR / "model.pkl").stat().st_size
    print(f"Modèle sauvegardé : {MODEL_DIR / 'model.pkl'} ({size / 1024:.1f} Ko)")


def summarize_patient_prediction(model: RandomForestClassifier, le_sexe: LabelEncoder, le_region: LabelEncoder, patient: dict) -> None:
    features = [
        patient["age"],
        le_sexe.transform([patient["sexe"]])[0],
        patient["temperature"],
        patient["tension_sys"],
        int(patient["toux"]),
        int(patient["fatigue"]),
        int(patient["maux_tete"]),
        le_region.transform([patient["region"]])[0],
    ]
    row = pd.DataFrame([features], columns=FEATURE_COLS)
    prediction = model.predict(row)[0]
    probabilities = model.predict_proba(row)[0]
    print(f"Patient : {patient['sexe']}, {patient['age']} ans")
    print(f"Diagnostic prédit : {prediction}")
    for label, proba in zip(model.classes_, probabilities):
        bar = "#" * int(proba * 30)
        print(f"  {label:11s} : {proba:.1%} {bar}")


def main() -> None:
    df = load_data()
    print(f"Dataset : {len(df)} patients, {len(df.columns)} colonnes")
    print(f"Colonnes : {list(df.columns)}")
    print(f"Diagnostics :\n{df['diagnostic'].value_counts()}\n")

    df, le_sexe, le_region = encode_features(df)
    X_train, X_test, y_train, y_test = split_dataset(df)

    print(f"Entraînement : {X_train.shape[0]} patients")
    print(f"Test : {X_test.shape[0]} patients")

    model = create_model()
    model.fit(X_train, y_train)
    print("Modèle entraîné !")
    print(f"Nombre d'arbres : {model.n_estimators}")
    print(f"Nombre de features : {model.n_features_in_}")
    print(f"Classes : {list(model.classes_)}")

    y_pred, accuracy, report, cm = evaluate_model(model, X_test, y_test)
    print(f"Accuracy : {accuracy:.2%}\n")
    print("Matrice de confusion :")
    print(cm)
    print("\nRapport de classification :")
    print(report)
    plot_confusion_matrix(cm, list(model.classes_))

    save_artifacts(model, le_sexe, le_region)

    print("\n--- Exemple de prédiction ---")
    example_patient = {
        "age": 28,
        "sexe": "F",
        "temperature": 39.5,
        "tension_sys": 110,
        "toux": True,
        "fatigue": True,
        "maux_tete": True,
        "region": "Dakar",
    }
    summarize_patient_prediction(model, le_sexe, le_region, example_patient)

    print("\n--- Prévisions de test ---")
    test_patients = [
        {
            "nom": "Jeune sans symptômes",
            "age": 20,
            "sexe": "M",
            "temperature": 36.5,
            "tension_sys": 120,
            "toux": False,
            "fatigue": False,
            "maux_tete": False,
            "region": "Dakar",
        },
        {
            "nom": "Adulte avec forte fièvre",
            "age": 45,
            "sexe": "F",
            "temperature": 40.2,
            "tension_sys": 110,
            "toux": False,
            "fatigue": True,
            "maux_tete": True,
            "region": "Diourbel",
        },
        {
            "nom": "Patient âgé avec toux",
            "age": 75,
            "sexe": "M",
            "temperature": 38.3,
            "tension_sys": 130,
            "toux": True,
            "fatigue": False,
            "maux_tete": False,
            "region": "Kaolack",
        },
    ]
    for patient in test_patients:
        print(f"\nPatient test : {patient['nom']}")
        summarize_patient_prediction(model, le_sexe, le_region, patient)


if __name__ == "__main__":
    main()

