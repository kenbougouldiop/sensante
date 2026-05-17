# api/main.py
# SenSante API - Assistant pre-diagnostic medical
# Lab 3 - Integration de Modeles IA - ESP/UCAD
from fastapi import FastAPI
from pydantic import BaseModel, Field
import joblib
import numpy as np

#--- Schemas Pydantic --------------------------------------------------
class PatientInput(BaseModel):
    age: int = Field(..., ge=0, le=120, description="Age en annees")
    sexe: str = Field(..., description="Sexe : M ou F")
    temperature: float = Field(..., ge=35.0, le=42.0, description="Temperature en Celsius")
    tension_sys: int = Field(..., ge=60, le=250, description="Tension systolique")
    toux: bool = Field(..., description="Presence de toux")
    fatigue: bool = Field(..., description="Presence de fatigue")
    maux_tete: bool = Field(..., description="Presence de maux de tete")
    region: str = Field(..., description="Region du Senegal")


class DiagnosticOutput(BaseModel):
    diagnostic: str
    probabilite: float
    confiance: str
    message: str


#--- Application FastAPI ----------------------------------------------
app = FastAPI(
    title="SenSante API",
    description="Assistant pre-diagnostic medical pour le Senegal",
    version="0.2.0",
)


#--- Chargement du modele (une seule fois) ----------------------------
print("Chargement du modele...")
model = joblib.load("models/model.pkl")
le_sexe = joblib.load("models/encoder_sexe.pkl")
le_region = joblib.load("models/encoder_region.pkl")
feature_cols = joblib.load("models/feature_cols.pkl")
print(f"Modele charge : {list(model.classes_)}")


#--- Routes -----------------------------------------------------------
@app.get("/health")
def health_check():
    return {"status": "ok", "message": "SenSante API is running"}


@app.post("/predict", response_model=DiagnosticOutput)
def predict(patient: PatientInput):
    """Predire un diagnostic a partir des symptomes d'un patient.
    Recoit les symptomes en JSON, renvoie le diagnostic, la probabilite et une recommandation.
    """

    # 1. Encoder les variables categoriques
    try:
        sexe_enc = le_sexe.transform([patient.sexe])[0]
    except ValueError:
        return DiagnosticOutput(
            diagnostic="erreur",
            probabilite=0.0,
            confiance="aucune",
            message=f"Sexe invalide : {patient.sexe}"
        )

    try:
        region_enc = le_region.transform([patient.region])[0]
    except ValueError:
        return DiagnosticOutput(
            diagnostic="erreur",
            probabilite=0.0,
            confiance="aucune",
            message=f"Region inconnue : {patient.region}"
        )

    # 2. Construire le vecteur de features
    features = np.array([[
        patient.age,
        sexe_enc,
        patient.temperature,
        patient.tension_sys,
        int(patient.toux),
        int(patient.fatigue),
        int(patient.maux_tete),
        region_enc,
    ]])

    # 3. Prediction
    diagnostic = model.predict(features)[0]
    proba_max = float(model.predict_proba(features)[0].max())

    # 4. Determiner le niveau de confiance
    confiance = (
        "haute" if proba_max >= 0.7 else
        "moyenne" if proba_max >= 0.4 else
        "faible"
    )

    # 5. Generer la recommandation
    messages = {
        "paludisme": "Suspicion de paludisme. Consultez rapidement.",
        "palu": "Suspicion de paludisme. Consultez rapidement.",
        "grippe": "Suspicion de grippe. Repos et hydratation.",
        "typh": "Suspicion de typhoide. Consultation necessaire.",
        "sain": "Pas de pathologie detectee.",
    }

    # 6. Renvoyer le resultat
    return DiagnosticOutput(
        diagnostic=diagnostic,
        probabilite=round(proba_max, 2),
        confiance=confiance,
        message=messages.get(diagnostic, "Consultez un medecin."),
    )


@app.get("/model-info")
def model_info():
    """Retourne des informations sur le modele charge."""
    model_type = type(model).__name__
    n_trees = getattr(model, "n_estimators", None)
    classes = list(getattr(model, "classes_", []))
    n_features = getattr(model, "n_features_in_", None)
    return {
        "type": model_type,
        "n_trees": n_trees,
        "classes": classes,
        "n_features": n_features,
    }
from fastapi.middleware.cors import CORSMiddleware
# Autoriser les requetes depuis le frontend
app.add_middleware(
CORSMiddleware,
allow_origins=["*"],
# En dev : tout accepter
allow_credentials=True,
allow_methods=["*"],
allow_headers=["*"],
)