# api/main.py
# SenSante API - Assistant pre-diagnostic medical
# Lab 3 - Integration de Modeles IA - ESP/UCAD
import os
import unicodedata
from pathlib import Path

import joblib
import numpy as np
from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from groq import Groq
from pydantic import BaseModel, Field


ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR = ROOT / "models"
FEATURE_COLS_FILE = MODEL_DIR / "feature_cols.pkl"
MODEL_FILE = MODEL_DIR / "model.pkl"
ENCODER_SEXE_FILE = MODEL_DIR / "encoder_sexe.pkl"
ENCODER_REGION_FILE = MODEL_DIR / "encoder_region.pkl"


def load_environment() -> None:
    load_dotenv()


def create_groq_client() -> Groq | None:
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        print("ATTENTION : GROQ_API_KEY non trouvee. /explain sera desactive.")
        return None

    client = Groq(api_key=api_key)
    print("Client Groq initialise.")
    return client


def normalize_text(value: str) -> str:
    text = value.strip().lower()
    normalized = unicodedata.normalize("NFKD", text)
    return "".join(ch for ch in normalized if unicodedata.category(ch) != "Mn")


def normalize_region(region: str, known_regions: list[str]) -> str:
    normalized_region = normalize_text(region)
    for candidate in known_regions:
        if normalize_text(candidate) == normalized_region:
            return candidate

    aliases = {
        "thies": "Thiès",
        "dakar": "Dakar",
        "saintlouis": "Saint-Louis",
        "saint-louis": "Saint-Louis",
        "matam": "Matam",
        "ziguinchor": "Ziguinchor",
    }
    return aliases.get(normalized_region, region)


class ExplainInput(BaseModel):
    diagnostic: str = Field(..., description="Diagnostic predit par le modele")
    probabilite: float = Field(..., description="Probabilite du diagnostic")
    age: int = Field(..., ge=0, le=120)
    sexe: str = Field(...)
    temperature: float = Field(...)
    region: str = Field(...)


class ExplainOutput(BaseModel):
    explication: str = Field(..., description="Explication en francais")
    modele_llm: str = Field(
        default="llama-3.1-8b-instant",
        description="Modele LLM utilise",
    )


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


app = FastAPI(
    title="SenSante API",
    description="Assistant pre-diagnostic medical pour le Senegal",
    version="0.2.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


SYSTEM_PROMPT = """Tu es un assistant medical senegalais.
Tu recois un diagnostic et des donnees patient.
Explique le resultat en francais simple,
comme un medecin parlerait a son patient.
Sois rassurant mais recommande toujours
une consultation medicale.
Maximum 3 phrases.
Ne fais JAMAIS de diagnostic toi-meme.
Tu expliques uniquement le diagnostic fourni."""


def load_model_artifacts() -> tuple:
    print("Chargement du modele...")
    model = joblib.load(MODEL_FILE)
    le_sexe = joblib.load(ENCODER_SEXE_FILE)
    le_region = joblib.load(ENCODER_REGION_FILE)
    feature_cols = joblib.load(FEATURE_COLS_FILE)
    print(f"Modele charge : {list(model.classes_)}")
    return model, le_sexe, le_region, feature_cols


load_environment()
groq_client = create_groq_client()
model, le_sexe, le_region, feature_cols = load_model_artifacts()


@app.get("/health")
def health_check() -> dict[str, str]:
    return {"status": "ok", "message": "SenSante API is running"}


@app.post("/predict", response_model=DiagnosticOutput)
def predict(patient: PatientInput) -> DiagnosticOutput:
    """Predire un diagnostic a partir des symptomes d'un patient."""
    try:
        sexe_enc = le_sexe.transform([patient.sexe])[0]
    except ValueError:
        return DiagnosticOutput(
            diagnostic="erreur",
            probabilite=0.0,
            confiance="aucune",
            message=f"Sexe invalide : {patient.sexe}",
        )

    region_value = normalize_region(patient.region, list(le_region.classes_))
    try:
        region_enc = le_region.transform([region_value])[0]
    except ValueError:
        return DiagnosticOutput(
            diagnostic="erreur",
            probabilite=0.0,
            confiance="aucune",
            message=f"Region inconnue : {patient.region}",
        )

    features = np.array([
        [
            patient.age,
            sexe_enc,
            patient.temperature,
            patient.tension_sys,
            int(patient.toux),
            int(patient.fatigue),
            int(patient.maux_tete),
            region_enc,
        ]
    ])

    diagnostic = model.predict(features)[0]
    proba_max = float(model.predict_proba(features)[0].max())

    confiance = (
        "haute" if proba_max >= 0.7 else "moyenne" if proba_max >= 0.4 else "faible"
    )

    messages = {
        "paludisme": "Suspicion de paludisme. Consultez rapidement.",
        "palu": "Suspicion de paludisme. Consultez rapidement.",
        "grippe": "Suspicion de grippe. Repos et hydratation.",
        "typh": "Suspicion de typhoide. Consultation necessaire.",
        "sain": "Pas de pathologie detectee.",
    }

    return DiagnosticOutput(
        diagnostic=diagnostic,
        probabilite=round(proba_max, 2),
        confiance=confiance,
        message=messages.get(diagnostic, "Consultez un medecin."),
    )


@app.post("/explain", response_model=ExplainOutput)
def explain(data: ExplainInput) -> ExplainOutput:
    """Expliquer un diagnostic en francais avec un LLM."""
    if not groq_client:
        return ExplainOutput(
            explication="Service d'explication indisponible. Cle API non configuree.",
            modele_llm="aucun",
        )

    user_prompt = (
        f"Patient : {data.sexe}, {data.age} ans, region {data.region}\n"
        f"Temperature : {data.temperature} C\n"
        f"Diagnostic du modele : {data.diagnostic} "
        f"(probabilite {data.probabilite:.0%})\n"
        f"Explique ce resultat au patient."
    )

    try:
        response = groq_client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            max_tokens=200,
            temperature=0.3,
        )
        explication = response.choices[0].message.content
    except Exception as e:
        explication = f"Erreur lors de l'appel au LLM : {str(e)}"

    return ExplainOutput(explication=explication)


@app.get("/model-info")
def model_info() -> dict[str, object]:
    """Retourne des informations sur le modele charge."""
    return {
        "type": type(model).__name__,
        "n_trees": getattr(model, "n_estimators", None),
        "classes": list(getattr(model, "classes_", [])),
        "n_features": getattr(model, "n_features_in_", None),
    }
app.mount("/static", StaticFiles(directory=ROOT / "frontend"), name="static")

@app.get("/")
def serve_frontend() -> FileResponse:
    """Servir la page d'accueil du frontend."""
    return FileResponse(ROOT / "frontend" / "index.html")
