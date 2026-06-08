"""API de prédiction des matchs de Coupe du Monde (FastAPI + XGBoost).

Lancement : uvicorn app:app --reload   (→ http://localhost:8000)
Pré-requis : model.pkl + stats.pkl générés par `python train.py`.

Endpoints :
  POST /predict  {team1, team2, stage?, stadiumId?}
  GET  /health
"""
import os

import joblib
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from config import CORS_ORIGINS, MODEL_PATH, STATS_PATH
from features import is_knockout

app = FastAPI(title="WC 2026 — ML Predictions", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Artefacts chargés au démarrage (None si entraînement pas encore fait).
_model = None
_stats = None
_elevations: dict[str, float] = {}


@app.on_event("startup")
def _load() -> None:
    global _model, _stats, _elevations
    if os.path.exists(MODEL_PATH) and os.path.exists(STATS_PATH):
        _model = joblib.load(MODEL_PATH)
        bundle = joblib.load(STATS_PATH)
        _stats = bundle["stats"]
        _elevations = bundle.get("elevations", {})


class PredictRequest(BaseModel):
    team1: str
    team2: str
    stage: str | None = None
    stadiumId: str | None = None


class PredictResponse(BaseModel):
    team1_win: float
    draw: float
    team2_win: float
    favorite: str
    confidence: str  # low / medium / high


def _confidence(p: float) -> str:
    if p < 0.55:
        return "low"
    if p <= 0.65:
        return "medium"
    return "high"


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "model_loaded": _model is not None}


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest) -> PredictResponse:
    if _model is None or _stats is None:
        raise HTTPException(
            status_code=503,
            detail="Modèle non chargé. Lance `python train.py` d'abord.",
        )

    elevation = _elevations.get(req.stadiumId or "", 0.0)
    feats = _stats.features(
        req.team1, req.team2, is_knockout(req.stage), elevation
    )
    proba = _model.predict_proba(np.array([feats], dtype=float))[0]
    p1, draw, p2 = (round(float(x) * 100, 1) for x in proba)

    favorite = req.team1 if proba[0] >= proba[2] else req.team2
    return PredictResponse(
        team1_win=p1,
        draw=draw,
        team2_win=p2,
        favorite=favorite,
        confidence=_confidence(float(max(proba[0], proba[2]))),
    )
