# api/main.py
# FastAPI application for Sepsis Early Warning System

import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from sepsis.inference import MAX_HOURS, SepsisPredictor

# ── Model location ─────────────────────────────────────
BASE_DIR  = Path(__file__).parent.parent
MODEL_DIR = os.environ.get('MODEL_DIR', str(BASE_DIR / 'models' / 'production'))

predictor: Optional[SepsisPredictor] = None
load_error: Optional[str] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global predictor, load_error
    try:
        predictor = SepsisPredictor(MODEL_DIR)
        load_error = None
        print(f"Model {predictor.config['model_version']} loaded from {MODEL_DIR}")
    except Exception as e:   # keep serving /health so the failure is visible
        predictor, load_error = None, f'{type(e).__name__}: {e}'
        print(f'Model failed to load: {load_error}')
    yield


# ── FastAPI app ────────────────────────────────────────
app = FastAPI(
    title       = "Sepsis Early Warning API",
    description = "Predicts sepsis up to 6 hours before clinical onset from hourly "
                  "ICU vitals and labs (PhysioNet/CinC 2019 task)",
    version     = "2.0.0",
    lifespan    = lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins = ["*"],
    allow_methods = ["GET", "POST"],
    allow_headers = ["*"],
)


# ── Request/Response models ────────────────────────────
class HourlyReading(BaseModel):
    """One hour of patient vitals and labs"""
    # Vitals
    HR:    Optional[float] = None
    O2Sat: Optional[float] = None
    SBP:   Optional[float] = None
    MAP:   Optional[float] = None
    DBP:   Optional[float] = None
    Resp:  Optional[float] = None
    Temp:  Optional[float] = None
    # Labs
    Lactate:    Optional[float] = None
    WBC:        Optional[float] = None
    Creatinine: Optional[float] = None
    Glucose:    Optional[float] = None
    pH:         Optional[float] = None
    Hgb:        Optional[float] = None
    # Demographics
    Age:         Optional[float] = None
    Gender:      Optional[float] = None
    HospAdmTime: Optional[float] = None
    ICULOS:      Optional[float] = None


class PredictionRequest(BaseModel):
    """Hourly readings, oldest first. Send all available history (6+ hours ideal)"""
    patient_id: str
    readings:   list[HourlyReading] = Field(min_length=1, max_length=MAX_HOURS)


class SHAPFactor(BaseModel):
    feature:      str
    display_name: str
    shap_value:   float
    contribution: str


class PredictionResponse(BaseModel):
    """Risk prediction result"""
    patient_id:      str
    risk_score:      float
    alert_level:     str
    alert_message:   str
    sepsis_in_6h:    bool
    threshold_used:  float
    hours_of_data:   int
    model_version:   str
    top_risk_factors:   list[SHAPFactor] = []
    protective_factors: list[SHAPFactor] = []


def get_predictor() -> SepsisPredictor:
    if predictor is None:
        raise HTTPException(status_code=503,
                            detail=f'Model not loaded: {load_error}')
    return predictor


# ── API Endpoints ──────────────────────────────────────
@app.get("/")
def root():
    return {
        "name":    "Sepsis Early Warning API",
        "version": app.version,
        "status":  "running" if predictor else "degraded",
        "model":   predictor.info() if predictor else None,
    }


@app.get("/health")
def health():
    if predictor is None:
        raise HTTPException(status_code=503,
                            detail={"status": "unhealthy", "model_loaded": False,
                                    "error": load_error})
    return {"status": "healthy", "model_loaded": True,
            "model_version": predictor.config['model_version']}


@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest):
    readings = [r.model_dump() for r in request.readings]
    try:
        result = get_predictor().predict(readings)
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    return PredictionResponse(patient_id=request.patient_id, **result)


# ── Run server ─────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api.main:app", host="0.0.0.0", port=8000, reload=True)
