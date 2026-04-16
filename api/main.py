# api/main.py
# FastAPI application for Sepsis Early Warning System

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import numpy as np
import torch
import torch.nn as nn
import pickle
from pathlib import Path
from typing import Optional
import uvicorn

# ── Paths ──────────────────────────────────────────────
BASE_DIR  = Path(__file__).parent.parent
MODEL_DIR = BASE_DIR / 'models'

# ── Explainer import ───────────────────────────────────
import sys
sys.path.append(str(BASE_DIR / 'src'))
from explainer import load_explainer, SepsisExplainer

# ── Load model config ──────────────────────────────────
with open(MODEL_DIR / 'model_config.pkl', 'rb') as f:
    config = pickle.load(f)

INPUT_SIZE    = config['input_size']
HIDDEN_SIZE   = config['hidden_size']
NUM_LAYERS    = config['num_layers']
DROPOUT       = config['dropout']
WINDOW_SIZE   = config['window_size']
FEATURE_COLS  = config['feature_cols']
THRESHOLD     = config['threshold']

# ── Load scaler ────────────────────────────────────────
with open(MODEL_DIR / 'scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

SCALE_FEATURES = config['scale_features']

# ── Define LSTM model ──────────────────────────────────
class SepsisLSTM(nn.Module):
    def __init__(self, input_size, hidden_size,
                 num_layers, dropout):
        super(SepsisLSTM, self).__init__()
        self.lstm = nn.LSTM(
            input_size  = input_size,
            hidden_size = hidden_size,
            num_layers  = num_layers,
            dropout     = dropout,
            batch_first = True
        )
        self.dropout = nn.Dropout(dropout)
        self.fc      = nn.Linear(hidden_size, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        out, _ = self.lstm(x)
        out    = out[:, -1, :]
        out    = self.dropout(out)
        out    = self.fc(out)
        return self.sigmoid(out).squeeze(1)

# ── Load trained model ─────────────────────────────────
device = torch.device('cpu')  # API runs on CPU
model  = SepsisLSTM(
    INPUT_SIZE, HIDDEN_SIZE, NUM_LAYERS, DROPOUT
).to(device)

model.load_state_dict(
    torch.load(
        MODEL_DIR / 'production_model.pt',
        map_location = device
    )
)
model.eval()
print("Model loaded successfully")

# ── Load SHAP explainer ────────────────────────────────
print("Loading SHAP explainer...")
explainer = load_explainer(model, FEATURE_COLS)
if explainer:
    print("SHAP explainer ready")
else:
    print("SHAP explainer not available")

# ── FastAPI app ────────────────────────────────────────
app = FastAPI(
    title       = "Sepsis Early Warning API",
    description = "Predicts sepsis risk 6 hours before onset",
    version     = "1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins     = ["*"],
    allow_credentials = True,
    allow_methods     = ["*"],
    allow_headers     = ["*"],
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
    """6 hours of patient readings"""
    patient_id: str
    readings:   list[HourlyReading]

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
    top_risk_factors:   list[SHAPFactor] = []
    protective_factors: list[SHAPFactor] = []

# ── Helper functions ───────────────────────────────────
LAB_FEATURES = ['Lactate','WBC','Creatinine',
                 'Glucose','pH','Hgb']

MEDIANS = {
    'HR': 84.0, 'O2Sat': 98.0, 'SBP': 118.5,
    'MAP': 77.0, 'DBP': 58.5, 'Resp': 18.0,
    'Temp': 37.06, 'Lactate': 1.8, 'WBC': 10.8,
    'Creatinine': 0.9, 'Glucose': 124.0,
    'pH': 7.39, 'Hgb': 10.4,
    'Age': 63.0, 'HospAdmTime': -0.5, 'ICULOS': 20.0
}

def preprocess_readings(readings: list[HourlyReading]):
    """
    Convert raw readings into scaled feature array
    ready for LSTM input
    """
    import pandas as pd

    # Convert to dataframe
    rows = []
    for i, r in enumerate(readings):
        row = {
            'HR':          r.HR,
            'O2Sat':       r.O2Sat,
            'SBP':         r.SBP,
            'MAP':         r.MAP,
            'DBP':         r.DBP,
            'Resp':        r.Resp,
            'Temp':        r.Temp,
            'Lactate':     r.Lactate,
            'WBC':         r.WBC,
            'Creatinine':  r.Creatinine,
            'Glucose':     r.Glucose,
            'pH':          r.pH,
            'Hgb':         r.Hgb,
            'Age':         r.Age,
            'Gender':      r.Gender,
            'HospAdmTime': r.HospAdmTime,
            'ICULOS':      r.ICULOS if r.ICULOS 
                           else i + 1,
        }
        rows.append(row)

    df = pd.DataFrame(rows)

    # Missingness indicators BEFORE imputation
    for lab in LAB_FEATURES:
        df[f'{lab}_obs'] = df[lab].notna().astype(float)

    # Forward fill then median fill
    vital_lab_cols = [
        'HR','O2Sat','SBP','MAP','DBP','Resp','Temp',
        'Lactate','WBC','Creatinine','Glucose','pH','Hgb'
    ]
    df[vital_lab_cols] = df[vital_lab_cols].ffill().bfill()
    for col in vital_lab_cols:
        df[col] = df[col].fillna(MEDIANS.get(col, 0))

    # Fill demographics
    for col in ['Age','Gender','HospAdmTime']:
        df[col] = df[col].fillna(MEDIANS.get(col, 0))

    # Scale
    df[SCALE_FEATURES] = scaler.transform(df[SCALE_FEATURES])

    # Build feature array in correct order
    feature_array = df[FEATURE_COLS].values
    feature_array = np.nan_to_num(feature_array, nan=0.0)

    return feature_array

def get_alert_level(risk_score: float):
    """Convert risk score to clinical alert level"""
    if risk_score >= 0.4:
        return "RED",    "HIGH RISK — Immediate clinical review required"
    elif risk_score >= 0.2:
        return "AMBER",  "MODERATE RISK — Increased monitoring recommended"
    elif risk_score >= THRESHOLD:
        return "YELLOW", "LOW RISK — Continue standard monitoring"
    else:
        return "GREEN",  "MINIMAL RISK — No immediate action required"

# ── API Endpoints ──────────────────────────────────────
@app.get("/")
def root():
    return {
        "name":        "Sepsis Early Warning API",
        "version":     "1.0.0",
        "status":      "running",
        "model_auroc": 0.7796,
        "threshold":   THRESHOLD,
    }

@app.get("/health")
def health():
    return {"status": "healthy", "model_loaded": True}

@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest):
    """
    Predict sepsis risk for a patient.
    Requires exactly 6 hourly readings.
    """
    # Validate input
    if len(request.readings) != WINDOW_SIZE:
        raise HTTPException(
            status_code = 422,
            detail      = f"Exactly {WINDOW_SIZE} hourly "
                          f"readings required. "
                          f"Got {len(request.readings)}."
        )

    # Preprocess
    try:
        features = preprocess_readings(request.readings)
    except Exception as e:
        raise HTTPException(
            status_code = 500,
            detail      = f"Preprocessing error: {str(e)}"
        )

    # Run model
    with torch.no_grad():
        tensor = torch.FloatTensor(features).unsqueeze(0)
        risk_score = model(tensor).item()

    # Get alert level
    alert_level, alert_message = get_alert_level(risk_score)

    # Get SHAP explanations
    top_risk   = []
    protective = []

    if explainer:
        try:
            factors = explainer.get_top_factors(
                features, top_n=5
            )

            for f in factors['risk_increasing'][:3]:
                top_risk.append(SHAPFactor(
                    feature      = f['feature'],
                    display_name = f['display_name'],
                    shap_value   = f['shap_value'],
                    contribution = f"+{f['shap_value']*100:.1f}%"
                ))

            for f in factors['risk_decreasing'][:3]:
                protective.append(SHAPFactor(
                    feature      = f['feature'],
                    display_name = f['display_name'],
                    shap_value   = f['shap_value'],
                    contribution = f"{f['shap_value']*100:.1f}%"
                ))
        except Exception as e:
            print(f"SHAP error: {str(e)}")
            import traceback
            traceback.print_exc()

    return PredictionResponse(
        patient_id          = request.patient_id,
        risk_score          = round(risk_score, 4),
        alert_level         = alert_level,
        alert_message       = alert_message,
        sepsis_in_6h        = risk_score >= THRESHOLD,
        threshold_used      = THRESHOLD,
        hours_of_data       = len(request.readings),
        top_risk_factors    = top_risk,
        protective_factors  = protective,
    )

# ── Run server ─────────────────────────────────────────
if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host     = "0.0.0.0",
        port     = 8000,
        reload   = True
    )

