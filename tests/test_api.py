# tests/test_api.py
# Automated tests for Sepsis Early Warning API

from fastapi.testclient import TestClient
import sys
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from api.main import app

client = TestClient(app)

# ── Test 1: Root endpoint ──────────────────────────────
def test_root():
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert data["name"] == "Sepsis Early Warning API"
    assert data["status"] == "running"
    assert "model_auroc" in data
    print("✓ Root endpoint test passed")

# ── Test 2: Health endpoint ────────────────────────────
def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["model_loaded"] == True
    print("✓ Health endpoint test passed")

# ── Test 3: Predict endpoint - valid input ─────────────
def test_predict_valid():
    payload = {
        "patient_id": "test_001",
        "readings": [
            {"HR":95,"O2Sat":94,"SBP":105,"MAP":72,
             "DBP":55,"Resp":22,"Temp":38.2,
             "Age":67,"Gender":1,"HospAdmTime":-2,
             "ICULOS":1},
            {"HR":98,"O2Sat":93,"SBP":102,"MAP":70,
             "DBP":53,"Resp":23,"Temp":38.4,
             "Age":67,"Gender":1,"HospAdmTime":-2,
             "ICULOS":2},
            {"HR":102,"O2Sat":92,"SBP":98,"MAP":68,
             "DBP":50,"Resp":24,"Temp":38.6,
             "Age":67,"Gender":1,"HospAdmTime":-2,
             "ICULOS":3},
            {"HR":105,"O2Sat":91,"SBP":95,"MAP":65,
             "DBP":48,"Resp":25,"Temp":38.7,
             "Age":67,"Gender":1,"HospAdmTime":-2,
             "ICULOS":4},
            {"HR":108,"O2Sat":90,"SBP":92,"MAP":63,
             "DBP":46,"Resp":26,"Temp":38.8,
             "Age":67,"Gender":1,"HospAdmTime":-2,
             "ICULOS":5},
            {"HR":112,"O2Sat":89,"SBP":88,"MAP":60,
             "DBP":44,"Resp":28,"Temp":39.0,
             "Age":67,"Gender":1,"HospAdmTime":-2,
             "ICULOS":6},
        ]
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "risk_score" in data
    assert "alert_level" in data
    assert "sepsis_in_6h" in data
    assert 0 <= data["risk_score"] <= 1
    assert data["alert_level"] in ["RED","AMBER",
                                    "YELLOW","GREEN"]
    print(f"✓ Predict test passed - "
          f"Risk: {data['risk_score']:.3f}, "
          f"Alert: {data['alert_level']}")

# ── Test 4: Predict endpoint - wrong number of readings
def test_predict_wrong_readings():
    payload = {
        "patient_id": "test_002",
        "readings": [
            {"HR":95,"O2Sat":94,"SBP":105,"MAP":72,
             "DBP":55,"Resp":22,"Temp":38.2,
             "Age":67,"Gender":1,"HospAdmTime":-2,
             "ICULOS":1}
        ]  # Only 1 reading instead of 6
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 422
    print("✓ Validation test passed - "
          "correctly rejected wrong input")