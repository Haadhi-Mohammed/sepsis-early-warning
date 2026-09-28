import importlib

import pytest
from fastapi.testclient import TestClient

READING = {"HR": 95, "O2Sat": 94, "SBP": 105, "MAP": 72, "DBP": 55,
           "Resp": 22, "Temp": 38.2, "Age": 67, "Gender": 1,
           "HospAdmTime": -2}


def readings(n):
    return [{**READING, "ICULOS": i + 1, "HR": 90 + i} for i in range(n)]


def make_client(monkeypatch, model_dir):
    monkeypatch.setenv('MODEL_DIR', str(model_dir))
    import api.main
    importlib.reload(api.main)   # re-read MODEL_DIR
    return TestClient(api.main.app)


@pytest.fixture
def client(monkeypatch, bundle_dir):
    with make_client(monkeypatch, bundle_dir) as c:
        yield c


def test_health(client):
    r = client.get('/health')
    assert r.status_code == 200
    assert r.json() == {'status': 'healthy', 'model_loaded': True,
                        'model_version': 'test'}


def test_root_reports_model_from_config(client):
    model = client.get('/').json()['model']
    assert model['decision_threshold'] == 0.3
    assert model['explanations'] is True


def test_predict_returns_consistent_alert(client):
    r = client.post('/predict', json={'patient_id': 'P1', 'readings': readings(6)})
    assert r.status_code == 200
    body = r.json()
    assert 0 <= body['risk_score'] <= 1
    assert body['alert_level'] in {'GREEN', 'YELLOW', 'AMBER', 'RED'}
    assert body['sepsis_in_6h'] == (body['risk_score'] >= 0.3)
    assert body['hours_of_data'] == 6
    factors = body['top_risk_factors'] + body['protective_factors']
    assert factors and all(f['contribution'].endswith('pp') for f in factors)


def test_longer_history_is_accepted(client):
    r = client.post('/predict', json={'patient_id': 'P1', 'readings': readings(24)})
    assert r.status_code == 200
    assert r.json()['hours_of_data'] == 24


def test_missing_values_are_imputed(client):
    sparse = [{"ICULOS": i + 1} for i in range(6)]
    sparse[2]["Lactate"] = 3.1
    r = client.post('/predict', json={'patient_id': 'P1', 'readings': sparse})
    assert r.status_code == 200


def test_short_history_is_padded(client):
    # A new admission can be scored from its first hour
    r = client.post('/predict', json={'patient_id': 'P1', 'readings': readings(1)})
    assert r.status_code == 200
    assert r.json()['hours_of_data'] == 1


def test_empty_readings_rejected(client):
    r = client.post('/predict', json={'patient_id': 'P1', 'readings': []})
    assert r.status_code == 422


def test_missing_model_reports_unhealthy(monkeypatch, tmp_path):
    with make_client(monkeypatch, tmp_path / 'nope') as c:
        assert c.get('/health').status_code == 503
        r = c.post('/predict', json={'patient_id': 'P1', 'readings': readings(6)})
        assert r.status_code == 503
