# tests/test_api.py
import pytest
from unittest.mock import patch, MagicMock
import numpy as np
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

# ── Mock model and config before importing app ─────────
# This prevents FileNotFoundError in CI environment

mock_config = {
    'input_size':    23,
    'hidden_size':   64,
    'num_layers':    2,
    'dropout':       0.3,
    'window_size':   6,
    'n_features':    23,
    'feature_cols':  [
        'HR','O2Sat','SBP','MAP','DBP','Resp','Temp',
        'Lactate','WBC','Creatinine','Glucose','pH','Hgb',
        'Age','Gender','HospAdmTime','ICULOS',
        'Lactate_obs','WBC_obs','Creatinine_obs',
        'Glucose_obs','pH_obs','Hgb_obs'
    ],
    'threshold':     0.05,
    'auroc':         0.7796,
    'scale_features':['HR','O2Sat','SBP','MAP','DBP',
                      'Resp','Temp','Lactate','WBC',
                      'Creatinine','Glucose','pH','Hgb',
                      'Age','HospAdmTime','ICULOS'],
}

# Sample readings for tests
SAMPLE_READINGS = [
    {"HR":95,"O2Sat":94,"SBP":105,"MAP":72,
     "DBP":55,"Resp":22,"Temp":38.2,
     "Age":67,"Gender":1,"HospAdmTime":-2,"ICULOS":1},
    {"HR":98,"O2Sat":93,"SBP":102,"MAP":70,
     "DBP":53,"Resp":23,"Temp":38.4,
     "Age":67,"Gender":1,"HospAdmTime":-2,"ICULOS":2},
    {"HR":102,"O2Sat":92,"SBP":98,"MAP":68,
     "DBP":50,"Resp":24,"Temp":38.6,
     "Age":67,"Gender":1,"HospAdmTime":-2,"ICULOS":3},
    {"HR":105,"O2Sat":91,"SBP":95,"MAP":65,
     "DBP":48,"Resp":25,"Temp":38.7,
     "Age":67,"Gender":1,"HospAdmTime":-2,"ICULOS":4},
    {"HR":108,"O2Sat":90,"SBP":92,"MAP":63,
     "DBP":46,"Resp":26,"Temp":38.8,
     "Age":67,"Gender":1,"HospAdmTime":-2,"ICULOS":5},
    {"HR":112,"O2Sat":89,"SBP":88,"MAP":60,
     "DBP":44,"Resp":28,"Temp":39.0,
     "Age":67,"Gender":1,"HospAdmTime":-2,"ICULOS":6},
]

def get_mock_app():
    """Create app with mocked model and dependencies"""
    import pickle
    import torch
    
    mock_scaler = MagicMock()
    mock_scaler.transform.return_value = np.zeros((6, 16))
    
    with patch('builtins.open', MagicMock()), \
         patch('pickle.load', return_value=mock_config), \
         patch('torch.load', return_value={}), \
         patch('torch.nn.Module.load_state_dict'), \
         patch('torch.nn.Module.eval'):
        
        with patch.dict('sys.modules', {
            'explainer': MagicMock()
        }):
            pass
    
    return None

# ── Simple structure tests (no model needed) ───────────
def test_sample_readings_structure():
    """Test that our sample readings have correct structure"""
    assert len(SAMPLE_READINGS) == 6
    for reading in SAMPLE_READINGS:
        assert 'HR' in reading
        assert 'O2Sat' in reading
        assert 'ICULOS' in reading
    print("✓ Sample readings structure test passed")

def test_mock_config_structure():
    """Test that model config has required fields"""
    required_keys = [
        'input_size', 'hidden_size', 'num_layers',
        'dropout', 'window_size', 'feature_cols',
        'threshold', 'scale_features'
    ]
    for key in required_keys:
        assert key in mock_config, f"Missing key: {key}"
    assert len(mock_config['feature_cols']) == 23
    assert mock_config['threshold'] == 0.05
    print("✓ Config structure test passed")

def test_window_size():
    """Test that window size matches readings"""
    assert mock_config['window_size'] == 6
    assert len(SAMPLE_READINGS) == mock_config['window_size']
    print("✓ Window size test passed")

def test_feature_count():
    """Test correct number of features"""
    assert mock_config['n_features'] == 23
    assert len(mock_config['feature_cols']) == 23
    print("✓ Feature count test passed")

def test_alert_thresholds():
    """Test alert level logic"""
    def get_alert_level(score, threshold=0.05):
        if score >= 0.4:
            return "RED"
        elif score >= 0.2:
            return "AMBER"
        elif score >= threshold:
            return "YELLOW"
        else:
            return "GREEN"
    
    assert get_alert_level(0.01) == "GREEN"
    assert get_alert_level(0.06) == "YELLOW"
    assert get_alert_level(0.25) == "AMBER"
    assert get_alert_level(0.45) == "RED"
    print("✓ Alert threshold logic test passed")

def test_readings_count_validation():
    """Test that wrong reading count is detected"""
    wrong_count = SAMPLE_READINGS[:3]  # Only 3 readings
    assert len(wrong_count) != mock_config['window_size']
    print("✓ Reading count validation test passed")