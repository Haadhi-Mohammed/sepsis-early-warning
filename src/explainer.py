# src/explainer.py
# SHAP explainability for Sepsis Early Warning System

import shap
import numpy as np
import torch
import torch.nn as nn
import pickle
from pathlib import Path

BASE_DIR  = Path(__file__).parent.parent
MODEL_DIR = BASE_DIR / 'models'

# ── Feature display names ──────────────────────────────
FEATURE_DISPLAY_NAMES = {
    'HR':             'Heart Rate',
    'O2Sat':          'O2 Saturation',
    'SBP':            'Systolic BP',
    'MAP':            'Mean Art. Pressure',
    'DBP':            'Diastolic BP',
    'Resp':           'Respiratory Rate',
    'Temp':           'Temperature',
    'Lactate':        'Lactate',
    'WBC':            'White Blood Cells',
    'Creatinine':     'Creatinine',
    'Glucose':        'Glucose',
    'pH':             'Blood pH',
    'Hgb':            'Haemoglobin',
    'Age':            'Age',
    'Gender':         'Gender',
    'HospAdmTime':    'Hosp Admission Time',
    'ICULOS':         'ICU Hours',
    'Lactate_obs':    'Lactate Recorded',
    'WBC_obs':        'WBC Recorded',
    'Creatinine_obs': 'Creatinine Recorded',
    'Glucose_obs':    'Glucose Recorded',
    'pH_obs':         'pH Recorded',
    'Hgb_obs':        'Hgb Recorded',
}

class SepsisExplainer:
    """
    SHAP-based explainer for the Sepsis LSTM model.
    
    Uses GradientExplainer which works with PyTorch
    neural networks including LSTMs.
    
    For each prediction, returns the top features
    that pushed the risk score up or down.
    """
    
    def __init__(self, model, feature_cols, 
                 background_data=None):
        self.model        = model
        self.feature_cols = feature_cols
        self.explainer    = None
        
        if background_data is not None:
            self._setup_explainer(background_data)
    
    def _setup_explainer(self, background_data):
        """
        Initialise SHAP GradientExplainer with a 
        wrapped model that outputs 2D tensor.
        """
        print("Setting up SHAP explainer...")
    
        # Wrap model to output 2D tensor
        # SHAP GradientExplainer needs (batch, output)
        # Our model outputs (batch,) which causes the error
        class ModelWrapper(nn.Module):
            def __init__(self, model):
                super().__init__()
                self.model = model
        
            def forward(self, x):
                out = self.model(x)
                # Reshape from (batch,) to (batch, 1)
                return out.unsqueeze(1)
    
        self.wrapped_model = ModelWrapper(self.model)
        self.wrapped_model.eval()
    
        background_tensor = torch.FloatTensor(
            background_data
        )
    
        self.explainer = shap.GradientExplainer(
            self.wrapped_model, background_tensor
        )
        print("SHAP explainer ready")
    
    
    def explain(self, sequence):
        """
        Generate SHAP values for a single sequence.
        """
        if self.explainer is None:
            raise ValueError("Explainer not initialised.")
    
        tensor = torch.FloatTensor(
            sequence
        ).unsqueeze(0)
    
        # Get SHAP values
        shap_values = self.explainer.shap_values(tensor)
    
        # Debug - print shapes
        print(f"shap_values type: {type(shap_values)}")
        if isinstance(shap_values, list):
            print(f"shap_values is list, len: {len(shap_values)}")
            for i, sv in enumerate(shap_values):
                print(f"  shap_values[{i}] shape: {np.array(sv).shape}")
        else:
            print(f"shap_values shape: {np.array(shap_values).shape}")
    
        # Get array
        if isinstance(shap_values, list):
            shap_array = np.array(shap_values[0])
        else:
            shap_array = np.array(shap_values)
    
        print(f"shap_array shape after extraction: {shap_array.shape}")
    
        # Mean across all dimensions except features
        feature_shap = shap_array.reshape(-1, len(self.feature_cols)).mean(axis=0)
        print(f"feature_shap shape: {feature_shap.shape}")
    
        # Build results
        results = []
        for i, feat in enumerate(self.feature_cols):
            display = FEATURE_DISPLAY_NAMES.get(feat, feat)
            results.append({
                'feature':      feat,
                'display_name': display,
                'shap_value':   float(feature_shap[i]),
                'abs_value':    abs(float(feature_shap[i]))
            })
    
        results.sort(
            key=lambda x: x['abs_value'], reverse=True
        )
    
        return results
    
    def get_top_factors(self, sequence, top_n=5):
        """
        Get top N factors driving the prediction.
        Returns separate lists for risk-increasing
        and risk-decreasing factors.
        """
        all_shap = self.explain(sequence)
        
        # Split into positive (risk increasing) 
        # and negative (risk decreasing)
        positive = [
            s for s in all_shap 
            if s['shap_value'] > 0
        ][:top_n]
        
        negative = [
            s for s in all_shap 
            if s['shap_value'] < 0
        ][:top_n]
        
        return {
            'risk_increasing': positive,
            'risk_decreasing': negative,
            'all_factors':     all_shap[:top_n]
        }


def load_explainer(model, feature_cols, 
                   n_background=100):
    """
    Load or create SHAP explainer with background data.
    Uses a small sample of test data as background.
    """
    import os
    
    # Try to load saved background data
    background_path = MODEL_DIR / 'shap_background.npy'
    
    if background_path.exists():
        print("Loading saved SHAP background data...")
        background = np.load(str(background_path))
    else:
        print("No background data found.")
        print("Run setup_shap_background() first.")
        return None
    
    explainer = SepsisExplainer(
        model        = model,
        feature_cols = feature_cols,
        background_data = background
    )
    
    return explainer


def setup_shap_background(X_test, n_samples=100):
    """
    Save a small background dataset for SHAP.
    Should be called once during setup.
    """
    # Use random sample of non-sepsis sequences
    # as the reference background
    idx = np.random.choice(
        len(X_test), n_samples, replace=False
    )
    background = X_test[idx]
    
    save_path = MODEL_DIR / 'shap_background.npy'
    np.save(str(save_path), background)
    print(f"Saved {n_samples} background samples to "
          f"{save_path}")
    
    return background