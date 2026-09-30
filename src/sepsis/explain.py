"""Per-prediction SHAP explanations."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

DISPLAY_NAMES = {
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


class _ProbabilityModel(nn.Module):
    """Wraps the logit model so SHAP explains the probability, shape (batch, 1)."""

    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model

    def forward(self, x):
        return torch.sigmoid(self.model(x)).unsqueeze(1)


class SepsisExplainer:
    """
    SHAP GradientExplainer over the probability output.

    A feature's contribution is its SHAP value summed over the window's
    timesteps (averaging would shrink it and let opposite-signed hours
    cancel). Values are in probability units: +0.03 means that feature
    moved the risk up by about 3 percentage points versus the background.
    """

    def __init__(self, model: nn.Module, feature_cols: list[str],
                 background: np.ndarray, nsamples: int = 200):
        import shap   # optional dependency, only needed when explaining

        self.feature_cols = feature_cols
        self.nsamples = nsamples
        wrapped = _ProbabilityModel(model).eval()
        self._explainer = shap.GradientExplainer(
            wrapped, torch.from_numpy(background.astype(np.float32)))

    def contributions(self, window: np.ndarray) -> np.ndarray:
        """window (time, features) -> per-feature contribution (features,)."""
        x = torch.from_numpy(window[None].astype(np.float32))
        values = np.asarray(self._explainer.shap_values(
            x, nsamples=self.nsamples, rseed=0))
        values = values.reshape(window.shape[0], len(self.feature_cols), -1)[..., 0]
        return values.sum(axis=0)

    def top_factors(self, window: np.ndarray, top_n: int = 3) -> dict[str, list[dict]]:
        contrib = self.contributions(window)
        factors = [{
            'feature':      feat,
            'display_name': DISPLAY_NAMES.get(feat, feat),
            'shap_value':   float(v),
            'contribution': f'{v * 100:+.1f} pp',
        } for feat, v in zip(self.feature_cols, contrib)]
        factors.sort(key=lambda f: abs(f['shap_value']), reverse=True)
        return {
            'risk_increasing': [f for f in factors if f['shap_value'] > 0][:top_n],
            'risk_decreasing': [f for f in factors if f['shap_value'] < 0][:top_n],
        }
