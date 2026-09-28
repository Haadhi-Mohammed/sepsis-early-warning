"""
Prediction service logic, independent of the web framework.

Used by the FastAPI app now and by the SageMaker endpoint's inference
script later, so both serve identical predictions.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .features import RAW_COLS, last_window
from .model import load_bundle

ALERT_MESSAGES = {
    'RED':    'HIGH RISK: immediate clinical review required',
    'AMBER':  'MODERATE RISK: increased monitoring recommended',
    'YELLOW': 'LOW RISK: continue standard monitoring',
    'GREEN':  'MINIMAL RISK: no immediate action required',
}

MAX_HOURS = 336   # longest ICU stay in the training data


class SepsisPredictor:

    def __init__(self, model_dir: str | Path, explain: bool = True):
        self.model, self.config, self.preprocessor, background = load_bundle(model_dir)
        self.window_size = self.config['window_size']
        self.explainer = None
        if explain and background is not None:
            try:
                from .explain import SepsisExplainer
                self.explainer = SepsisExplainer(
                    self.model, self.config['feature_cols'], background)
            except ImportError:
                pass   # shap not installed: predictions still work

    def alert_level(self, risk: float) -> str:
        for tier in ('RED', 'AMBER', 'YELLOW'):
            if risk >= self.config['alert_thresholds'][tier]:
                return tier
        return 'GREEN'

    def prepare(self, readings: list[dict]) -> np.ndarray:
        """
        Hourly readings, oldest first -> scaled model window (time, features).

        Callers should send as much history as they have: labs measured
        before the last 6 hours are carried forward, exactly as in training.
        With fewer than 6 hours the window is padded by repeating the first
        hour, also as in training.
        """
        if not 1 <= len(readings) <= MAX_HOURS:
            raise ValueError(f'Need between 1 and {MAX_HOURS} hourly readings, '
                             f'got {len(readings)}')
        raw = pd.DataFrame(readings).reindex(columns=RAW_COLS).astype(float)

        # Missing ICU hour: continue counting from the previous reading
        iculos = raw['ICULOS'].to_numpy()
        for i in range(len(iculos)):
            if np.isnan(iculos[i]):
                iculos[i] = iculos[i - 1] + 1 if i else 1.0
        raw['ICULOS'] = iculos

        return last_window(self.preprocessor.transform(raw), self.window_size)

    def predict(self, readings: list[dict], explain: bool = True) -> dict:
        window = self.prepare(readings)
        risk = float(self.model.predict_proba(window[None])[0])
        level = self.alert_level(risk)

        factors = {'risk_increasing': [], 'risk_decreasing': []}
        if explain and self.explainer is not None:
            factors = self.explainer.top_factors(window)

        return {
            'risk_score':         round(risk, 4),
            'alert_level':        level,
            'alert_message':      ALERT_MESSAGES[level],
            'sepsis_in_6h':       level != 'GREEN',
            'threshold_used':     self.config['decision_threshold'],
            'hours_of_data':      len(readings),
            'model_version':      self.config['model_version'],
            'top_risk_factors':   factors['risk_increasing'],
            'protective_factors': factors['risk_decreasing'],
        }

    def info(self) -> dict:
        val = self.config['validation']
        return {
            'model_version':      self.config['model_version'],
            'window_size':        self.window_size,
            'horizon_hours':      self.config['horizon_hours'],
            'decision_threshold': self.config['decision_threshold'],
            'alert_thresholds':   self.config['alert_thresholds'],
            'validation_utility': val['utility'],
            'validation_auroc':   val['auroc'],
            'validation_sensitivity': val['at_threshold']['sensitivity'],
            'validation_specificity': val['at_threshold']['specificity'],
            'explanations':       self.explainer is not None,
        }
