"""
Feature engineering shared by training and serving.

Everything that turns raw hourly readings into model input lives here, so
the training pipeline and the API cannot drift apart. The only fitted state
(medians for imputation, mean/std for scaling) is kept in a JSON-serialisable
Preprocessor instead of a pickled sklearn object, so it loads the same way in
any environment.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

VITALS       = ['HR', 'O2Sat', 'SBP', 'MAP', 'DBP', 'Resp', 'Temp']
LABS         = ['Lactate', 'WBC', 'Creatinine', 'Glucose', 'pH', 'Hgb']
DEMOGRAPHICS = ['Age', 'Gender', 'HospAdmTime', 'ICULOS']
OBS_FLAGS    = [f'{lab}_obs' for lab in LABS]

# Raw columns a caller supplies for each hour
RAW_COLS = VITALS + LABS + DEMOGRAPHICS

# Model input, in order
FEATURE_COLS = VITALS + LABS + DEMOGRAPHICS + OBS_FLAGS

# Already 0/1 - left unscaled
UNSCALED = {'Gender', *OBS_FLAGS}

WINDOW_SIZE = 6
CLIP        = 5.0   # scaled values are clipped to +/- CLIP std


def fill_features(raw: pd.DataFrame, medians: dict[str, float],
                  group_col: str | None = None) -> pd.DataFrame:
    """
    Raw hourly readings -> the 23 unscaled model features.

    Rows must be sorted by time (within each patient if group_col is given).
    Imputation is strictly causal: each hour only sees itself and earlier
    hours, which is exactly what is available at prediction time.

      1. <lab>_obs = 1 if that lab was measured this hour
      2. forward-fill within the patient (carry last known value)
      3. anything still missing (never measured so far) -> training median
    """
    raw = raw.reindex(columns=RAW_COLS if group_col is None
                      else RAW_COLS + [group_col])

    obs = pd.DataFrame(
        {f'{lab}_obs': raw[lab].notna().astype(np.float32) for lab in LABS},
        index=raw.index,
    )

    if group_col is None:
        filled = raw[RAW_COLS].ffill()
    else:
        filled = raw.groupby(group_col, sort=False)[RAW_COLS].ffill()
    filled = filled.fillna(value=medians)

    return pd.concat([filled, obs], axis=1)[FEATURE_COLS].astype(np.float32)


def window_rows(ends: np.ndarray, starts: np.ndarray, window: int) -> np.ndarray:
    """
    Row indices (n, window) of the window ending at each row in `ends`.

    Hours before the patient's first row (`starts`) are padded by repeating
    that first row, so a prediction can be made from hour 1 of a record,
    as the challenge requires.
    """
    idx = ends[:, None] + np.arange(-window + 1, 1)
    return np.maximum(idx, starts[:, None])


def last_window(X: np.ndarray, window: int = WINDOW_SIZE) -> np.ndarray:
    """One patient's hourly features (hours, features) -> latest window."""
    rows = window_rows(np.array([len(X) - 1]), np.array([0]), window)[0]
    return X[rows]


@dataclass
class Preprocessor:
    """Fitted imputation + scaling parameters (fit on the training split only)."""
    medians: dict[str, float]
    means:   dict[str, float]
    stds:    dict[str, float]
    feature_cols: list[str] = field(default_factory=lambda: list(FEATURE_COLS))

    @classmethod
    def fit(cls, raw: pd.DataFrame, group_col: str | None = None) -> 'Preprocessor':
        # Medians of values that were actually recorded
        medians = {c: float(raw[c].median()) for c in RAW_COLS}
        medians = {c: (0.0 if np.isnan(v) else v) for c, v in medians.items()}

        filled = fill_features(raw, medians, group_col)
        means, stds = {}, {}
        for c in FEATURE_COLS:
            if c in UNSCALED:
                means[c], stds[c] = 0.0, 1.0
            else:
                std = float(filled[c].std())
                means[c] = float(filled[c].mean())
                stds[c]  = std if std > 1e-6 else 1.0
        return cls(medians, means, stds)

    def transform(self, raw: pd.DataFrame,
                  group_col: str | None = None) -> np.ndarray:
        """Raw hourly readings -> scaled float32 array (hours, n_features)."""
        filled = fill_features(raw, self.medians, group_col)
        mean = np.array([self.means[c] for c in self.feature_cols], np.float32)
        std  = np.array([self.stds[c]  for c in self.feature_cols], np.float32)
        scaled = (filled[self.feature_cols].to_numpy(np.float32) - mean) / std
        return np.clip(scaled, -CLIP, CLIP)

    def to_json(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps({
            'medians': self.medians, 'means': self.means,
            'stds': self.stds, 'feature_cols': self.feature_cols,
        }, indent=2))

    @classmethod
    def from_json(cls, path: str | Path) -> 'Preprocessor':
        return cls(**json.loads(Path(path).read_text()))
