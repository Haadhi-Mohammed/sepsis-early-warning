"""
Prediction target: the official PhysioNet/CinC 2019 challenge label.

The dataset already shifts the label 6 hours ahead of clinical sepsis
(Sepsis-3 onset, t_sepsis):

    SepsisLabel = 1  if t >= t_sepsis - 6     (septic patients)
    SepsisLabel = 0  otherwise                (and always for non-septic)

Septic records end around t_sepsis + 3, so each septic patient has about
10 positive hours (t_sepsis - 6 .. t_sepsis + 3). Predicting SepsisLabel at
each hour from current and past data therefore IS the 6-hour early warning
task, and it is scored with the challenge's utility function
(see sepsis.utility).
"""

import numpy as np
import pandas as pd

HORIZON = 6   # hours of warning built into SepsisLabel


def make_labels(df: pd.DataFrame) -> np.ndarray:
    """Return int8 labels aligned with df's rows."""
    return df['SepsisLabel'].fillna(0).to_numpy(np.int8)
