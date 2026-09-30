import numpy as np
import pandas as pd
import pytest
import torch

from sepsis.features import FEATURE_COLS, RAW_COLS, WINDOW_SIZE, Preprocessor
from sepsis.model import SepsisLSTM, save_bundle


def synthetic_raw(n_patients: int = 4, hours: int = 12, seed: int = 0) -> pd.DataFrame:
    """Small raw PhysioNet-like frame with sparse labs, sorted by patient/hour."""
    rng = np.random.default_rng(seed)
    rows = []
    for p in range(n_patients):
        for h in range(1, hours + 1):
            row = {c: rng.normal(50, 10) for c in RAW_COLS}
            for lab in ('Lactate', 'WBC', 'Creatinine', 'Glucose', 'pH', 'Hgb'):
                if rng.random() < 0.7:
                    row[lab] = np.nan
            row.update(Gender=p % 2, ICULOS=h, patient_id=f'p{p:03d}',
                       SepsisLabel=int(p == 0 and h >= 9))
            rows.append(row)
    return pd.DataFrame(rows)


@pytest.fixture
def raw_df():
    return synthetic_raw()


@pytest.fixture(scope='session')
def bundle_dir(tmp_path_factory):
    """A tiny, randomly initialised model bundle in the real on-disk format."""
    torch.manual_seed(0)
    path = tmp_path_factory.mktemp('bundle')
    preprocessor = Preprocessor.fit(synthetic_raw(), group_col='patient_id')
    model = SepsisLSTM(len(FEATURE_COLS), hidden_size=8, num_layers=1, dropout=0.0)
    config = {
        'model_version': 'test',
        'feature_cols':  FEATURE_COLS,
        'window_size':   WINDOW_SIZE,
        'horizon_hours': 6,
        'architecture':  {'hidden_size': 8, 'num_layers': 1, 'dropout': 0.0},
        'decision_threshold': 0.3,
        'alert_thresholds': {'YELLOW': 0.3, 'AMBER': 0.5, 'RED': 0.7},
        'validation': {'auroc': 0.5, 'utility': 0.1,
                       'at_threshold': {'sensitivity': 0.8, 'specificity': 0.5}},
    }
    background = np.random.default_rng(0).normal(
        size=(10, WINDOW_SIZE, len(FEATURE_COLS))).astype(np.float32)
    save_bundle(path, model, config, preprocessor, background)
    return path
