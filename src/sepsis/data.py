"""Loading raw PhysioNet files, patient-level splitting and window building."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .features import window_rows


def load_patient_dir(folder: str | Path, workers: int = 8) -> pd.DataFrame:
    """Load every .psv file in a folder into one frame sorted by patient/hour."""
    files = sorted(Path(folder).glob('*.psv'))
    if not files:
        raise FileNotFoundError(f'No .psv files in {folder}')

    def read(path: Path) -> pd.DataFrame:
        df = pd.read_csv(path, sep='|')
        df['patient_id'] = path.stem
        return df

    with ThreadPoolExecutor(workers) as pool:
        frames = list(pool.map(read, files))
    df = pd.concat(frames, ignore_index=True)
    return df.sort_values(['patient_id', 'ICULOS'], kind='stable',
                          ignore_index=True)


def split_patients(df: pd.DataFrame, val_frac: float = 0.2,
                   seed: int = 42) -> tuple[set[str], set[str]]:
    """
    Patient-level split, stratified by whether the patient ever develops
    sepsis, so no patient's hours end up on both sides.
    """
    septic = df.groupby('patient_id')['SepsisLabel'].max()
    rng = np.random.default_rng(seed)
    train, val = set(), set()
    for flag in (0, 1):
        ids = np.array(sorted(septic.index[septic == flag]))
        rng.shuffle(ids)
        n_val = int(round(len(ids) * val_frac))
        val.update(ids[:n_val])
        train.update(ids[n_val:])
    return train, val


@dataclass
class HourlySplit:
    """Scaled hourly features for one split, grouped by patient."""
    X: np.ndarray            # (hours, features) float32, scaled
    y: np.ndarray            # (hours,) int8 SepsisLabel
    patient_ptr: np.ndarray  # (patients + 1,) row offsets per patient
    patient_ids: np.ndarray  # (patients,)

    def save(self, path: str | Path) -> None:
        np.savez_compressed(path, X=self.X, y=self.y,
                            patient_ptr=self.patient_ptr,
                            patient_ids=self.patient_ids)

    @classmethod
    def load(cls, path: str | Path) -> 'HourlySplit':
        with np.load(path, allow_pickle=False) as d:
            return cls(d['X'], d['y'], d['patient_ptr'], d['patient_ids'])

    def patient_of_row(self) -> np.ndarray:
        return np.repeat(np.arange(len(self.patient_ids)), np.diff(self.patient_ptr))

    def windows(self, window: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        One window ending at every hour (early hours padded by repeating the
        patient's first row).

        Returns (X_windows (hours, window, features), y (hours,),
        patient index (hours,)), in the same row order as the split.
        """
        patient = self.patient_of_row()
        rows = window_rows(np.arange(len(self.y)), self.patient_ptr[patient], window)
        return self.X[rows], self.y.astype(np.float32), patient


def patient_ptr_from(ids: pd.Series) -> tuple[np.ndarray, np.ndarray]:
    """Row offsets for contiguous patient blocks in a sorted frame."""
    ids = ids.to_numpy()
    starts = np.flatnonzero(np.r_[True, ids[1:] != ids[:-1]])
    return np.r_[starts, len(ids)].astype(np.int64), ids[starts].astype(str)
