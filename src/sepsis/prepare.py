"""
Data preparation step (runs locally or as a SageMaker Processing job).

  Set A -> patient-level 80/20 split -> train / val
  Set B -> test (never touched until final evaluation)

Outputs to --out-dir:
  train.npz, val.npz, test.npz   scaled hourly features + labels
  preprocessor.json              medians / means / stds fitted on train only
  prepare_summary.json           row and label counts

Usage:
  python -m sepsis.prepare --raw-dir data/raw --out-dir data/processed
"""

import argparse
import json
import os
from pathlib import Path

import numpy as np

from .data import HourlySplit, load_patient_dir, patient_ptr_from, split_patients
from .features import Preprocessor
from .labels import make_labels


def build_split(df, preprocessor) -> HourlySplit:
    ptr, ids = patient_ptr_from(df['patient_id'])
    return HourlySplit(
        X=preprocessor.transform(df, group_col='patient_id'),
        y=make_labels(df),
        patient_ptr=ptr,
        patient_ids=ids,
    )


def summarise(split: HourlySplit) -> dict:
    y = split.y
    return {
        'patients':        int(len(split.patient_ids)),
        'septic_patients': int(len(np.unique(split.patient_of_row()[y == 1]))),
        'hours':           int(len(y)),
        'positive_hours':  int((y == 1).sum()),
    }


def main():
    # Defaults follow SageMaker Processing's conventional mount points
    p = argparse.ArgumentParser()
    p.add_argument('--raw-dir', default=os.environ.get(
        'RAW_DIR', '/opt/ml/processing/input/raw'))
    p.add_argument('--out-dir', default=os.environ.get(
        'OUT_DIR', '/opt/ml/processing/output'))
    p.add_argument('--val-frac', type=float, default=0.2)
    p.add_argument('--seed', type=int, default=42)
    args = p.parse_args()

    raw_dir, out_dir = Path(args.raw_dir), Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print('Loading Set A...')
    set_a = load_patient_dir(raw_dir / 'training_setA')
    print('Loading Set B...')
    set_b = load_patient_dir(raw_dir / 'training_setB')

    train_ids, val_ids = split_patients(set_a, args.val_frac, args.seed)
    train_df = set_a[set_a['patient_id'].isin(train_ids)]
    val_df   = set_a[set_a['patient_id'].isin(val_ids)]

    print('Fitting preprocessor on train split...')
    preprocessor = Preprocessor.fit(train_df, group_col='patient_id')
    preprocessor.to_json(out_dir / 'preprocessor.json')

    summary = {}
    for name, df in [('train', train_df), ('val', val_df), ('test', set_b)]:
        split = build_split(df, preprocessor)
        split.save(out_dir / f'{name}.npz')
        summary[name] = summarise(split)
        print(f'{name}: {summary[name]}')

    (out_dir / 'prepare_summary.json').write_text(json.dumps(summary, indent=2))
    print(f'Done -> {out_dir}')


if __name__ == '__main__':
    main()
