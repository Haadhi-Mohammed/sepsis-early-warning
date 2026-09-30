"""
Evaluation step: score the held-out test set (PhysioNet Set B) once.

Writes evaluation.json in the shape SageMaker Model Registry expects for
model quality metrics, plus the detailed numbers.

Usage:
  python -m sepsis.evaluate --model-dir models/production \
      --data-dir data/processed --out-dir reports
"""

import argparse
import json
import os
from pathlib import Path

from . import metrics
from .data import HourlySplit
from .model import load_bundle
from .utility import UtilityScorer


def main():
    # Defaults follow SageMaker Processing's conventional mount points
    p = argparse.ArgumentParser()
    p.add_argument('--model-dir', default=os.environ.get('MODEL_DIR', '/opt/ml/processing/model'))
    p.add_argument('--data-dir',  default=os.environ.get('DATA_DIR',  '/opt/ml/processing/test'))
    p.add_argument('--out-dir',   default=os.environ.get('OUT_DIR',   '/opt/ml/processing/evaluation'))
    args = p.parse_args()

    model, config, _, _ = load_bundle(args.model_dir)
    test = HourlySplit.load(Path(args.data_dir) / 'test.npz')
    X, y, patient = test.windows(config['window_size'])
    probs = model.predict_proba(X)

    decision = config['decision_threshold']
    scorer = UtilityScorer(y, patient)
    report = {
        'model_version': config['model_version'],
        'test': {
            **metrics.summary(y, probs),
            'utility': scorer.score(probs >= decision),
            'at_threshold': metrics.at_threshold(y, probs, decision),
            'patient_level': metrics.patient_level(y, probs, patient, decision,
                                                   scorer.hours_to_sepsis),
            'tiers': {tier: metrics.at_threshold(y, probs, t)
                      for tier, t in config['alert_thresholds'].items()},
        },
    }
    test_m = report['test']
    # SageMaker Model Registry "ModelQuality" format
    report['binary_classification_metrics'] = {
        'utility':     {'value': test_m['utility']},
        'auc':         {'value': test_m['auroc']},
        'recall':      {'value': test_m['at_threshold']['sensitivity']},
        'precision':   {'value': test_m['at_threshold']['ppv']},
        'specificity': {'value': test_m['at_threshold']['specificity']},
    }

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / 'evaluation.json').write_text(json.dumps(report, indent=2))

    at = test_m['at_threshold']
    print(f"test_utility={test_m['utility']:.4f} "
          f"test_auroc={test_m['auroc']:.4f} test_auprc={test_m['auprc']:.4f} "
          f"sensitivity={at['sensitivity']:.3f} specificity={at['specificity']:.3f} "
          f"ppv={at['ppv']:.3f} threshold={decision:.4f}")


if __name__ == '__main__':
    main()
