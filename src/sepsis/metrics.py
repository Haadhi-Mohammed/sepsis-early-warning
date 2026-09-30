"""Threshold selection and evaluation metrics."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

# Alert tiers. YELLOW is the decision threshold that maximises the challenge
# utility on validation. AMBER and RED are stricter levels that keep the 50%
# and 25% most confident true alerts (by validation sensitivity), so a
# higher tier means fewer, surer alerts.
ESCALATION_SENSITIVITY = {'AMBER': 0.50, 'RED': 0.25}


def threshold_for_sensitivity(y: np.ndarray, p: np.ndarray,
                              target: float) -> float:
    """Highest threshold t such that sensitivity(p >= t) >= target."""
    pos = np.sort(p[y == 1])[::-1]
    if len(pos) == 0:
        raise ValueError('No positive examples to choose a threshold from')
    k = int(np.ceil(target * len(pos)))
    return float(pos[max(k, 1) - 1])


def alert_thresholds(y: np.ndarray, p: np.ndarray,
                     decision: float) -> dict[str, float]:
    thresholds = {'YELLOW': float(decision)}
    floor = decision
    for tier, sens in ESCALATION_SENSITIVITY.items():
        floor = max(floor, threshold_for_sensitivity(y, p, sens))
        thresholds[tier] = float(floor)
    return thresholds


def at_threshold(y: np.ndarray, p: np.ndarray, t: float) -> dict[str, float]:
    pred = p >= t
    tp = int((pred & (y == 1)).sum())
    fp = int((pred & (y == 0)).sum())
    fn = int((~pred & (y == 1)).sum())
    tn = int((~pred & (y == 0)).sum())
    return {
        'threshold':    float(t),
        'sensitivity':  tp / max(tp + fn, 1),
        'specificity':  tn / max(tn + fp, 1),
        'ppv':          tp / max(tp + fp, 1),
        'flagged_rate': float(pred.mean()),
        'tp': tp, 'fp': fp, 'tn': tn, 'fn': fn,
    }


def patient_level(y: np.ndarray, p: np.ndarray, patient: np.ndarray,
                  t: float, hours_to_sepsis: np.ndarray) -> dict[str, float]:
    """
    Per-patient view at threshold t (hours_to_sepsis from UtilityScorer):
      septic_warned_early - share of septic patients alerted at least once in
                            the 12 hours BEFORE clinical onset (t_sepsis)
      septic_warned       - share alerted at any point in their positive window
      nonseptic_alarm     - share of never-septic patients alerted at least once
    """
    pred = p >= t
    septic = np.unique(patient[y == 1])
    early = (hours_to_sepsis >= -12) & (hours_to_sepsis < 0)
    warned_early = np.unique(patient[early & pred])
    warned = np.unique(patient[(y == 1) & pred])
    nonseptic = np.setdiff1d(np.unique(patient), septic)
    alarmed = np.intersect1d(np.unique(patient[pred]), nonseptic)
    n_septic = max(len(septic), 1)
    return {
        'septic_patients':     int(len(septic)),
        'septic_warned_early': len(warned_early) / n_septic,
        'septic_warned':       len(warned) / n_septic,
        'nonseptic_patients':  int(len(nonseptic)),
        'nonseptic_alarm':     len(alarmed) / max(len(nonseptic), 1),
    }


def summary(y: np.ndarray, p: np.ndarray) -> dict[str, float]:
    return {
        'auroc':      float(roc_auc_score(y, p)),
        'auprc':      float(average_precision_score(y, p)),
        'prevalence': float(y.mean()),
        'n':          int(len(y)),
    }
