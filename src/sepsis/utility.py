"""
PhysioNet/CinC 2019 normalised utility score, vectorised.

Equivalent to compute_prediction_utility / evaluate_sepsis_score in
https://github.com/physionetchallenges/evaluation-2019 (verified against a
direct port in tests/test_utility.py). Relative to t_sepsis = first
SepsisLabel hour + 6, an alert earns:

  up to +1     between t_sepsis - 12 and t_sepsis + 3 (peak at -6)
  -0.05        any other alert (false alarm)
  down to -2   for each missed hour between t_sepsis - 6 and t_sepsis + 3

Normalised so that 1 = perfect alerting and 0 = never alerting.

Because the utility is a sum over hours, each hour's payoff for "alert" and
"no alert" is computed once; scoring any threshold is then a single pass.
"""

from __future__ import annotations

import numpy as np

DT_EARLY, DT_OPTIMAL, DT_LATE = -12, -6, 3.0
MAX_U_TP, MIN_U_FN, U_FP, U_TN = 1.0, -2.0, -0.05, 0.0


class UtilityScorer:
    """Precomputes per-hour payoffs for rows ordered by patient, then time."""

    def __init__(self, y: np.ndarray, patient: np.ndarray):
        n = len(y)
        rows = np.arange(n)
        n_patients = int(patient.max()) + 1 if n else 0

        start = np.full(n_patients, n, dtype=np.int64)
        np.minimum.at(start, patient, rows)
        t = rows - start[patient]                       # hour index within patient

        first_pos = np.full(n_patients, np.inf)
        pos = y == 1
        np.minimum.at(first_pos, patient[pos], t[pos])
        t_sepsis = first_pos[patient] - DT_OPTIMAL      # inf for non-septic
        septic = np.isfinite(t_sepsis)
        d = np.where(septic, t - t_sepsis, 0.0)         # hours relative to t_sepsis

        m_1 = MAX_U_TP / (DT_OPTIMAL - DT_EARLY);  b_1 = -m_1 * DT_EARLY
        m_2 = -MAX_U_TP / (DT_LATE - DT_OPTIMAL);  b_2 = -m_2 * DT_LATE
        m_3 = MIN_U_FN / (DT_LATE - DT_OPTIMAL);   b_3 = -m_3 * DT_OPTIMAL

        in_scope = ~septic | (d <= DT_LATE)
        before_opt = d <= DT_OPTIMAL

        alert = np.where(before_opt, np.maximum(m_1 * d + b_1, U_FP), m_2 * d + b_2)
        miss  = np.where(before_opt, 0.0, m_3 * d + b_3)
        self.u_alert = np.where(septic, np.where(in_scope, alert, 0.0), U_FP)
        self.u_quiet = np.where(septic, np.where(in_scope, miss, 0.0), U_TN)

        # Hours relative to clinical onset (NaN for non-septic patients)
        self.hours_to_sepsis = np.where(septic, d, np.nan)

        best = septic & (d >= DT_EARLY) & (d <= DT_LATE)
        self.best_total     = float(np.where(best, self.u_alert, self.u_quiet).sum())
        self.inaction_total = float(self.u_quiet.sum())

    def score(self, alerts: np.ndarray) -> float:
        observed = float(np.where(alerts, self.u_alert, self.u_quiet).sum())
        return (observed - self.inaction_total) / (self.best_total - self.inaction_total)

    def best_threshold(self, probs: np.ndarray, n_candidates: int = 200) -> tuple[float, float]:
        """Threshold on probs that maximises utility -> (threshold, utility)."""
        candidates = np.unique(np.quantile(probs, np.linspace(0.5, 0.999, n_candidates)))
        scores = [self.score(probs >= c) for c in candidates]
        i = int(np.argmax(scores))
        return float(candidates[i]), float(scores[i])
