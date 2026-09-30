import numpy as np
import pytest

from sepsis.utility import UtilityScorer


# ── Reference: the official challenge scoring code, ported directly ───────
# https://github.com/physionetchallenges/evaluation-2019 (evaluate_sepsis_score.py)
def official_utility(labels, predictions, dt_early=-12, dt_optimal=-6, dt_late=3.0,
                     max_u_tp=1, min_u_fn=-2, u_fp=-0.05, u_tn=0):
    if np.any(labels):
        is_septic = True
        t_sepsis = np.argmax(labels) - dt_optimal
    else:
        is_septic = False
        t_sepsis = float('inf')
    n = len(labels)
    m_1 = float(max_u_tp) / float(dt_optimal - dt_early)
    b_1 = -m_1 * dt_early
    m_2 = float(-max_u_tp) / float(dt_late - dt_optimal)
    b_2 = -m_2 * dt_late
    m_3 = float(min_u_fn) / float(dt_late - dt_optimal)
    b_3 = -m_3 * dt_optimal
    u = np.zeros(n)
    for t in range(n):
        if t <= t_sepsis + dt_late:
            if is_septic and predictions[t]:
                if t <= t_sepsis + dt_optimal:
                    u[t] = max(m_1 * (t - t_sepsis) + b_1, u_fp)
                elif t <= t_sepsis + dt_late:
                    u[t] = m_2 * (t - t_sepsis) + b_2
            elif not is_septic and predictions[t]:
                u[t] = u_fp
            elif is_septic and not predictions[t]:
                if t <= t_sepsis + dt_optimal:
                    u[t] = 0
                elif t <= t_sepsis + dt_late:
                    u[t] = m_3 * (t - t_sepsis) + b_3
            elif not is_septic and not predictions[t]:
                u[t] = u_tn
    return np.sum(u)


def official_normalised(cohort_labels, cohort_preds):
    obs = best = inaction = 0.0
    for labels, preds in zip(cohort_labels, cohort_preds):
        n = len(labels)
        best_preds = np.zeros(n)
        if np.any(labels):
            t_sepsis = np.argmax(labels) + 6
            best_preds[max(0, t_sepsis - 12):min(t_sepsis + 3 + 1, n)] = 1
        obs      += official_utility(labels, preds)
        best     += official_utility(labels, best_preds)
        inaction += official_utility(labels, np.zeros(n))
    return (obs - inaction) / (best - inaction)


def random_cohort(rng, n_patients=60):
    labels, preds = [], []
    for _ in range(n_patients):
        n = int(rng.integers(8, 60))
        y = np.zeros(n, dtype=int)
        if rng.random() < 0.3:                     # septic: label from some hour to the end
            y[int(rng.integers(0, n)):] = 1
        labels.append(y)
        preds.append((rng.random(n) < 0.2).astype(int))
    return labels, preds


@pytest.mark.parametrize('seed', range(5))
def test_matches_official_scoring(seed):
    labels, preds = random_cohort(np.random.default_rng(seed))
    y = np.concatenate(labels)
    patient = np.concatenate([np.full(len(l), i) for i, l in enumerate(labels)])
    scorer = UtilityScorer(y, patient)
    assert scorer.score(np.concatenate(preds).astype(bool)) == pytest.approx(
        official_normalised(labels, preds), abs=1e-9)


def test_reference_points():
    rng = np.random.default_rng(0)
    labels, _ = random_cohort(rng)
    y = np.concatenate(labels)
    patient = np.concatenate([np.full(len(l), i) for i, l in enumerate(labels)])
    scorer = UtilityScorer(y, patient)
    assert scorer.score(np.zeros(len(y), bool)) == pytest.approx(0.0)   # never alert
    d = scorer.hours_to_sepsis
    perfect = (d >= -12) & (d <= 3)                                     # NaN -> False
    assert scorer.score(perfect) == pytest.approx(1.0)


def test_best_threshold_prefers_informative_scores():
    labels, _ = random_cohort(np.random.default_rng(1))
    y = np.concatenate(labels)
    patient = np.concatenate([np.full(len(l), i) for i, l in enumerate(labels)])
    scorer = UtilityScorer(y, patient)
    probs = np.where(y == 1, 0.9, 0.1) + np.random.default_rng(2).normal(0, 0.01, len(y))
    t, u = scorer.best_threshold(probs)
    assert 0.1 < t < 0.9 and u > 0
