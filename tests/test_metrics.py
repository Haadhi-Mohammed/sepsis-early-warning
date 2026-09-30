import numpy as np
import pytest

from sepsis import metrics


def test_threshold_achieves_target_sensitivity():
    y = np.array([1, 1, 1, 1, 0, 0])
    p = np.array([0.9, 0.8, 0.4, 0.2, 0.5, 0.1])
    t = metrics.threshold_for_sensitivity(y, p, 0.75)
    assert t == 0.4
    assert metrics.at_threshold(y, p, t)['sensitivity'] == 0.75


def test_alert_thresholds_are_ordered():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 1000)
    p = rng.random(1000)
    t = metrics.alert_thresholds(y, p, decision=0.3)
    assert t['YELLOW'] == 0.3
    assert t['YELLOW'] <= t['AMBER'] <= t['RED']
    # AMBER never drops below the decision threshold
    assert metrics.alert_thresholds(y, p, decision=0.99)['AMBER'] == 0.99


def test_threshold_needs_positives():
    with pytest.raises(ValueError):
        metrics.threshold_for_sensitivity(np.zeros(3), np.ones(3), 0.8)


def test_patient_level():
    y       = np.array([1, 1, 0, 0, 0])
    p       = np.array([0.1, 0.9, 0.9, 0.1, 0.1])
    patient = np.array([0, 0, 1, 2, 2])
    hours_to_sepsis = np.array([-7, -6, np.nan, np.nan, np.nan])
    r = metrics.patient_level(y, p, patient, 0.5, hours_to_sepsis)
    assert r['septic_warned'] == 1.0          # patient 0 alerted once
    assert r['septic_warned_early'] == 1.0    # ... 6 hours before onset
    assert r['nonseptic_alarm'] == 0.5        # patient 1 of {1, 2}
