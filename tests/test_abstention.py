"""Selective-prediction / abstention gate + monitor wiring."""

import numpy as np

from localguard.abstention import confidence, is_committed, random_curve, selective_curve
from localguard.calibrate import Calibrator
from localguard.monitor import Monitor, PolicyConfig
from localguard.schemas import StepEvent


def test_confidence_peaks_away_from_half():
    c = confidence(np.array([0.5, 0.9, 0.1, 0.7]))
    assert c[0] == 0.0
    assert abs(c[1] - 0.4) < 1e-9 and abs(c[2] - 0.4) < 1e-9


def test_is_committed_requires_both_gates():
    assert not is_committed(prefix_step=3, calibrated_risk=0.95, step_floor=10, conf_floor=0.2)  # too early
    assert not is_committed(prefix_step=12, calibrated_risk=0.52, step_floor=10, conf_floor=0.2)  # unsure
    assert is_committed(prefix_step=12, calibrated_risk=0.95, step_floor=10, conf_floor=0.2)       # judgeable


def test_selective_curve_beats_random_with_signal():
    """Confident tails are reliable; the middle (p~0.5) is genuine noise — so
    confidence-selective abstention should lift AUC while random does not."""
    rng = np.random.default_rng(0)
    n = 6000
    p = rng.random(n)
    conf = np.abs(p - 0.5) * 2.0                 # 0 at p=0.5, 1 at the tails
    correct_prob = 0.5 + 0.5 * conf              # random in the middle, reliable at tails
    ideal = (p > 0.5).astype(int)
    flip = rng.random(n) > correct_prob
    y = np.where(flip, 1 - ideal, ideal)
    sel = {round(d["coverage"], 1): d["auc"] for d in selective_curve(y, p)}
    rnd = {round(d["coverage"], 1): d["auc"] for d in random_curve(y, p)}
    # confidence-selective clearly beats random abstention at 40% coverage
    assert sel.get(0.4, 0) > rnd.get(0.4, 1) + 0.05
    # random abstention barely changes AUC vs full coverage
    assert abs(rnd.get(0.4, 0) - rnd.get(1.0, 0)) < 0.03


class _FixedModel:
    kind = "majority"

    def __init__(self, risk):
        self._r = risk

    def predict_proba_fail(self, df):
        return np.array([self._r])


def _steps(n):
    return [StepEvent(trajectory_id="t", instance_id="i", step_index=k, action_text="cat a.py")
            for k in range(n)]


def test_monitor_abstains_when_too_early():
    mon = Monitor(_FixedModel(0.95), Calibrator(method="identity"),
                  PolicyConfig(min_step=5, threshold=0.7, abstain_conf_floor=0.1))
    v = mon.assess(_steps(3))
    assert v.abstain and not v.alarm


def test_monitor_abstains_when_unconfident():
    mon = Monitor(_FixedModel(0.52), Calibrator(method="identity"),
                  PolicyConfig(min_step=5, threshold=0.7, abstain_conf_floor=0.1))
    v = mon.assess(_steps(10))
    assert v.abstain and not v.alarm


def test_monitor_commits_when_confident_and_late():
    mon = Monitor(_FixedModel(0.95), Calibrator(method="identity"),
                  PolicyConfig(min_step=5, threshold=0.7, abstain_conf_floor=0.1))
    v = mon.assess(_steps(10))
    assert not v.abstain and v.alarm
