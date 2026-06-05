"""First-alert evaluation + threshold policies + ECE (Phase 8)."""

import numpy as np
import pandas as pd

from localguard.evaluate import expected_calibration_error
from localguard.thresholding import (
    first_alert_metrics,
    first_alert_per_trajectory,
    threshold_for_success_false_alarm,
)


def _toy_predictions():
    # two failed trajectories (risk rises), two successful (risk low)
    rows = []
    # failed traj f1: risk crosses 0.6 at step 3
    for step, risk in [(1, 0.2), (2, 0.4), (3, 0.7), (4, 0.9)]:
        rows.append(dict(trajectory_id="f1", n_total_steps=4, y_fail=1, prefix_step=step, risk=risk))
    for step, risk in [(1, 0.3), (2, 0.5), (3, 0.65), (4, 0.8)]:
        rows.append(dict(trajectory_id="f2", n_total_steps=4, y_fail=1, prefix_step=step, risk=risk))
    # successful trajectories stay low
    for step, risk in [(1, 0.1), (2, 0.2), (3, 0.2), (4, 0.3)]:
        rows.append(dict(trajectory_id="s1", n_total_steps=4, y_fail=0, prefix_step=step, risk=risk))
    for step, risk in [(1, 0.1), (2, 0.15), (3, 0.25), (4, 0.35)]:
        rows.append(dict(trajectory_id="s2", n_total_steps=4, y_fail=0, prefix_step=step, risk=risk))
    return pd.DataFrame(rows)


def test_first_alert_lead_time():
    df = _toy_predictions()
    per = first_alert_per_trajectory(df, threshold=0.6)
    f1 = per[per["trajectory_id"] == "f1"].iloc[0]
    assert f1["first_alarm_step"] == 3
    assert f1["lead_steps"] == 1  # 4 total - 3 = 1


def test_first_alert_metrics():
    df = _toy_predictions()
    m = first_alert_metrics(df, threshold=0.6)
    assert m.n_failed == 2 and m.n_success == 2
    assert m.failed_coverage == 1.0           # both failures alerted
    assert m.success_false_alarm_rate == 0.0  # no success alerted at 0.6


def test_threshold_for_success_far_respects_budget():
    df = _toy_predictions()
    thr = threshold_for_success_false_alarm(df, max_far=0.0)
    m = first_alert_metrics(df, thr)
    assert m.success_false_alarm_rate <= 0.0 + 1e-9


def test_ece_perfect_calibration():
    rng = np.random.default_rng(0)
    p = rng.random(5000)
    y = (rng.random(5000) < p).astype(int)  # perfectly calibrated by construction
    assert expected_calibration_error(y, p) < 0.05


def test_ece_miscalibrated_high():
    p = np.full(1000, 0.9)
    y = np.zeros(1000, dtype=int)  # predicts 0.9 but always negative
    assert expected_calibration_error(y, p) > 0.5
