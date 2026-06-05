"""Weak-supervision label construction (cleaner-label direction)."""

import pandas as pd

from localguard.weak_labels import (
    relabel_trouble_gated,
    trouble_indicator,
    weights_downweight_noisy,
)


def _df(rows):
    return pd.DataFrame(rows)


def test_trouble_indicator_fires_on_loops_and_tests():
    df = _df([
        {"f__max_command_repeat_count": 4, "y_fail": 1},   # loop -> trouble
        {"f__n_test_runs": 3, "f__last_test_fail_count": 2, "y_fail": 1},  # persistent fail -> trouble
        {"f__n_tracebacks_seen": 3, "y_fail": 1},          # repeated errors -> trouble
        {"f__n_read": 2, "y_fail": 1},                     # clean -> no trouble
    ])
    t = trouble_indicator(df)
    assert t[0] and t[1] and t[2] and not t[3]


def test_w1_relabels_healthy_early_failures_to_zero():
    df = _df([
        {"f__max_command_repeat_count": 4, "y_fail": 1},  # failed + trouble -> 1
        {"f__n_read": 1, "y_fail": 1},                    # failed + no trouble -> 0 (de-noised)
        {"f__max_command_repeat_count": 4, "y_fail": 0},  # success + trouble -> 0 (recovered)
    ])
    y = relabel_trouble_gated(df)
    assert list(y) == [1, 0, 0]


def test_w2_downweights_only_noisy_positives():
    df = _df([
        {"f__max_command_repeat_count": 4, "y_fail": 1},  # failed + trouble -> full weight
        {"f__n_read": 1, "y_fail": 1},                    # failed + no trouble -> low weight
        {"f__n_read": 1, "y_fail": 0},                    # success -> full weight
    ])
    w = weights_downweight_noisy(df, low=0.3)
    assert w[0] == 1.0 and abs(w[1] - 0.3) < 1e-9 and w[2] == 1.0
