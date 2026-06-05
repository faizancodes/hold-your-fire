"""Leakage guards: forbidden fields, shuffled-label AUC, future-feature (Phase 19)."""

import numpy as np
import pytest
from conftest import make_synthetic_prefix_df

from localguard.evaluate import roc_auc_metric
from localguard.train import train_model
from localguard.utils import assert_no_leakage


def test_assert_no_leakage_rejects_outcome_fields():
    with pytest.raises(ValueError):
        assert_no_leakage({"n_edit": 1, "target": 0})
    with pytest.raises(ValueError):
        assert_no_leakage({"eval_logs_len": 5})
    # clean feature dict passes
    assert_no_leakage({"n_edit": 1, "n_read": 2, "max_command_repeat_count": 3})


def test_shuffled_label_auc_near_half():
    """Training on shuffled labels must yield ~0.5 AUC on average (no leakage).

    A single permutation has high variance when one feature dominates, so we
    average several shuffles: the expectation is exactly 0.5. The leakage failure
    mode we guard against is a *high* shuffled AUC, which averaging exposes.
    """
    from localguard.split import make_splits

    # many instances + many trajectories so shuffled-label AUC concentrates at 0.5
    df = make_synthetic_prefix_df(n_traj=700, signal=1.5, seed=1, n_instances=180)
    splits = make_splits(df, regime="instance", seed=7)  # grouped, no instance overlap
    tr, te = splits.train, splits.test
    y_te = te["y_fail"].to_numpy()

    # real labels => clearly better than chance (there is an injected signal)
    real = train_model("logistic_regression", tr, seed=0)
    real_auc = roc_auc_metric(y_te, real.predict_proba_fail(te))
    assert real_auc > 0.62, real_auc

    # shuffled labels => mean AUC near chance over many permutations
    aucs = [
        roc_auc_metric(y_te, train_model("logistic_regression", tr, seed=s,
                                         shuffle_labels=True).predict_proba_fail(te))
        for s in range(15)
    ]
    mean_auc = float(np.mean(aucs))
    assert abs(mean_auc - 0.5) < 0.07, (mean_auc, aucs)


def test_majority_baseline_is_constant():
    df = make_synthetic_prefix_df(n_traj=60, seed=2)
    m = train_model("baseline_majority", df, seed=0)
    p = m.predict_proba_fail(df)
    assert np.allclose(p, p[0])
