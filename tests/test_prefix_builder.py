"""Prefix builder + schedule + leakage gates (Phase 4 verification gate)."""

from conftest import make_trajectory

from localguard.prefix_builder import (
    FEATURE_PREFIX,
    build_prefix_examples,
    prefix_schedule,
)
from localguard.utils import LEAKAGE_FIELDS


def test_prefix_schedule_basic():
    sched = prefix_schedule(40)
    assert sched[0] == 1
    assert sched[-1] == 40
    assert all(1 <= s <= 40 for s in sched)
    assert sched == sorted(sched)
    # every prefix length is unique
    assert len(sched) == len(set(sched))


def test_prefix_schedule_short_and_empty():
    assert prefix_schedule(0) == []
    assert prefix_schedule(1) == [1]
    assert prefix_schedule(3) == [1, 2, 3]


def test_prefix_step_never_exceeds_total():
    traj = make_trajectory(False, [("cat a.py", "x")] * 30)
    for ex in build_prefix_examples(traj):
        assert ex.prefix_step <= ex.n_total_steps


def test_label_is_terminal_outcome():
    fail = make_trajectory(False, [("cat a.py", "x")] * 6)
    ok = make_trajectory(True, [("cat a.py", "x")] * 6)
    assert all(ex.y_fail == 1 for ex in build_prefix_examples(fail))
    assert all(ex.y_fail == 0 for ex in build_prefix_examples(ok))


def test_no_outcome_field_in_features():
    traj = make_trajectory(False, [("pytest", "1 failed, 2 passed")] * 8)
    for ex in build_prefix_examples(traj):
        for key in ex.feature_dict:
            assert not any(bad in key.lower() for bad in LEAKAGE_FIELDS), key
        assert "y_fail" not in ex.feature_dict
        assert "target" not in ex.feature_dict


def test_feature_columns_namespaced():
    from localguard.prefix_builder import prefix_example_to_row

    traj = make_trajectory(True, [("cat a.py", "x")] * 5)
    row = prefix_example_to_row(build_prefix_examples(traj)[0])
    feat_cols = [c for c in row if c.startswith(FEATURE_PREFIX)]
    assert feat_cols
    assert "y_fail" in row and not row.get("f__y_fail")
