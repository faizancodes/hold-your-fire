"""Feature extraction correctness + no-future-leakage (Phases 5 & 19)."""

from conftest import make_step, make_trajectory

from localguard.features import TEXT_FEATURE_KEY, extract_features


def test_basic_counts():
    steps = [
        make_step(0, "cat a.py", "code"),
        make_step(1, "rg foo", "match"),
        make_step(2, "sed -i 's/a/b/' a.py", ""),
        make_step(3, "python -m pytest", "1 failed, 2 passed"),
    ]
    f = extract_features(steps)
    assert f["prefix_step"] == 4
    assert f["n_read"] == 1
    assert f["n_search"] == 1
    assert f["n_edit"] == 1
    assert f["n_test"] == 1
    assert f["last_test_fail_count"] == 1
    assert f["last_test_pass_count"] == 2


def test_context_before_edit():
    # edit happens before any read/search
    steps = [make_step(0, "sed -i 's/a/b/' a.py", ""), make_step(1, "cat a.py", "x")]
    f = extract_features(steps)
    assert f["edited_before_any_read"] == 1
    assert f["first_edit_step"] == 1


def test_read_before_edit_not_flagged():
    steps = [make_step(0, "cat a.py", "x"), make_step(1, "sed -i 's/a/b/' a.py", "")]
    f = extract_features(steps)
    assert f["edited_before_any_read"] == 0
    assert f["n_reads_before_first_edit"] == 1


def test_loop_detection():
    steps = [make_step(i, "python -m pytest", "1 failed") for i in range(4)]
    f = extract_features(steps)
    assert f["repeated_exact_command_last_3"] == 1
    assert f["max_command_repeat_count"] >= 3
    assert f["same_action_type_streak"] == 4


def test_text_blob_present_and_string():
    steps = [make_step(0, "cat a.py", "hello world")]
    f = extract_features(steps)
    assert TEXT_FEATURE_KEY in f
    assert isinstance(f[TEXT_FEATURE_KEY], str)


def test_no_future_leakage():
    """Features at prefix k must not change when steps after k are appended."""
    base_actions = [("cat a.py", "x"), ("rg foo", "y"), ("sed -i 's/a/b/' a.py", ""),
                    ("python -m pytest", "1 failed, 1 passed")]
    traj = make_trajectory(False, base_actions + [("git diff", ""), ("submit", "done")])
    k = 4
    f_at_k = extract_features(traj.steps[:k])
    # extend the trajectory with more steps; recompute features at the SAME prefix
    longer = make_trajectory(False, base_actions + [("pytest", "5 failed")] * 10)
    f_at_k_again = extract_features(longer.steps[:k])
    assert f_at_k == f_at_k_again


def test_no_total_length_feature():
    """The deployable feature set must not include the (future) total length."""
    f = extract_features([make_step(0, "cat a.py", "x")])
    assert "n_total_steps" not in f
    assert "fraction_of_total_steps_seen" not in f
    assert not any("total_steps" in k for k in f)
