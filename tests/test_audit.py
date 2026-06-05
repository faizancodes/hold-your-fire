"""Failure-mode heuristic labeler (used by the human-validated observability audit)."""

from localguard.audit import FAILURE_MODES, RUBRIC, heuristic_failure_mode


def test_looping_takes_priority_on_repeats():
    assert heuristic_failure_mode({"f__max_command_repeat_count": 5}) == "looping"
    assert heuristic_failure_mode({"f__repeated_exact_command_last_3": 1}) == "looping"
    assert heuristic_failure_mode({"f__same_action_type_streak": 6}) == "looping"


def test_other_modes_when_no_loop():
    assert heuristic_failure_mode({"f__edited_before_any_read": 1}) == "insufficient_context"
    assert heuristic_failure_mode({"f__n_edit": 2, "f__n_test_runs": 0}) == "test_neglect"
    assert heuristic_failure_mode({"f__n_submit": 1, "f__n_edit": 0}) == "submission_too_early"
    assert heuristic_failure_mode({"f__tests_worsening": 1}) == "patch_churn"


def test_defaults_to_not_observable():
    assert heuristic_failure_mode({}) == "not_observable"


def test_accepts_bare_and_prefixed_keys_and_bad_values():
    # bare key (no f__) and an unparseable value both handled
    assert heuristic_failure_mode({"max_command_repeat_count": 3}) == "looping"
    assert heuristic_failure_mode({"f__max_command_repeat_count": "n/a"}) == "not_observable"


def test_taxonomy_and_rubric_consistent():
    assert set(RUBRIC) == set(FAILURE_MODES)
