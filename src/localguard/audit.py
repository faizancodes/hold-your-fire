"""Failure-mode taxonomy + heuristic labeler (Phase 18), importable + testable.

The heuristic assigns a failure mode from prefix-visible FEATURE values. It is
used both for the automatic audit and as the thing a human read validates against.
"""

from __future__ import annotations

from typing import Any

FAILURE_MODES = [
    "insufficient_context", "wrong_file", "test_neglect", "looping",
    "patch_churn", "environment_distraction", "submission_too_early", "not_observable",
]

# One-line rubric definitions (used by the human annotator and in the writeup).
RUBRIC = {
    "looping": "repeats the same/near-identical command or action with no new evidence",
    "insufficient_context": "edits/acts before reading or searching enough to understand the bug",
    "wrong_file": "works in/on a file that is not where the bug lives",
    "test_neglect": "edits but never runs a test/reproduction to verify",
    "patch_churn": "keeps changing code; tests not improving or getting worse",
    "environment_distraction": "stuck on env/setup (install, path, command-not-found) not the bug",
    "submission_too_early": "submits/finishes without adequately solving",
    "not_observable": "nothing clearly wrong is visible in the prefix yet",
}


def heuristic_failure_mode(f: dict[str, Any]) -> str:
    """Assign a failure mode from prefix features (f__-prefixed or bare keys)."""
    def g(k: str, d: float = 0.0) -> float:
        v = f.get(f"f__{k}", f.get(k, d))
        try:
            return float(v)
        except (TypeError, ValueError):
            return d

    if (g("max_command_repeat_count") >= 3 or g("repeated_exact_command_last_3") > 0
            or g("same_action_type_streak") >= 5):
        return "looping"
    if g("tests_worsening") > 0 or g("same_test_command_repeated") >= 2:
        return "patch_churn"
    if g("edited_before_any_read") > 0 or g("edited_file_never_read_count") > 0:
        return "insufficient_context"
    if g("n_edit") > 0 and g("n_test_runs") == 0:
        return "test_neglect"
    if g("n_submit") > 0 and g("n_edit") == 0:
        return "submission_too_early"
    if g("n_install") >= 2 or g("n_environment") >= g("n_edit") + g("n_read") + 1:
        return "environment_distraction"
    if g("n_unique_files_edited") >= 1 and g("n_unique_files_read") == 0:
        return "wrong_file"
    return "not_observable"
