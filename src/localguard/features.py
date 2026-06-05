"""Prefix feature extraction (Phase 5).

``extract_features`` consumes ONLY the prefix steps it is given (steps 0..t) and
returns a flat ``feature_dict``. It never receives the trajectory's total length,
final patch, eval logs, or terminal label, so by construction the features cannot
leak the future. The one diagnostic quantity that *would* need future knowledge
(fraction of total steps seen) is computed elsewhere from metadata, never here.

Feature families (see research plan):
  A length/pace   B action counts   C context-before-edit   D file behavior
  E testing       F loop behavior   G patch/churn (online, via ``extra``)
  H text (a single ``text_blob`` string consumed by a TF-IDF vectorizer)
"""

from __future__ import annotations

import re
from collections import Counter
from typing import Any, Sequence

from .action_parser import is_test_path
from .schemas import StepEvent

_ASSERTION_RE = re.compile(r"\bAssertionError\b")

# The single text column. Kept separate from numeric features by name so the
# training ColumnTransformer can route it to a TF-IDF vectorizer.
TEXT_FEATURE_KEY = "text_blob"

ACTION_TYPES = (
    "read", "search", "edit", "test", "git", "install", "submit", "environment", "other",
)


def _norm_cmd(step: StepEvent) -> str:
    """A normalized command string for loop/repeat detection."""
    base = (step.command or step.action_text or "").strip()
    # collapse whitespace so trivially reformatted repeats still match
    return re.sub(r"\s+", " ", base)[:200]


def _dirname(path: str) -> str:
    return path.rsplit("/", 1)[0] if "/" in path else ""


def _build_text_blob(steps: Sequence[StepEvent], max_chars: int = 4000) -> str:
    """Recent action text + short observation excerpts, capped, newest-last."""
    parts: list[str] = []
    for s in steps:
        act = (s.action_text or "").strip()
        if act:
            parts.append(act[:400])
        obs = (s.observation_text or "").strip()
        if obs:
            parts.append(obs[:200])
    blob = " \n ".join(parts)
    if len(blob) > max_chars:
        blob = blob[-max_chars:]  # keep the most recent context
    return blob


def extract_features(
    steps: Sequence[StepEvent],
    extra: dict[str, Any] | None = None,
) -> dict[str, float | int | str | bool]:
    """Extract the full prefix feature vector from ``steps`` (the prefix only)."""
    f: dict[str, float | int | str | bool] = {}
    n = len(steps)

    # ---- Family A: length / pace -----------------------------------------
    f["prefix_step"] = n
    f["n_actions_seen"] = sum(1 for s in steps if (s.action_text or "").strip())
    f["n_model_tokens_approx"] = sum(
        (len(s.thought_text) + len(s.action_text)) // 4 for s in steps
    )
    obs_chars = sum(len(s.observation_text) for s in steps)
    f["n_observation_chars"] = obs_chars
    f["avg_observation_chars"] = obs_chars / max(1, n)

    # ---- Family B: action counts -----------------------------------------
    type_counts = Counter(s.action_type for s in steps)
    for t in ACTION_TYPES:
        f[f"n_{t}"] = type_counts.get(t, 0)
    n_read = type_counts.get("read", 0)
    n_edit = type_counts.get("edit", 0)
    n_search = type_counts.get("search", 0)
    n_test = type_counts.get("test", 0)
    f["edit_to_read_ratio"] = n_edit / max(1, n_read)
    f["test_to_edit_ratio"] = n_test / max(1, n_edit)
    f["search_to_edit_ratio"] = n_search / max(1, n_edit)

    # ---- Family C/D: context-before-edit + file behavior -----------------
    files_seen: set[str] = set()
    files_read: set[str] = set()
    files_edited: set[str] = set()
    per_file_edit = Counter()
    per_file_read = Counter()
    first_edit_idx = -1
    reads_before_first_edit = 0
    searches_before_first_edit = 0
    seen_any_read = False
    seen_any_search = False
    edited_before_any_read = 0
    edited_before_any_search = 0

    for i, s in enumerate(steps):
        if s.action_type == "read":
            if first_edit_idx < 0:
                reads_before_first_edit += 1
            seen_any_read = True
        elif s.action_type == "search":
            if first_edit_idx < 0:
                searches_before_first_edit += 1
            seen_any_search = True
        elif s.action_type == "edit":
            if first_edit_idx < 0:
                first_edit_idx = i
                if not seen_any_read:
                    edited_before_any_read = 1
                if not seen_any_search:
                    edited_before_any_search = 1
            for p in s.file_paths:
                files_edited.add(p)
                per_file_edit[p] += 1
        if s.action_type == "read":
            for p in s.file_paths:
                files_read.add(p)
                per_file_read[p] += 1
        for p in s.file_paths:
            files_seen.add(p)

    f["first_edit_step"] = (first_edit_idx + 1) if first_edit_idx >= 0 else -1
    f["n_reads_before_first_edit"] = reads_before_first_edit
    f["n_searches_before_first_edit"] = searches_before_first_edit
    f["edited_before_any_read"] = edited_before_any_read
    f["edited_before_any_search"] = edited_before_any_search

    f["n_unique_files_seen"] = len(files_seen)
    f["n_unique_files_read"] = len(files_read)
    f["n_unique_files_edited"] = len(files_edited)
    f["n_unique_dirs_edited"] = len({_dirname(p) for p in files_edited})
    f["n_test_files_touched"] = sum(1 for p in files_seen if is_test_path(p))
    f["n_src_files_touched"] = sum(1 for p in files_seen if not is_test_path(p))
    f["edited_file_never_read_count"] = len(files_edited - files_read)
    f["same_file_edit_count_max"] = max(per_file_edit.values()) if per_file_edit else 0

    # ---- Family E: testing behavior --------------------------------------
    test_fail_series: list[int] = []
    test_pass_series: list[int] = []
    last_test_returncode = -1
    n_test_runs = 0
    test_cmds: list[str] = []
    n_tracebacks = 0
    n_assertion = 0
    for s in steps:
        if s.contains_traceback:
            n_tracebacks += 1
        if _ASSERTION_RE.search(s.observation_text or ""):
            n_assertion += 1
        if s.is_test_command:
            n_test_runs += 1
            test_cmds.append(_norm_cmd(s))
            if s.test_fail_count is not None:
                test_fail_series.append(s.test_fail_count)
            if s.test_pass_count is not None:
                test_pass_series.append(s.test_pass_count)
            # pseudo returncode (offline has no real one): nonzero on failure signal
            if s.returncode is not None:
                last_test_returncode = s.returncode
            else:
                bad = (
                    (s.test_fail_count or 0) > 0
                    or s.contains_traceback
                    or s.contains_exception
                )
                last_test_returncode = 1 if bad else 0

    f["n_test_runs"] = n_test_runs
    f["last_test_returncode"] = last_test_returncode
    f["last_test_fail_count"] = test_fail_series[-1] if test_fail_series else -1
    f["last_test_pass_count"] = test_pass_series[-1] if test_pass_series else -1
    if len(test_fail_series) >= 2:
        delta = test_fail_series[-1] - test_fail_series[-2]
    else:
        delta = 0
    f["test_fail_count_delta"] = delta
    f["tests_improving"] = int(delta < 0)
    f["tests_worsening"] = int(delta > 0)
    f["same_test_command_repeated"] = n_test_runs - len(set(test_cmds))
    f["n_tracebacks_seen"] = n_tracebacks
    f["n_assertion_errors_seen"] = n_assertion

    # ---- Family F: loop behavior -----------------------------------------
    cmds = [_norm_cmd(s) for s in steps]
    types = [s.action_type for s in steps]
    f["repeated_exact_command_last_3"] = int(_has_dup(cmds[-3:]))
    f["repeated_exact_command_last_5"] = int(_has_dup(cmds[-5:]))
    cmd_counts = Counter(c for c in cmds if c)
    f["max_command_repeat_count"] = max(cmd_counts.values()) if cmd_counts else 0
    f["same_action_type_streak"] = _trailing_streak(types)
    f["edit_test_edit_test_loop_count"] = sum(
        1 for a, b in zip(types, types[1:]) if a == "edit" and b == "test"
    )
    f["read_same_file_repeatedly"] = max(per_file_read.values()) if per_file_read else 0

    # ---- Family H: text --------------------------------------------------
    f[TEXT_FEATURE_KEY] = _build_text_blob(steps)

    # ---- Family G: patch/churn (online only; merged from extra) ----------
    if extra:
        for k, v in extra.items():
            f[k] = v

    return f


def _has_dup(seq: Sequence[str]) -> bool:
    seen: set[str] = set()
    for x in seq:
        if not x:
            continue
        if x in seen:
            return True
        seen.add(x)
    return False


def _trailing_streak(types: Sequence[str]) -> int:
    if not types:
        return 0
    last = types[-1]
    streak = 0
    for t in reversed(types):
        if t == last:
            streak += 1
        else:
            break
    return streak


# Columns that are diagnostic-only / non-deployable and must be dropped before
# training a *deployable* monitor (none are produced here, but the training code
# imports this so the contract is explicit and testable).
NON_DEPLOYABLE_FEATURES: frozenset[str] = frozenset()


def numeric_feature_names(feature_dict: dict[str, Any]) -> list[str]:
    """Numeric feature keys (everything except the text column)."""
    return [k for k in feature_dict if k != TEXT_FEATURE_KEY]
