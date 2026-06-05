"""Disruption-aware interventions (Phase 12).

Three low-to-moderate disruption interventions. Each has an explicit trigger over
prefix-visible features, an injected message, and (via the monitor) cooldown +
budget control. The first version only *suggests* rollback; forced rollback is a
later experiment. Triggers are deliberately conservative because the Intervention
Paradox shows over-eager interventions disrupt runs that would have succeeded.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

LOOP_GUARD = "loop_guard"
LOOP_BREAK = "loop_break"
EVIDENCE_GATE = "evidence_gate"
ROLLBACK_SUGGEST = "rollback_suggest"
NONE = "none"

# Default strength for the white-box logit-penalty form of the targeted loop-break
# (the value validated in the mechanistic-interpretability study, mech_interp/).
DEFAULT_LOOP_PENALTY = 8.0

LOOP_GUARD_MESSAGE = """[MONITOR] A possible loop was detected: you appear to be repeating similar actions without new evidence.
Before making another edit:
1. Summarize the exact failing behavior.
2. State what evidence supports your current hypothesis.
3. Identify one NEW file, test, or error message to inspect.
4. Then run one targeted command."""

EVIDENCE_GATE_MESSAGE = """[MONITOR] Edits may be happening before enough evidence has been gathered.
Pause editing. First inspect the relevant implementation and test files. Then state:
1. the likely root cause,
2. the smallest code region involved,
3. the test or reproduction command that will verify the fix."""

ROLLBACK_MESSAGE = """[MONITOR] The latest edits may have worsened the trajectory (more failing tests / growing churn).
Consider rolling back to the previous checkpoint and trying a smaller patch. Before continuing:
1. inspect the latest test failure,
2. compare it to the previous failure,
3. decide whether to revert the last edit."""

MESSAGES = {
    LOOP_GUARD: LOOP_GUARD_MESSAGE,
    EVIDENCE_GATE: EVIDENCE_GATE_MESSAGE,
    ROLLBACK_SUGGEST: ROLLBACK_MESSAGE,
}


@dataclass
class InterventionDecision:
    kind: str                 # none | loop_guard | loop_break | evidence_gate | rollback_suggest
    message: str
    reason: str
    triggers: dict[str, Any]
    # Targeted loop-break payload (set only for ``loop_break``). ``message`` above is the
    # deployable form for message-only backends (e.g. Ollama): it names ``target_command``
    # and forbids it. ``target_command`` + ``penalty`` are the mechanistically-validated
    # form for white-box backends: subtract ``penalty`` from the logits of the repeated
    # command's tokens (mech_interp/: this breaks 100% of real loops, beats generic
    # repetition_penalty / no_repeat_ngram, and is ~zero-disruption because it fires only
    # when the monitor flags an unproductive loop and targets one specific command).
    target_command: str | None = None
    penalty: float = 0.0


def _g(features: Mapping[str, Any], key: str, default: float = 0.0) -> float:
    """Read a feature that may be stored with or without the ``f__`` prefix."""
    if key in features:
        v = features[key]
    elif f"f__{key}" in features:
        v = features[f"f__{key}"]
    else:
        return default
    try:
        return float(v)
    except (TypeError, ValueError):
        return default


def loop_triggered(features: Mapping[str, Any]) -> bool:
    return (
        _g(features, "repeated_exact_command_last_3") > 0
        or _g(features, "max_command_repeat_count") >= 3
        or _g(features, "same_action_type_streak") >= 4
        or _g(features, "edit_test_edit_test_loop_count") >= 2
        or _g(features, "read_same_file_repeatedly") >= 3
    )


def evidence_gate_triggered(features: Mapping[str, Any]) -> bool:
    edited = _g(features, "n_edit") > 0
    return edited and (
        _g(features, "edited_before_any_read") > 0
        or _g(features, "edited_file_never_read_count") > 0
        or (_g(features, "n_reads_before_first_edit") == 0 and _g(features, "n_searches_before_first_edit") == 0)
    )


def rollback_triggered(features: Mapping[str, Any], has_checkpoint: bool) -> bool:
    if not has_checkpoint:
        return False
    worsening = _g(features, "tests_worsening") > 0 or _g(features, "test_fail_count_delta") > 0
    churn = _g(features, "patch_growth_without_test_improvement") > 0
    return worsening or churn


def most_repeated_command(recent_commands: Sequence[str] | None,
                          window: int = 5, min_count: int = 2) -> tuple[str | None, int]:
    """The command repeated most often in the recent window (the one driving the loop)."""
    from collections import Counter
    recent = [str(c).strip() for c in (recent_commands or [])[-window:] if c and str(c).strip()]
    if not recent:
        return None, 0
    cmd, n = Counter(recent).most_common(1)[0]
    return (cmd, n) if n >= min_count else (None, 0)


def loop_break_message(cmd: str, n: int) -> str:
    return (f"[MONITOR] You have run `{cmd}` {n} times with the same result and no progress. "
            f"This is an unproductive loop. Do NOT run `{cmd}` again. Take a different action: "
            f"inspect a NEW file, read the actual error message, or run a different command.")


def targeted_loop_break(features: Mapping[str, Any],
                        recent_commands: Sequence[str] | None) -> InterventionDecision | None:
    """Monitor-gated, *targeted* loop break: if a loop is flagged AND one specific command is
    being repeated, name and forbid exactly that command (and expose it for white-box logit
    suppression). Returns ``None`` when no single repeated command is identifiable.

    This is the deployable form of the mechanistic-interpretability finding (mech_interp/):
    targeting the specific looped command is what actually breaks loops, whereas generic
    repetition penalties fail on strong loops and disrupt productive repetition.
    """
    if not loop_triggered(features):
        return None
    cmd, n = most_repeated_command(recent_commands)
    if not cmd:
        return None
    return InterventionDecision(
        LOOP_BREAK, loop_break_message(cmd, n), f"loop on repeated command (x{n})",
        {"loop": True, "repeated_command": cmd, "count": n},
        target_command=cmd, penalty=DEFAULT_LOOP_PENALTY,
    )


def select_intervention(
    features: Mapping[str, Any],
    risk: float,
    *,
    high_risk: float = 0.5,
    very_high_risk: float = 0.8,
    has_checkpoint: bool = False,
    recent_commands: Sequence[str] | None = None,
) -> InterventionDecision:
    """Pick at most one intervention given risk level + active triggers.

    Priority: rollback (very-high risk + worsening + checkpoint) > evidence gate
    (high risk + premature edits) > loop guard (high risk + looping). Returns a
    ``none`` decision when no trigger fires at the required risk level.
    """
    triggers = {
        "loop": loop_triggered(features),
        "evidence_gate": evidence_gate_triggered(features),
        "rollback": rollback_triggered(features, has_checkpoint),
        "risk": risk,
    }

    if risk >= very_high_risk and triggers["rollback"]:
        return InterventionDecision(ROLLBACK_SUGGEST, ROLLBACK_MESSAGE,
                                    "very-high risk + worsening tests/churn + checkpoint", triggers)
    if risk >= high_risk and triggers["evidence_gate"]:
        return InterventionDecision(EVIDENCE_GATE, EVIDENCE_GATE_MESSAGE,
                                    "high risk + edits before sufficient evidence", triggers)
    if risk >= high_risk and triggers["loop"]:
        # Prefer the *targeted* loop break when the trajectory identifies one repeated
        # command (the mechanistically-validated, low-disruption form). Fall back to the
        # generic loop guard when no single command stands out, or when the caller did not
        # provide recent commands (backward compatible).
        targeted = targeted_loop_break(features, recent_commands)
        if targeted is not None:
            targeted.triggers.update(triggers)
            return targeted
        return InterventionDecision(LOOP_GUARD, LOOP_GUARD_MESSAGE,
                                    "high risk + repeated actions / loop", triggers)
    return InterventionDecision(NONE, "", "no trigger at required risk level", triggers)


# For experiment configs that force a specific single intervention family.
def forced_intervention(kind: str, features: Mapping[str, Any], risk: float, has_checkpoint: bool,
                        recent_commands: Sequence[str] | None = None) -> InterventionDecision:
    """Return the named intervention if its trigger fires, else ``none``.

    Used by the per-policy online runs (loop_guard / loop_break / evidence_gate / rollback),
    which each enable only one intervention family.
    """
    if kind == LOOP_BREAK and loop_triggered(features):
        targeted = targeted_loop_break(features, recent_commands)
        if targeted is not None:
            return targeted
        # No single repeated command identifiable -> generic loop guard.
        return InterventionDecision(LOOP_GUARD, LOOP_GUARD_MESSAGE, "loop trigger (no single command)", {"risk": risk})
    if kind == LOOP_GUARD and loop_triggered(features):
        return InterventionDecision(LOOP_GUARD, LOOP_GUARD_MESSAGE, "loop trigger", {"risk": risk})
    if kind == EVIDENCE_GATE and evidence_gate_triggered(features):
        return InterventionDecision(EVIDENCE_GATE, EVIDENCE_GATE_MESSAGE, "evidence-gate trigger", {"risk": risk})
    if kind == ROLLBACK_SUGGEST and rollback_triggered(features, has_checkpoint):
        return InterventionDecision(ROLLBACK_SUGGEST, ROLLBACK_MESSAGE, "rollback trigger", {"risk": risk})
    return InterventionDecision(NONE, "", "forced kind trigger not active", {"risk": risk})
