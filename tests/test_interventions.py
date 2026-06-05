"""Intervention triggers, selection, messages, and monitor policy (Phase 12)."""

from localguard.interventions import (
    DEFAULT_LOOP_PENALTY,
    EVIDENCE_GATE,
    LOOP_BREAK,
    LOOP_GUARD,
    MESSAGES,
    NONE,
    ROLLBACK_SUGGEST,
    evidence_gate_triggered,
    forced_intervention,
    loop_triggered,
    most_repeated_command,
    rollback_triggered,
    select_intervention,
    targeted_loop_break,
)
from localguard.monitor import PolicyConfig, should_alarm


def test_loop_trigger():
    assert loop_triggered({"repeated_exact_command_last_3": 1})
    assert loop_triggered({"max_command_repeat_count": 3})
    assert not loop_triggered({"max_command_repeat_count": 1})


def test_evidence_gate_trigger():
    assert evidence_gate_triggered({"n_edit": 2, "edited_before_any_read": 1})
    assert evidence_gate_triggered({"n_edit": 1, "edited_file_never_read_count": 1})
    assert not evidence_gate_triggered({"n_edit": 0, "edited_before_any_read": 1})


def test_rollback_requires_checkpoint():
    feats = {"tests_worsening": 1}
    assert not rollback_triggered(feats, has_checkpoint=False)
    assert rollback_triggered(feats, has_checkpoint=True)


def test_select_intervention_priority():
    # very-high risk + worsening + checkpoint => rollback
    d = select_intervention({"tests_worsening": 1}, risk=0.9, has_checkpoint=True)
    assert d.kind == ROLLBACK_SUGGEST
    # high risk + premature edit => evidence gate
    d = select_intervention({"n_edit": 1, "edited_before_any_read": 1}, risk=0.6)
    assert d.kind == EVIDENCE_GATE
    # high risk + loop => loop guard
    d = select_intervention({"max_command_repeat_count": 4}, risk=0.6)
    assert d.kind == LOOP_GUARD
    # low risk => none even if a trigger is active
    d = select_intervention({"max_command_repeat_count": 4}, risk=0.1)
    assert d.kind == NONE


def test_every_intervention_has_message():
    for kind in (LOOP_GUARD, EVIDENCE_GATE, ROLLBACK_SUGGEST):
        assert MESSAGES[kind].strip()


def test_most_repeated_command():
    # the command repeated most in the recent window
    cmd, n = most_repeated_command(["cat a", "find x", "find x", "find x", "ls"])
    assert cmd == "find x" and n == 3
    # nothing repeated >= min_count
    assert most_repeated_command(["a", "b", "c"]) == (None, 0)
    # empty / None
    assert most_repeated_command(None) == (None, 0)
    assert most_repeated_command([]) == (None, 0)


def test_targeted_loop_break_names_command():
    feats = {"max_command_repeat_count": 4}
    cmds = ["find_file SecretStr", "find_file SecretStr", "find_file SecretStr"]
    d = targeted_loop_break(feats, cmds)
    assert d is not None
    assert d.kind == LOOP_BREAK
    # the deployable message names AND forbids the exact repeated command
    assert "find_file SecretStr" in d.message
    assert "Do NOT" in d.message or "do not" in d.message.lower()
    # white-box payload is populated for backends that support logit penalties
    assert d.target_command == "find_file SecretStr"
    assert d.penalty == DEFAULT_LOOP_PENALTY


def test_targeted_loop_break_requires_loop_and_repeat():
    # loop trigger but no single repeated command -> None (falls back to generic upstream)
    assert targeted_loop_break({"max_command_repeat_count": 4}, ["a", "b", "c"]) is None
    # repeated command but no loop trigger -> None
    assert targeted_loop_break({}, ["a", "a", "a"]) is None


def test_select_intervention_prefers_targeted_loop_break():
    feats = {"max_command_repeat_count": 4}
    cmds = ["pytest -k foo", "pytest -k foo", "pytest -k foo"]
    d = select_intervention(feats, risk=0.6, recent_commands=cmds)
    assert d.kind == LOOP_BREAK
    assert "pytest -k foo" in d.message
    assert d.target_command == "pytest -k foo"
    # triggers carry the repeated-command evidence
    assert d.triggers.get("repeated_command") == "pytest -k foo"


def test_select_intervention_backward_compatible_without_recent_commands():
    # no recent_commands -> generic loop guard (existing callers unchanged)
    d = select_intervention({"max_command_repeat_count": 4}, risk=0.6)
    assert d.kind == LOOP_GUARD
    # recent commands present but nothing clearly repeated -> generic loop guard
    d = select_intervention({"max_command_repeat_count": 4}, risk=0.6,
                            recent_commands=["a", "b", "c", "d"])
    assert d.kind == LOOP_GUARD


def test_forced_loop_break():
    feats = {"max_command_repeat_count": 4}
    cmds = ["grep foo", "grep foo", "grep foo"]
    d = forced_intervention(LOOP_BREAK, feats, risk=0.6, has_checkpoint=False, recent_commands=cmds)
    assert d.kind == LOOP_BREAK
    assert d.target_command == "grep foo"
    # forced loop_break with no identifiable command falls back to generic loop guard
    d = forced_intervention(LOOP_BREAK, feats, risk=0.6, has_checkpoint=False,
                            recent_commands=["a", "b", "c"])
    assert d.kind == LOOP_GUARD


def test_policy_refuses_before_min_step():
    cfg = PolicyConfig(min_step=5, threshold=0.5)
    # high risk but too early
    assert not should_alarm(0.99, step=2, last_alarm_step=None, n_interventions=0, cfg=cfg)
    # high risk, past min_step
    assert should_alarm(0.99, step=6, last_alarm_step=None, n_interventions=0, cfg=cfg)


def test_policy_cooldown_and_budget():
    cfg = PolicyConfig(min_step=3, cooldown_steps=5, max_interventions=2, threshold=0.5)
    # within cooldown
    assert not should_alarm(0.9, step=6, last_alarm_step=4, n_interventions=1, cfg=cfg)
    # budget exhausted
    assert not should_alarm(0.9, step=20, last_alarm_step=None, n_interventions=2, cfg=cfg)
