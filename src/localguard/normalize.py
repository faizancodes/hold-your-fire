"""Normalize raw trajectory rows into ordered StepEvent lists (Phase 3).

The Nebius ``trajectory`` field may arrive as:
  * a JSON string or a python list;
  * a list of *turn* dicts (each with thought/action/observation keys); or
  * a list of *chat message* dicts (alternating role/content) that must be
    paired into (assistant action -> following observation) turns; or
  * a dict wrapping one of the above under a ``trajectory``/``history`` key.

We detect the shape and reduce everything to one StepEvent per agent turn.
"""

from __future__ import annotations

import hashlib
import json
import re
from typing import Any

from .action_parser import (
    classify_action,
    detect_exception,
    detect_traceback,
    parse_test_counts,
)
from .schemas import NormalizedTrajectory, RawTrajectoryRow, StepEvent
from .utils import stable_hash

# An assistant message in SWE-agent style usually ends with a fenced command
# block; the prose before it is the "thought".
_FENCE_RE = re.compile(r"```(?:bash|sh|python|py)?\s*\n(.*?)```", re.DOTALL)
_THOUGHT_LABEL_RE = re.compile(r"(?is)\b(?:thought|discussion)\s*:?\s*(.*?)(?:\baction\s*:|\Z)")
_ACTION_LABEL_RE = re.compile(r"(?is)\baction\s*:?\s*(.*)\Z")


def _coerce_target(value: Any) -> bool:
    """Coerce the heterogeneous ``target`` field to a success boolean."""
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        return value.strip().lower() in {"true", "1", "yes", "resolved", "success", "pass", "passed"}
    return bool(value)


def _load_trajectory_payload(traj: str | list | dict) -> list[dict[str, Any]]:
    """Return a flat list of step/message dicts from any supported shape."""
    if isinstance(traj, str):
        traj = traj.strip()
        if not traj:
            return []
        try:
            traj = json.loads(traj)
        except json.JSONDecodeError:
            # Not JSON: treat the whole blob as a single observation-ish step.
            return [{"observation": traj}]
    if isinstance(traj, dict):
        for key in ("trajectory", "history", "steps", "messages"):
            if isinstance(traj.get(key), list):
                traj = traj[key]
                break
        else:
            return [traj]
    if isinstance(traj, list):
        return [x for x in traj if isinstance(x, dict)]
    return []


def _split_thought_action(content: str) -> tuple[str, str]:
    """Split a free-form assistant message into (thought, action)."""
    if not content:
        return "", ""
    # Prefer an explicit fenced command block as the action.
    fences = _FENCE_RE.findall(content)
    if fences:
        action = fences[-1].strip()
        thought = _FENCE_RE.sub("", content).strip()
        return thought, action
    # Fall back to THOUGHT:/ACTION: labels.
    a = _ACTION_LABEL_RE.search(content)
    if a:
        action = a.group(1).strip()
        t = _THOUGHT_LABEL_RE.search(content)
        thought = t.group(1).strip() if t else content[: a.start()].strip()
        return thought, action
    # No structure: the message is its own action (and thought).
    return content.strip(), content.strip()


def _is_turn_shape(items: list[dict[str, Any]]) -> bool:
    return any(("action" in it) or ("observation" in it) for it in items)


def _msg_text(msg: dict[str, Any]) -> str:
    """Message body across schemas: Nebius uses ``text``; chat APIs use ``content``."""
    return str(msg.get("content") or msg.get("text") or "")


def _step_from_turn(
    item: dict[str, Any], *, trajectory_id: str, instance_id: str,
    model_name: str | None, step_index: int,
) -> StepEvent:
    thought = str(item.get("thought") or "")
    action = str(item.get("action") or item.get("response") or "")
    observation = str(item.get("observation") or "")
    role = item.get("role")
    raw = str(item.get("response") or item.get("content") or action)
    return _finalize_step(
        trajectory_id, instance_id, model_name, step_index,
        role, raw, thought, action, observation,
        explicit_rc=item.get("returncode"),
    )


def _finalize_step(
    trajectory_id: str, instance_id: str, model_name: str | None, step_index: int,
    role: str | None, raw: str, thought: str, action: str, observation: str,
    explicit_rc: int | None = None,
) -> StepEvent:
    pa = classify_action(action)
    pass_count, fail_count = parse_test_counts(observation)
    return StepEvent(
        trajectory_id=trajectory_id,
        instance_id=instance_id,
        model_name=model_name,
        step_index=step_index,
        role=role,
        raw_text=raw[:8000],
        thought_text=thought[:8000],
        action_text=action[:8000],
        observation_text=observation[:8000],
        action_type=pa.action_type,
        command=pa.command,
        file_paths=pa.file_paths,
        returncode=explicit_rc,
        is_test_command=pa.is_test_command,
        is_search_command=pa.is_search_command,
        is_read_command=pa.is_read_command,
        is_edit_command=pa.is_edit_command,
        is_git_command=pa.is_git_command,
        is_install_command=pa.is_install_command,
        is_submit_command=pa.is_submit_command,
        test_pass_count=pass_count,
        test_fail_count=fail_count,
        contains_traceback=detect_traceback(observation),
        contains_exception=detect_exception(observation),
    )


def _steps_from_messages(
    items: list[dict[str, Any]], *, trajectory_id: str, instance_id: str,
    model_name: str | None,
) -> list[StepEvent]:
    """Pair assistant messages with their following observation message."""
    steps: list[StepEvent] = []
    idx = 0
    i = 0
    n = len(items)
    while i < n:
        msg = items[i]
        role = (msg.get("role") or "").lower()
        content = _msg_text(msg)
        if role in ("assistant", "ai"):
            thought, action = _split_thought_action(content)
            observation = ""
            # consume following non-assistant message(s) as the observation
            j = i + 1
            obs_parts: list[str] = []
            while j < n and (items[j].get("role") or "").lower() not in ("assistant", "ai"):
                obs_parts.append(_msg_text(items[j]))
                j += 1
            observation = "\n".join(p for p in obs_parts if p)
            steps.append(
                _finalize_step(
                    trajectory_id, instance_id, model_name, idx,
                    "assistant", content, thought, action, observation,
                )
            )
            idx += 1
            i = j
        else:
            # leading system/user message with no preceding assistant turn
            i += 1
    return steps


def normalize_row(raw: dict[str, Any] | RawTrajectoryRow) -> NormalizedTrajectory:
    """Convert one raw dataset row into a :class:`NormalizedTrajectory`."""
    if isinstance(raw, RawTrajectoryRow):
        row = raw
    else:
        # tolerate target arriving as int/str
        data = dict(raw)
        if "target" in data:
            data["target"] = _coerce_target(data["target"])
        row = RawTrajectoryRow(**data)

    instance_id = row.instance_id
    model_name = row.model_name
    # The dataset stores MANY distinct rollouts per (instance_id, model_name), so
    # the id must also depend on trajectory *content*; otherwise distinct runs
    # collapse to one id and per-trajectory evaluation silently merges them.
    traj_text = (
        row.trajectory
        if isinstance(row.trajectory, str)
        else json.dumps(row.trajectory, ensure_ascii=False, default=str)
    )
    content_sig = hashlib.sha1(traj_text.encode("utf-8", "ignore")).hexdigest()[:10]
    trajectory_id = stable_hash(instance_id, model_name or "?", content_sig)

    items = _load_trajectory_payload(row.trajectory)
    if _is_turn_shape(items):
        steps = [
            _step_from_turn(
                it, trajectory_id=trajectory_id, instance_id=instance_id,
                model_name=model_name, step_index=k,
            )
            for k, it in enumerate(items)
        ]
    else:
        steps = _steps_from_messages(
            items, trajectory_id=trajectory_id, instance_id=instance_id,
            model_name=model_name,
        )

    return NormalizedTrajectory(
        trajectory_id=trajectory_id,
        instance_id=instance_id,
        model_name=model_name,
        target=_coerce_target(row.target),
        steps=steps,
        n_steps=len(steps),
    )


def normalize_rows(rows: list[dict[str, Any]]) -> list[NormalizedTrajectory]:
    out: list[NormalizedTrajectory] = []
    for r in rows:
        try:
            out.append(normalize_row(r))
        except Exception as exc:  # keep going; report at the call site
            out.append(
                NormalizedTrajectory(
                    trajectory_id=stable_hash(r.get("instance_id", "?"), "err"),
                    instance_id=str(r.get("instance_id", "unknown")),
                    model_name=r.get("model_name"),
                    target=_coerce_target(r.get("target", False)),
                    steps=[],
                    n_steps=0,
                )
            )
            _ = exc
    return out
