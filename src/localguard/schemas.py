"""Pydantic data schemas for the offline trajectory pipeline (Phase 2).

These types are the contract between ingest -> normalize -> prefix -> features.
``feature_dict`` on :class:`PrefixExample` is the ONLY thing that ever reaches a
model, so it must contain prefix-visible values exclusively (see
``utils.assert_no_leakage``).
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class RawTrajectoryRow(BaseModel):
    """One row of the Nebius SWE-agent-trajectories dataset (tolerant schema)."""

    instance_id: str
    model_name: str | None = None
    # `target` is the terminal success label in the Nebius schema. We accept
    # bool / int / str forms and coerce in the validator below.
    target: bool
    trajectory: str | list[dict[str, Any]]
    exit_status: str | None = None
    generated_patch: str | None = None
    eval_logs: str | None = None

    model_config = {"extra": "ignore"}


class StepEvent(BaseModel):
    """A single normalized step (thought / action / observation) of a run."""

    trajectory_id: str
    instance_id: str
    model_name: str | None = None
    step_index: int

    role: str | None = None
    raw_text: str = ""
    thought_text: str = ""
    action_text: str = ""
    observation_text: str = ""

    action_type: str = "unknown"
    command: str | None = None
    file_paths: list[str] = Field(default_factory=list)
    returncode: int | None = None

    is_test_command: bool = False
    is_search_command: bool = False
    is_read_command: bool = False
    is_edit_command: bool = False
    is_git_command: bool = False
    is_install_command: bool = False
    is_submit_command: bool = False

    test_pass_count: int | None = None
    test_fail_count: int | None = None
    contains_traceback: bool = False
    contains_exception: bool = False


class NormalizedTrajectory(BaseModel):
    """A full run reduced to an ordered list of :class:`StepEvent`."""

    trajectory_id: str
    instance_id: str
    model_name: str | None = None
    target: bool
    steps: list[StepEvent]
    n_steps: int

    def prefix(self, k: int) -> list[StepEvent]:
        """Return steps 0..k (1-indexed prefix length ``k``). No future leak."""
        return self.steps[: max(0, k)]


class PrefixExample(BaseModel):
    """One training/eval row: features visible at ``prefix_step`` + the label.

    ``y_fail`` is a *terminal* label (the whole trajectory failed), attached to a
    partial prefix. It is intentionally the only outcome-derived value here and
    lives outside ``feature_dict``.
    """

    prefix_id: str
    trajectory_id: str
    instance_id: str
    model_name: str | None = None
    prefix_step: int
    n_total_steps: int
    y_fail: int
    feature_dict: dict[str, float | int | str | bool]
