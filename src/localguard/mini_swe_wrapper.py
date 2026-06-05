"""Integrate the monitor with mini-SWE-agent for local online runs (Phase 13).

We subclass ``DefaultAgent`` and hook ``step()``:
  * baseline      — no monitor, agent runs untouched.
  * shadow        — after each step the monitor logs a verdict; the agent never
                    sees it, so baseline and shadow are identical at temperature 0.
  * intervention  — on an alarm, the monitor's message is injected as a user turn
                    (and a git checkpoint is taken after edits for rollback).

Uses the text-based LiteLLM model (```mswea_bash_command``` blocks) so local Ollama
models work without tool-calling. No paid API is ever called.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

from .git_checkpoints import diff_stats, has_git, make_checkpoint, patch_features
from .interventions import MESSAGES
from .monitor import Monitor
from .normalize import _finalize_step
from .schemas import StepEvent
from .utils import stable_hash

DEFAULT_OLLAMA_BASE = "http://localhost:11434"


def _ensure_python_shim() -> str:
    """Provide a ``python`` -> ``python3`` shim on PATH (macOS lacks ``python``).

    Returns a bin directory to prepend to the agent's PATH so the agent's own
    reproduce/test commands work regardless of the ``python`` vs ``python3`` split.
    """
    import os
    import stat
    import sys

    bindir = Path.home() / ".cache" / "localguard" / "bin"
    bindir.mkdir(parents=True, exist_ok=True)
    shim = bindir / "python"
    if not shim.exists():
        shim.write_text(f'#!/bin/sh\nexec "{sys.executable}" "$@"\n')
        shim.chmod(shim.stat().st_mode | stat.S_IEXEC | stat.S_IXGRP | stat.S_IXOTH)
    _ = os
    return str(bindir)


def make_local_model(
    model_name: str = "ollama_chat/qwen2.5-coder:7b",
    api_base: str = DEFAULT_OLLAMA_BASE,
    temperature: float = 0.0,
    num_ctx: int = 32768,
):
    """Build a text-based LiteLLM model pointed at the local Ollama endpoint."""
    from minisweagent.models.litellm_textbased_model import LitellmTextbasedModel

    return LitellmTextbasedModel(
        model_name=model_name,
        cost_tracking="ignore_errors",
        model_kwargs={
            "api_base": api_base,
            "temperature": temperature,
            "num_ctx": num_ctx,
        },
    )


def agent_messages_to_steps(
    messages: Sequence[dict], instance_id: str, model_name: str | None = None
) -> list[StepEvent]:
    """Convert live mini-SWE-agent messages into StepEvents (monitor-injected
    messages are skipped so they never pollute prefix features)."""
    trajectory_id = stable_hash(instance_id, model_name or "?", "online")
    steps: list[StepEvent] = []
    idx = 0
    n = len(messages)
    i = 0
    while i < n:
        msg = messages[i]
        role = (msg.get("role") or "").lower()
        extra = msg.get("extra") or {}
        if role == "assistant":
            actions = extra.get("actions") or []
            action_text = "\n".join(a.get("command", "") for a in actions) if actions else ""
            content = str(msg.get("content") or "")
            thought = content
            if not action_text:
                action_text = content
            # gather following non-assistant, non-injected messages as observation
            obs_parts: list[str] = []
            j = i + 1
            while j < n and (messages[j].get("role") or "").lower() not in ("assistant", "exit"):
                if not (messages[j].get("extra") or {}).get("localguard_injected"):
                    obs_parts.append(str(messages[j].get("content") or ""))
                j += 1
            observation = "\n".join(p for p in obs_parts if p)
            steps.append(_finalize_step(
                trajectory_id, instance_id, model_name, idx,
                "assistant", content, thought, action_text, observation,
            ))
            idx += 1
            i = j
        else:
            i += 1
    return steps


@dataclass
class RunResult:
    instance_id: str
    model: str
    policy: str
    mode: str
    success: bool
    n_steps: int
    n_interventions: int
    first_alarm_step: int | None
    exit_status: str
    diff_files_changed: int
    diff_lines_added: int
    diff_lines_deleted: int
    n_test_runs: int
    total_tokens_approx: int
    runtime_s: float
    final_patch: str
    trajectory_path: str
    verdicts: list[dict] = field(default_factory=list)
    injected: list[dict] = field(default_factory=list)

    def summary_row(self) -> dict[str, Any]:
        d = self.__dict__.copy()
        d.pop("verdicts", None)
        d.pop("final_patch", None)
        return d


def _build_agent_class():
    """Defer importing DefaultAgent until mini-swe-agent is needed."""
    from minisweagent.agents.default import DefaultAgent

    class MonitoredAgent(DefaultAgent):
        def __init__(self, model, env, **kwargs):
            super().__init__(model, env, **kwargs)
            # monitoring state — configured via configure() after construction so
            # our attributes never collide with mini-swe-agent's own config keys.
            self.monitor: Monitor | None = None
            self.mode = "baseline"
            self.repo_dir = Path(".")
            self.instance_id = "task"
            self.model_name = "ollama_chat/qwen2.5-coder:7b"
            self.verdicts: list[dict] = []
            self.injected: list[dict] = []
            self.last_alarm_step: int | None = None
            self.n_interventions = 0
            self.prev_patch_stats = None
            self.checkpoint_sha: str | None = None
            self._prev_fail_count: int | None = None

        def configure(self, *, monitor, mode, repo_dir, instance_id, model_name):
            self.monitor = monitor
            self.mode = mode
            self.repo_dir = Path(repo_dir)
            self.instance_id = instance_id
            self.model_name = model_name
            return self

        def step(self) -> list[dict]:
            out = super().step()
            if self.mode == "baseline" or self.monitor is None:
                return out
            try:
                self._assess_and_maybe_intervene()
            except Exception as exc:  # never let monitoring crash the agent
                self.verdicts.append({"error": str(exc)[:200]})
            return out

        def _assess_and_maybe_intervene(self) -> None:
            assert self.monitor is not None
            steps = agent_messages_to_steps(self.messages, self.instance_id, self.model_name)
            if not steps:
                return
            extra = None
            has_ckpt = self.checkpoint_sha is not None
            if has_git(self.repo_dir):
                last = steps[-1]
                test_improved = (
                    last.test_fail_count is not None
                    and self._prev_fail_count is not None
                    and last.test_fail_count < self._prev_fail_count
                )
                extra, self.prev_patch_stats = patch_features(
                    self.repo_dir, self.prev_patch_stats, test_improved)
                if last.test_fail_count is not None:
                    self._prev_fail_count = last.test_fail_count

            verdict = self.monitor.assess(
                steps, extra_features=extra, last_alarm_step=self.last_alarm_step,
                n_interventions=self.n_interventions, has_checkpoint=has_ckpt,
            )
            self.verdicts.append(verdict.model_dump())

            # checkpoint after edits so a later rollback suggestion is actionable
            if has_git(self.repo_dir) and (verdict.evidence.get("n_edit") or 0):
                self.checkpoint_sha = make_checkpoint(self.repo_dir, f"step{verdict.step}") or self.checkpoint_sha

            if self.mode == "intervention" and verdict.alarm and verdict.recommended_intervention != "none":
                # Prefer the dynamic, trajectory-specific message (e.g. loop_break names
                # the exact repeated command); fall back to the static MESSAGES table.
                msg = verdict.intervention_message or MESSAGES.get(verdict.recommended_intervention, "")
                if msg:
                    self.add_messages({"role": "user", "content": msg,
                                       "extra": {"localguard_injected": True}})
                    self.injected.append({"step": verdict.step,
                                          "kind": verdict.recommended_intervention,
                                          "target_command": verdict.target_command})
                    self.last_alarm_step = verdict.step
                    self.n_interventions += 1

    return MonitoredAgent


def run_task(
    task_prompt: str,
    instance_id: str,
    repo_dir: str,
    verify_cmd: str,
    *,
    model=None,
    model_name: str = "ollama_chat/qwen2.5-coder:7b",
    monitor: Monitor | None = None,
    mode: str = "baseline",
    policy: str = "baseline",
    step_limit: int = 40,
    out_dir: Path | None = None,
) -> RunResult:
    """Run one task end-to-end and evaluate success via ``verify_cmd``."""
    import subprocess

    import yaml
    from minisweagent import package_dir
    from minisweagent.environments.local import LocalEnvironment

    import os

    model = model or make_local_model(model_name)
    shim_path = _ensure_python_shim()
    env = LocalEnvironment(
        cwd=str(repo_dir), timeout=60,
        env={"PATH": shim_path + os.pathsep + os.environ.get("PATH", "")},
    )
    cfg = yaml.safe_load((Path(package_dir) / "config" / "mini_textbased.yaml").read_text())["agent"]
    cfg["step_limit"] = step_limit
    cfg["cost_limit"] = 0.0  # disable cost limit (0 disables the check)

    out_dir = Path(out_dir) if out_dir else Path(repo_dir).parent
    traj_path = out_dir / f"{instance_id}__{policy}.traj.json"
    cfg["output_path"] = traj_path

    AgentCls = _build_agent_class()
    agent = AgentCls(model, env, **cfg)
    agent.configure(monitor=monitor, mode=mode, repo_dir=str(repo_dir),
                    instance_id=instance_id, model_name=model_name)

    t0 = time.time()
    exit_status = "ok"
    try:
        result = agent.run(task_prompt)
        exit_status = result.get("exit_status", "ok")
    except Exception as exc:
        exit_status = f"error:{type(exc).__name__}"
    runtime = time.time() - t0

    # success = verify command exits 0
    success = False
    try:
        vr = subprocess.run(verify_cmd, shell=True, cwd=str(repo_dir), text=True,
                            capture_output=True, timeout=120)
        success = vr.returncode == 0
    except Exception:
        success = False

    stats = diff_stats(Path(repo_dir)) if has_git(Path(repo_dir)) else diff_stats(Path(repo_dir))
    final_patch = ""
    if has_git(Path(repo_dir)):
        import subprocess as sp
        final_patch = sp.run(["git", "-C", str(repo_dir), "diff", "HEAD"],
                             capture_output=True, text=True).stdout[:20000]

    steps = agent_messages_to_steps(agent.messages, instance_id, model_name)
    n_test_runs = sum(1 for s in steps if s.is_test_command)
    tokens = sum((len(s.thought_text) + len(s.action_text) + len(s.observation_text)) // 4 for s in steps)
    injected = getattr(agent, "injected", [])
    first_alarm = injected[0]["step"] if injected else None

    return RunResult(
        instance_id=instance_id, model=model_name, policy=policy, mode=mode,
        success=success, n_steps=len(steps), n_interventions=len(injected),
        first_alarm_step=first_alarm, exit_status=exit_status,
        diff_files_changed=stats.files_changed, diff_lines_added=stats.lines_added,
        diff_lines_deleted=stats.lines_deleted, n_test_runs=n_test_runs,
        total_tokens_approx=tokens, runtime_s=round(runtime, 1), final_patch=final_patch,
        trajectory_path=str(traj_path), verdicts=getattr(agent, "verdicts", []),
        injected=injected,
    )
