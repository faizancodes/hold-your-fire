"""Drive paired online experiments and do recovery/disruption accounting (Phase 15).

For each task we run a baseline plus one or more monitored policies, then pair
them to measure whether intervention *recovered* runs (baseline failed -> policy
succeeded) or *disrupted* them (baseline succeeded -> policy failed). Per the
research framing, we never claim improvement from success rate alone — we report
the recovery/disruption breakdown and resource savings.
"""

from __future__ import annotations

import tempfile
from dataclasses import asdict
from pathlib import Path
from typing import Any

from .mini_swe_wrapper import RunResult, make_local_model, run_task
from .monitor import Monitor, PolicyConfig, load_monitor
from .toy_tasks import materialize, tasks_for_level
from .utils import RESULTS_ONLINE, append_jsonl, ensure_dirs


def build_monitor(cfg: dict[str, Any], threshold: float | None = None) -> Monitor | None:
    """Construct a Monitor from an online config's ``monitor`` + ``policy`` block."""
    mon_cfg = cfg.get("monitor") or {}
    if not mon_cfg:
        return None
    pol = cfg.get("policy") or {}
    policy = PolicyConfig(
        min_step=int(pol.get("min_step", 5)),
        cooldown_steps=int(pol.get("cooldown_steps", 5)),
        max_interventions=int(pol.get("max_interventions", 2)),
        threshold=float(threshold if threshold is not None else pol.get("threshold", 0.6)),
        high_risk=float(pol.get("high_risk", 0.5)),
        very_high_risk=float(pol.get("very_high_risk", 0.8)),
        shadow=(cfg.get("mode") == "shadow"),
        forced_kind=cfg.get("forced_kind"),
    )
    from .utils import REPO_ROOT

    return load_monitor(
        REPO_ROOT / mon_cfg["models_dir"],
        REPO_ROOT / mon_cfg["calibrators_dir"],
        mon_cfg.get("model", "hist_gradient_boosting"),
        policy=policy,
    )


def run_policy(
    cfg: dict[str, Any],
    *,
    model=None,
    workdir: Path | None = None,
    log_path: Path | None = None,
) -> list[RunResult]:
    """Run every task in the configured level under one policy/mode."""
    model_name = cfg.get("model_name", "ollama_chat/qwen2.5-coder:7b")
    level = int(cfg.get("level", 1))
    mode = cfg.get("mode", "baseline")
    policy_name = cfg.get("name", mode)
    step_limit = int(cfg.get("step_limit", 40))

    workdir = Path(workdir or tempfile.mkdtemp(prefix=f"lg_online_{policy_name}_"))
    model = model or make_local_model(model_name, num_ctx=int(cfg.get("num_ctx", 32768)))
    monitor = build_monitor(cfg) if mode != "baseline" else None

    log_path = log_path or (RESULTS_ONLINE / "online_runs.jsonl")
    ensure_dirs(RESULTS_ONLINE, workdir)

    results: list[RunResult] = []
    for task in tasks_for_level(level):
        repo = materialize(task, workdir)
        res = run_task(
            task.prompt, task.task_id, str(repo), task.verify_cmd,
            model=model, model_name=model_name, monitor=monitor,
            mode=mode, policy=policy_name, step_limit=step_limit, out_dir=workdir,
        )
        results.append(res)
        row = res.summary_row()
        row["level"] = level
        append_jsonl(log_path, row)
    return results


def accounting(baseline: list[RunResult], policy: list[RunResult]) -> dict[str, Any]:
    """Pair baseline vs policy runs by task and compute the accounting."""
    b = {r.instance_id: r for r in baseline}
    p = {r.instance_id: r for r in policy}
    common = sorted(set(b) & set(p))

    recovery = disruption = unchanged_success = unchanged_failure = 0
    safe_saving = harmful_waste = 0
    steps_b = steps_p = 0
    for tid in common:
        rb, rp = b[tid], p[tid]
        steps_b += rb.n_steps
        steps_p += rp.n_steps
        if not rb.success and rp.success:
            recovery += 1
        elif rb.success and not rp.success:
            disruption += 1
        elif rb.success and rp.success:
            unchanged_success += 1
            if rp.n_steps < rb.n_steps:
                safe_saving += 1
            elif rp.n_steps > rb.n_steps:
                harmful_waste += 1
        else:
            unchanged_failure += 1

    n = max(1, len(common))
    return {
        "policy": policy[0].policy if policy else "?",
        "n_tasks": len(common),
        "baseline_success_rate": round(sum(r.success for r in baseline) / max(1, len(baseline)), 4),
        "policy_success_rate": round(sum(r.success for r in policy) / max(1, len(policy)), 4),
        "recovery_count": recovery,
        "disruption_count": disruption,
        "unchanged_success_count": unchanged_success,
        "unchanged_failure_count": unchanged_failure,
        "safe_saving_count": safe_saving,
        "harmful_waste_count": harmful_waste,
        "avg_steps_baseline": round(steps_b / n, 2),
        "avg_steps_policy": round(steps_p / n, 2),
        "interventions_per_run": round(
            sum(r.n_interventions for r in policy) / max(1, len(policy)), 2),
    }


def behaviorally_identical(baseline: list[RunResult], shadow: list[RunResult]) -> dict[str, Any]:
    """Check the shadow-mode invariant: shadow must not change agent behavior."""
    b = {r.instance_id: r for r in baseline}
    s = {r.instance_id: r for r in shadow}
    rows = []
    for tid in sorted(set(b) & set(s)):
        rb, rs = b[tid], s[tid]
        rows.append({
            "task": tid,
            "same_n_steps": rb.n_steps == rs.n_steps,
            "same_success": rb.success == rs.success,
            "baseline_steps": rb.n_steps,
            "shadow_steps": rs.n_steps,
        })
    return {
        "identical": all(r["same_n_steps"] and r["same_success"] for r in rows),
        "per_task": rows,
    }


def runresult_dict(r: RunResult) -> dict[str, Any]:
    d = asdict(r)
    d.pop("final_patch", None)
    return d
