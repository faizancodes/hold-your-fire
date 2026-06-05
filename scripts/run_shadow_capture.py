#!/usr/bin/env python3
"""Shadow-capture: run the agent once per task with the monitor in shadow mode and
record per-step calibrated risk + outcome (online evidence for abstention).

Only N agent runs (one per task); the gated-vs-ungated comparison is then a
deterministic offline replay (run_monitor_replay.py) over these fixed trajectories
— no nondeterminism confound. The capture monitor never abstains (min_step=1,
conf_floor=0) so every step's risk is logged for replay.

  python scripts/run_shadow_capture.py --level 2 --model-name ollama_chat/qwen2.5-coder:7b
"""

from __future__ import annotations

import argparse
import logging
import os
import tempfile
from pathlib import Path

import _bootstrap  # noqa: F401

os.environ.setdefault("MSWEA_COST_TRACKING", "ignore_errors")
logging.disable(logging.WARNING)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--level", type=int, default=2)
    ap.add_argument("--model-name", default="ollama_chat/qwen2.5-coder:7b")
    ap.add_argument("--step-limit", type=int, default=30)
    ap.add_argument("--num-ctx", type=int, default=16384)
    ap.add_argument("--models-dir", default="models/monitor/full")
    ap.add_argument("--calibrators-dir", default="models/calibrators/full")
    ap.add_argument("--model", default="hist_gradient_boosting")
    ap.add_argument("--out", default="results/online/shadow_capture.jsonl")
    args = ap.parse_args()

    from localguard.mini_swe_wrapper import make_local_model, run_task
    from localguard.monitor import PolicyConfig, load_monitor
    from localguard.toy_tasks import materialize, tasks_for_level
    from localguard.utils import REPO_ROOT, append_jsonl, ensure_dirs

    # capture monitor: never abstain -> log risk at EVERY step for later replay
    monitor = load_monitor(
        REPO_ROOT / args.models_dir, REPO_ROOT / args.calibrators_dir, args.model,
        policy=PolicyConfig(min_step=1, abstain_conf_floor=0.0, shadow=True, threshold=0.5),
    )
    model = make_local_model(args.model_name, num_ctx=args.num_ctx)
    workdir = Path(tempfile.mkdtemp(prefix="lg_capture_"))
    out = REPO_ROOT / args.out
    ensure_dirs(out.parent)
    if out.exists():
        out.unlink()

    tasks = tasks_for_level(args.level)
    print(f"[capture] {len(tasks)} tasks, model={args.model_name}, monitor={args.model} (shadow, no-abstain)")
    for t in tasks:
        repo = materialize(t, workdir)
        res = run_task(t.prompt, t.task_id, str(repo), t.verify_cmd, model=model,
                       model_name=args.model_name, monitor=monitor, mode="shadow",
                       policy="capture", step_limit=args.step_limit, out_dir=workdir)
        risks = [[int(v["step"]), float(v["calibrated_risk"])]
                 for v in res.verdicts if v.get("calibrated_risk") is not None]
        append_jsonl(out, {"instance_id": t.task_id, "success": bool(res.success),
                           "n_steps": res.n_steps, "runtime_s": res.runtime_s, "risks": risks})
        print(f"  {t.task_id:14s} success={res.success} steps={res.n_steps} "
              f"risk_steps={len(risks)} ({res.runtime_s:.0f}s)")
    print(f"[capture] wrote {out}")


if __name__ == "__main__":
    main()
