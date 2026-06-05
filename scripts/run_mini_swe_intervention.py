#!/usr/bin/env python3
"""Online intervention run + recovery/disruption accounting (Phase 15).

  python scripts/run_mini_swe_intervention.py --level 1 --config configs/online_loop_guard.yaml

Pairs the policy against a baseline (cached per level+model at temperature 0) and
reports recovery (baseline failed -> policy succeeded), disruption (baseline
succeeded -> policy failed), and resource changes. Never claims improvement from
success rate alone.
"""

from __future__ import annotations

import argparse
import logging
import os

import _bootstrap  # noqa: F401

os.environ.setdefault("MSWEA_COST_TRACKING", "ignore_errors")
logging.disable(logging.WARNING)


def _baseline_cache_path(level: int, model_name: str):
    from localguard.utils import RESULTS_ONLINE

    safe = model_name.replace("/", "_").replace(":", "_")
    return RESULTS_ONLINE / f"baseline_level{level}_{safe}.json"


def _load_or_run_baseline(cfg: dict, refresh: bool):
    from localguard.mini_swe_wrapper import RunResult, make_local_model
    from localguard.online_runner import run_policy, runresult_dict
    from localguard.utils import read_json, write_json

    path = _baseline_cache_path(cfg["level"], cfg["model_name"])
    if path.exists() and not refresh:
        rows = read_json(path)
        return [RunResult(**{**r, "verdicts": [], "injected": [], "final_patch": ""}) for r in rows]
    model = make_local_model(cfg["model_name"], num_ctx=int(cfg.get("num_ctx", 32768)))
    baseline = run_policy(dict(cfg, mode="baseline", name="baseline"), model=model)
    write_json(path, [runresult_dict(r) for r in baseline])
    return baseline


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", required=True)
    ap.add_argument("--level", type=int, default=None)
    ap.add_argument("--model-name", default=None)
    ap.add_argument("--refresh-baseline", action="store_true")
    args = ap.parse_args()

    from localguard.online_runner import accounting, run_policy, runresult_dict
    from localguard.utils import RESULTS_ONLINE, ensure_dirs, load_config, write_json

    cfg = load_config(args.config).data
    if args.level is not None:
        cfg["level"] = args.level
    if args.model_name:
        cfg["model_name"] = args.model_name
    ensure_dirs(RESULTS_ONLINE)

    print(f"[online] baseline (cached) for level={cfg['level']}, model={cfg['model_name']}")
    baseline = _load_or_run_baseline(cfg, args.refresh_baseline)
    for r in baseline:
        print(f"  baseline {r.instance_id:14s} success={r.success} steps={r.n_steps}")

    print(f"[online] policy={cfg['name']} (forced_kind={cfg.get('forced_kind')})")
    policy = run_policy(cfg)
    for r in policy:
        print(f"  {cfg['name']:18s} {r.instance_id:14s} success={r.success} steps={r.n_steps} "
              f"interventions={r.n_interventions} first_alarm={r.first_alarm_step}")

    acc = accounting(baseline, policy)
    out = {
        "accounting": acc,
        "baseline": [runresult_dict(r) for r in baseline],
        "policy": [runresult_dict(r) for r in policy],
    }
    write_json(RESULTS_ONLINE / f"accounting_{cfg['name']}_level{cfg['level']}.json", out)
    print("\n[online] === accounting ===")
    for k, v in acc.items():
        print(f"  {k}: {v}")
    print(f"[online] wrote accounting_{cfg['name']}_level{cfg['level']}.json")


if __name__ == "__main__":
    main()
