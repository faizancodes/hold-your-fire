#!/usr/bin/env python3
"""Online baseline + shadow-mode run; verifies shadow doesn't change behavior (Phase 13).

  python scripts/run_mini_swe_shadow.py --level 1 --config configs/online_shadow.yaml

Shadow mode logs what the monitor *would* do without ever injecting a message, so
at temperature 0 baseline and shadow runs must be behaviorally identical.
"""

from __future__ import annotations

import argparse
import logging
import os

import _bootstrap  # noqa: F401

os.environ.setdefault("MSWEA_COST_TRACKING", "ignore_errors")
logging.disable(logging.WARNING)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="configs/online_shadow.yaml")
    ap.add_argument("--level", type=int, default=None)
    ap.add_argument("--model-name", default=None)
    args = ap.parse_args()

    from localguard.online_runner import (
        behaviorally_identical,
        run_policy,
        runresult_dict,
    )
    from localguard.utils import RESULTS_ONLINE, ensure_dirs, load_config, write_json

    cfg = load_config(args.config).data
    if args.level is not None:
        cfg["level"] = args.level
    if args.model_name:
        cfg["model_name"] = args.model_name

    ensure_dirs(RESULTS_ONLINE)
    print(f"[online] baseline run, level={cfg['level']}, model={cfg['model_name']}")
    base_cfg = dict(cfg, mode="baseline", name="baseline")
    baseline = run_policy(base_cfg)
    for r in baseline:
        print(f"  baseline {r.instance_id:14s} success={r.success} steps={r.n_steps} ({r.runtime_s}s)")

    print(f"[online] shadow run, level={cfg['level']}")
    shadow = run_policy(dict(cfg, mode="shadow"))
    for r in shadow:
        n_would = sum(1 for v in r.verdicts if v.get("would_alarm"))
        print(f"  shadow   {r.instance_id:14s} success={r.success} steps={r.n_steps} "
              f"would_alarm_steps={n_would}")

    check = behaviorally_identical(baseline, shadow)
    write_json(RESULTS_ONLINE / f"shadow_check_level{cfg['level']}.json", {
        "behaviorally_identical": check["identical"],
        "per_task": check["per_task"],
        "baseline": [runresult_dict(r) for r in baseline],
        "shadow": [runresult_dict(r) for r in shadow],
    })
    print(f"[online] shadow behaviorally identical to baseline: {check['identical']}")
    print(f"[online] wrote {RESULTS_ONLINE/('shadow_check_level'+str(cfg['level'])+'.json')}")


if __name__ == "__main__":
    main()
