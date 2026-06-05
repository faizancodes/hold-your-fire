#!/usr/bin/env python3
"""Qualitative prefix audit tool (Phase 18).

Surfaces true/false positives and false negatives from the test split with their
last actions/observations and a semi-automatic failure-mode label, to answer
"which coding-agent failures are observable early?". Prints examples and writes a
JSONL + a failure-mode distribution to results/audits/.

  python scripts/audit_prefixes.py --config configs/offline_full.yaml --kind true_positive --n 50
"""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

import _bootstrap  # noqa: F401
import pandas as pd

from localguard.audit import FAILURE_MODES, heuristic_failure_mode  # noqa: F401
from localguard.calibrate import Calibrator
from localguard.schemas import NormalizedTrajectory
from localguard.thresholding import DEFAULT_DEPLOY_POLICY
from localguard.train import MonitorModel
from localguard.utils import (
    INTERIM_DIR,
    REPO_ROOT,
    RESULTS_AUDITS,
    ensure_dirs,
    load_config,
    read_json,
    read_jsonl,
    write_jsonl,
)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="configs/offline_full.yaml")
    ap.add_argument("--kind", default="true_positive",
                    choices=["true_positive", "false_positive", "false_negative",
                             "true_negative", "high_risk"])
    ap.add_argument("--n", type=int, default=50)
    ap.add_argument("--min-step", type=int, default=3)
    args = ap.parse_args()

    cfg = load_config(args.config).data
    results_dir = REPO_ROOT / cfg["results_dir"]
    df = pd.read_parquet(REPO_ROOT / cfg["dataset"])
    folds = pd.read_parquet(results_dir / "split_assignment.parquet")
    test = df.merge(folds, on="prefix_id").query("fold == 'test'")
    test = test[test["prefix_step"] >= args.min_step]

    best = read_json(results_dir / "offline_results.json").get("best_model", "random_forest")
    model = MonitorModel.load(REPO_ROOT / cfg["models_dir"] / f"{MonitorModel.safe_filename(best)}.joblib")
    cal_path = REPO_ROOT / cfg["calibrators_dir"] / f"{MonitorModel.safe_filename(best)}.joblib"
    cal = Calibrator.load(cal_path) if cal_path.exists() else Calibrator(method="identity")

    thr = read_json(results_dir / "offline_results.json")
    threshold = 0.5
    for r in thr.get("main", []):
        if r["name"] == best:
            threshold = r["thresholds"].get(DEFAULT_DEPLOY_POLICY, 0.5)
    risk = cal.transform(model.predict_proba_fail(test))
    test = test.assign(risk=risk, alarm=(risk >= threshold))

    if args.kind == "true_positive":
        sel = test[(test["alarm"]) & (test["y_fail"] == 1)]
    elif args.kind == "false_positive":
        sel = test[(test["alarm"]) & (test["y_fail"] == 0)]
    elif args.kind == "false_negative":
        sel = test[(~test["alarm"]) & (test["y_fail"] == 1)]
    elif args.kind == "true_negative":
        sel = test[(~test["alarm"]) & (test["y_fail"] == 0)]
    else:  # high_risk
        sel = test.sort_values("risk", ascending=False)
    sel = sel.sort_values("risk", ascending=False).head(args.n)
    print(f"[audit] {args.kind}: {len(sel)} prefixes (deploy threshold={threshold:.3f}, model={best})")

    normalized = _load_normalized(cfg["dataset"])
    records = []
    modes = Counter()
    for _, r in sel.iterrows():
        feats = {c: r[c] for c in test.columns if c.startswith("f__") and c != "f__text_blob"}
        mode = heuristic_failure_mode(feats)
        modes[mode] += 1
        traj = normalized.get(r["trajectory_id"])
        last_actions, last_obs = [], []
        if traj is not None:
            steps = traj.steps[: int(r["prefix_step"])]
            last_actions = [f"({s.action_type}) {(s.action_text or '').strip()[:80]}" for s in steps[-5:]]
            last_obs = [(s.observation_text or "").strip().replace("\n", " ")[:120] for s in steps[-3:]]
        records.append({
            "instance_id": r["instance_id"], "model_name": r["model_name"],
            "y_fail": int(r["y_fail"]), "prefix_step": int(r["prefix_step"]),
            "n_total_steps": int(r["n_total_steps"]), "risk": round(float(r["risk"]), 3),
            "heuristic_failure_mode": mode,
            "last_actions": last_actions, "last_observations": last_obs,
        })

    ensure_dirs(RESULTS_AUDITS)
    out = RESULTS_AUDITS / f"audit_{cfg['name']}_{args.kind}.jsonl"
    write_jsonl(out, records)

    # print a few examples
    for rec in records[:6]:
        print(f"\n  {rec['instance_id']} step {rec['prefix_step']}/{rec['n_total_steps']} "
              f"risk={rec['risk']} y_fail={rec['y_fail']} mode={rec['heuristic_failure_mode']}")
        for a in rec["last_actions"]:
            print(f"     $ {a}")
        for o in rec["last_observations"]:
            if o:
                print(f"     > {o}")
    print(f"\n[audit] failure-mode distribution ({args.kind}):")
    for m, c in modes.most_common():
        print(f"    {m:26s} {c:4d}  ({c/max(1,len(records)):.0%})")
    print(f"[audit] wrote {out}")


def _load_normalized(dataset_path: str) -> dict[str, NormalizedTrajectory]:
    stem = Path(dataset_path).stem
    path = INTERIM_DIR / f"normalized_{stem}.jsonl"
    out: dict[str, NormalizedTrajectory] = {}
    if path.exists():
        for row in read_jsonl(path):
            t = NormalizedTrajectory(**row)
            out[t.trajectory_id] = t
    return out


if __name__ == "__main__":
    main()
