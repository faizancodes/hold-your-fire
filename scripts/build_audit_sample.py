#!/usr/bin/env python3
"""Build a BLIND human-annotation sample for the observability audit (#2 strengthening).

Renders the actual agent behavior (task + recent action/observation steps) for a
sample of (a) confidently-flagged failures [high-risk true positives] and (b) missed
failures [false negatives, at their most-developed prefix], so a human can label the
failure mode from the *trajectory* — not the features. Writes two files:

  results/audits/human_blind.jsonl  — id, kind, instance, step, risk, rendered (NO label)
  results/audits/human_key.jsonl    — id, heuristic_failure_mode, y_fail (the key)

The annotator reads human_blind.jsonl, labels each id, then we join on the key.
"""

from __future__ import annotations

import argparse
import re

import _bootstrap  # noqa: F401
import numpy as np
import pandas as pd

from localguard.audit import heuristic_failure_mode
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

_WS = re.compile(r"\s+")


def _clean(text: str, n: int) -> str:
    return _WS.sub(" ", (text or "").strip())[:n]


def _render(traj: NormalizedTrajectory, prefix_step: int, n_recent: int = 12) -> str:
    """Task goal (step-0 observation) + the last n_recent action/observation steps."""
    steps = traj.steps[:prefix_step]
    lines = []
    if steps:
        goal = _clean(steps[0].observation_text, 600)
        lines.append(f"TASK/CONTEXT (step 0 obs): {goal}")
        lines.append("--- recent steps ---")
    start = max(0, len(steps) - n_recent)
    for i, s in enumerate(steps[start:], start=start):
        act = _clean(s.action_text, 160) or "(no action text)"
        obs = _clean(s.observation_text, 240)
        lines.append(f"[{i}] ({s.action_type}) $ {act}")
        if obs:
            lines.append(f"      -> {obs}")
    return "\n".join(lines)


def _pick_one_per_trajectory(sel: pd.DataFrame, by: str) -> pd.DataFrame:
    """One prefix per trajectory: highest-risk (confident TP) or latest-step (FN)."""
    asc = by == "prefix_step"  # for FN we want the LAST prefix -> sort asc, take last
    sel = sel.sort_values(by, ascending=asc)
    return sel.groupby("trajectory_id", as_index=False).last()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="configs/offline_full.yaml")
    ap.add_argument("--n-tp", type=int, default=40)
    ap.add_argument("--n-fn", type=int, default=10)
    ap.add_argument("--min-step", type=int, default=5)
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    cfg = load_config(args.config).data
    results_dir = REPO_ROOT / cfg["results_dir"]
    df = pd.read_parquet(REPO_ROOT / cfg["dataset"])
    folds = pd.read_parquet(results_dir / "split_assignment.parquet")
    test = df.merge(folds, on="prefix_id").query("fold == 'test'")
    test = test[test["prefix_step"] >= args.min_step]

    res = read_json(results_dir / "offline_results.json")
    best = res.get("best_model", "random_forest")
    model = MonitorModel.load(REPO_ROOT / cfg["models_dir"] / f"{MonitorModel.safe_filename(best)}.joblib")
    cal_path = REPO_ROOT / cfg["calibrators_dir"] / f"{MonitorModel.safe_filename(best)}.joblib"
    cal = Calibrator.load(cal_path) if cal_path.exists() else Calibrator(method="identity")
    threshold = 0.5
    for r in res.get("main", []):
        if r["name"] == best:
            threshold = r["thresholds"].get(DEFAULT_DEPLOY_POLICY, 0.5)
    risk = cal.transform(model.predict_proba_fail(test))
    test = test.assign(risk=risk, alarm=(risk >= threshold))

    rng = np.random.default_rng(args.seed)
    # (a) confident true positives: alarmed failures, one (highest-risk) prefix per run
    tp = _pick_one_per_trajectory(test[test["alarm"] & (test["y_fail"] == 1)], "risk")
    tp = tp.drop_duplicates("instance_id")
    tp = tp.iloc[rng.permutation(len(tp))[: args.n_tp]]
    # (b) missed failures: un-alarmed failures, one (latest-step) prefix per run
    fn = _pick_one_per_trajectory(test[(~test["alarm"]) & (test["y_fail"] == 1)], "prefix_step")
    fn = fn.drop_duplicates("instance_id")
    fn = fn.iloc[rng.permutation(len(fn))[: args.n_fn]]

    sample = pd.concat([tp.assign(kind="flagged_failure"), fn.assign(kind="missed_failure")])
    need = set(sample["trajectory_id"])
    print(f"[build] sampled {len(tp)} flagged + {len(fn)} missed = {len(sample)} prefixes "
          f"(threshold={threshold:.3f}, model={best}); loading {len(need)} trajectories...")

    normalized = _load_needed(cfg["dataset"], need)

    blind, key = [], []
    feat_cols = [c for c in test.columns if c.startswith("f__") and c != "f__text_blob"]
    # shuffle so flagged/missed are interleaved (annotator can't infer kind from order)
    sample = sample.iloc[rng.permutation(len(sample))]
    for k, (_, r) in enumerate(sample.iterrows()):
        aid = f"A{k:02d}"
        traj = normalized.get(r["trajectory_id"])
        rendered = _render(traj, int(r["prefix_step"])) if traj is not None else "(trajectory unavailable)"
        feats = {c: r[c] for c in feat_cols}
        blind.append({
            "id": aid, "instance_id": r["instance_id"], "model_name": r["model_name"],
            "prefix_step": int(r["prefix_step"]), "n_total_steps": int(r["n_total_steps"]),
            "risk": round(float(r["risk"]), 3), "rendered": rendered,
        })
        key.append({
            "id": aid, "kind": r["kind"], "y_fail": int(r["y_fail"]),
            "heuristic_failure_mode": heuristic_failure_mode(feats),
        })

    ensure_dirs(RESULTS_AUDITS)
    write_jsonl(RESULTS_AUDITS / "human_blind.jsonl", blind)
    write_jsonl(RESULTS_AUDITS / "human_key.jsonl", key)
    print(f"[build] wrote {RESULTS_AUDITS/'human_blind.jsonl'} ({len(blind)} items) + human_key.jsonl")


def _load_needed(dataset_path: str, need: set) -> dict[str, NormalizedTrajectory]:
    from pathlib import Path
    path = INTERIM_DIR / f"normalized_{Path(dataset_path).stem}.jsonl"
    out: dict[str, NormalizedTrajectory] = {}
    if not path.exists():
        return out
    for row in read_jsonl(path):
        if row.get("trajectory_id") in need:
            t = NormalizedTrajectory(**row)
            out[t.trajectory_id] = t
            if len(out) == len(need):
                break
    return out


if __name__ == "__main__":
    main()
