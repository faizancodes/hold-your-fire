#!/usr/bin/env python3
"""Local Ollama LLM-judge baseline on a held-out subset (Phase 10).

Evaluates the local model on the SAME test split as the offline classifier, then
compares risk AUC/AUPRC, JSON validity, and latency.

  python scripts/run_ollama_judge_subset.py --config configs/offline_small.yaml \
      --n 150 --model qwen2.5-coder:7b
"""

from __future__ import annotations

import argparse
from pathlib import Path

import _bootstrap  # noqa: F401
import numpy as np
import pandas as pd

from localguard.evaluate import prefix_metrics
from localguard.ollama_judge import DEFAULT_MODEL, judge_prefix
from localguard.schemas import NormalizedTrajectory
from localguard.train import MonitorModel
from localguard.utils import (
    INTERIM_DIR,
    REPO_ROOT,
    ensure_dirs,
    read_json,
    read_jsonl,
    write_json,
)


def _load_normalized(dataset_path: str) -> dict[str, NormalizedTrajectory]:
    stem = Path(dataset_path).stem
    path = INTERIM_DIR / f"normalized_{stem}.jsonl"
    out: dict[str, NormalizedTrajectory] = {}
    for row in read_jsonl(path):
        t = NormalizedTrajectory(**row)
        out[t.trajectory_id] = t
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="configs/offline_small.yaml")
    ap.add_argument("--n", type=int, default=150)
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--min-step", type=int, default=4)
    ap.add_argument("--max-step", type=int, default=40)
    args = ap.parse_args()

    from localguard.utils import load_config

    cfg = load_config(args.config)
    results_dir = REPO_ROOT / cfg["results_dir"]
    models_dir = REPO_ROOT / cfg["models_dir"]
    df = pd.read_parquet(REPO_ROOT / cfg["dataset"])

    folds = pd.read_parquet(results_dir / "split_assignment.parquet")
    test = df.merge(folds, on="prefix_id").query("fold == 'test'")
    test = test[(test["prefix_step"] >= args.min_step) & (test["prefix_step"] <= args.max_step)]

    # stratified sample by label
    rng = np.random.default_rng(args.seed)
    per = args.n // 2
    pos = test[test["y_fail"] == 1]
    neg = test[test["y_fail"] == 0]
    take_pos = pos.sample(min(per, len(pos)), random_state=args.seed)
    take_neg = neg.sample(min(args.n - len(take_pos), len(neg)), random_state=args.seed)
    sample = pd.concat([take_pos, take_neg]).sample(frac=1.0, random_state=args.seed).reset_index(drop=True)
    print(f"[judge] sampling {len(sample)} test prefixes "
          f"({int(sample['y_fail'].sum())} fail / {int((1-sample['y_fail']).sum())} success), model={args.model}")

    normalized = _load_normalized(cfg["dataset"])

    rows = []
    n_valid = 0
    total_latency = 0.0
    for i, r in sample.iterrows():
        traj = normalized.get(r["trajectory_id"])
        if traj is None:
            continue
        steps = traj.steps[: int(r["prefix_step"])]
        out = judge_prefix(steps, model=args.model)
        n_valid += int(out.valid_json)
        total_latency += out.latency_s
        rows.append({
            "prefix_id": r["prefix_id"],
            "trajectory_id": r["trajectory_id"],
            "prefix_step": int(r["prefix_step"]),
            "y_fail": int(r["y_fail"]),
            "judge_risk": out.risk_score,
            "valid_json": out.valid_json,
            "latency_s": round(out.latency_s, 2),
            "should_intervene": bool(out.judgment.should_intervene) if out.judgment else False,
            "intervention_type": out.judgment.intervention_type if out.judgment else "none",
        })
        if (i + 1) % 25 == 0:
            print(f"  judged {i+1}/{len(sample)}  valid={n_valid}  avg_latency={total_latency/(i+1):.1f}s")

    jdf = pd.DataFrame(rows)
    y = jdf["y_fail"].to_numpy(int)
    judge_m = prefix_metrics(y, jdf["judge_risk"].to_numpy(float))
    # valid-only metrics
    vmask = jdf["valid_json"].to_numpy(bool)
    judge_valid_m = prefix_metrics(y[vmask], jdf["judge_risk"].to_numpy(float)[vmask]) if vmask.sum() > 2 else {}

    # classifier on the SAME prefixes
    best_name = read_json(results_dir / "offline_results.json").get("best_model", "random_forest")
    clf = MonitorModel.load(models_dir / f"{MonitorModel.safe_filename(best_name)}.joblib")
    clf_df = df[df["prefix_id"].isin(jdf["prefix_id"])].set_index("prefix_id").loc[jdf["prefix_id"]].reset_index()
    clf_risk = clf.predict_proba_fail(clf_df)
    jdf["clf_risk"] = clf_risk
    clf_m = prefix_metrics(y, clf_risk)

    out_obj = {
        "model": args.model,
        "n": len(jdf),
        "n_pos": int(y.sum()),
        "invalid_json_rate": round(1 - n_valid / max(1, len(jdf)), 4),
        "avg_latency_s": round(total_latency / max(1, len(jdf)), 2),
        "judge_auc": judge_m.get("roc_auc"),
        "judge_auprc": judge_m.get("auprc"),
        "judge_auc_valid_only": judge_valid_m.get("roc_auc"),
        "classifier": best_name,
        "classifier_auc": clf_m.get("roc_auc"),
        "classifier_auprc": clf_m.get("auprc"),
        "should_intervene_rate": round(float(jdf["should_intervene"].mean()), 4),
    }
    ensure_dirs(results_dir / "figdata")
    write_json(results_dir / "ollama_judge.json", out_obj)
    jdf.to_csv(results_dir / "figdata" / "ollama_judge_predictions.csv", index=False)

    print("\n[judge] === results ===")
    for k, v in out_obj.items():
        print(f"  {k}: {v}")
    print(f"[judge] wrote {results_dir/'ollama_judge.json'}")


if __name__ == "__main__":
    main()
