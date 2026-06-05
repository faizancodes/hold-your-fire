#!/usr/bin/env python3
"""AUC-lift experiment #3: more failure data (train-only enrichment).

Keeps the v1 TEST instances/prefixes fixed; re-samples the TRAIN instances from
the full corpus with a higher per-instance failure cap, rebuilds v1 features, and
retrains HGB. Isolates the effect of more training data. Paired vs the same
v1-HGB baseline.

  python scripts/run_more_data.py --levels 10 25
"""

from __future__ import annotations

import argparse

import _bootstrap  # noqa: F401
import pandas as pd

from localguard.evaluate import paired_bootstrap_auc_delta, roc_auc_metric
from localguard.ingest_nebius import load_sampled_full
from localguard.normalize import normalize_rows
from localguard.prefix_builder import trajectories_to_rows
from localguard.train import _to_numeric_matrix, numeric_columns
from localguard.utils import REPO_ROOT, RESULTS_OFFLINE, DEFAULT_SEED, read_json, write_json


def _hgb(seed=DEFAULT_SEED):
    from sklearn.ensemble import HistGradientBoostingClassifier
    return HistGradientBoostingClassifier(max_iter=300, learning_rate=0.08,
                                          l2_regularization=1.0, random_state=seed)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--v1", default="data/processed/prefix_offline_full.parquet")
    ap.add_argument("--split", default="results/offline/full/split_assignment.parquet")
    ap.add_argument("--levels", type=int, nargs="+", default=[10, 25])
    ap.add_argument("--n-boot", type=int, default=1000)
    args = ap.parse_args()

    folds = pd.read_parquet(REPO_ROOT / args.split)
    v1 = pd.read_parquet(REPO_ROOT / args.v1).merge(folds, on="prefix_id").sort_values("prefix_id").reset_index(drop=True)
    cols = numeric_columns(v1)
    te = v1[v1["fold"] == "test"]
    train_instances = set(v1[v1["fold"] == "train"]["instance_id"].unique())
    orig_train_fail = int(v1[(v1["fold"] == "train")].drop_duplicates("trajectory_id")["y_fail"].sum())
    print(f"[more-data] {len(train_instances)} train instances; baseline train uses "
          f"≤2 fail/instance. Test fixed: {len(te)} rows / {te['instance_id'].nunique()} instances")

    base = pd.read_parquet(RESULTS_OFFLINE / "full" / "auc_lift_baseline.parquet").sort_values("prefix_id").reset_index(drop=True)
    te = te.sort_values("prefix_id").reset_index(drop=True)
    assert (te["prefix_id"].values == base["prefix_id"].values).all(), "test misalignment"
    yt = te["y_fail"].to_numpy(int)
    inst_te = te["instance_id"].to_numpy()
    pt_base = base["p_base"].to_numpy()

    results = {}
    for K in args.levels:
        raw = load_sampled_full(only_instances=train_instances, max_fail_per_instance=K,
                                max_success_per_instance=12, seed=43)
        norm = normalize_rows(raw)
        rows = list(trajectories_to_rows(norm, schedule_mode="default"))
        enr = pd.DataFrame(rows)
        n_traj = enr["trajectory_id"].nunique()
        n_fail_traj = int(enr.drop_duplicates("trajectory_id")["y_fail"].sum())
        # ensure enriched columns cover the model's feature cols
        miss = [c for c in cols if c not in enr.columns]
        for c in miss:
            enr[c] = 0
        est = _hgb()
        est.fit(_to_numeric_matrix(enr, cols), enr["y_fail"].to_numpy(int))
        pos = list(est.classes_).index(1)
        pt = est.predict_proba(_to_numeric_matrix(te, cols))[:, pos]
        r = paired_bootstrap_auc_delta(inst_te, yt, pt_base, pt, n_boot=args.n_boot)
        print(f"[more-data] K_fail={K}: enriched_train_traj={n_traj} (fail={n_fail_traj}, "
              f"~{n_fail_traj/orig_train_fail:.1f}x baseline failures), rows={len(enr)}  TEST {r}")
        results[f"more_data_K{K}"] = {"enriched_train_traj": n_traj, "enriched_fail_traj": n_fail_traj,
                                      "baseline_fail_traj": orig_train_fail, **r}

    out = RESULTS_OFFLINE / "full" / "auc_lift_results.json"
    existing = read_json(out) if out.exists() else {}
    existing.update(results)
    write_json(out, existing)
    print(f"[more-data] wrote {out}")


if __name__ == "__main__":
    main()
