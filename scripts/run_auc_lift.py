#!/usr/bin/env python3
"""AUC-lift experiments #4 (ensemble) and #1 (cleaner label).

All comparisons are paired against the SAME fixed v1-HGB baseline on the SAME
held-out test instances. Selection (ensemble weights, weighting scheme) is done
on validation; test is scored once per config.

Writes baseline test predictions to results/offline/full/auc_lift_baseline.parquet
(reused by run_more_data.py and run_seq_model.py) and appends results to
results/offline/full/auc_lift_results.json.
"""

from __future__ import annotations

import argparse

import _bootstrap  # noqa: F401
import numpy as np
import pandas as pd

from localguard.evaluate import paired_bootstrap_auc_delta, prefix_metrics, roc_auc_metric
from localguard.train import _to_numeric_matrix, numeric_columns
from localguard.utils import REPO_ROOT, RESULTS_OFFLINE, DEFAULT_SEED, read_json, write_json


def _load(v1_path, split_path):
    folds = pd.read_parquet(REPO_ROOT / split_path)
    df = pd.read_parquet(REPO_ROOT / v1_path).merge(folds, on="prefix_id")
    df = df.sort_values("prefix_id").reset_index(drop=True)
    tr = df[df["fold"] == "train"]
    va = df[df["fold"] == "val"]
    te = df[df["fold"] == "test"]
    return df, tr, va, te, numeric_columns(df)


def _hgb(seed=DEFAULT_SEED):
    from sklearn.ensemble import HistGradientBoostingClassifier
    return HistGradientBoostingClassifier(max_iter=300, learning_rate=0.08,
                                          l2_regularization=1.0, random_state=seed)


def _rf(seed=DEFAULT_SEED):
    from sklearn.ensemble import RandomForestClassifier
    return RandomForestClassifier(n_estimators=300, min_samples_leaf=5,
                                  class_weight="balanced", n_jobs=-1, random_state=seed)


def _lr(seed=DEFAULT_SEED):
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    return Pipeline([("s", StandardScaler()),
                     ("c", LogisticRegression(max_iter=2000, class_weight="balanced"))])


def _fit_pred(est, tr, va, te, cols, sample_weight=None):
    Xtr = _to_numeric_matrix(tr, cols)
    est.fit(Xtr, tr["y_fail"].to_numpy(int), **({"sample_weight": sample_weight} if sample_weight is not None else {}))
    classes = list(getattr(est, "classes_", getattr(est[-1] if hasattr(est, "__getitem__") else est, "classes_", [0, 1])))
    pos = classes.index(1) if 1 in classes else len(classes) - 1
    pv = est.predict_proba(_to_numeric_matrix(va, cols))[:, pos]
    pt = est.predict_proba(_to_numeric_matrix(te, cols))[:, pos]
    return pv, pt


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--v1", default="data/processed/prefix_offline_full.parquet")
    ap.add_argument("--split", default="results/offline/full/split_assignment.parquet")
    ap.add_argument("--n-boot", type=int, default=1000)
    args = ap.parse_args()

    df, tr, va, te, cols = _load(args.v1, args.split)
    yv = va["y_fail"].to_numpy(int)
    yt = te["y_fail"].to_numpy(int)
    inst_te = te["instance_id"].to_numpy()
    print(f"[auc-lift] train={len(tr)} val={len(va)} test={len(te)} ({te['instance_id'].nunique()} test instances), {len(cols)} features")

    # ---- fixed baseline: v1 HGB -----------------------------------------
    pv_h, pt_h = _fit_pred(_hgb(), tr, va, te, cols)
    base_test_auc = roc_auc_metric(yt, pt_h)
    print(f"[baseline] v1 HGB  val={roc_auc_metric(yv, pv_h):.4f}  test={base_test_auc:.4f}")
    # persist baseline test preds for the other experiment scripts
    base_df = te[["prefix_id", "instance_id", "prefix_step", "n_total_steps", "y_fail"]].copy()
    base_df["p_base"] = pt_h
    base_df.to_parquet(RESULTS_OFFLINE / "full" / "auc_lift_baseline.parquet", index=False)

    results = {}

    # ===================================================================
    # Experiment 4: decorrelated ensemble (HGB + RF + linear)
    # ===================================================================
    pv_r, pt_r = _fit_pred(_rf(), tr, va, te, cols)
    pv_l, pt_l = _fit_pred(_lr(), tr, va, te, cols)
    print(f"[ensemble] members val AUC  HGB={roc_auc_metric(yv,pv_h):.4f} RF={roc_auc_metric(yv,pv_r):.4f} LR={roc_auc_metric(yv,pv_l):.4f}")

    cand = {}
    cand["mean"] = ((pv_h + pv_r + pv_l) / 3, (pt_h + pt_r + pt_l) / 3)
    # weighted: small grid on val (weights sum to 1, step 0.1)
    best_w, best_wauc = None, -1
    for a in np.arange(0, 1.01, 0.1):
        for b in np.arange(0, 1.01 - a, 0.1):
            c = 1 - a - b
            if c < -1e-9:
                continue
            pv = a * pv_h + b * pv_r + c * pv_l
            au = roc_auc_metric(yv, pv)
            if au > best_wauc:
                best_wauc, best_w = au, (a, b, c)
    a, b, c = best_w
    cand["weighted"] = (a * pv_h + b * pv_r + c * pv_l, a * pt_h + b * pt_r + c * pt_l)
    # stacking: logistic on member val preds
    from sklearn.linear_model import LogisticRegression
    stk = LogisticRegression(max_iter=1000)
    stk.fit(np.column_stack([pv_h, pv_r, pv_l]), yv)
    cand["stack"] = (stk.predict_proba(np.column_stack([pv_h, pv_r, pv_l]))[:, 1],
                     stk.predict_proba(np.column_stack([pt_h, pt_r, pt_l]))[:, 1])

    print("[ensemble] val AUC by combiner:")
    for name, (pv, _) in cand.items():
        print(f"   {name:9s} val={roc_auc_metric(yv, pv):.4f}" + (f"  weights={best_w}" if name == "weighted" else ""))
    winner = max(cand, key=lambda k: roc_auc_metric(yv, cand[k][0]))
    pt_ens = cand[winner][1]
    r = paired_bootstrap_auc_delta(inst_te, yt, pt_h, pt_ens, n_boot=args.n_boot)
    print(f"[ensemble] winner={winner}  TEST {r}")
    results["ensemble"] = {"combiner": winner, **r, "auprc": prefix_metrics(yt, pt_ens)["auprc"]}

    # ===================================================================
    # Experiment 1: cleaner label (position stratification + weighting)
    # ===================================================================
    pos = (te["prefix_step"] / te["n_total_steps"].clip(lower=1)).to_numpy()
    bins = {"early(<=0.33)": pos <= 0.33, "mid(0.33-0.66)": (pos > 0.33) & (pos <= 0.66), "late(>0.66)": pos > 0.66}
    print("[label] baseline HGB AUC by normalized position (ceiling diagnostic):")
    strat = {}
    for name, mask in bins.items():
        if mask.sum() > 50 and len(np.unique(yt[mask])) > 1:
            strat[name] = round(roc_auc_metric(yt[mask], pt_h[mask]), 4)
            print(f"   {name:16s} n={int(mask.sum()):6d}  AUC={strat[name]:.4f}")

    # position-weighted training: emphasize determinable (later) prefixes
    fracs_tr = (tr["prefix_step"] / tr["n_total_steps"].clip(lower=1)).to_numpy()
    schemes = {
        "linear": fracs_tr,
        "quadratic": fracs_tr ** 2,
        "late_emph": 0.3 + fracs_tr,  # floor so early prefixes aren't ignored
    }
    best_scheme, best_sval, best_pt = None, roc_auc_metric(yv, pv_h), None
    sval_all = {}
    for name, w in schemes.items():
        w = np.clip(w, 1e-3, None)
        pv_w, pt_w = _fit_pred(_hgb(), tr, va, te, cols, sample_weight=w)
        sval_all[name] = round(roc_auc_metric(yv, pv_w), 4)
        if sval_all[name] > best_sval:
            best_sval, best_scheme, best_pt = sval_all[name], name, pt_w
    print(f"[label] position-weighting val AUC: {sval_all} (baseline val={roc_auc_metric(yv,pv_h):.4f})")
    if best_pt is not None:
        rw = paired_bootstrap_auc_delta(inst_te, yt, pt_h, best_pt, n_boot=args.n_boot)
        # late-subset AUC for the weighted model
        late = bins["late(>0.66)"]
        late_auc = round(roc_auc_metric(yt[late], best_pt[late]), 4) if late.sum() > 50 else None
        print(f"[label] best weighting={best_scheme}  TEST {rw}  late_subset_AUC={late_auc}")
        results["label_weighted"] = {"scheme": best_scheme, **rw}
    else:
        print("[label] no position-weighting scheme beat baseline on validation")
        results["label_weighted"] = {"scheme": None, "delta": 0.0, "significant": False,
                                     "note": "no scheme improved val"}
    results["label_position_strata"] = strat

    # ---- persist ---------------------------------------------------------
    out = RESULTS_OFFLINE / "full" / "auc_lift_results.json"
    existing = read_json(out) if out.exists() else {}
    existing.update({"baseline_test_auc": round(base_test_auc, 4), **results})
    write_json(out, existing)
    print(f"\n[auc-lift] wrote {out}")


if __name__ == "__main__":
    main()
