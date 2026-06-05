#!/usr/bin/env python3
"""Controlled A/B: do the advanced (v2) features beat v1? (Phase: feature eng.)

Same split, same model class (HGB), same hyperparameters — only the features
change. Decision is made on VALIDATION; the held-out TEST is scored once with a
*paired* instance-bootstrap of the AUC delta (far more powerful than comparing
overlapping CIs).

  python scripts/evaluate_v2.py
"""

from __future__ import annotations

import argparse

import _bootstrap  # noqa: F401
import numpy as np
import pandas as pd

from localguard.evaluate import prefix_metrics, roc_auc_metric
from localguard.features_advanced import ADV_FAMILY_PREFIXES, advanced_family_columns
from localguard.train import (
    _build_sklearn_numeric,
    _pos_index,
    _to_numeric_matrix,
    numeric_columns,
)
from localguard.utils import REPO_ROOT, DEFAULT_SEED


def _hgb(tuned=False, seed=DEFAULT_SEED):
    from sklearn.ensemble import HistGradientBoostingClassifier

    if tuned:
        return HistGradientBoostingClassifier(
            max_iter=600, learning_rate=0.05, max_leaf_nodes=63,
            l2_regularization=2.0, min_samples_leaf=40, random_state=seed,
        )
    return _build_sklearn_numeric("hist_gradient_boosting", seed)


def _fit_predict(train_df, val_df, test_df, cols, seed=DEFAULT_SEED, tuned=False):
    est = _hgb(tuned=tuned, seed=seed)
    est.fit(_to_numeric_matrix(train_df, cols), train_df["y_fail"].to_numpy(int))
    pos = _pos_index(est)
    pv = est.predict_proba(_to_numeric_matrix(val_df, cols))[:, pos]
    pt = est.predict_proba(_to_numeric_matrix(test_df, cols))[:, pos]
    return pv, pt


def paired_bootstrap_delta(test_df, p1, p2, n_boot=1000, seed=DEFAULT_SEED, alpha=0.05):
    """Bootstrap the per-instance AUC(p2) - AUC(p1) on the SAME test rows."""
    rng = np.random.default_rng(seed)
    groups = test_df["instance_id"].astype(str).to_numpy()
    uniq = np.unique(groups)
    by = {g: np.where(groups == g)[0] for g in uniq}
    y = test_df["y_fail"].to_numpy(int)

    base1 = roc_auc_metric(y, p1)
    base2 = roc_auc_metric(y, p2)
    deltas, wins = [], 0
    for _ in range(n_boot):
        chosen = rng.choice(uniq, size=len(uniq), replace=True)
        idx = np.concatenate([by[g] for g in chosen])
        yb = y[idx]
        if len(np.unique(yb)) < 2:
            continue
        a1 = roc_auc_metric(yb, p1[idx])
        a2 = roc_auc_metric(yb, p2[idx])
        deltas.append(a2 - a1)
        wins += int(a2 > a1)
    deltas = np.array(deltas)
    return {
        "auc_v1": base1, "auc_v2": base2, "delta_point": base2 - base1,
        "delta_lo": float(np.quantile(deltas, alpha / 2)),
        "delta_hi": float(np.quantile(deltas, 1 - alpha / 2)),
        "frac_v2_better": wins / max(1, len(deltas)),
        "n_boot": len(deltas),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--v1", default="data/processed/prefix_offline_full.parquet")
    ap.add_argument("--v2", default="data/processed/prefix_offline_full_v2.parquet")
    ap.add_argument("--split", default="results/offline/full/split_assignment.parquet")
    ap.add_argument("--n-boot", type=int, default=1000)
    args = ap.parse_args()

    folds = pd.read_parquet(REPO_ROOT / args.split)
    v1 = pd.read_parquet(REPO_ROOT / args.v1).merge(folds, on="prefix_id")
    v2 = pd.read_parquet(REPO_ROOT / args.v2).merge(folds, on="prefix_id")
    # align v2 to v1 row order by prefix_id for paired comparison
    v1 = v1.sort_values("prefix_id").reset_index(drop=True)
    v2 = v2.sort_values("prefix_id").reset_index(drop=True)
    assert (v1["prefix_id"].values == v2["prefix_id"].values).all(), "prefix_id misalignment"

    def folddf(df, f):
        return df[df["fold"] == f]
    v1_tr, v1_va, v1_te = folddf(v1, "train"), folddf(v1, "val"), folddf(v1, "test")
    v2_tr, v2_va, v2_te = folddf(v2, "train"), folddf(v2, "val"), folddf(v2, "test")

    cols_v1 = numeric_columns(v1)
    cols_v2 = numeric_columns(v2)
    print(f"[v2] v1 features={len(cols_v1)}  v2 features={len(cols_v2)} "
          f"(+{len(cols_v2)-len(cols_v1)} advanced)")

    # ---- DEVELOPMENT: decide on validation ------------------------------
    yv = v1_va["y_fail"].to_numpy(int)
    pv1, pt1 = _fit_predict(v1_tr, v1_va, v1_te, cols_v1)
    pv2, pt2 = _fit_predict(v2_tr, v2_va, v2_te, cols_v2)
    val_auc_v1 = roc_auc_metric(yv, pv1)
    val_auc_v2 = roc_auc_metric(yv, pv2)
    print(f"\n[VAL decision]  v1 AUC={val_auc_v1:.4f}   v2 AUC={val_auc_v2:.4f}   "
          f"Δ={val_auc_v2-val_auc_v1:+.4f}")

    # ablation: each advanced family on top of v1 (val AUC), to see what helps
    print("\n[VAL ablation] v1 + one advanced family:")
    base_val = val_auc_v1
    contrib = []
    for fam in ADV_FAMILY_PREFIXES:
        cols = cols_v1 + advanced_family_columns(cols_v2, [fam])
        pv, _ = _fit_predict(v2_tr, v2_va, v2_te, cols)  # v2_tr has both v1 + adv cols
        a = roc_auc_metric(yv, pv)
        contrib.append((fam, a - base_val))
        print(f"  +{fam:18s} val AUC={a:.4f}  ({a-base_val:+.4f})")
    contrib.sort(key=lambda x: -x[1])

    # ---- candidate configs — pick the winner on VALIDATION --------------
    helpful = ["adv_timesince", "adv_workflow", "adv_sequence"]
    cols_focused = cols_v1 + advanced_family_columns(cols_v2, helpful)
    candidates: dict[str, tuple] = {"full_v2": (val_auc_v2, pv2, pt2, len(cols_v2), False)}
    pvf, ptf = _fit_predict(v2_tr, v2_va, v2_te, cols_focused)
    candidates["focused"] = (roc_auc_metric(yv, pvf), pvf, ptf, len(cols_focused), False)
    pvt, ptt = _fit_predict(v2_tr, v2_va, v2_te, cols_focused, tuned=True)
    candidates["focused_tuned"] = (roc_auc_metric(yv, pvt), pvt, ptt, len(cols_focused), True)

    print("\n[VAL candidates] (selection is on validation only)")
    for name, (va, _, _, ncol, tu) in candidates.items():
        print(f"  {name:16s} cols={ncol:3d} tuned={tu!s:5s} val AUC={va:.4f}")
    winner = max(candidates, key=lambda k: candidates[k][0])
    wval, _, wpt, wncol, wtuned = candidates[winner]
    print(f"  winner by VAL: {winner} (val AUC={wval:.4f}, {wncol} features)")

    # ---- TEST: winner vs v1, paired, scored once ------------------------
    print("\n[TEST — winner vs v1, paired instance-bootstrap, scored once]")
    res = paired_bootstrap_delta(v1_te, pt1, wpt, n_boot=args.n_boot)
    yt = v1_te["y_fail"].to_numpy(int)
    m2 = prefix_metrics(yt, wpt)
    print(f"  v1 test AUC      = {res['auc_v1']:.4f}")
    print(f"  {winner} test AUC = {res['auc_v2']:.4f}   (AUPRC={m2['auprc']:.4f}, ECE={m2['ece']:.4f})")
    print(f"  Δ AUC (winner - v1) = {res['delta_point']:+.4f}  "
          f"95% CI [{res['delta_lo']:+.4f}, {res['delta_hi']:+.4f}]")
    print(f"  winner better in {res['frac_v2_better']*100:.1f}% of {res['n_boot']} bootstraps")
    sig = res["delta_lo"] > 0
    print(f"  => {'SIGNIFICANT improvement (CI excludes 0)' if sig else 'NOT significant at 95%'}")

    from localguard.utils import RESULTS_OFFLINE, write_json
    write_json(RESULTS_OFFLINE / "full" / "v2_feature_ab.json", {
        "n_features_v1": len(cols_v1), "n_features_v2": len(cols_v2),
        "val_auc_v1": val_auc_v1, "val_auc_v2": val_auc_v2,
        "val_ablation_sorted": contrib,
        "candidates_val": {k: v[0] for k, v in candidates.items()},
        "winner": winner, "winner_val_auc": wval,
        "test": res, "winner_test_auprc": m2["auprc"], "winner_test_ece": m2["ece"],
    })
    print("\n[v2] wrote results/offline/full/v2_feature_ab.json")


if __name__ == "__main__":
    main()
