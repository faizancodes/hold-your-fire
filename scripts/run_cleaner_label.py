#!/usr/bin/env python3
"""Cleaner-label study: can weak supervision LIFT early-prefix AUC? (deeper #3)

Baseline = terminal label. Treatments:
  W1 trouble-gated relabel (failed prefix positive only once trouble is visible)
  W2 down-weight noisy positives (failed + no observable trouble)
Success criterion is STRATIFIED early/mid/late AUC vs baseline (paired on test),
NOT overall AUC — the whole point is the early prefixes. All training-label shaping;
evaluation is always vs the original terminal label on held-out instances.
"""

from __future__ import annotations

import argparse

import _bootstrap  # noqa: F401
import numpy as np
import pandas as pd

from localguard.evaluate import paired_bootstrap_auc_delta, roc_auc_metric
from localguard.train import _build_sklearn_numeric, _to_numeric_matrix, numeric_columns
from localguard.utils import REPO_ROOT, RESULTS_OFFLINE, ensure_dirs, write_json
from localguard.weak_labels import (
    relabel_trouble_gated,
    trouble_indicator,
    weights_downweight_noisy,
)


def _fit(tr, cols, y, sample_weight=None):
    est = _build_sklearn_numeric("hist_gradient_boosting", 42)
    est.fit(_to_numeric_matrix(tr, cols), y,
            **({"sample_weight": sample_weight} if sample_weight is not None else {}))
    return est, list(est.classes_).index(1)


def _strata(te, pt, yt):
    pos = (te["prefix_step"] / te["n_total_steps"].clip(lower=1)).to_numpy()
    out = {}
    for name, m in [("early(<=.33)", pos <= 0.33), ("mid(.33-.66)", (pos > 0.33) & (pos <= 0.66)),
                    ("late(>.66)", pos > 0.66), ("overall", np.ones(len(pos), bool))]:
        if m.sum() > 50 and len(np.unique(yt[m])) > 1:
            out[name] = round(roc_auc_metric(yt[m], pt[m]), 4)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--v1", default="data/processed/prefix_offline_full.parquet")
    ap.add_argument("--split", default="results/offline/full/split_assignment.parquet")
    ap.add_argument("--n-boot", type=int, default=1000)
    args = ap.parse_args()

    folds = pd.read_parquet(REPO_ROOT / args.split)
    df = pd.read_parquet(REPO_ROOT / args.v1).merge(folds, on="prefix_id")
    cols = numeric_columns(df)
    tr, te = df[df.fold == "train"].reset_index(drop=True), df[df.fold == "test"].reset_index(drop=True)
    yt = te["y_fail"].to_numpy(int)
    inst_te = te["instance_id"].to_numpy()

    # ---- diagnostics: how often does trouble appear, and where? ----------
    trb_tr = trouble_indicator(tr)
    y_tr = tr["y_fail"].to_numpy(int)
    pos_tr = (tr["prefix_step"] / tr["n_total_steps"].clip(lower=1)).to_numpy()
    failed = y_tr == 1
    print(f"[diag] train failed prefixes: {failed.mean():.1%} of rows")
    for nm, m in [("early", pos_tr <= 0.33), ("mid", (pos_tr > 0.33) & (pos_tr <= 0.66)), ("late", pos_tr > 0.66)]:
        fm = failed & m
        print(f"   failed & {nm:5s}: {fm.sum():6d} prefixes, {trb_tr[fm].mean():.1%} show observable trouble")
    y_w1 = relabel_trouble_gated(tr)
    print(f"[diag] positive rate  terminal={y_tr.mean():.3f}  W1(trouble-gated)={y_w1.mean():.3f}")

    # ---- baseline (terminal label) --------------------------------------
    est, pos = _fit(tr, cols, y_tr)
    pt_base = est.predict_proba(_to_numeric_matrix(te, cols))[:, pos]
    base_str = _strata(te, pt_base, yt)
    print(f"\n[baseline] stratified AUC: {base_str}")

    results = {"baseline_strata": base_str}

    # ---- W1 trouble-gated relabel ---------------------------------------
    est1, pos1 = _fit(tr, cols, y_w1)
    pt1 = est1.predict_proba(_to_numeric_matrix(te, cols))[:, pos1]
    s1 = _strata(te, pt1, yt)
    r1 = paired_bootstrap_auc_delta(inst_te, yt, pt_base, pt1, n_boot=args.n_boot)
    print(f"[W1] stratified AUC: {s1}")
    print(f"[W1] overall paired vs baseline: Δ={r1['delta']:+.4f} CI[{r1['delta_lo']:+.4f},{r1['delta_hi']:+.4f}] sig={r1['significant']}")
    results["W1_strata"] = s1; results["W1_overall_paired"] = r1

    # ---- W2 down-weight noisy positives ---------------------------------
    w = weights_downweight_noisy(tr)
    est2, pos2 = _fit(tr, cols, y_tr, sample_weight=w)
    pt2 = est2.predict_proba(_to_numeric_matrix(te, cols))[:, pos2]
    s2 = _strata(te, pt2, yt)
    r2 = paired_bootstrap_auc_delta(inst_te, yt, pt_base, pt2, n_boot=args.n_boot)
    print(f"[W2] stratified AUC: {s2}")
    print(f"[W2] overall paired vs baseline: Δ={r2['delta']:+.4f} CI[{r2['delta_lo']:+.4f},{r2['delta_hi']:+.4f}] sig={r2['significant']}")
    results["W2_strata"] = s2; results["W2_overall_paired"] = r2

    # ---- per-stratum paired deltas (the real question) ------------------
    print("\n[stratified Δ vs baseline] (the success criterion is EARLY/MID lift):")
    pos_te = (te["prefix_step"] / te["n_total_steps"].clip(lower=1)).to_numpy()
    strata_masks = {"early": pos_te <= 0.33, "mid": (pos_te > 0.33) & (pos_te <= 0.66), "late": pos_te > 0.66}
    strat_paired = {}
    for nm, m in strata_masks.items():
        for lbl, pt in [("W1", pt1), ("W2", pt2)]:
            rr = paired_bootstrap_auc_delta(inst_te[m], yt[m], pt_base[m], pt[m], n_boot=400)
            strat_paired[f"{lbl}_{nm}"] = rr
            print(f"   {lbl} {nm:5s}: baseline={rr['auc_base']:.4f} -> {rr['auc_new']:.4f} "
                  f"Δ={rr['delta']:+.4f} CI[{rr['delta_lo']:+.4f},{rr['delta_hi']:+.4f}] sig={rr['significant']}")
    results["stratified_paired"] = strat_paired

    ensure_dirs(RESULTS_OFFLINE / "full")
    write_json(RESULTS_OFFLINE / "full" / "cleaner_label.json", results)
    print(f"\n[cleaner] wrote {RESULTS_OFFLINE/'full'/'cleaner_label.json'}")


if __name__ == "__main__":
    main()
