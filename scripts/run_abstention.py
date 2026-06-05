#!/usr/bin/env python3
"""Selective-prediction / abstention study (tasks 1-3).

(1) Define the abstention gate on VALIDATION (step floor + confidence floor);
    compare variant (a) full model vs (b) model retrained on the step>=S regime.
(2) Risk-coverage curve as the headline artifact (AUC + precision@FAR vs coverage),
    with a random-abstention baseline to prove the gain is real.
(3) Recompute calibration + thresholds + first-alert on the committed regime.

All gate/threshold choices are made on validation; test is scored once.
"""

from __future__ import annotations

import argparse

import _bootstrap  # noqa: F401
import numpy as np
import pandas as pd

from localguard.abstention import (
    confidence,
    random_curve,
    selective_curve,
    step_gate_curve,
)
from localguard.calibrate import fit_calibrator
from localguard.evaluate import (
    paired_bootstrap_auc_delta,
    precision_recall_at_far,
    prefix_metrics,
    roc_auc_metric,
)
from localguard.thresholding import first_alert_metrics, threshold_for_success_false_alarm
from localguard.train import _build_sklearn_numeric, _to_numeric_matrix, numeric_columns
from localguard.utils import REPO_ROOT, RESULTS_OFFLINE, ensure_dirs, write_json


def _fit_cal(tr, va, te, cols, calib="isotonic"):
    est = _build_sklearn_numeric("hist_gradient_boosting", 42)
    est.fit(_to_numeric_matrix(tr, cols), tr["y_fail"].to_numpy(int))
    pos = list(est.classes_).index(1)
    pv = est.predict_proba(_to_numeric_matrix(va, cols))[:, pos]
    pt = est.predict_proba(_to_numeric_matrix(te, cols))[:, pos]
    cal = fit_calibrator(calib, pv, va["y_fail"].to_numpy(int))
    return cal.transform(pv), cal.transform(pt)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--v1", default="data/processed/prefix_offline_full.parquet")
    ap.add_argument("--split", default="results/offline/full/split_assignment.parquet")
    ap.add_argument("--step-floor", type=int, default=10)
    ap.add_argument("--n-boot", type=int, default=1000)
    args = ap.parse_args()

    folds = pd.read_parquet(REPO_ROOT / args.split)
    df = pd.read_parquet(REPO_ROOT / args.v1).merge(folds, on="prefix_id")
    cols = numeric_columns(df)
    tr, va, te = (df[df.fold == f].reset_index(drop=True) for f in ("train", "val", "test"))
    S = args.step_floor
    figdir = RESULTS_OFFLINE / "full" / "figdata"
    ensure_dirs(figdir)

    # ---- variant (a) full model -----------------------------------------
    cav_a, cat_a = _fit_cal(tr, va, te, cols)
    # ---- variant (b) regime model (train+calibrate on step>=S) ----------
    cav_b, cat_b = _fit_cal(tr[tr.prefix_step >= S], va[va.prefix_step >= S],
                            te[te.prefix_step >= S], cols)

    yv, yt = va["y_fail"].to_numpy(int), te["y_fail"].to_numpy(int)
    ps_t = te["prefix_step"].to_numpy()
    overall_auc = roc_auc_metric(yt, cat_a)
    print(f"[abstain] unconditional test AUC={overall_auc:.4f}  ({len(te)} prefixes)")

    # ===== Task 1: variant (a) vs (b) on the SAME committed regime (step>=S)
    mask_a = ps_t >= S
    te_reg = te[mask_a].reset_index(drop=True)
    auc_a_reg = roc_auc_metric(yt[mask_a], cat_a[mask_a])
    auc_b_reg = roc_auc_metric(te_reg["y_fail"].to_numpy(int), cat_b)
    pair = paired_bootstrap_auc_delta(te_reg["instance_id"].to_numpy(),
                                      te_reg["y_fail"].to_numpy(int),
                                      cat_a[mask_a], cat_b, n_boot=args.n_boot)
    print(f"[task1] regime step>={S} (coverage={mask_a.mean():.1%}): "
          f"(a) full={auc_a_reg:.4f}  (b) regime-retrained={auc_b_reg:.4f}  "
          f"Δ(b-a)={pair['delta']:+.4f} CI[{pair['delta_lo']:+.4f},{pair['delta_hi']:+.4f}] "
          f"sig={pair['significant']}")
    better = "b" if auc_b_reg >= auc_a_reg else "a"
    cat_best = cat_b if better == "b" else cat_a[mask_a]  # committed-regime preds
    print(f"[task1] using variant ({better}) for the committed regime")

    # ===== Task 2: risk-coverage curves (variant a calibrated, on TEST) =====
    sel = selective_curve(yt, cat_a)
    rnd = random_curve(yt, cat_a)
    stp = step_gate_curve(yt, cat_a, ps_t, [0, 3, 5, 8, 10, 15, 20, 25, 30])
    # save figdata
    rc = pd.DataFrame(sel).rename(columns={"auc": "auc_selective"})
    rc["auc_random"] = pd.DataFrame(rnd)["auc"]
    rc.to_csv(figdir / "risk_coverage.csv", index=False)
    pd.DataFrame(stp).to_csv(figdir / "risk_coverage_stepgate.csv", index=False)
    print("\n[task2] risk-coverage (confidence-selective vs random):")
    for s, r in zip(sel, rnd):
        if abs(s["coverage"] - round(s["coverage"], 1)) < 0.03:
            print(f"   coverage={s['coverage']:.0%}  selective AUC={s['auc']:.4f}  random AUC={r['auc']:.4f}")
    print("[task2] step-gate curve:")
    for s in stp:
        print(f"   step>={s['step_floor']:2d}  coverage={s['coverage']:.0%}  AUC={s['auc']:.4f}")

    # ===== Operating point: pick conf_floor on VAL for ~50% coverage within regime
    conf_v = confidence(cav_a)[va["prefix_step"].to_numpy() >= S]
    c_thresh = float(np.quantile(conf_v, 0.5)) if len(conf_v) else 0.0  # keep top 50% by confidence
    commit_te = (ps_t >= S) & (confidence(cat_a) >= c_thresh)
    op_auc = roc_auc_metric(yt[commit_te], cat_a[commit_te]) if commit_te.sum() > 50 else float("nan")
    op_cov = commit_te.mean()
    pr = precision_recall_at_far(yt[commit_te], cat_a[commit_te], 0.10)
    print(f"\n[op] gate step>={S} & conf>={c_thresh:.3f} (chosen on val): "
          f"test coverage={op_cov:.1%}  AUC={op_auc:.4f}  precision@FAR10={pr['precision']:.3f} recall={pr['recall']:.3f}")

    # ===== Task 3: thresholds + first-alert on regime vs unconditional =====
    print("\n[task3] first-alert (T3, threshold chosen on each regime's val):")
    # NB: thresholds chosen on VAL of each regime, applied to TEST
    va_reg = va[va.prefix_step >= S]
    thr_uncond = threshold_for_success_false_alarm(_attach(va, cav_a), max_far=0.10)
    thr_reg = threshold_for_success_false_alarm(
        _attach(va_reg, cav_a[va["prefix_step"].to_numpy() >= S]), max_far=0.10)
    a_unc = _first_alert_at(te, cat_a, thr_uncond, "unconditional (all prefixes)")
    a_reg = _first_alert_at(te[mask_a], cat_a[mask_a], thr_reg, f"abstain step<{S}")
    # calibration on regime vs unconditional
    cal_unc = prefix_metrics(yt, cat_a)
    cal_reg = prefix_metrics(yt[mask_a], cat_a[mask_a])
    print(f"[task3] calibration  unconditional ECE={cal_unc['ece']:.4f} Brier={cal_unc['brier']:.4f} | "
          f"regime ECE={cal_reg['ece']:.4f} Brier={cal_reg['brier']:.4f}")

    write_json(RESULTS_OFFLINE / "full" / "abstention.json", {
        "step_floor": S,
        "unconditional_test_auc": round(overall_auc, 4),
        "task1_variant_compare": {"auc_full_on_regime": round(auc_a_reg, 4),
                                  "auc_regime_retrained": round(auc_b_reg, 4),
                                  "paired": pair, "regime_coverage": round(float(mask_a.mean()), 4)},
        "operating_point": {"step_floor": S, "conf_floor": round(c_thresh, 4),
                            "test_coverage": round(float(op_cov), 4), "test_auc": round(float(op_auc), 4),
                            "precision_at_far10": round(pr["precision"], 4), "recall_at_far10": round(pr["recall"], 4)},
        "first_alert_unconditional": a_unc,
        "first_alert_regime": a_reg,
        "calibration_unconditional": {"ece": cal_unc["ece"], "brier": cal_unc["brier"]},
        "calibration_regime": {"ece": cal_reg["ece"], "brier": cal_reg["brier"]},
        "risk_coverage_selective": sel, "risk_coverage_random": rnd, "step_gate_curve": stp,
    })
    print(f"\n[abstain] wrote {RESULTS_OFFLINE/'full'/'abstention.json'}")


def _attach(df, risk):
    d = df.copy(); d["risk"] = risk
    return d


def _first_alert_at(df, risk, thr, label):
    d = df.copy(); d["risk"] = risk
    m = first_alert_metrics(d, thr)
    print(f"   {label:30s} thr={thr:.3f}  success_FAR={m.success_false_alarm_rate:.3f} "
          f"failed_cov={m.failed_coverage:.3f}  median_lead={m.median_lead_steps:.1f}")
    return {"label": label, "threshold": round(float(thr), 4),
            "success_far": round(m.success_false_alarm_rate, 4),
            "failed_coverage": round(m.failed_coverage, 4),
            "median_lead": round(m.median_lead_steps, 2), "n_failed": m.n_failed, "n_success": m.n_success}


if __name__ == "__main__":
    main()
