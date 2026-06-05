#!/usr/bin/env python3
"""Evaluate trained monitors: metrics, calibration, thresholds, first-alert,
and figure data (Phases 8-9, plus inputs for Phases 16-17).

  python scripts/evaluate_monitor.py --config configs/offline_small.yaml

Thresholds are selected on validation (calibrated risk) and applied once to test.
All confidence intervals bootstrap whole instance_id groups.
"""

from __future__ import annotations

import argparse

import _bootstrap  # noqa: F401
import numpy as np
import pandas as pd

from localguard.calibrate import fit_calibrator, reliability_curve
from localguard.evaluate import (
    auprc_metric,
    bootstrap_ci,
    precision_recall_at_far,
    prefix_metrics,
    risk_by_normalized_position,
    roc_auc_metric,
)
from localguard.thresholding import (
    DEFAULT_DEPLOY_POLICY,
    THRESHOLD_POLICIES,
    first_alert_metrics,
    first_alert_per_trajectory,
    select_thresholds,
)
from localguard.train import MonitorModel
from localguard.utils import REPO_ROOT, ensure_dirs, load_config, write_json

MAIN_FEATURE_DESC = {
    "baseline_majority": "none (constant)",
    "baseline_step_count_only": "step count only",
    "heuristic_rule_monitor": "hand rules",
    "logistic_regression": "structured",
    "random_forest": "structured",
    "hist_gradient_boosting": "structured",
    "structured_plus_tfidf_logistic": "structured + text",
}


def _load_split(df: pd.DataFrame, results_dir) -> dict[str, pd.DataFrame]:
    folds = pd.read_parquet(results_dir / "split_assignment.parquet")
    merged = df.merge(folds, on="prefix_id", how="inner")
    return {f: merged[merged["fold"] == f].copy() for f in ("train", "val", "test")}


def _attach_risk(df: pd.DataFrame, risk: np.ndarray) -> pd.DataFrame:
    out = df.copy()
    out["risk"] = risk
    return out


def evaluate_model(
    model: MonitorModel,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    calibration: str,
    far_targets: list[float],
    bootstrap: int,
    seed: int,
    calibrators_dir,
) -> dict:
    y_val = val_df["y_fail"].to_numpy(int)
    y_test = test_df["y_fail"].to_numpy(int)
    p_val = model.predict_proba_fail(val_df)
    p_test = model.predict_proba_fail(test_df)

    calib = fit_calibrator(calibration, p_val, y_val)
    ensure_dirs(calibrators_dir)
    calib.save(calibrators_dir / f"{MonitorModel.safe_filename(model.name)}.joblib")
    cp_val = calib.transform(p_val)
    cp_test = calib.transform(p_test)

    raw = prefix_metrics(y_test, p_test)
    cal = prefix_metrics(y_test, cp_test)

    test_risk = _attach_risk(test_df, p_test)
    auc_ci = bootstrap_ci(test_risk, roc_auc_metric, n_boot=bootstrap, seed=seed)
    ap_ci = bootstrap_ci(test_risk, auprc_metric, n_boot=bootstrap, seed=seed)

    # thresholds on calibrated validation risk; applied to calibrated test risk
    val_cal = _attach_risk(val_df, cp_val)
    test_cal = _attach_risk(test_df, cp_test)
    thresholds = select_thresholds(val_cal)
    fa: dict[str, dict] = {}
    for policy, thr in thresholds.items():
        fa[policy] = first_alert_metrics(test_cal, thr).as_row()

    pr_at_far = {f"far_{int(t*100)}": precision_recall_at_far(y_test, cp_test, t) for t in far_targets}

    return {
        "name": model.name,
        "kind": model.kind,
        "features": MAIN_FEATURE_DESC.get(model.name, "+".join(model.meta.get("families", []))),
        "roc_auc": raw.get("roc_auc"),
        "roc_auc_ci": [auc_ci["lo"], auc_ci["hi"]],
        "auprc": raw.get("auprc"),
        "auprc_ci": [ap_ci["lo"], ap_ci["hi"]],
        "brier_raw": raw.get("brier"),
        "ece_raw": raw.get("ece"),
        "brier_cal": cal.get("brier"),
        "ece_cal": cal.get("ece"),
        "thresholds": thresholds,
        "first_alert": fa,
        "pr_at_far": pr_at_far,
        "n_test": int(len(test_df)),
        "test_pos_rate": float(y_test.mean()),
        "_p_test": p_test,
        "_cp_test": cp_test,
        "_cp_val": cp_val,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", required=True)
    args = ap.parse_args()

    cfg = load_config(args.config)
    seed = int(cfg.get("seed", 42))
    calibration = str(cfg.get("calibration", "isotonic"))
    far_targets = [float(x) for x in cfg.get("threshold_far_targets", [0.2, 0.1, 0.05])]
    bootstrap = int(cfg.get("bootstrap", 300))

    df = pd.read_parquet(REPO_ROOT / cfg["dataset"] if not str(cfg["dataset"]).startswith("/") else cfg["dataset"])
    results_dir = REPO_ROOT / cfg["results_dir"]
    models_dir = REPO_ROOT / cfg["models_dir"]
    calibrators_dir = REPO_ROOT / cfg["calibrators_dir"]
    figdir = results_dir / "figdata"
    ensure_dirs(results_dir, figdir)

    split = _load_split(df, results_dir)
    val_df, test_df = split["val"], split["test"]
    print(f"[eval] {cfg['name']}: val={len(val_df)} test={len(test_df)} "
          f"(test instances={test_df['instance_id'].nunique()}, pos_rate={test_df['y_fail'].mean():.3f})")

    # ---- main models -----------------------------------------------------
    main_results: list[dict] = []
    for name in cfg["models"]:
        path = models_dir / f"{MonitorModel.safe_filename(name)}.joblib"
        if not path.exists():
            print(f"[eval] WARN missing model {path}")
            continue
        model = MonitorModel.load(path)
        res = evaluate_model(model, val_df, test_df, calibration, far_targets, bootstrap, seed, calibrators_dir)
        main_results.append(res)
        ci = res["roc_auc_ci"]
        print(f"  {name:34s} AUC={res['roc_auc']:.3f} [{ci[0]:.3f},{ci[1]:.3f}] "
              f"AUPRC={res['auprc']:.3f} Brier={res['brier_cal']:.3f} ECE={res['ece_cal']:.3f}")

    # pick best deployable model by test AUC (exclude trivial baselines)
    deployable = [r for r in main_results if r["name"] not in ("baseline_majority",)]
    best = max(deployable, key=lambda r: (r["roc_auc"] or 0.0))
    print(f"[eval] best deployable model: {best['name']} (AUC={best['roc_auc']:.3f})")

    # ---- ablation models -------------------------------------------------
    ablation_results: list[dict] = []
    for families in cfg.get("ablation_families", []):
        label = "+".join(families)
        path = models_dir / f"ablation__{MonitorModel.safe_filename(label)}.joblib"
        if not path.exists():
            continue
        model = MonitorModel.load(path)
        res = evaluate_model(model, val_df, test_df, calibration, far_targets, bootstrap, seed, calibrators_dir)
        res["feature_set"] = label
        ablation_results.append(res)

    # ---- Table 1: offline monitor performance ----------------------------
    deploy_policy = DEFAULT_DEPLOY_POLICY
    t1_rows = []
    for r in main_results:
        fa = r["first_alert"].get(deploy_policy, {})
        t1_rows.append({
            "model": r["name"],
            "features": r["features"],
            "split": cfg.get("split.regime", "instance"),
            "ROC_AUC": _fmt(r["roc_auc"]),
            "ROC_AUC_lo": _fmt(r["roc_auc_ci"][0]),
            "ROC_AUC_hi": _fmt(r["roc_auc_ci"][1]),
            "AUPRC": _fmt(r["auprc"]),
            "Brier": _fmt(r["brier_cal"]),
            "ECE": _fmt(r["ece_cal"]),
            "success_FAR@T3": fa.get("success_false_alarm_rate"),
            "failed_coverage@T3": fa.get("failed_coverage"),
            "median_lead_steps@T3": fa.get("median_lead_steps"),
        })
    pd.DataFrame(t1_rows).to_csv(results_dir / "table1_offline.csv", index=False)

    # ---- Table 2: feature ablation ---------------------------------------
    t2_rows = []
    for r in ablation_results:
        fa = r["first_alert"].get(deploy_policy, {})
        t2_rows.append({
            "feature_set": r["feature_set"],
            "ROC_AUC": _fmt(r["roc_auc"]),
            "AUPRC": _fmt(r["auprc"]),
            "success_FAR@T3": fa.get("success_false_alarm_rate"),
            "median_lead_steps@T3": fa.get("median_lead_steps"),
        })
    pd.DataFrame(t2_rows).to_csv(results_dir / "table2_ablation.csv", index=False)

    # ---- Table 3: threshold tradeoff (best model) ------------------------
    t3_rows = []
    for policy in THRESHOLD_POLICIES:
        fa = best["first_alert"].get(policy, {})
        t3_rows.append({
            "threshold_policy": policy,
            "threshold": _fmt(best["thresholds"].get(policy)),
            "success_false_alarm_rate": fa.get("success_false_alarm_rate"),
            "failed_alert_rate": fa.get("failed_coverage"),
            "median_lead_steps": fa.get("median_lead_steps"),
        })
    pd.DataFrame(t3_rows).to_csv(results_dir / "table3_thresholds.csv", index=False)

    # ---- figure data -----------------------------------------------------
    _write_figure_data(figdir, best, main_results, test_df, models_dir, deploy_policy)

    # ---- consolidated JSON (strip heavy arrays) --------------------------
    def clean(r: dict) -> dict:
        return {k: v for k, v in r.items() if not k.startswith("_")}

    write_json(results_dir / "offline_results.json", {
        "config": cfg.data,
        "deploy_policy": deploy_policy,
        "best_model": best["name"],
        "main": [clean(r) for r in main_results],
        "ablation": [clean(r) for r in ablation_results],
    })
    print(f"[eval] wrote tables + figure data + offline_results.json -> {results_dir}")


def _fmt(x, nd: int = 4):
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return None
    return round(float(x), nd)


def _write_figure_data(figdir, best, main_results, test_df, models_dir, deploy_policy):
    y_test = test_df["y_fail"].to_numpy(int)

    # Figure 1: calibrated risk vs normalized position (best model)
    tr = test_df.copy()
    tr["risk"] = best["_cp_test"]
    risk_by_normalized_position(tr).to_csv(figdir / "risk_by_position.csv", index=False)

    # Figure 2: PR curves for a representative model set
    from sklearn.metrics import precision_recall_curve

    pr_models = {"heuristic_rule_monitor", "logistic_regression", "random_forest",
                 "hist_gradient_boosting", "structured_plus_tfidf_logistic"}
    pr_data = {}
    for r in main_results:
        if r["name"] in pr_models:
            prec, rec, _ = precision_recall_curve(y_test, r["_cp_test"])
            # subsample to keep files small
            step = max(1, len(prec) // 300)
            pr_data[r["name"]] = {
                "precision": [round(float(x), 4) for x in prec[::step]],
                "recall": [round(float(x), 4) for x in rec[::step]],
                "auprc": _fmt(r["auprc"]),
            }
    write_json(figdir / "pr_curves.json", pr_data)

    # Figure 3: lead-time histogram (best model at deploy threshold)
    thr = best["thresholds"].get(deploy_policy, 0.5)
    per = first_alert_per_trajectory(_attach_risk(test_df, best["_cp_test"]), thr)
    failed_alerted = per[(per["y_fail"] == 1) & (per["alerted"])]
    failed_alerted[["trajectory_id", "n_total_steps", "first_alarm_step", "lead_steps", "lead_fraction"]].to_csv(
        figdir / "leadtime.csv", index=False)

    # reliability curve raw vs calibrated (best model)
    rel_raw = reliability_curve(y_test, best["_p_test"])
    rel_cal = reliability_curve(y_test, best["_cp_test"])
    write_json(figdir / "reliability.json", {"raw": rel_raw, "calibrated": rel_cal,
                                             "model": best["name"]})

    # Figure 5: logistic-regression feature importance
    lr_path = models_dir / "logistic_regression.joblib"
    if lr_path.exists():
        lr = MonitorModel.load(lr_path)
        try:
            clf = lr.estimator[-1]
            coefs = clf.coef_[0]
            names = [c.replace("f__", "") for c in lr.numeric_cols]
            imp = pd.DataFrame({"feature": names, "coef": coefs})
            imp["abs"] = imp["coef"].abs()
            imp.sort_values("abs", ascending=False).drop(columns="abs").to_csv(
                figdir / "feature_importance.csv", index=False)
        except Exception as exc:
            print(f"[eval] feature importance skipped: {exc}")


if __name__ == "__main__":
    main()
