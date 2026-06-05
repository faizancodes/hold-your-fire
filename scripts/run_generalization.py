#!/usr/bin/env python3
"""Generalization study (#1): does the monitor transfer beyond its training distribution?

(a) Cross-MODEL (within the swe-agent-llama family): train on 70b (excluding the
    instances the 8b/405b traces attempt -> pure model+task shift), test on 8b+405b.
    Report AUC transfer gap and calibration before/after recalibration.
(b) Cross-FAMILY + cross-SCAFFOLD: the offline monitor applied to the live
    qwen2.5-coder / mini-SWE-agent shadow captures. Quantify over-firing and show
    abstention mitigates it.
"""

from __future__ import annotations

import argparse

import _bootstrap  # noqa: F401
import numpy as np
import pandas as pd

from localguard.abstention import confidence
from localguard.calibrate import fit_calibrator
from localguard.evaluate import expected_calibration_error, prefix_metrics, roc_auc_metric
from localguard.train import _build_sklearn_numeric, _to_numeric_matrix, numeric_columns
from localguard.utils import REPO_ROOT, RESULTS_OFFLINE, DEFAULT_SEED, read_jsonl, write_json


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--v1", default="data/processed/prefix_offline_full.parquet")
    ap.add_argument("--capture", default="results/online/shadow_capture.jsonl")
    args = ap.parse_args()

    df = pd.read_parquet(REPO_ROOT / args.v1)
    cols = numeric_columns(df)
    rng = np.random.default_rng(DEFAULT_SEED)
    results = {}

    # ===================== (a) cross-MODEL (size) shift ==================
    big = "swe-agent-llama-70b"
    test_models = sorted(m for m in df.model_name.unique() if m != big)
    test_inst = set(df[df.model_name.isin(test_models)]["instance_id"].unique())
    # pure shift: 70b training excludes any instance the test models attempt
    train_pool = df[(df.model_name == big) & (~df.instance_id.isin(test_inst))]
    te = df[df.model_name.isin(test_models)]
    # split 70b pool by instance into train / holdout (within-dist test + calibration)
    insts = list(train_pool.groupby("instance_id").groups.keys())
    rng.shuffle(insts)
    cut = int(0.85 * len(insts))
    tr = train_pool[train_pool.instance_id.isin(set(insts[:cut]))]
    ho = train_pool[train_pool.instance_id.isin(set(insts[cut:]))]
    print(f"[xmodel] train(70b, no test-task overlap)={tr.trajectory_id.nunique()} traj; "
          f"within-dist holdout={ho.trajectory_id.nunique()} traj; "
          f"cross-model test({'+'.join(m.split('-')[-1] for m in test_models)})={te.trajectory_id.nunique()} traj "
          f"({te.instance_id.nunique()} unseen instances)")

    est = _build_sklearn_numeric("hist_gradient_boosting", DEFAULT_SEED)
    est.fit(_to_numeric_matrix(tr, cols), tr["y_fail"].to_numpy(int))
    pos = list(est.classes_).index(1)
    p_ho = est.predict_proba(_to_numeric_matrix(ho, cols))[:, pos]
    p_te = est.predict_proba(_to_numeric_matrix(te, cols))[:, pos]
    y_ho, y_te = ho["y_fail"].to_numpy(int), te["y_fail"].to_numpy(int)

    auc_within = roc_auc_metric(y_ho, p_ho)
    auc_cross = roc_auc_metric(y_te, p_te)
    # calibration: fit on within-dist holdout, apply cross-model
    cal = fit_calibrator("isotonic", p_ho, y_ho)
    ece_cross_naive = expected_calibration_error(y_te, cal.transform(p_te))
    # recalibrate on half the cross-model data, eval on the other half
    n = len(te); idx = rng.permutation(n); half = idx[: n // 2]; rest = idx[n // 2:]
    cal2 = fit_calibrator("isotonic", p_te[half], y_te[half])
    ece_cross_recal = expected_calibration_error(y_te[rest], cal2.transform(p_te[rest]))
    ece_within = expected_calibration_error(y_ho, cal.transform(p_ho))

    print(f"[xmodel] AUC within-70b={auc_within:.4f}  cross-model(8b+405b)={auc_cross:.4f}  "
          f"transfer gap={auc_within-auc_cross:+.4f}")
    print(f"[xmodel] ECE within={ece_within:.4f}  cross(naive 70b-calibrator)={ece_cross_naive:.4f}  "
          f"cross(recalibrated)={ece_cross_recal:.4f}")
    results["cross_model"] = {
        "test_models": test_models, "n_test_traj": int(te.trajectory_id.nunique()),
        "auc_within_70b": round(auc_within, 4), "auc_cross_model": round(auc_cross, 4),
        "transfer_gap": round(auc_within - auc_cross, 4),
        "ece_within": round(ece_within, 4), "ece_cross_naive": round(ece_cross_naive, 4),
        "ece_cross_recalibrated": round(ece_cross_recal, 4),
    }

    # ===================== (b) cross-FAMILY / SCAFFOLD (live qwen) ========
    cap_path = REPO_ROOT / args.capture
    if cap_path.exists():
        runs = list(read_jsonl(cap_path))
        rows = [(s, cr, int(not r["success"])) for r in runs for s, cr in r["risks"]]
        steps, risk, yfail = (np.array([x[i] for x in rows]) for i in range(3))
        succ_mask = yfail == 0
        n_succ_runs = sum(1 for r in runs if r["success"])
        auc_qwen = roc_auc_metric(yfail, risk) if len(np.unique(yfail)) > 1 else float("nan")
        print(f"\n[xfamily] live qwen2.5-coder/mini-SWE capture: {len(runs)} runs "
              f"({n_succ_runs} success), {len(rows)} prefixes")
        print(f"[xfamily] monitor prefix-AUC on qwen traces = {auc_qwen:.4f} (N small)")
        print(f"[xfamily] OVER-FIRING: mean risk on SUCCESS prefixes={risk[succ_mask].mean():.3f} "
              f"vs FAIL prefixes={risk[~succ_mask].mean():.3f}  "
              f"(Nebius offline: success-prefix risk ~0.45) -> shift inflates success risk")
        # abstention mitigation: fraction of success prefixes that clear ungated vs gated bar
        over_ung = (risk[succ_mask] >= 0.778).mean()
        over_gat = ((steps[succ_mask] >= 10) & (risk[succ_mask] >= 0.851)).mean()
        print(f"[xfamily] MITIGATION: success prefixes above alarm bar: "
              f"ungated={over_ung:.0%} -> gated(abstention)={over_gat:.0%}")
        results["cross_family_qwen"] = {
            "n_runs": len(runs), "n_success_runs": n_succ_runs, "n_prefixes": len(rows),
            "monitor_auc_on_qwen": round(float(auc_qwen), 4),
            "mean_risk_success_prefixes": round(float(risk[succ_mask].mean()), 3),
            "mean_risk_fail_prefixes": round(float(risk[~succ_mask].mean()), 3),
            "success_above_bar_ungated": round(float(over_ung), 3),
            "success_above_bar_gated": round(float(over_gat), 3),
        }

    write_json(RESULTS_OFFLINE / "full" / "generalization.json", results)
    print(f"\n[gen] wrote {RESULTS_OFFLINE/'full'/'generalization.json'}")


if __name__ == "__main__":
    main()
