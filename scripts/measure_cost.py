#!/usr/bin/env python3
"""Measure the monitor's deployment cost — latency AND memory — on CPU (#2).

Backs "local, lightweight, CPU-only" with hard numbers, honestly:
  - LATENCY via the *real* deployment path `Monitor.assess(prefix_steps)` (feature
    extraction + 1-row DataFrame + predict + calibrate), reported as median + p10/p90
    over many distinct prefixes and repeated trials — not a single noisy sample. Also
    an optimized numpy-only path (lower bound) and full-test-set throughput.
  - MEMORY via psutil RSS checkpoints (interpreter+libs baseline -> +model -> serving),
    plus peak RSS. Memory checkpoints are taken BEFORE the 25k-row eval parquet loads,
    so the eval data does not pollute the deployment footprint.

Writes results/offline/full/cost.json.
"""

from __future__ import annotations

import os
import resource
import sys
import time
from pathlib import Path

import _bootstrap  # noqa: F401
import numpy as np
import pandas as pd
import psutil

from localguard.calibrate import Calibrator
from localguard.features import extract_features
from localguard.monitor import Monitor, PolicyConfig
from localguard.schemas import NormalizedTrajectory
from localguard.train import MonitorModel, _to_numeric_matrix
from localguard.utils import INTERIM_DIR, REPO_ROOT, load_config, read_jsonl, write_json

PROC = psutil.Process()


def rss_mb() -> float:
    return PROC.memory_info().rss / 1e6


def peak_rss_mb() -> float:
    r = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return r / 1e6 if sys.platform == "darwin" else r / 1e3  # bytes (mac) vs KB (linux)


def stats_ms(times_s: list[float]) -> dict[str, float]:
    a = np.array(times_s) * 1e3
    return {"median": round(float(np.median(a)), 3),
            "p10": round(float(np.percentile(a, 10)), 3),
            "p90": round(float(np.percentile(a, 90)), 3)}


def _load_prefixes(dataset_path: str, n: int, k_steps: int = 10):
    path = INTERIM_DIR / f"normalized_{Path(dataset_path).stem}.jsonl"
    out = []
    if not path.exists():
        return out
    for row in read_jsonl(path):
        t = NormalizedTrajectory(**row)
        if len(t.steps) >= k_steps:
            out.append(t.steps[:k_steps])
        if len(out) >= n:
            break
    return out


def main() -> None:
    cfg = load_config("configs/offline_full.yaml").data
    results_dir = REPO_ROOT / cfg["results_dir"]
    best = "hist_gradient_boosting"
    model_path = REPO_ROOT / cfg["models_dir"] / f"{MonitorModel.safe_filename(best)}.joblib"

    # ---- MEMORY checkpoints (before the big eval parquet) -------------------
    rss_baseline = rss_mb()  # interpreter + numpy/pandas/sklearn imported, no model
    model = MonitorModel.load(model_path)
    cal_path = REPO_ROOT / cfg["calibrators_dir"] / f"{MonitorModel.safe_filename(best)}.joblib"
    cal = Calibrator.load(cal_path) if cal_path.exists() else Calibrator(method="identity")
    rss_after_model = rss_mb()
    monitor = Monitor(model, cal, PolicyConfig(min_step=1, abstain_conf_floor=0.0))

    prefixes = _load_prefixes(cfg["dataset"], n=300)
    if prefixes:
        monitor.assess(prefixes[0])  # warm up (lazy alloc + JIT of code paths)
    rss_serving = rss_mb()  # libs + model + monitor, ready to score one prefix at a time
    model_resident = rss_after_model - rss_baseline

    # ---- LATENCY: real deployment path Monitor.assess() --------------------
    assess_s = []
    for _ in range(3):  # repeated trials to expose jitter
        for p in prefixes:
            t0 = time.perf_counter()
            monitor.assess(p)
            assess_s.append(time.perf_counter() - t0)
    # feature extraction alone
    feat_s = []
    for p in prefixes:
        t0 = time.perf_counter()
        extract_features(p)
        feat_s.append(time.perf_counter() - t0)
    # optimized numpy-only path (no pandas, no feature extraction) — a lower bound
    rows = [{f"f__{k}": v for k, v in extract_features(p).items()} for p in prefixes]
    X = _to_numeric_matrix(pd.DataFrame(rows), model.numeric_cols)
    model.estimator.predict_proba(X[:1])  # warm up
    raw_s = []
    for i in range(len(X)):
        t0 = time.perf_counter()
        model.estimator.predict_proba(X[i:i + 1])
        raw_s.append(time.perf_counter() - t0)

    # ---- THROUGHPUT + full-test-set (loads the 25k parquet; inflates RSS) ---
    df = pd.read_parquet(REPO_ROOT / cfg["dataset"])
    folds = pd.read_parquet(results_dir / "split_assignment.parquet")
    test = df.merge(folds, on="prefix_id").query("fold == 'test'").reset_index(drop=True)
    cal.transform(model.predict_proba_fail(test.iloc[:100]))  # warm up
    full_times = []
    for _ in range(5):
        t0 = time.perf_counter()
        cal.transform(model.predict_proba_fail(test))
        full_times.append(time.perf_counter() - t0)
    full_s = float(np.median(full_times))
    throughput = len(test) / full_s
    peak = peak_rss_mb()

    out = {
        "hardware": "Apple M4 MacBook (10 cores, 32 GB RAM), CPU-only (sklearn HistGradientBoosting)",
        "model": best,
        # ---- memory ----
        "model_size_on_disk_mb": round(os.path.getsize(model_path) / 1e6, 2),
        "rss_baseline_libs_mb": round(rss_baseline, 1),
        "model_resident_mb": round(model_resident, 1),
        "rss_serving_mb": round(rss_serving, 1),
        "peak_rss_mb": round(peak, 1),
        "peak_rss_note": "peak includes loading the 25,166-row eval parquet for throughput; NOT a deployment cost",
        # ---- latency (n_prefixes timed, median/p10/p90 ms) ----
        "n_prefixes_timed": len(prefixes),
        "assess_end_to_end_ms": stats_ms(assess_s),
        "feature_extraction_ms": stats_ms(feat_s),
        "raw_numpy_predict_ms": stats_ms(raw_s),
        "amortized_model_latency_ms": round(1000.0 / throughput, 4),
        "batch_throughput_prefixes_per_s": round(throughput, 0),
        "full_test_set_scoring_s_median": round(full_s, 2),
        "n_test_prefixes": int(len(test)),
        # ---- comparison ----
        "judge_latency_s_per_prefix": 11.6,
        "judge_model_weights_gb": 4.7,
    }
    out["speedup_vs_judge_realpath"] = round(out["judge_latency_s_per_prefix"] * 1000 / max(out["assess_end_to_end_ms"]["median"], 1e-6))
    out["memory_ratio_vs_judge"] = round(out["judge_model_weights_gb"] * 1000 / max(model_resident, 1e-6))
    write_json(results_dir / "cost.json", out)
    print("=== Monitor deployment cost (CPU-only) ===")
    for k, v in out.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
