#!/usr/bin/env python3
"""Train all offline monitors for a config (Phase 7).

Trains the main model roster + the feature-ablation models, runs the
shuffled-label leakage sanity check, and persists the train/val/test split so
that evaluate_monitor.py scores the *identical* split.

  python scripts/train_monitor.py --config configs/offline_small.yaml
"""

from __future__ import annotations

import argparse
import time

import _bootstrap  # noqa: F401
import pandas as pd

from localguard.evaluate import roc_auc_metric
from localguard.split import make_splits, verify_disjoint
from localguard.train import MonitorModel, train_family_model, train_model
from localguard.utils import REPO_ROOT, ensure_dirs, write_json


def _safe_auc(model: MonitorModel, df: pd.DataFrame) -> float:
    y = df["y_fail"].to_numpy()
    if len(set(y.tolist())) < 2:
        return float("nan")
    try:
        return roc_auc_metric(y, model.predict_proba_fail(df))
    except Exception:
        return float("nan")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", required=True)
    args = ap.parse_args()

    from localguard.utils import load_config

    cfg = load_config(args.config)
    seed = int(cfg.get("seed", 42))
    df = pd.read_parquet(REPO_ROOT / cfg["dataset"] if not str(cfg["dataset"]).startswith("/") else cfg["dataset"])
    print(f"[train] {cfg['name']}: {len(df)} prefix rows, "
          f"{df['instance_id'].nunique()} instances, fail_rate={df['y_fail'].mean():.3f}")

    splits = make_splits(
        df,
        regime=cfg.get("split.regime", "instance"),
        test_size=float(cfg.get("split.test_size", 0.2)),
        val_size=float(cfg.get("split.val_size", 0.2)),
        seed=seed,
    )
    verify_disjoint(splits)
    print("[train] split:", splits.summary())

    results_dir = REPO_ROOT / cfg["results_dir"]
    models_dir = REPO_ROOT / cfg["models_dir"]
    ensure_dirs(results_dir, models_dir)

    # Persist the exact fold assignment (prefix_id -> fold) for evaluation.
    fold_rows = []
    for fold, part in (("train", splits.train), ("val", splits.val), ("test", splits.test)):
        for pid in part["prefix_id"].tolist():
            fold_rows.append({"prefix_id": pid, "fold": fold})
    pd.DataFrame(fold_rows).to_parquet(results_dir / "split_assignment.parquet", index=False)
    write_json(results_dir / "split_summary.json", splits.summary())

    # ---- main models -----------------------------------------------------
    print("\n[train] main models (val ROC AUC):")
    for name in cfg["models"]:
        t0 = time.time()
        model = train_model(
            name, splits.train, seed=seed,
            text_max_features=int(cfg.get("text_max_features", 50000)),
        )
        model.save(models_dir)
        print(f"  {name:34s} val_auc={_safe_auc(model, splits.val):.3f}  ({time.time()-t0:.1f}s)")

    # ---- ablation models -------------------------------------------------
    print("\n[train] ablation feature-family models (val ROC AUC):")
    for families in cfg.get("ablation_families", []):
        model = train_family_model(splits.train, list(families), seed=seed)
        model.save(models_dir)
        print(f"  {'+'.join(families):34s} val_auc={_safe_auc(model, splits.val):.3f}")

    # ---- shuffled-label leakage sanity (expect AUC ~ 0.5) ----------------
    shuf = train_model("logistic_regression", splits.train, seed=seed, shuffle_labels=True)
    shuf_auc = _safe_auc(shuf, splits.val)
    write_json(results_dir / "shuffle_sanity.json", {
        "model": "logistic_regression",
        "shuffled_label_val_auc": shuf_auc,
        "interpretation": "AUC near 0.5 => no leakage via features",
    })
    print(f"\n[train] shuffled-label sanity: val_auc={shuf_auc:.3f} (want ~0.5)")
    print(f"[train] saved models -> {models_dir}")


if __name__ == "__main__":
    main()
