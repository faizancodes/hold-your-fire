#!/usr/bin/env python3
"""Build the v2 (advanced-feature) prefix dataset from saved normalized trajectories.

Reuses the on-disk normalized trajectories so prefix_ids are IDENTICAL to v1 →
the exact same train/val/test split can be reused for a controlled A/B.

  python scripts/build_features_v2.py \
      --normalized data/interim/normalized_prefix_offline_full.jsonl \
      --output data/processed/prefix_offline_full_v2.parquet \
      --split-assignment results/offline/full/split_assignment.parquet
"""

from __future__ import annotations

import argparse
from pathlib import Path

import _bootstrap  # noqa: F401
import pandas as pd

from localguard.features_advanced import extract_features_advanced
from localguard.prefix_builder import FEATURE_PREFIX, META_COLUMNS, TEXT_COLUMN, trajectories_to_rows
from localguard.schemas import NormalizedTrajectory
from localguard.utils import LEAKAGE_FIELDS, ensure_dirs, read_jsonl


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--normalized", default="data/interim/normalized_prefix_offline_full.jsonl")
    ap.add_argument("--output", default="data/processed/prefix_offline_full_v2.parquet")
    ap.add_argument("--schedule-mode", default="default")
    ap.add_argument("--split-assignment", default="results/offline/full/split_assignment.parquet")
    args = ap.parse_args()

    trajs = [NormalizedTrajectory(**r) for r in read_jsonl(args.normalized)]
    print(f"[v2] loaded {len(trajs)} normalized trajectories from {args.normalized}")

    rows = list(trajectories_to_rows(trajs, schedule_mode=args.schedule_mode,
                                     extractor=extract_features_advanced))
    df = pd.DataFrame(rows)
    feat_cols = sorted(c for c in df.columns if c.startswith(FEATURE_PREFIX) and c != TEXT_COLUMN)
    ordered = [c for c in META_COLUMNS if c in df.columns] + feat_cols
    if TEXT_COLUMN in df.columns:
        ordered.append(TEXT_COLUMN)
    df = df[ordered]

    # leakage + integrity gates
    bad = df[df["prefix_step"] > df["n_total_steps"]]
    assert bad.empty, f"{len(bad)} rows prefix_step>n_total_steps"
    for c in feat_cols:
        name = c[len(FEATURE_PREFIX):].lower()
        assert not any(b in name for b in LEAKAGE_FIELDS), f"leak: {c}"
    print(f"[v2] {len(feat_cols)} feature columns ({sum(c.startswith(FEATURE_PREFIX+'adv') for c in feat_cols)} advanced), "
          f"no leakage, {len(df)} rows")

    # verify prefix_ids match the existing split (so the split is reusable)
    if Path(args.split_assignment).exists():
        folds = pd.read_parquet(args.split_assignment)
        v2_ids = set(df["prefix_id"])
        split_ids = set(folds["prefix_id"])
        inter = v2_ids & split_ids
        print(f"[v2] prefix_id match vs split: {len(inter)}/{len(split_ids)} split ids present "
              f"({'IDENTICAL' if inter == split_ids else 'MISMATCH'})")
        assert inter == split_ids, "prefix_ids do not match the v1 split — cannot reuse it!"

    out = Path(args.output)
    ensure_dirs(out.parent)
    df.to_parquet(out, index=False)
    print(f"[v2] wrote {out}  fail_rate={df['y_fail'].mean():.3f}")


if __name__ == "__main__":
    main()
