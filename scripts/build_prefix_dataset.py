#!/usr/bin/env python3
"""Normalize raw trajectories and build the prefix dataset. Phases 3-4.

  python scripts/build_prefix_dataset.py --input sample \
      --output data/processed/prefix_sample.parquet

Writes:
  data/interim/normalized_<input>.jsonl   (full step text; used by the audit tool)
  <output>.parquet                        (flat prefix rows: metadata + f__ features)

Runs the Phase 4 verification gate inline (no leakage columns; prefix_step <=
n_total_steps; y_fail present only as the label).
"""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

import _bootstrap  # noqa: F401
import pandas as pd

from localguard.ingest_nebius import load_raw_rows, load_sampled_full
from localguard.normalize import normalize_rows
from localguard.prefix_builder import (
    FEATURE_PREFIX,
    META_COLUMNS,
    TEXT_COLUMN,
    trajectories_to_rows,
)
from localguard.utils import (
    INTERIM_DIR,
    LEAKAGE_FIELDS,
    ensure_dirs,
    write_jsonl,
)


def _verify(df: pd.DataFrame) -> None:
    # prefix_step <= n_total_steps for every row
    bad = df[df["prefix_step"] > df["n_total_steps"]]
    assert bad.empty, f"{len(bad)} rows have prefix_step > n_total_steps"

    # y_fail only as label; never as a feature column
    feature_cols = [c for c in df.columns if c.startswith(FEATURE_PREFIX)]
    for c in feature_cols:
        name = c[len(FEATURE_PREFIX):].lower()
        assert not any(bad_word in name for bad_word in LEAKAGE_FIELDS), (
            f"leaked outcome field in feature column: {c}"
        )
    assert "y_fail" in df.columns and not any(
        c.startswith(FEATURE_PREFIX) and "y_fail" in c for c in feature_cols
    )
    print(f"[verify] OK: {len(feature_cols)} feature columns, no leakage, "
          f"prefix_step <= n_total_steps for all {len(df)} rows")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input", default="sample", help="sample | full | fixtures | <path>")
    ap.add_argument("--output", default="data/processed/prefix_sample.parquet")
    ap.add_argument("--schedule-mode", default="default", choices=["default", "dense", "sparse"])
    ap.add_argument("--limit", type=int, default=None, help="cap number of raw rows")
    # full-corpus instance-diverse sampling (ignored unless --input full)
    ap.add_argument("--max-instances", type=int, default=None)
    ap.add_argument("--max-success-per-instance", type=int, default=4)
    ap.add_argument("--max-fail-per-instance", type=int, default=4)
    ap.add_argument("--sample-seed", type=int, default=42)
    ap.add_argument("--save-normalized", action="store_true", default=True)
    ap.add_argument("--no-save-normalized", dest="save_normalized", action="store_false")
    args = ap.parse_args()

    if args.input == "full":
        raw = load_sampled_full(
            max_instances=args.max_instances,
            max_success_per_instance=args.max_success_per_instance,
            max_fail_per_instance=args.max_fail_per_instance,
            seed=args.sample_seed,
        )
        print(f"[build] sampled {len(raw)} instance-diverse rows from full corpus "
              f"(<= {args.max_fail_per_instance} fail + {args.max_success_per_instance} "
              f"success per instance, max_instances={args.max_instances})")
    else:
        raw = load_raw_rows(args.input, limit=args.limit)
        print(f"[build] loaded {len(raw)} raw rows from '{args.input}'")
    norm = normalize_rows(raw)
    n_nonempty = sum(1 for t in norm if t.n_steps > 0)
    print(f"[build] normalized {len(norm)} trajectories ({n_nonempty} non-empty)")

    if args.save_normalized:
        ensure_dirs(INTERIM_DIR)
        norm_path = INTERIM_DIR / f"normalized_{Path(args.output).stem}.jsonl"
        write_jsonl(norm_path, (t.model_dump() for t in norm))
        print(f"[build] wrote normalized trajectories -> {norm_path}")

    rows = list(trajectories_to_rows(norm, schedule_mode=args.schedule_mode))
    df = pd.DataFrame(rows)
    if df.empty:
        print("[build] no prefix rows produced (all trajectories empty?)")
        return

    # stable column order: metadata, then sorted feature columns (text last)
    feat_cols = sorted(c for c in df.columns if c.startswith(FEATURE_PREFIX) and c != TEXT_COLUMN)
    ordered = [c for c in META_COLUMNS if c in df.columns] + feat_cols
    if TEXT_COLUMN in df.columns:
        ordered.append(TEXT_COLUMN)
    df = df[ordered]

    _verify(df)

    out = Path(args.output)
    ensure_dirs(out.parent)
    df.to_parquet(out, index=False)

    label_balance = Counter(df["y_fail"].tolist())
    n_traj = df["trajectory_id"].nunique()
    n_inst = df["instance_id"].nunique()
    print(f"[build] wrote {len(df)} prefix rows -> {out}")
    print(f"[build] trajectories={n_traj} instances={n_inst} "
          f"prefixes/traj={len(df)/max(1,n_traj):.1f}")
    print(f"[build] label balance y_fail: {dict(label_balance)} "
          f"(fail_rate={label_balance.get(1,0)/max(1,len(df)):.3f})")
    print(f"[build] prefix_step stats: min={df['prefix_step'].min()} "
          f"median={int(df['prefix_step'].median())} max={df['prefix_step'].max()}")


if __name__ == "__main__":
    main()
