#!/usr/bin/env python3
"""Inspect the downloaded dataset sample and save a schema report. Phase 2.

Reports column names/types, label balance, model_name distribution, trajectory
length distribution, and normalization success on the sample. Writes a JSON
report to results/offline/dataset_schema.json.
"""

from __future__ import annotations

import argparse
from collections import Counter

import _bootstrap  # noqa: F401
from localguard.ingest_nebius import load_raw_rows
from localguard.normalize import normalize_rows
from localguard.utils import RESULTS_OFFLINE, ensure_dirs, write_json


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input", default="sample", help="sample | full | fixtures | <path>")
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    rows = load_raw_rows(args.input, limit=args.limit)
    print(f"[inspect] loaded {len(rows)} raw rows from '{args.input}'")
    if not rows:
        print("[inspect] no rows; run download_data.py first.")
        return

    col_types: dict[str, Counter] = {}
    for r in rows:
        for k, v in r.items():
            col_types.setdefault(k, Counter())[type(v).__name__] += 1

    targets = Counter(bool(r.get("target")) for r in rows)
    models = Counter(r.get("model_name") for r in rows)
    exit_status = Counter(r.get("exit_status") for r in rows)

    norm = normalize_rows(rows)
    n_steps = [t.n_steps for t in norm]
    n_empty = sum(1 for t in norm if t.n_steps == 0)
    action_types: Counter = Counter()
    for t in norm:
        for s in t.steps:
            action_types[s.action_type] += 1

    n_steps_sorted = sorted(n_steps)

    def pct(p: float) -> int:
        if not n_steps_sorted:
            return 0
        idx = min(len(n_steps_sorted) - 1, int(p * len(n_steps_sorted)))
        return n_steps_sorted[idx]

    report = {
        "n_rows": len(rows),
        "columns": {k: dict(v) for k, v in col_types.items()},
        "label_balance": {str(k): v for k, v in targets.items()},
        "fail_rate": round(targets.get(False, 0) / max(1, len(rows)), 4),
        "model_name_counts": {str(k): v for k, v in models.most_common(20)},
        "exit_status_counts": {str(k): v for k, v in exit_status.most_common(20)},
        "n_steps": {
            "min": min(n_steps) if n_steps else 0,
            "p50": pct(0.50),
            "p90": pct(0.90),
            "p99": pct(0.99),
            "max": max(n_steps) if n_steps else 0,
            "mean": round(sum(n_steps) / max(1, len(n_steps)), 2),
            "n_empty_trajectories": n_empty,
        },
        "action_type_counts": dict(action_types.most_common()),
    }

    ensure_dirs(RESULTS_OFFLINE)
    out = RESULTS_OFFLINE / "dataset_schema.json"
    write_json(out, report)

    print(f"[inspect] columns: {list(col_types)}")
    print(f"[inspect] label balance (target): {dict(targets)}  fail_rate={report['fail_rate']}")
    print(f"[inspect] n_steps p50={report['n_steps']['p50']} p90={report['n_steps']['p90']} "
          f"max={report['n_steps']['max']} empty={n_empty}")
    print(f"[inspect] action types: {dict(action_types.most_common())}")
    print(f"[inspect] wrote {out}")


if __name__ == "__main__":
    main()
