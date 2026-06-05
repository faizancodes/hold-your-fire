#!/usr/bin/env python3
"""Download the Nebius trajectory dataset (sample or full). Phase 2.

Examples:
  python scripts/download_data.py --mode sample --n 1000
  python scripts/download_data.py --mode full
"""

from __future__ import annotations

import argparse

import _bootstrap  # noqa: F401  (path setup)
from localguard.ingest_nebius import DEFAULT_DATASET, download_full, download_sample


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", default=DEFAULT_DATASET)
    ap.add_argument("--mode", choices=["sample", "full"], default="sample")
    ap.add_argument("--n", type=int, default=1000, help="rows for sample mode")
    ap.add_argument("--split", default="train")
    args = ap.parse_args()

    if args.mode == "sample":
        path = download_sample(args.dataset, n=args.n, split=args.split)
        print(f"[download] wrote sample ({args.n} rows requested) -> {path}")
    else:
        print(f"[download] downloading full dataset {args.dataset} (this may take a while)...")
        path = download_full(args.dataset, split=args.split)
        print(f"[download] wrote full dataset -> {path}")


if __name__ == "__main__":
    main()
