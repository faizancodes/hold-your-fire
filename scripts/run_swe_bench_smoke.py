#!/usr/bin/env python3
"""SWE-bench Verified Mini smoke test (Phase 14, gated).

Loads the 50-task list and reports local prerequisites for the heavy
containerized path. Does NOT run Docker images or official grading.

  python scripts/run_swe_bench_smoke.py --n 5
"""

from __future__ import annotations

import argparse

import _bootstrap  # noqa: F401

from localguard.swe_bench_mini import gate_or_explain, load_tasks, prerequisites


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n", type=int, default=5)
    args = ap.parse_args()

    print("[swe-bench-mini] prerequisites:")
    for k, v in prerequisites().items():
        print(f"  {k}: {v}")

    tasks = load_tasks(n=args.n)
    print(f"\n[swe-bench-mini] loaded {len(tasks)} tasks:")
    for t in tasks:
        print(f"  {t.instance_id:35s} repo={t.repo} commit={t.base_commit[:10]}")
        print(f"    issue: {t.problem_statement[:100].strip()}...")

    if gate_or_explain():
        print("\n[swe-bench-mini] local prerequisites MET — heavy path can run.")
    else:
        print("\n[swe-bench-mini] heavy path GATED (insufficient local resources / Docker). "
              "Use the toy-task online path (run_mini_swe_shadow.py) for local experiments.")


if __name__ == "__main__":
    main()
