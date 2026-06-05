#!/usr/bin/env python3
"""Rank the 4 AUC-lift experiments by their paired test-set gain vs v1-HGB."""

from __future__ import annotations

import _bootstrap  # noqa: F401

from localguard.utils import RESULTS_OFFLINE, read_json


def main() -> None:
    res = read_json(RESULTS_OFFLINE / "full" / "auc_lift_results.json")
    base = res.get("baseline_test_auc")
    print(f"\n=== AUC-lift ranking (baseline v1-HGB test AUC = {base}) ===\n")
    print(f"{'experiment':28s} {'test AUC':>9s} {'Δ vs base':>10s} {'95% CI':>20s} {'%win':>6s} {'sig':>4s}")
    print("-" * 84)

    rows = []
    for key, label in [
        ("ensemble", "#4 ensemble (HGB+RF+LR)"),
        ("more_data_K10", "#3 more data (≤10 fail)"),
        ("more_data_K25", "#3 more data (≤25 fail)"),
        ("seq_model_gru", "#2 sequence model (GRU)"),
        ("label_weighted", "#1 label: pos-weighting"),
    ]:
        r = res.get(key)
        if not r or "auc_new" not in r:
            continue
        ci = f"[{r['delta_lo']:+.4f},{r['delta_hi']:+.4f}]"
        rows.append((r["delta"], label, r["auc_new"], ci, r.get("frac_new_better"), r.get("significant")))
    for delta, label, auc, ci, win, sig in sorted(rows, key=lambda x: -x[0]):
        print(f"{label:28s} {auc:9.4f} {delta:+10.4f} {ci:>20s} {str(win):>6s} {'YES' if sig else 'no':>4s}")

    strata = res.get("label_position_strata")
    if strata:
        print("\n#1 cleaner label — diagnostic: baseline AUC by normalized prefix position")
        print("   (this is a DIFFERENT denominator — a reframing, not a same-test gain)")
        for k, v in strata.items():
            print(f"   {k:18s} AUC={v}")
        late = strata.get("late(>0.66)")
        if late:
            print(f"   => On determinable (late) prefixes the ceiling is ~{late} (+{late-base:.3f} vs {base})")

    print("\nNote: '#1 pos-weighting' and '#3/#2' are same-test paired comparisons; the position")
    print("strata above are a reframing of the evaluation, not a model on the same rows.")


if __name__ == "__main__":
    main()
