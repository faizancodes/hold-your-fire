#!/usr/bin/env python3
"""Deterministic monitor-replay over captured live trajectories (online abstention test).

Replays the UNGATED vs GATED (abstaining) monitor over the SAME shadow-captured
risk sequences and counts interventions on successful runs (disruption risk) vs
failed runs (coverage). Gate + thresholds are taken from the OFFLINE abstention
study (never tuned on this live data). The comparison is confound-free because both
configs see identical trajectories.

  python scripts/run_monitor_replay.py
"""

from __future__ import annotations

import argparse

import _bootstrap  # noqa: F401

from localguard.utils import REPO_ROOT, RESULTS_ONLINE, read_json, read_jsonl, write_json


def replay(risks, step_floor, conf_floor, threshold, cooldown=3, max_int=2):
    """Return the list of intervention steps under a gate + alarm policy."""
    interventions, last = [], None
    for step, cr in sorted(risks):
        if step < step_floor:
            continue
        if abs(cr - 0.5) < conf_floor:
            continue
        if cr < threshold:
            continue
        if last is not None and (step - last) < cooldown:
            continue
        if len(interventions) >= max_int:
            break
        interventions.append(step)
        last = step
    return interventions


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--capture", default="results/online/shadow_capture.jsonl")
    ap.add_argument("--abstention", default="results/offline/full/abstention.json")
    args = ap.parse_args()

    runs = list(read_jsonl(REPO_ROOT / args.capture))
    ab = read_json(REPO_ROOT / args.abstention)
    thr_unc = ab["first_alert_unconditional"]["threshold"]
    thr_reg = ab["first_alert_regime"]["threshold"]
    S = ab["operating_point"]["step_floor"]
    C = ab["operating_point"]["conf_floor"]
    print(f"[replay] {len(runs)} captured runs | ungated: step>=5 risk>={thr_unc} | "
          f"gated: step>={S} conf>={C} risk>={thr_reg}")

    rows = []
    for r in runs:
        ung = replay(r["risks"], step_floor=5, conf_floor=0.0, threshold=thr_unc)
        gat = replay(r["risks"], step_floor=S, conf_floor=C, threshold=thr_reg)
        rows.append({**{k: r[k] for k in ("instance_id", "success", "n_steps")},
                     "ungated_n": len(ung), "ungated_first": ung[0] if ung else None,
                     "gated_n": len(gat), "gated_first": gat[0] if gat else None})

    succ = [x for x in rows if x["success"]]
    fail = [x for x in rows if not x["success"]]
    print(f"\n[replay] outcomes: {len(succ)} succeeded, {len(fail)} failed\n")
    print(f"{'task':14s} {'outcome':8s} {'ungated fires@':>14s} {'gated fires@':>14s}")
    for x in rows:
        print(f"  {x['instance_id']:12s} {'success' if x['success'] else 'FAIL':8s} "
              f"{str(x['ungated_first'])+' (x'+str(x['ungated_n'])+')':>14s} "
              f"{str(x['gated_first'])+' (x'+str(x['gated_n'])+')':>14s}")

    def rate(rows_, key):
        return sum(1 for x in rows_ if x[key] > 0) / max(1, len(rows_))

    def total(rows_, key):
        return sum(x[key] for x in rows_)

    def early_ints(rows_, cfg):
        """Interventions in the FIRST HALF of a run — the most disruptive ones."""
        return sum(1 for x in rows_ if x[f"{cfg}_first"] is not None and x[f"{cfg}_first"] < 0.5 * x["n_steps"])

    def first_frac(rows_, cfg):
        fr = [x[f"{cfg}_first"] / x["n_steps"] for x in rows_ if x[f"{cfg}_first"] is not None and x["n_steps"]]
        return round(sum(fr) / len(fr), 3) if fr else None

    out = {
        "n_success": len(succ), "n_fail": len(fail),
        "disruption_success_fire_rate": {"ungated": round(rate(succ, "ungated_n"), 3),
                                         "gated": round(rate(succ, "gated_n"), 3)},
        "disruption_total_interventions_on_success": {"ungated": total(succ, "ungated_n"),
                                                      "gated": total(succ, "gated_n")},
        "disruption_EARLY_interventions_on_success": {"ungated": early_ints(succ, "ungated"),
                                                     "gated": early_ints(succ, "gated")},
        "disruption_mean_first_alarm_fraction_on_success": {"ungated": first_frac(succ, "ungated"),
                                                          "gated": first_frac(succ, "gated")},
        "coverage_fail_fire_rate": {"ungated": round(rate(fail, "ungated_n"), 3),
                                    "gated": round(rate(fail, "gated_n"), 3)},
        "coverage_mean_first_alarm_fraction_on_fail": {"ungated": first_frac(fail, "ungated"),
                                                      "gated": first_frac(fail, "gated")},
    }
    print("\n[replay] === DISRUPTION on runs that SUCCEEDED (lower / later is safer) ===")
    print(f"  ever-fired rate:           ungated {out['disruption_success_fire_rate']['ungated']:.0%}   gated {out['disruption_success_fire_rate']['gated']:.0%}")
    print(f"  total interventions:       ungated {out['disruption_total_interventions_on_success']['ungated']}      gated {out['disruption_total_interventions_on_success']['gated']}")
    print(f"  EARLY (first-half) interv: ungated {out['disruption_EARLY_interventions_on_success']['ungated']}      gated {out['disruption_EARLY_interventions_on_success']['gated']}   <- the disruptive ones")
    print(f"  mean first-alarm position: ungated {out['disruption_mean_first_alarm_fraction_on_success']['ungated']}   gated {out['disruption_mean_first_alarm_fraction_on_success']['gated']}   (later = safer)")
    print("[replay] === COVERAGE on runs that FAILED (higher is better) ===")
    print(f"  caught rate:               ungated {out['coverage_fail_fire_rate']['ungated']:.0%}   gated {out['coverage_fail_fire_rate']['gated']:.0%}")

    write_json(RESULTS_ONLINE / "monitor_replay.json", {"summary": out, "per_run": rows})
    print(f"\n[replay] wrote results/online/monitor_replay.json")


if __name__ == "__main__":
    main()
