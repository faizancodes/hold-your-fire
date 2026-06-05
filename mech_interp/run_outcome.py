"""Tier-2 outcome experiment: does breaking the loop IMPROVE OUTCOMES (real task success)?

A real agent loop on single-edit bug-fix tasks (agent_env.py). The agent issues shell commands;
we execute them and feed back real output. A loop = the same command repeated while the test
still fails (the monitor signal).

  control   : no intervention (let it loop to the step budget)
  treatment : on loop-detection, penalize the repeated command's tokens for the next action

Metric: REAL task success (test passes), plus escape + steps. Same tasks, greedy -> paired A/B.

    mech_interp/.venv/bin/python -m mech_interp.run_outcome [temperature] [seeds]
"""
from __future__ import annotations

import json, sys, time
from collections import Counter
from pathlib import Path

import numpy as np


def is_test_runner(c: str) -> bool:
    c = c.strip()
    return c.startswith("python test") or c.startswith("python3 test") or c.startswith("pytest")

from mech_interp.model_wrapper import ModelWrapper
from mech_interp.agent_env import TASKS, setup_task, run_cmd, check_solved, cleanup
from mech_interp.run_onpolicy import parse_cmd

OUT = Path(__file__).parent / "results"
BUDGET, PEN = 10, 8.0
SYS = ("You are fixing a bug in sol.py so that test.py passes. Each turn reply with EXACTLY ONE "
       "shell command in backticks, nothing else. You can use: `cat sol.py`, "
       "`sed -i 's/old/new/' sol.py` to edit, and `python test.py` to run the test.")


def episode(mw, task, treatment, temp, seed, exempt_test=True):
    d = setup_task(task)
    init = run_cmd("python test.py", d)
    msgs = [{"role": "system", "content": SYS},
            {"role": "user", "content": f"The test fails:\n{init[:240]}\nFix sol.py so test.py passes."}]
    hist, trace = [], []
    solved = False
    n_loops = n_interv = n_escape = 0
    for step in range(BUDGET):
        # loop = a command repeated within the recent window while the test still fails.
        # exempt_test=True leaves `python test.py` un-penalized (running tests is legitimate);
        # exempt_test=False (aggressive) penalizes ANY repeated command.
        recent = Counter(hist[-5:])
        looped = [c for c, n in recent.items() if n >= 2 and c and (not is_test_runner(c) if exempt_test else True)]
        is_loop = bool(looped) and not solved
        n_loops += int(is_loop)
        bad = None
        if is_loop and treatment:
            bad = set()
            for c in looped:
                bad |= set(mw.tok(c, add_special_tokens=False).input_ids)
            n_interv += 1
        gen = mw.generate_kv(mw.render(msgs), max_new_tokens=40, temperature=temp,
                             seed=seed + step, bad_ids=bad, penalty=PEN if bad else 0.0)
        cmd = parse_cmd(gen)
        if bad and cmd not in looped:
            n_escape += 1
        obs = run_cmd(cmd, d)
        solved = check_solved(task, d)
        msgs.append({"role": "assistant", "content": gen[:120]})
        msgs.append({"role": "user", "content": obs[:240]})
        hist.append(cmd); trace.append((cmd[:40], obs[:40]))
        if solved:
            break
    cleanup(d)
    return {"solved": solved, "steps": len(hist), "n_loops": n_loops,
            "n_interventions": n_interv, "n_escape": n_escape, "trace": trace}


def main():
    t0 = time.time()
    temp = float(sys.argv[1]) if len(sys.argv) > 1 else 0.0
    seeds = [int(x) for x in sys.argv[2].split(",")] if len(sys.argv) > 2 else [0]
    modes = sys.argv[3].split(",") if len(sys.argv) > 3 else ["control", "treatment"]
    exempt = sys.argv[4] != "0" if len(sys.argv) > 4 else True
    mw = ModelWrapper()
    print(f"[setup] {len(TASKS)} tasks x {modes} x seeds={seeds} temp={temp} exempt_test={exempt} | {time.time()-t0:.0f}s", flush=True)

    rows = []
    for task in TASKS:
        for seed in seeds:
            for mode in modes:
                r = episode(mw, task, mode != "control", temp, seed * 100, exempt_test=exempt)
                r.update(task=task["name"], seed=seed, mode=mode)
                rows.append(r)
                print(f"  {task['name']:12s} s{seed} {mode:9s} solved={int(r['solved'])} "
                      f"loops={r['n_loops']} interv={r['n_interventions']} esc={r['n_escape']} steps={r['steps']}",
                      flush=True)

    def agg(mode):
        rs = [r for r in rows if r["mode"] == mode]
        return {"solve_rate": float(np.mean([r["solved"] for r in rs])),
                "loop_rate": float(np.mean([r["n_loops"] > 0 for r in rs])),
                "mean_steps": float(np.mean([r["steps"] for r in rs]))}
    # recovery: paired runs where control looped & failed -> did treatment solve?
    pairs = {}
    for r in rows:
        pairs.setdefault((r["task"], r["seed"]), {})[r["mode"]] = r
    looped_fail = [(k, p) for k, p in pairs.items()
                   if "control" in p and "treatment" in p
                   and p["control"]["n_loops"] > 0 and not p["control"]["solved"]]
    recovered = sum(1 for _, p in looped_fail if p["treatment"]["solved"])
    res = {"temp": temp, "seeds": seeds, "budget": BUDGET,
           "control": agg("control"), "treatment": agg("treatment"),
           "n_control_loop_and_fail": len(looped_fail), "n_recovered_by_treatment": recovered,
           "rows": rows}
    (OUT / "outcome_results.json").write_text(json.dumps(res, indent=2))
    print(f"\n[control]   {res['control']}", flush=True)
    print(f"[treatment] {res['treatment']}", flush=True)
    print(f"[recovery] control looped+failed on {len(looped_fail)} runs; "
          f"treatment then solved {recovered}", flush=True)
    print(f"[done] wrote outcome_results.json | {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
