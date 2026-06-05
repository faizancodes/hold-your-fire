"""Does a monitor-gated 'reconsider' MESSAGE flip outcomes on real hard tasks?

The MI work proved you can't STEER a capable model out of a competence failure. But a *message*
can inject a STRATEGY the model has but isn't using. So: when the agent is detected flailing on a
LiveCodeBench problem, inject a 'stop tweaking, reconsider from scratch' nudge and measure whether
it solves more than the control (normal test-failure feedback).

Paired design: for each (problem, seed), control and treatment share an identical first attempt
(same seed) and diverge only at the intervention -> the lift isolates the message's effect. We run
treatment only on episodes where control FLAILED (failed attempt 0) -- the ones where it applies.

  mech_interp/.venv/bin/python -u -m mech_interp.lcb_intervene <repo> <difficulties> <n> <seeds> <temp>
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

from mech_interp.mlx_wrapper import MLXModel
from mech_interp.lcb_data import load_problems
from mech_interp.lcb_env import run_tests, solved
from mech_interp.recovery_lcb import SYS, fmt_problem, extract_code

OUT = Path(__file__).parent / "results"
BUDGET = int(os.environ.get("LCB_BUDGET", "4"))
MAXTOK = int(os.environ.get("LCB_MAXTOK", "340"))
FLAIL_AFTER = int(os.environ.get("LCB_FLAIL_AFTER", "1"))   # inject after this many failures

RECONSIDER = (
    "\n\n[MONITOR] You have failed this problem more than once with the same approach. Do NOT just "
    "tweak your previous code. Step back and reconsider from scratch: (1) re-read the problem "
    "statement and EVERY constraint carefully; (2) check edge cases you may be missing (large "
    "inputs, zero / negative / boundary values, ties, empty input); (3) reconsider whether your "
    "overall ALGORITHM is correct and try a fundamentally different approach if needed. Then write "
    "the complete corrected program.")


def feedback(res):
    ff = res["first_fail"] or {}
    return (f"Your program is wrong. For input:\n{(ff.get('input') or '')[:200]}\nexpected:\n"
            f"{(ff.get('expected') or '')[:160]}\nbut it produced:\n"
            f"{((ff.get('got') or ff.get('err')) or '')[:200]}\nFind the bug and reply with the full "
            "corrected Python program.")


def episode(mw, p, seed, temp, intervene):
    msgs = [{"role": "system", "content": SYS}, {"role": "user", "content": fmt_problem(p)}]
    attempt0_failed = False
    for attempt in range(BUDGET):
        gen = mw.generate(mw.render(msgs), max_tokens=MAXTOK, temperature=temp, seed=seed + attempt)
        code = extract_code(gen)
        res = run_tests(code, p["public"], stop_on_fail=True)
        sv = bool(res["passed"]) and solved(code, p)
        if attempt == 0:
            attempt0_failed = not sv
        if sv:
            return {"solved": True, "attempts": attempt + 1, "flailed": attempt0_failed}
        fb = feedback(res)
        if intervene and attempt >= FLAIL_AFTER:
            fb += RECONSIDER
        msgs.append({"role": "assistant", "content": gen[:1100]})
        msgs.append({"role": "user", "content": fb})
    return {"solved": False, "attempts": BUDGET, "flailed": attempt0_failed}


def main():
    t0 = time.time()
    repo = sys.argv[1] if len(sys.argv) > 1 else "mlx-community/Qwen2.5-Coder-7B-Instruct-4bit"
    diffs = sys.argv[2].split(",") if len(sys.argv) > 2 else ["easy", "medium"]
    n = int(sys.argv[3]) if len(sys.argv) > 3 else 8
    seeds = [int(x) for x in sys.argv[4].split(",")] if len(sys.argv) > 4 else [0, 1]
    temp = float(sys.argv[5]) if len(sys.argv) > 5 else 0.7

    print(f"[load] {repo}", flush=True)
    mw = MLXModel(repo)
    problems = []
    for d in diffs:
        problems += load_problems(difficulties=(d,), after="2024-12", limit=n)
    probmap = {p["id"]: p for p in problems}
    print(f"[model] {mw.n_layers}L | {len(problems)} problems {diffs} x seeds {seeds} | budget={BUDGET} "
          f"reconsider-after={FLAIL_AFTER} | {time.time()-t0:.0f}s", flush=True)

    # ---- pass 1: control (normal feedback) ----
    control = {}
    for p in problems:
        for s in seeds:
            r = episode(mw, p, s, temp, intervene=False)
            control[(p["id"], s)] = {**r, "difficulty": p["difficulty"]}
            print(f"   [ctrl] {p['id']:11} {p['difficulty']:6} s{s} solved={int(r['solved'])} "
                  f"flailed={int(r['flailed'])} | {time.time()-t0:.0f}s", flush=True)

    # ---- pass 2: treatment (reconsider nudge) on the FLAILED episodes only ----
    flailed = [k for k, r in control.items() if r["flailed"]]
    print(f"\n[treatment] re-running {len(flailed)} flailed episodes with the reconsider nudge", flush=True)
    treat = {}
    for (pid, s) in flailed:
        r = episode(mw, probmap[pid], s, temp, intervene=True)
        treat[(pid, s)] = {**r, "difficulty": probmap[pid]["difficulty"]}
        print(f"   [treat] {pid:11} {control[(pid,s)]['difficulty']:6} s{s} "
              f"ctrl_solved={int(control[(pid,s)]['solved'])} treat_solved={int(r['solved'])} | {time.time()-t0:.0f}s", flush=True)

    # ---- compare on the flailed subset (where the intervention applies) ----
    def lift(subset):
        if not subset:
            return (0, 0, 0)
        cs = sum(control[k]["solved"] for k in subset)
        ts = sum(treat[k]["solved"] for k in subset)
        return cs, ts, len(subset)

    print("\n[result] solve count on FLAILED episodes (control normal feedback vs treatment reconsider):", flush=True)
    for d in diffs + ["ALL"]:
        sub = [k for k in flailed if d == "ALL" or control[k]["difficulty"] == d]
        cs, ts, nn = lift(sub)
        print(f"   {d:7}: control {cs}/{nn}  ->  treatment {ts}/{nn}   (lift {ts-cs:+d})", flush=True)

    safe = repo.replace("/", "_").replace(":", "_")
    OUT.mkdir(exist_ok=True)
    (OUT / f"lcb_intervene_{safe}.json").write_text(json.dumps({
        "repo": repo, "diffs": diffs, "n": n, "seeds": seeds, "budget": BUDGET, "flail_after": FLAIL_AFTER,
        "n_flailed": len(flailed),
        "by_difficulty": {d: lift([k for k in flailed if control[k]["difficulty"] == d]) for d in diffs},
        "overall": lift(flailed),
        "control": {f"{k[0]}|{k[1]}": v for k, v in control.items()},
        "treatment": {f"{k[0]}|{k[1]}": v for k, v in treat.items()}}, indent=2, default=float))
    print(f"\n[done] wrote lcb_intervene_{safe}.json | {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
