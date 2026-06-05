"""Recovery direction on REAL hard tasks (LiveCodeBench) at the model's frontier.

The capable model doesn't get stuck on toy bugs — but on genuinely hard, contamination-free
competitive-programming problems it does: it writes a wrong solution, sees the failing test,
revises, and often keeps failing. Tuned to the model's ~50% frontier, the SAME problem solves on
some seeds and stays stuck on others. That gives the clean, difficulty-controlled contrast the
recovery lever needs: in-trouble decisions on trajectories that RECOVER (eventually solve) vs
decisions on trajectories that stay STUCK.

Modes:
  calibrate : run the agentic loop (no capture), report solve rate by difficulty -> pick frontier.
  harvest   : capture per-attempt activations, harvest recover-vs-stuck decisions, discover +
              causally validate the recovery direction.

  mech_interp/.venv/bin/python -u -m mech_interp.recovery_lcb <repo> <mode> <difficulties> <n> <seeds>
"""
from __future__ import annotations

import json
import os
import re
import sys
import time
from pathlib import Path

import numpy as np

from mech_interp.mlx_wrapper import MLXModel
from mech_interp.lcb_data import load_problems
from mech_interp.lcb_env import run_tests, solved
from mech_interp.run_recovery_direction import _unit, _logreg_cv_auc

OUT = Path(__file__).parent / "results"
SYS = ("You are an expert competitive programmer. Read the problem and write a COMPLETE Python 3 "
       "program that reads from standard input and prints the answer to standard output. Put the "
       "entire program inside one ```python code block and nothing else.")
BUDGET = int(os.environ.get("LCB_BUDGET", "5"))
MAXTOK = int(os.environ.get("LCB_MAXTOK", "420"))


def extract_code(gen: str) -> str:
    blocks = re.findall(r"```(?:python|py)?\s*\n(.*?)```", gen, re.S)
    if blocks:
        return blocks[-1].strip()
    return gen.strip().strip("`").strip()


def fmt_problem(p: dict) -> str:
    ex = "\n\n".join(f"Input:\n{t['input']}\nExpected output:\n{t['output']}" for t in p["public"][:2])
    return f"{p['statement'].strip()}\n\nExamples:\n{ex}\n\nWrite the complete program."


def episode(mw, p, seed, temp, capture=False, steer=None):
    """steer = (layer, vec, alpha) applies the steering vector during every generation in the
    episode (used by the causal outcome test: does steering a stuck episode make it solve?)."""
    msgs = [{"role": "system", "content": SYS},
            {"role": "user", "content": fmt_problem(p)}]
    decs, solved_flag = [], False
    for attempt in range(BUDGET):
        ids = mw.render(msgs)
        ptxt = mw.tokenizer.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False) if capture else None
        caps = mw.capture_resid(ids, list(range(mw.n_layers))) if capture else None
        if steer is not None:
            with mw.steering(steer[0], steer[1], steer[2]):
                gen = mw.generate(ids, max_tokens=MAXTOK, temperature=temp, seed=seed + attempt)
        else:
            gen = mw.generate(ids, max_tokens=MAXTOK, temperature=temp, seed=seed + attempt)
        code = extract_code(gen)
        res = run_tests(code, p["public"], stop_on_fail=True)
        sv = bool(res["passed"]) and solved(code, p)
        decs.append({"caps": caps, "prompt": ptxt, "attempt": attempt, "pub_ok": res["n_ok"],
                     "pub_total": res["total"], "passed_public": bool(res["passed"]), "solved_now": sv,
                     "in_trouble": attempt >= 1, "id": p["id"], "difficulty": p["difficulty"], "seed": seed})
        if sv:
            solved_flag = True
            break
        ff = res["first_fail"] or {}
        fb = (f"Your program is wrong. For input:\n{(ff.get('input') or '')[:200]}\nexpected output:\n"
              f"{(ff.get('expected') or '')[:160]}\nbut it produced:\n{((ff.get('got') or ff.get('err')) or '')[:200]}\n"
              "Find the bug and reply with the full corrected Python program.")
        msgs.append({"role": "assistant", "content": gen[:1100]})
        msgs.append({"role": "user", "content": fb})
    return decs, solved_flag


def run_episodes(mw, problems, seeds, temp, capture, t0):
    rows = []  # (problem, seed, solved, n_attempts, decisions)
    for p in problems:
        for s in seeds:
            decs, sv = episode(mw, p, s, temp, capture=capture)
            rows.append({"id": p["id"], "difficulty": p["difficulty"], "seed": s,
                         "solved": sv, "attempts": len(decs), "decs": decs})
            print(f"   {p['id']:12} {p['difficulty']:6} s{s} solved={int(sv)} attempts={len(decs)} | {time.time()-t0:.0f}s", flush=True)
    return rows


def main():
    t0 = time.time()
    repo = sys.argv[1] if len(sys.argv) > 1 else "mlx-community/Qwen2.5-Coder-7B-Instruct-4bit"
    mode = sys.argv[2] if len(sys.argv) > 2 else "calibrate"
    diffs = sys.argv[3].split(",") if len(sys.argv) > 3 else ["easy", "medium", "hard"]
    n = int(sys.argv[4]) if len(sys.argv) > 4 else 4
    seeds = [int(x) for x in sys.argv[5].split(",")] if len(sys.argv) > 5 else [0, 1]
    temp = float(sys.argv[6]) if len(sys.argv) > 6 else 0.7

    print(f"[load] {repo}", flush=True)
    mw = MLXModel(repo)
    print(f"[model] layers={mw.n_layers} d={mw.d_model} | mode={mode} diffs={diffs} n={n} seeds={seeds} temp={temp}", flush=True)

    problems = []
    for diff in diffs:
        problems += load_problems(difficulties=(diff,), after="2024-12", limit=n)
    print(f"[data] {len(problems)} problems ({[ (d, sum(p['difficulty']==d for p in problems)) for d in diffs]})", flush=True)

    rows = run_episodes(mw, problems, seeds, temp, capture=(mode == "harvest"), t0=t0)

    # solve-rate report (frontier calibration)
    print("\n[calibration] solve rate by difficulty:", flush=True)
    for diff in diffs:
        rs = [r for r in rows if r["difficulty"] == diff]
        sr = np.mean([r["solved"] for r in rs]) if rs else float("nan")
        # frontier problems = solved on some seeds, not all
        byprob = {}
        for r in rs:
            byprob.setdefault(r["id"], []).append(r["solved"])
        mixed = sum(1 for v in byprob.values() if 0 < sum(v) < len(v))
        print(f"   {diff:7}: solve {sr*100:4.0f}%  | mixed(frontier) problems {mixed}/{len(byprob)}", flush=True)

    safe = repo.replace("/", "_").replace(":", "_")
    summary = {"repo": repo, "mode": mode, "temp": temp, "seeds": seeds,
               "by_difficulty": {d: float(np.mean([r["solved"] for r in rows if r["difficulty"] == d]) or 0)
                                 for d in diffs},
               "rows": [{k: v for k, v in r.items() if k != "decs"} for r in rows]}
    OUT.mkdir(exist_ok=True)
    (OUT / f"lcb_calibrate_{safe}.json").write_text(json.dumps(summary, indent=2, default=float))

    if mode != "harvest":
        print(f"\n[done] calibration written | {time.time()-t0:.0f}s", flush=True)
        return

    # ---- harvest: PROGRESS-level contrast (abundant at the frontier) ----
    # Episode-level solve/fail is too sparse (the model one-shots or stays stuck). Instead label
    # each in-trouble revision by whether it was PRODUCTIVE: did the attempt it produced pass more
    # public tests than the previous attempt? Productive revision (recover) vs flailing (stuck) is
    # the real recovery disposition, and every struggling episode supplies both.
    recover, stuck = [], []
    for r in rows:
        ds = r["decs"]
        for i, d in enumerate(ds):
            if not d["in_trouble"]:
                continue
            prev = ds[i - 1]["pub_ok"] if i > 0 else 0
            (recover if d["pub_ok"] > prev else stuck).append(d)
    print(f"\n[harvest] PRODUCTIVE-revision={len(recover)} FLAILING-revision={len(stuck)}", flush=True)
    if len(recover) < 8 or len(stuck) < 8:
        print("[abort] too few in-trouble decisions on both sides for a reliable contrast.", flush=True)
        return

    pool = stuck + recover
    y = np.array([0] * len(stuck) + [1] * len(recover))
    groups = [d["id"] for d in pool]
    N, D = mw.n_layers, mw.d_model
    acts = {L: np.stack([d["caps"][L] for d in pool]) for L in range(N)}
    print("[B] per-layer probe (problem-grouped CV AUC) ...", flush=True)
    perlayer = []
    for L in range(N):
        auc, _ = _logreg_cv_auc(acts[L], y, groups)
        perlayer.append({"L": L, "auc": auc, "dom": _unit(acts[L][y == 1].mean(0) - acts[L][y == 0].mean(0))})
    ranked = sorted([p for p in perlayer if not np.isnan(p["auc"])], key=lambda p: p["auc"], reverse=True)
    print("[B] top layers: " + ", ".join(f"L{p['L']}={p['auc']:.2f}" for p in ranked[:6]), flush=True)

    # save a reusable bundle so the (slow) causal test can run OFFLINE without re-harvesting
    safe = repo.replace("/", "_").replace(":", "_")
    _bundle = {f"L{L}": acts[L] for L in range(N)}
    _bundle["y"] = y
    np.savez_compressed(OUT / f"lcb_acts_{safe}.npz", **_bundle)
    (OUT / f"lcb_meta_{safe}.json").write_text(json.dumps(
        [{"id": d["id"], "pub_ok": d["pub_ok"], "prompt": d.get("prompt") or "", "label": int(yy)}
         for d, yy in zip(pool, y)]))
    bL = ranked[0]["L"]
    np.save(OUT / f"recovery_lcb_dir_{safe}_L{bL}.npy", perlayer[bL]["dom"])
    (OUT / f"recovery_lcb_{safe}.json").write_text(json.dumps(
        {"repo": repo, "productive": len(recover), "flailing": len(stuck), "best_layer": bL,
         "stageB_top": [{"L": p["L"], "auc": p["auc"]} for p in ranked[:10]]}, indent=2, default=float))
    print(f"[save] acts+meta+direction saved | best L{bL} grouped-CV AUC {ranked[0]['auc']:.3f}", flush=True)
    if not os.environ.get("LCB_OUTCOME"):
        print(f"\n[done] discovery complete; activations saved for offline causal test "
              f"(set LCB_OUTCOME=1 to run the slow in-line version) | {time.time()-t0:.0f}s", flush=True)
        return

    # ---- Stage C: the decisive OUTCOME-FLIP causal test ----------------------------
    # Hold out ~1/3 of problems. Build the direction on the REST. Re-run the held-out problems'
    # STUCK episodes with the recovery direction steered in (and random/orthogonal controls at the
    # same norm) — does steering make a previously-stuck episode actually SOLVE? This is the real
    # outcome flip on real hard tasks, not a teacher-forced proxy.
    best = ranked[0]["L"]
    all_ids = sorted(set(groups))
    test_ids = set(all_ids[::3])               # held-out problems (grouped split)
    train_mask = np.array([g not in test_ids for g in groups])
    Xb = acts[best]
    if not (train_mask & (y == 1)).any() or not (train_mask & (y == 0)).any():
        train_mask = np.ones(len(groups), bool)
    dir_best = _unit(Xb[train_mask & (y == 1)].mean(0) - Xb[train_mask & (y == 0)].mean(0))
    rng = np.random.default_rng(0)
    rand = _unit(rng.standard_normal(D))
    ortho = _unit(rng.standard_normal(D) - (rng.standard_normal(D) @ dir_best) * dir_best)
    probmap = {p["id"]: p for p in problems}
    held_stuck = [(r["id"], r["seed"]) for r in rows if not r["solved"] and r["id"] in test_ids]
    print(f"[B] direction at L{best} (grouped-CV AUC {ranked[0]['auc']:.2f})", flush=True)
    print(f"[C] outcome test: re-run {len(held_stuck)} held-out STUCK episodes under steering "
          f"(held-out problems {sorted(test_ids)})", flush=True)

    alpha = 8.0
    outcomes = {}
    for name, vec in (("none", None), ("dir", dir_best), ("random", rand), ("ortho", ortho)):
        solves = 0
        for pid, seed in held_stuck:
            steer = None if vec is None else (best, vec, alpha)
            _, sv = episode(mw, probmap[pid], seed, temp, capture=False, steer=steer)
            solves += int(sv)
        outcomes[name] = solves
        print(f"   steer={name:7} @L{best} a{alpha:g}: {solves}/{len(held_stuck)} stuck episodes now SOLVE", flush=True)

    lift = outcomes["dir"] - max(outcomes["none"], outcomes["random"], outcomes["ortho"])
    np.save(OUT / f"recovery_lcb_dir_{safe}_L{best}.npy", dir_best)
    (OUT / f"recovery_lcb_{safe}.json").write_text(json.dumps(
        {"repo": repo, "recover": len(recover), "stuck": len(stuck), "best_layer": best,
         "stageB_top": [{"L": p["L"], "auc": p["auc"]} for p in ranked[:8]],
         "held_out_problems": sorted(test_ids), "n_held_stuck": len(held_stuck),
         "outcome_flip": outcomes, "dir_lift_over_best_control": lift, "alpha": alpha}, indent=2, default=float))
    print(f"\n[result] outcome flip: dir solved {outcomes['dir']} vs none {outcomes['none']} / "
          f"random {outcomes['random']} / ortho {outcomes['ortho']}  (lift {lift:+d})", flush=True)
    print(f"[done] best layer {best} | wrote recovery_lcb_{safe}.json | {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
