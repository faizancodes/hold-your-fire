"""The capable-model recovery lever: investigate-when-stuck vs act-when-stuck.

A capable model does not loop (it never repeats), so the persist-vs-recover contrast has no
data for it. But it still FAILS — by flailing: editing/running without understanding (the exact
'edits the test, never reads the source' failure from the start of this project). So the
recovery lever for a capable model is "when you're stuck, INVESTIGATE the evidence instead of
blindly ACTing."

Contrast (among decisions where the model is in trouble / the test still fails):
  recover  = INVESTIGATE  (cat / grep / ls / head / find ...  -> gather evidence)
  persist  = ACT          (sed / python / echo ...            -> change or re-run code)

We harvest these from the model's own real agent-loop trajectories, discover the direction
across all layers (task-grouped CV), and causally validate by steering held-out ACT decisions
and measuring the shift toward INVESTIGATE (recovery-margin dose-response + on-policy class
flip), against random/orthogonal controls. Decodability is cheap; the causal numbers carry it.

  mech_interp/.venv/bin/python -u -m mech_interp.recovery_capable [repo] [seeds] [temp]
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

from mech_interp.mlx_wrapper import MLXModel
from mech_interp.agent_env import TASKS, setup_task, run_cmd, check_solved, cleanup
from mech_interp.harder_tasks import HARD_TASKS
from mech_interp.run_recovery_direction import _extract_cmd, _unit, _logreg_cv_auc

OUT = Path(__file__).parent / "results"
SYS = ("You are an autonomous coding agent. Commands: `cat sol.py` to read the source, "
       "`sed -i \"s/old/new/\" sol.py` to edit it, `python test.py` to run the test. "
       "Read the source, find the bug, then fix it. Reply with EXACTLY ONE shell command.")
BUDGET = 8
INVESTIGATE = {"cat", "grep", "ls", "head", "tail", "find", "find_file", "nl", "less", "wc", "diff"}
ACT = {"sed", "python", "python3", "echo", "mv", "cp", "touch", "printf", "awk"}
INV_CMDS = ["cat sol.py", "grep def sol.py", "ls", "head sol.py"]   # evidence-gathering moves


def cmd_class(cmd: str) -> str:
    v = cmd.strip().split()[0] if cmd.strip() else ""
    if v in INVESTIGATE:
        return "investigate"
    if v in ACT:
        return "act"
    return "other"


def rollout(mw, task, seed, temp):
    d = setup_task(task)
    init = run_cmd("python test.py", d)
    msgs = [{"role": "system", "content": SYS},
            {"role": "user", "content": f"test.py fails:\n{init[:180]}\nFix sol.py."}]
    decisions, solved = [], False
    for step in range(BUDGET):
        prompt = mw.tokenizer.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)
        ids = mw.render(msgs)
        caps = mw.capture_resid(ids, list(range(mw.n_layers)))
        gen = mw.generate(ids, max_tokens=24, temperature=temp, seed=seed + step)
        cmd = _extract_cmd(gen)
        obs = run_cmd(cmd, d) if cmd else "(no command)"
        solved = check_solved(task, d)
        decisions.append({"prompt": prompt, "caps": caps, "cmd": cmd, "cls": cmd_class(cmd),
                          "in_trouble": step >= 1 and not solved, "task": task["name"], "seed": seed})
        msgs.append({"role": "assistant", "content": cmd})
        msgs.append({"role": "user", "content": obs[:160]})
        if solved:
            break
    cleanup(d)
    return decisions, solved


def margin(mw, prompt, act_cmd):
    lp_act = mw.continuation_logprob(prompt, act_cmd)
    lp_inv = max(mw.continuation_logprob(prompt, c) for c in INV_CMDS)
    return lp_inv - lp_act


def main():
    t0 = time.time()
    repo = sys.argv[1] if len(sys.argv) > 1 else "mlx-community/Qwen2.5-Coder-7B-Instruct-4bit"
    seeds = [int(x) for x in sys.argv[2].split(",")] if len(sys.argv) > 2 else [0, 1, 2]
    temp = float(sys.argv[3]) if len(sys.argv) > 3 else 0.7
    alphas = [0, 6, 12]
    print(f"[load] {repo}", flush=True)
    mw = MLXModel(repo)
    N, D = mw.n_layers, mw.d_model
    tasks = TASKS + HARD_TASKS
    print(f"[model] layers={N} d={D} | rollouts {len(tasks)} tasks x seeds {seeds} temp {temp}", flush=True)

    decs, n_solved = [], 0
    for task in tasks:
        for seed in seeds:
            ds, solved = rollout(mw, task, seed, temp)
            decs.extend(ds); n_solved += int(solved)
            cls = [x["cls"] for x in ds]
            print(f"   {task['name']:12} s{seed} solved={int(solved)} "
                  f"inv={cls.count('investigate')} act={cls.count('act')} oth={cls.count('other')} "
                  f"| {time.time()-t0:.0f}s", flush=True)

    recover = [x for x in decs if x["in_trouble"] and x["cls"] == "investigate"]
    persist = [x for x in decs if x["in_trouble"] and x["cls"] == "act"]
    print(f"\n[harvest] {len(decs)} decisions | INVESTIGATE(recover)={len(recover)} "
          f"ACT(persist)={len(persist)} | episodes solved {n_solved}/{len(tasks)*len(seeds)}", flush=True)
    if len(recover) < 6 or len(persist) < 6:
        print("[abort] too few investigate/act decisions for a reliable contrast.", flush=True)
        (OUT / f"recovery_capable_{repo.replace('/','_')}.json").write_text(json.dumps(
            {"repo": repo, "investigate": len(recover), "act": len(persist), "note": "insufficient"}, indent=2))
        return

    pool = persist + recover
    y = np.array([0] * len(persist) + [1] * len(recover))
    groups = [x["task"] for x in pool]
    acts = {L: np.stack([x["caps"][L] for x in pool]) for L in range(N)}

    print("[B] per-layer probe (task-grouped CV AUC) + diff-means ...", flush=True)
    perlayer = []
    for L in range(N):
        auc, _ = _logreg_cv_auc(acts[L], y, groups)
        dom = _unit(acts[L][y == 1].mean(0) - acts[L][y == 0].mean(0))
        perlayer.append({"L": L, "auc": auc, "dom": dom})
    ranked = sorted([p for p in perlayer if not np.isnan(p["auc"])], key=lambda p: p["auc"], reverse=True)
    print("[B] top layers: " + ", ".join(f"L{p['L']}={p['auc']:.2f}" for p in ranked[:6]), flush=True)

    test_task = max(set(x["task"] for x in persist), key=lambda t: sum(x["task"] == t for x in persist))
    held = [x for x in persist if x["task"] == test_task]
    train = np.array([g != test_task for g in groups])
    print(f"[C] validate on held-out task '{test_task}' ({len(held)} ACT decisions); dir from the rest", flush=True)

    results = {}
    for L in [p["L"] for p in ranked[:3]]:
        XL = acts[L]
        dir_tr = _unit(XL[train & (y == 1)].mean(0) - XL[train & (y == 0)].mean(0))
        rng = np.random.default_rng(L)
        rand = _unit(rng.standard_normal(D))
        ortho = _unit(rng.standard_normal(D) - (rng.standard_normal(D) @ dir_tr) * dir_tr)
        ser = {"dir": [], "random": [], "ortho": []}
        flip = {}
        for a in alphas:
            for name, vec in (("dir", dir_tr), ("random", rand), ("ortho", ortho)):
                ms = []
                for x in held:
                    with mw.steering(L, vec, a):
                        ms.append(margin(mw, x["prompt"], x["cmd"]))
                ser[name].append(float(np.mean(ms)))
            labs = []
            for x in held:
                with mw.steering(L, dir_tr, a):
                    labs.append(cmd_class(_extract_cmd(mw.generate(mw.tok(x["prompt"]), max_tokens=18, temperature=0.0))))
            flip[a] = labs.count("investigate") / len(labs)
        net, net_r = ser["dir"][-1] - ser["dir"][0], ser["random"][-1] - ser["random"][0]
        results[L] = {"alphas": alphas, "margin": ser, "onpolicy_investigate": flip, "net_dir": net, "net_random": net_r}
        print(f"\n[C] L{L}: investigate-margin on held-out ACT decisions", flush=True)
        print(f"    {'alpha':>5} | {'dir':>7} | {'random':>7} | {'ortho':>7} | investigate%", flush=True)
        for j, a in enumerate(alphas):
            print(f"    {a:>5} | {ser['dir'][j]:>+7.2f} | {ser['random'][j]:>+7.2f} | {ser['ortho'][j]:>+7.2f} | {flip[a]*100:4.0f}%", flush=True)
        print(f"    net dir {net:+.2f} vs random {net_r:+.2f}  (specific: {'yes' if net > net_r + 0.2 else 'weak'})", flush=True)

    best = max(results, key=lambda L: results[L]["net_dir"] - results[L]["net_random"]) if results else None
    safe = repo.replace("/", "_").replace(":", "_")
    OUT.mkdir(exist_ok=True)
    if best is not None:
        np.save(OUT / f"recovery_capable_dir_{safe}_L{best}.npy",
                _unit(acts[best][y == 1].mean(0) - acts[best][y == 0].mean(0)))
    (OUT / f"recovery_capable_{safe}.json").write_text(json.dumps(
        {"repo": repo, "investigate": len(recover), "act": len(persist), "solved": n_solved,
         "stageB_top": [{"L": p["L"], "auc": p["auc"]} for p in ranked[:8]],
         "stageC": {str(L): results[L] for L in results}, "best_layer": best}, indent=2, default=float))
    print(f"\n[done] best layer {best} | wrote recovery_capable_{safe}.json | {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
