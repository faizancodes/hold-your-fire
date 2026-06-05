"""Recovery direction from REAL on-policy trajectories (the deep substrate).

Instead of synthetic stuck prompts (which a capable model sees through), we let the model run
the actual bug-fix agent loop and harvest its OWN decision points. At each step we capture the
decision-token residuals at every layer, execute the command, and record whether the model
*persisted* (repeated a recent command while still failing) or *recovered* (took a new
productive action out of a failing/looping state). The diff-of-means(recover - persist) over
these real decisions is the candidate recovery axis; we then causally validate it by steering
held-out persist decisions and measuring the shift toward recovery, vs random/orthogonal
controls.

This is the rigorous version: the contrast comes from the model's genuine stuck behavior, not
constructed text.

  mech_interp/.venv/bin/python -m mech_interp.recovery_onpolicy [repo] [seeds] [temp]
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
from mech_interp.recovery_contexts import classify
from mech_interp.run_recovery_direction import _extract_cmd, _unit, _auc, _logreg_cv_auc

OUT = Path(__file__).parent / "results"
SYS = ("You are an autonomous coding agent. Commands: `cat sol.py` to read the source, "
       "`sed -i \"s/old/new/\" sol.py` to edit it, `python test.py` to run the test. "
       "Read the source, find the bug, then fix it. Reply with EXACTLY ONE shell command.")
BUDGET = 8
RECOVERS = ["cat sol.py", "ls", "cat test.py"]   # productive moves for the margin metric


def rollout(mw, task, seed, temp):
    d = setup_task(task)
    init = run_cmd("python test.py", d)
    msgs = [{"role": "system", "content": SYS},
            {"role": "user", "content": f"test.py fails:\n{init[:180]}\nFix sol.py."}]
    history, decisions = [], []
    solved = False
    for step in range(BUDGET):
        prompt = mw.tokenizer.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)
        ids = mw.render(msgs)
        caps = mw.capture_resid(ids, list(range(mw.n_layers)))
        gen = mw.generate(ids, max_tokens=24, temperature=temp, seed=seed + step)
        cmd = _extract_cmd(gen)
        obs = run_cmd(cmd, d) if cmd else "(no command)"
        solved = check_solved(task, d)
        repeated = bool(cmd) and cmd in history[-4:]
        productive = classify(cmd, "\0") == "recover"   # a productive, non-repeat command
        decisions.append({"prompt": prompt, "caps": caps, "cmd": cmd, "step": step,
                          "repeated": repeated, "productive": productive,
                          "in_trouble": step >= 1 and not solved,
                          "task": task["name"], "seed": seed, "temp": temp})
        msgs.append({"role": "assistant", "content": cmd})
        msgs.append({"role": "user", "content": obs[:160]})
        history.append(cmd)
        if solved:
            break
    cleanup(d)
    return decisions, solved


def margin(mw, prompt, persist_cmd):
    lp_p = mw.continuation_logprob(prompt, persist_cmd)
    lp_r = max(mw.continuation_logprob(prompt, r) for r in RECOVERS)
    return lp_r - lp_p


def main():
    t0 = time.time()
    repo = sys.argv[1] if len(sys.argv) > 1 else "mlx-community/Qwen2.5-Coder-1.5B-Instruct-4bit"
    seeds = [int(x) for x in sys.argv[2].split(",")] if len(sys.argv) > 2 else [0, 1]
    temp = float(sys.argv[3]) if len(sys.argv) > 3 else 0.7   # sampling -> behavioral diversity
    alphas = [0, 6, 12]
    print(f"[load] {repo}")
    mw = MLXModel(repo)
    N, D = mw.n_layers, mw.d_model
    tasks = TASKS + HARD_TASKS
    print(f"[model] layers={N} d={D} | rollouts: {len(tasks)} tasks x seeds {seeds} x temp {temp}")

    # ---- harvest real decisions --------------------------------------------------
    decs = []
    n_solved = 0
    for task in tasks:
        for seed in seeds:
            ds, solved = rollout(mw, task, seed, temp)
            decs.extend(ds); n_solved += int(solved)
            print(f"   {task['name']:12} s{seed} steps={len(ds)} solved={int(solved)} "
                  f"reps={sum(d['repeated'] for d in ds)} | {time.time()-t0:.0f}s", flush=True)

    # label persist vs recover among 'in trouble' decisions
    persist = [d for d in decs if d["in_trouble"] and d["repeated"]]
    recover = [d for d in decs if d["in_trouble"] and d["productive"] and not d["repeated"]]
    print(f"\n[harvest] {len(decs)} decisions | persist={len(persist)} recover={len(recover)} "
          f"| episodes solved {n_solved}/{len(tasks)*len(seeds)}")
    if len(persist) < 5 or len(recover) < 5:
        print("[abort] too few real persist/recover decisions to build a reliable contrast.")
        print("        (model rarely loops on these tasks -> need harder tasks or a stuck-prone model)")
        (OUT / f"recovery_onpolicy_{repo.replace('/','_')}.json").write_text(
            json.dumps({"repo": repo, "decisions": len(decs), "persist": len(persist),
                        "recover": len(recover), "note": "insufficient contrast"}, indent=2))
        return

    pool = persist + recover
    y = np.array([0] * len(persist) + [1] * len(recover))
    groups = [d["task"] for d in pool]
    acts = {L: np.stack([d["caps"][L] for d in pool]) for L in range(N)}

    # ---- Stage B: per-layer discovery (task-grouped CV) --------------------------
    print("[B] per-layer probe (task-grouped CV AUC) + diff-means ...")
    perlayer = []
    for L in range(N):
        auc, _ = _logreg_cv_auc(acts[L], y, groups)
        dom = _unit(acts[L][y == 1].mean(0) - acts[L][y == 0].mean(0))
        perlayer.append({"L": L, "auc": auc, "dom": dom})
    ranked = sorted([p for p in perlayer if not np.isnan(p["auc"])], key=lambda p: p["auc"], reverse=True)
    print("[B] top layers by grouped-CV AUC: " + ", ".join(f"L{p['L']}={p['auc']:.2f}" for p in ranked[:6]))

    # ---- Stage C: causal validation on held-out-task persist decisions ------------
    test_task = max(set(d["task"] for d in persist), key=lambda t: sum(d["task"] == t for d in persist))
    held = [d for d in persist if d["task"] == test_task]
    train_mask = np.array([g != test_task for g in groups])
    print(f"[C] validate on held-out task '{test_task}' ({len(held)} persist decisions); dir built on the rest")

    results = {}
    for L in [p["L"] for p in ranked[:3]]:
        XL = acts[L]
        dir_tr = _unit(XL[train_mask & (y == 1)].mean(0) - XL[train_mask & (y == 0)].mean(0)) \
            if (train_mask & (y == 1)).any() and (train_mask & (y == 0)).any() else perlayer[L]["dom"]
        rng = np.random.default_rng(L)
        rand = _unit(rng.standard_normal(D))
        ortho = _unit(rng.standard_normal(D) - (rng.standard_normal(D) @ dir_tr) * dir_tr)
        ser = {"dir": [], "random": [], "ortho": []}
        oprec = {}
        for a in alphas:
            for name, vec in (("dir", dir_tr), ("random", rand), ("ortho", ortho)):
                ms = []
                for d in held:
                    with mw.steering(L, vec, a):
                        ms.append(margin(mw, d["prompt"], d["cmd"]))
                ser[name].append(float(np.mean(ms)))
            labs = []
            for d in held:
                with mw.steering(L, dir_tr, a):
                    g = _extract_cmd(mw.generate(mw.tok(d["prompt"]), max_tokens=20, temperature=0.0))
                labs.append(classify(g, d["cmd"]))
            oprec[a] = labs.count("recover") / len(labs)
        net, net_r = ser["dir"][-1] - ser["dir"][0], ser["random"][-1] - ser["random"][0]
        results[L] = {"alphas": alphas, "margin": ser, "onpolicy_recover": oprec, "net_dir": net, "net_random": net_r}
        print(f"\n[C] L{L}: recovery-margin on held-out persist (real decisions)")
        print(f"    {'alpha':>5} | {'dir':>7} | {'random':>7} | {'ortho':>7} | recover%")
        for j, a in enumerate(alphas):
            print(f"    {a:>5} | {ser['dir'][j]:>+7.2f} | {ser['random'][j]:>+7.2f} | {ser['ortho'][j]:>+7.2f} | {oprec[a]*100:4.0f}%")
        print(f"    net dir {net:+.2f} vs random {net_r:+.2f}  (specific: {'yes' if net > net_r + 0.2 else 'weak'})")

    best = max(results, key=lambda L: results[L]["net_dir"] - results[L]["net_random"]) if results else None
    safe = repo.replace("/", "_").replace(":", "_")
    out = {"repo": repo, "decisions": len(decs), "persist": len(persist), "recover": len(recover),
           "stageB_top": [{"L": p["L"], "auc": p["auc"]} for p in ranked[:8]],
           "stageC": {str(L): results[L] for L in results}, "best_layer": best}
    OUT.mkdir(exist_ok=True)
    if best is not None:
        np.save(OUT / f"recovery_onpolicy_dir_{safe}_L{best}.npy",
                _unit(acts[best][y == 1].mean(0) - acts[best][y == 0].mean(0)))
    (OUT / f"recovery_onpolicy_{safe}.json").write_text(json.dumps(out, indent=2, default=float))
    print(f"\n[done] best layer {best} | wrote recovery_onpolicy_{safe}.json | {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
