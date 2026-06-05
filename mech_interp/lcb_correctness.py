"""Phase 2+3: build the model's code-CORRECTNESS direction and causally steer generation with it.

Premise (Phase 1): for problems the model fails greedily, the correct solution is REACHABLE
(pass@k > 0). So the failure is an elicitation gap, not a competence wall: the correct mode exists
in the distribution, just at low probability. If the model linearly represents "this code is
correct", steering toward it should AMPLIFY the correct mode -> raise the density of correct
samples (lift pass@1 toward pass@k).

Phase 2 (direction): from Phase 1's correct vs incorrect solution samples, teacher-force each into
the GENERATION context (messages = [sys, problem, assistant=code]) and capture the residual stream
at the final code token. diff-of-means(correct - incorrect) per layer, problem-grouped CV AUC,
length control. This is the "correct-solution" direction, built where it will be used.

Phase 3 (causal): on HELD-OUT reachable-but-greedily-failed problems, draw K samples WITH steering
+dir vs WITHOUT vs random/orthogonal controls; does steering raise the fraction that solve?

  mech_interp/.venv/bin/python -u -m mech_interp.lcb_correctness <repo> [alphas]
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import numpy as np

from mech_interp.mlx_wrapper import MLXModel
from mech_interp.lcb_data import load_problems
from mech_interp.lcb_env import solved
from mech_interp.recovery_lcb import SYS, fmt_problem, extract_code
from mech_interp.run_recovery_direction import _unit, _logreg_cv_auc

OUT = Path(__file__).parent / "results"
MAXTOK = int(os.environ.get("LCB_MAXTOK", "440"))
MC, MI = 4, 4   # max correct / incorrect codes per problem for the contrast


def code_state_ids(mw, problem, code):
    """Tokens for [sys, problem, assistant=code] WITHOUT a generation prompt -> the model's state
    'having just written this solution'. We capture the last-token residual here."""
    msgs = [{"role": "system", "content": SYS},
            {"role": "user", "content": fmt_problem(problem)},
            {"role": "assistant", "content": f"```python\n{code}\n```"}]
    return list(mw.tokenizer.apply_chat_template(msgs, add_generation_prompt=False))


def sample_solution(mw, problem, temp, seed, steer=None):
    msgs = [{"role": "system", "content": SYS}, {"role": "user", "content": fmt_problem(problem)}]
    ids = mw.render(msgs)
    if steer is not None:
        with mw.steering(steer[0], steer[1], steer[2]):
            gen = mw.generate(ids, max_tokens=MAXTOK, temperature=temp, seed=seed)
    else:
        gen = mw.generate(ids, max_tokens=MAXTOK, temperature=temp, seed=seed)
    code = extract_code(gen)
    return code, solved(code, problem)


def main():
    t0 = time.time()
    repo = sys.argv[1] if len(sys.argv) > 1 else "mlx-community/Qwen2.5-Coder-7B-Instruct-4bit"
    alphas = [float(x) for x in sys.argv[2].split(",")] if len(sys.argv) > 2 else [0, 6, 10, 14]
    safe = repo.replace("/", "_").replace(":", "_")

    pk = json.loads((OUT / f"lcb_passk_{safe}.json").read_text())
    rows = pk["rows"]
    statements = {}
    for d in ("easy", "medium", "hard"):
        for p in load_problems(difficulties=(d,), after="2024-12"):
            statements[p["id"]] = p
    # usable = reachable problems with BOTH correct and incorrect samples (so the contrast and the
    # held-out causal test both have headroom).
    usable = [r for r in rows if r["correct"] and r["incorrect"] and r["id"] in statements]
    # Held-out causal test = the greedy-fail-but-reachable problems (greedy fails, so steering has
    # the most headroom). Build the direction on the remaining usable problems (never the test set).
    gf = [r for r in usable if not r["greedy"]]
    test_probs = gf if len(gf) >= 2 else usable[: max(2, len(usable) // 3)]
    test_ids = {t["id"] for t in test_probs}
    train_probs = [r for r in usable if r["id"] not in test_ids]
    print(f"[data] {len(usable)} usable -> {len(train_probs)} train (build dir) / {len(test_probs)} "
          f"held-out greedy-fail (causal test: {sorted(test_ids)})", flush=True)
    if len(train_probs) < 2 or len(test_probs) < 2:
        print("[abort] too few usable problems; widen Phase 1 (more easy problems / higher K).", flush=True)
        return

    print(f"[load] {repo}", flush=True)
    mw = MLXModel(repo)
    N, D = mw.n_layers, mw.d_model

    # ---- Phase 2: capture the correctness contrast in the generation context (TRAIN only) ----
    # Per-problem PAIRED diff-of-means controls the problem-identity confound: within each problem,
    # correct vs incorrect codes differ mainly in correctness, not topic/length.
    per_prob_diff = {L: [] for L in range(N)}     # within-problem (correct-mean - incorrect-mean)
    X, y, groups = [], [], []                      # flat set for the grouped-CV decodability check
    for r in train_probs:
        p = statements[r["id"]]
        cor = [mw.capture_resid(code_state_ids(mw, p, c), list(range(N))) for c in r["correct"][:MC]]
        inc = [mw.capture_resid(code_state_ids(mw, p, c), list(range(N))) for c in r["incorrect"][:MI]]
        for L in range(N):
            cm = np.mean([c[L] for c in cor], 0); im = np.mean([c[L] for c in inc], 0)
            per_prob_diff[L].append(cm - im)
        for c in cor:
            X.append(c); y.append(1); groups.append(r["id"])
        for c in inc:
            X.append(c); y.append(0); groups.append(r["id"])
    y = np.array(y)
    acts = {L: np.stack([c[L] for c in X]) for L in range(N)}
    print(f"[2] captured {len(y)} solution-states from {len(train_probs)} train problems "
          f"(correct={int(y.sum())} incorrect={int((1-y).sum())})", flush=True)

    perlayer = []
    for L in range(N):
        auc, _ = _logreg_cv_auc(acts[L], y, groups)
        dom = _unit(np.mean(per_prob_diff[L], 0))   # paired, problem-confound-controlled direction
        perlayer.append({"L": L, "auc": auc, "dom": dom})
    ranked = sorted([p for p in perlayer if not np.isnan(p["auc"])], key=lambda p: p["auc"], reverse=True)
    print("[2] correctness decodable? top layers (problem-grouped CV AUC): "
          + ", ".join(f"L{p['L']}={p['auc']:.2f}" for p in ranked[:6]), flush=True)

    # ---- Phase 3: causal test on the HELD-OUT problems (direction never saw them) ----
    held = test_probs
    best = ranked[0]["L"]
    dirv = perlayer[best]["dom"]
    rng = np.random.default_rng(0)
    rand = _unit(rng.standard_normal(D))
    ortho = _unit(rng.standard_normal(D) - (rng.standard_normal(D) @ dirv) * dirv)
    print(f"\n[3] causal test at L{best} on {len(held)} held-out reachable-but-greedy-failed problems", flush=True)

    Ksteer = 6
    results = {}
    for name, vec in (("none", None), ("dir", dirv), ("random", rand), ("ortho", ortho)):
        for a in ([0] if vec is None else alphas[1:]):
            key = "baseline" if vec is None else f"{name}@{a:g}"
            tot = 0
            for r in held:
                p = statements[r["id"]]
                steer = None if vec is None else (best, vec, a)
                tot += sum(int(sample_solution(mw, p, 0.8, 7000 + k, steer=steer)[1]) for k in range(Ksteer))
            results[key] = tot
            print(f"   {key:12}: {tot}/{len(held)*Ksteer} samples solve | {time.time()-t0:.0f}s", flush=True)

    base = results.get("baseline", 0)
    best_dir = max((v for k, v in results.items() if k.startswith("dir")), default=0)
    best_ctrl = max((v for k, v in results.items() if k.startswith(("random", "ortho"))), default=0)
    (OUT / f"lcb_correctness_{safe}.json").write_text(json.dumps(
        {"repo": repo, "best_layer": best, "stageB_top": [{"L": p["L"], "auc": p["auc"]} for p in ranked[:8]],
         "n_held": len(held), "K_per": Ksteer, "results": results,
         "lift_dir_over_baseline": best_dir - base, "lift_dir_over_control": best_dir - best_ctrl}, indent=2, default=float))
    np.save(OUT / f"correctness_dir_{safe}_L{best}.npy", dirv)
    print(f"\n[result] correct-sample count: baseline {base}, best-dir {best_dir}, best-control {best_ctrl} "
          f"(of {len(held)*Ksteer}) | dir lift vs baseline {best_dir-base:+d}, vs control {best_dir-best_ctrl:+d}", flush=True)
    print(f"[done] wrote lcb_correctness_{safe}.json | {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
