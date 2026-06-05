"""Phase 4 — verifier-guided selection: use the model's OWN internal correctness representation to
pick the correct solution out of K samples (self-contained: builds the direction, then selects).

The deduction: additive steering of the correctness direction disrupts generation (Phase 3), but
the premises hold -- the correct solution is REACHABLE (pass@k>0, Phase 1) and "this code is correct"
is DECODABLE on held-out problems (Phase 2, AUC~0.75). So don't steer weights: generate K samples,
score each by projecting its solution-state activation onto the correctness direction, and emit the
top-scored one. If the probe carries held-out signal this lifts effective pass@1 toward pass@k --
a real way to make the model 'eventually write correct code' using its own internals.

  mech_interp/.venv/bin/python -u -m mech_interp.lcb_select <repo> <K>
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

from mech_interp.mlx_wrapper import MLXModel
from mech_interp.lcb_data import load_problems
from mech_interp.lcb_correctness import code_state_ids, sample_solution
from mech_interp.run_recovery_direction import _auc, _unit, _logreg_cv_auc

OUT = Path(__file__).parent / "results"
MC, MI = 4, 4


def build_direction(mw, train_rows, statements):
    """Within-problem paired diff-of-means correctness direction; layer chosen by grouped-CV AUC."""
    N = mw.n_layers
    per_prob = {L: [] for L in range(N)}
    X, y, groups = [], [], []
    for r in train_rows:
        p = statements[r["id"]]
        cor = [mw.capture_resid(code_state_ids(mw, p, c), list(range(N))) for c in r["correct"][:MC]]
        inc = [mw.capture_resid(code_state_ids(mw, p, c), list(range(N))) for c in r["incorrect"][:MI]]
        for L in range(N):
            per_prob[L].append(np.mean([c[L] for c in cor], 0) - np.mean([c[L] for c in inc], 0))
        for c in cor:
            X.append(c); y.append(1); groups.append(r["id"])
        for c in inc:
            X.append(c); y.append(0); groups.append(r["id"])
    y = np.array(y)
    acts = {L: np.stack([c[L] for c in X]) for L in range(N)}
    perlayer = []
    for L in range(N):
        auc, _ = _logreg_cv_auc(acts[L], y, groups)
        perlayer.append({"L": L, "auc": auc, "dom": _unit(np.mean(per_prob[L], 0))})
    ranked = sorted([p for p in perlayer if not np.isnan(p["auc"])], key=lambda p: p["auc"], reverse=True)
    return ranked


def main():
    t0 = time.time()
    repo = sys.argv[1] if len(sys.argv) > 1 else "mlx-community/Qwen2.5-Coder-7B-Instruct-4bit"
    K = int(sys.argv[2]) if len(sys.argv) > 2 else 16
    safe = repo.replace("/", "_").replace(":", "_")

    pk = json.loads((OUT / f"lcb_passk_{safe}.json").read_text())
    statements = {}
    for d in ("easy", "medium", "hard"):
        for p in load_problems(difficulties=(d,), after="2024-12"):
            statements[p["id"]] = p
    usable = [r for r in pk["rows"] if r["correct"] and r["incorrect"] and r["id"] in statements]
    clean = {r["id"] for r in usable if not r["greedy"]}        # greedy-fail-reachable = held-out
    train_rows = [r for r in usable if r["id"] not in clean]     # build the direction on these
    print(f"[setup] {len(usable)} mixed-sampling problems | train(dir)={[r['id'] for r in train_rows]} "
          f"held-out={sorted(clean)} | K={K}", flush=True)

    print(f"[load] {repo}", flush=True)
    mw = MLXModel(repo)
    ranked = build_direction(mw, train_rows, statements)
    L, dirv = ranked[0]["L"], ranked[0]["dom"]
    print("[dir] correctness decodable (grouped-CV AUC): "
          + ", ".join(f"L{p['L']}={p['auc']:.2f}" for p in ranked[:5]) + f"  -> using L{L}", flush=True)

    rows, all_scores, all_solved = [], [], []
    for r in usable:
        p = statements[r["id"]]
        solved, scores = [], []
        for k in range(K):
            code, sv = sample_solution(mw, p, 0.8, 9000 + k)
            act = mw.capture_resid(code_state_ids(mw, p, code), [L])[L]
            solved.append(int(sv)); scores.append(float(act @ dirv))
        sel = int(np.argmax(scores))
        tag = "held-out" if r["id"] in clean else "train"
        rows.append({"id": r["id"], "tag": tag, "sel_solved": solved[sel], "pass1": float(np.mean(solved)),
                     "passk": int(any(solved)), "n_solve": int(sum(solved))})
        all_scores += scores; all_solved += solved
        print(f"   {r['id']:11} [{tag:8}] selected_solved={solved[sel]} (sample {sel}) "
              f"pass@1={np.mean(solved):.2f} pass@{K}={int(any(solved))} ({sum(solved)}/{K}) | {time.time()-t0:.0f}s", flush=True)

    def agg(rs):
        return (float(np.mean([x["sel_solved"] for x in rs])), float(np.mean([x["pass1"] for x in rs])),
                float(np.mean([x["passk"] for x in rs]))) if rs else (0, 0, 0)
    s_all, p1_all, pk_all = agg(rows)
    s_h, p1_h, pk_h = agg([r for r in rows if r["tag"] == "held-out"])
    score_auc = _auc(np.array(all_solved), np.array(all_scores))
    print(f"\n[result] verifier-guided SELECTION (pick argmax internal-correctness score):", flush=True)
    print(f"   ALL {len(rows)}:        selection {s_all:.2f} | pass@1 (random) {p1_all:.2f} | pass@{K} (oracle) {pk_all:.2f}  (lift {s_all-p1_all:+.2f})", flush=True)
    print(f"   held-out {len([r for r in rows if r['tag']=='held-out'])}:   selection {s_h:.2f} | pass@1 {p1_h:.2f} | pass@{K} {pk_h:.2f}  (lift {s_h-p1_h:+.2f})", flush=True)
    print(f"   per-sample score AUC vs solved: {score_auc:.2f} (>0.5 = the internal verifier ranks correct samples higher)", flush=True)
    (OUT / f"lcb_select_{safe}.json").write_text(json.dumps(
        {"repo": repo, "L": L, "K": K, "rows": rows, "score_auc": score_auc,
         "all": {"selection": s_all, "pass1": p1_all, "passk": pk_all},
         "held_out": {"selection": s_h, "pass1": p1_h, "passk": pk_h}}, indent=2, default=float))
    print(f"[done] wrote lcb_select_{safe}.json | {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
