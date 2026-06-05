"""The demonstrated way: verification-guided selection. Generate K samples, rank each by the model's
OWN explicit correctness judgement (P(YES) it passes all tests), emit the top one. Exploits the
confirmed generation-verification gap (verify AUC 0.71 >> linear probe 0.45) to lift pass@1 toward
pass@k WITHOUT external tests or fragile steering -- purely the model's internals.

  mech_interp/.venv/bin/python -u -m mech_interp.lcb_verify_select <repo> <K>
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

from mech_interp.mlx_wrapper import MLXModel
from mech_interp.lcb_data import load_problems
from mech_interp.lcb_correctness import sample_solution
from mech_interp.lcb_verify_auc import verify_score
from mech_interp.run_recovery_direction import _auc

OUT = Path(__file__).parent / "results"


def main():
    t0 = time.time()
    repo = sys.argv[1] if len(sys.argv) > 1 else "mlx-community/Qwen2.5-Coder-7B-Instruct-4bit"
    K = int(sys.argv[2]) if len(sys.argv) > 2 else 12
    safe = repo.replace("/", "_").replace(":", "_")
    pk = json.loads((OUT / f"lcb_passk_{safe}.json").read_text())
    statements = {}
    for d in ("easy", "medium", "hard"):
        for p in load_problems(difficulties=(d,), after="2024-12"):
            statements[p["id"]] = p
    usable = [r for r in pk["rows"] if r["correct"] and r["incorrect"] and r["id"] in statements]
    clean = {r["id"] for r in usable if not r["greedy"]}
    print(f"[setup] verification-guided selection on {len(usable)} mixed problems (held-out: {sorted(clean)}) K={K}", flush=True)

    print(f"[load] {repo}", flush=True)
    mw = MLXModel(repo)

    rows, sc_all, sv_all = [], [], []
    for r in usable:
        p = statements[r["id"]]
        solved, vscores = [], []
        for k in range(K):
            code, sv = sample_solution(mw, p, 0.8, 9000 + k)   # same seeds as activation-probe run
            solved.append(int(sv)); vscores.append(verify_score(mw, p, code))
        sel = int(np.argmax(vscores))
        tag = "held-out" if r["id"] in clean else "train"
        rows.append({"id": r["id"], "tag": tag, "sel_solved": solved[sel], "pass1": float(np.mean(solved)),
                     "passk": int(any(solved)), "n_solve": int(sum(solved))})
        sc_all += vscores; sv_all += solved
        print(f"   {r['id']:11} [{tag:8}] verify-selected_solved={solved[sel]} (sample {sel}) "
              f"pass@1={np.mean(solved):.2f} pass@{K}={int(any(solved))} ({sum(solved)}/{K}) | {time.time()-t0:.0f}s", flush=True)

    def agg(rs):
        return (float(np.mean([x["sel_solved"] for x in rs])), float(np.mean([x["pass1"] for x in rs])),
                float(np.mean([x["passk"] for x in rs]))) if rs else (0, 0, 0)
    s_all, p1_all, pk_all = agg(rows)
    s_h, p1_h, pk_h = agg([r for r in rows if r["tag"] == "held-out"])
    auc = _auc(np.array(sv_all), np.array(sc_all))
    print(f"\n[result] VERIFICATION-guided selection (rank K samples by the model's own P(correct)):", flush=True)
    print(f"   ALL {len(rows)}:        selection {s_all:.2f} | pass@1 (random) {p1_all:.2f} | pass@{K} (oracle) {pk_all:.2f}  (lift {s_all-p1_all:+.2f})", flush=True)
    print(f"   held-out {sum(1 for r in rows if r['tag']=='held-out')}:   selection {s_h:.2f} | pass@1 {p1_h:.2f} | pass@{K} {pk_h:.2f}  (lift {s_h-p1_h:+.2f})", flush=True)
    print(f"   per-sample verify-score AUC vs solved (fresh): {auc:.2f}", flush=True)
    (OUT / f"lcb_verify_select_{safe}.json").write_text(json.dumps(
        {"repo": repo, "K": K, "rows": rows, "verify_auc_fresh": auc,
         "all": {"selection": s_all, "pass1": p1_all, "passk": pk_all},
         "held_out": {"selection": s_h, "pass1": p1_h, "passk": pk_h}}, indent=2, default=float))
    print(f"[done] wrote lcb_verify_select_{safe}.json | {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
