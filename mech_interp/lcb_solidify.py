"""Solidify verification-guided selection: scale it over more easy-frontier problems and benchmark
it against cheaper training-free baselines on the SAME samples.

Self-verification is zero-shot, so there is no train/test split -- every problem is held-out. For
each problem we draw K samples and compare four ways to pick which one to emit:
  random        : a random sample          (= pass@1, the do-nothing baseline)
  self_consist  : the sample agreeing with the MAJORITY output on the public examples (strong,
                  training-free, the real baseline to beat)
  verification  : the sample the model itself judges most likely correct (P(YES))
  oracle        : pass@k (any sample solves; the ceiling)
Usable problems = those with a MIX of correct/incorrect samples (0 < n_solve < K), i.e. selection
has something to do.

  mech_interp/.venv/bin/python -u -m mech_interp.lcb_solidify <repo> <N> <K>
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from collections import Counter
from pathlib import Path

import numpy as np

from mech_interp.mlx_wrapper import MLXModel
from mech_interp.lcb_data import load_problems
from mech_interp.lcb_env import _norm
from mech_interp.lcb_correctness import sample_solution
from mech_interp.lcb_verify_auc import verify_score_fast as verify_score
from mech_interp.run_recovery_direction import _auc

OUT = Path(__file__).parent / "results"
PY = sys.executable


def public_outputs(code, inputs, timeout=6):
    """The code's outputs on the public example inputs -> a signature for self-consistency."""
    d = tempfile.mkdtemp(prefix="sc_")
    (Path(d) / "sol.py").write_text(code)
    outs = []
    try:
        for inp in inputs:
            try:
                p = subprocess.run([PY, "sol.py"], input=inp, cwd=d, timeout=timeout, capture_output=True, text=True)
                outs.append(_norm(p.stdout) if p.returncode == 0 else "ERR")
            except Exception:
                outs.append("ERR")
    finally:
        shutil.rmtree(d, ignore_errors=True)
    return tuple(outs)


def main():
    t0 = time.time()
    repo = sys.argv[1] if len(sys.argv) > 1 else "mlx-community/Qwen2.5-Coder-7B-Instruct-4bit"
    N = int(sys.argv[2]) if len(sys.argv) > 2 else 20
    K = int(sys.argv[3]) if len(sys.argv) > 3 else 10
    safe = repo.replace("/", "_").replace(":", "_")

    problems = load_problems(difficulties=("easy",), after="2024-12", limit=N)
    print(f"[load] {repo} | {len(problems)} easy problems | K={K}", flush=True)
    mw = MLXModel(repo)

    usable, vlabels, vscores = [], [], []
    for p in problems:
        pub_in = [t["input"] for t in p["public"]]
        S = []
        for k in range(K):
            code, sv = sample_solution(mw, p, 0.8, 9000 + k)
            S.append({"solved": int(sv), "v": verify_score(mw, p, code), "pub": public_outputs(code, pub_in)})
        n = sum(s["solved"] for s in S)
        print(f"   {p['id']:11} n_solve={n}/{K} {'(usable)' if 0 < n < K else ''} | {time.time()-t0:.0f}s", flush=True)
        if 0 < n < K:
            usable.append({"id": p["id"], "S": S, "n": n})
            vlabels += [s["solved"] for s in S]; vscores += [s["v"] for s in S]

    if len(usable) < 4:
        print(f"[warn] only {len(usable)} usable problems; result will be noisy.", flush=True)

    def sel_solves(S, method):
        if method == "verification":
            return S[int(np.argmax([s["v"] for s in S]))]["solved"]
        if method == "self_consist":
            maj = Counter(s["pub"] for s in S).most_common(1)[0][0]
            return next(s for s in S if s["pub"] == maj)["solved"]
        return float(np.mean([s["solved"] for s in S]))  # random == pass@1 expectation

    rows = []
    for u in usable:
        rows.append({"id": u["id"], "pass1": u["n"] / K,
                     "verification": sel_solves(u["S"], "verification"),
                     "self_consist": sel_solves(u["S"], "self_consist"),
                     "passk": 1.0})
    def m(key):
        return float(np.mean([r[key] for r in rows])) if rows else float("nan")
    verify_auc = _auc(np.array(vlabels), np.array(vscores)) if vlabels else float("nan")

    print(f"\n[result] selection method solve-rate over {len(rows)} usable easy-frontier problems "
          f"(K={K}, verify fresh-AUC {verify_auc:.2f}):", flush=True)
    print(f"   random (pass@1)        : {m('pass1'):.2f}   <- do nothing", flush=True)
    print(f"   self-consistency       : {m('self_consist'):.2f}   <- majority public-output (baseline to beat)", flush=True)
    print(f"   VERIFICATION (self)    : {m('verification'):.2f}   <- model's own correctness judgement", flush=True)
    print(f"   oracle (pass@{K})        : {m('passk'):.2f}", flush=True)
    print(f"   >>> verification lift over pass@1 {m('verification')-m('pass1'):+.2f} | "
          f"over self-consistency {m('verification')-m('self_consist'):+.2f}", flush=True)
    (OUT / f"lcb_solidify_{safe}.json").write_text(json.dumps(
        {"repo": repo, "N": N, "K": K, "n_usable": len(rows), "verify_auc": verify_auc,
         "random_pass1": m("pass1"), "self_consistency": m("self_consist"),
         "verification": m("verification"), "oracle": m("passk"), "rows": rows}, indent=2, default=float))
    print(f"[done] wrote lcb_solidify_{safe}.json | {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
