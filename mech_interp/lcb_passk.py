"""Phase 1 — reachability: is the failure an elicitation gap (pass@k > pass@1) or a competence
wall (pass@k = 0)?

For each LiveCodeBench problem we take the greedy solution (pass@1) and K independent temperature
samples (single-shot, NOT agentic revisions, which anchor on a wrong approach). If a problem the
model fails greedily is solved within K samples, the correct solution is REACHABLE in its
distribution -> a steerable target exists (amplify the correct mode). We also save the correct and
incorrect code samples per problem, which Phase 2 uses to build the correctness direction.

  mech_interp/.venv/bin/python -u -m mech_interp.lcb_passk <repo> <difficulties> <n> <K> <temp>
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

from mech_interp.mlx_wrapper import MLXModel
from mech_interp.lcb_data import load_problems
from mech_interp.lcb_env import solved
from mech_interp.recovery_lcb import SYS, fmt_problem, extract_code

OUT = Path(__file__).parent / "results"
MAXTOK = int(os.environ.get("LCB_MAXTOK", "440"))


def one_sample(mw, p, temp, seed):
    msgs = [{"role": "system", "content": SYS}, {"role": "user", "content": fmt_problem(p)}]
    gen = mw.generate(mw.render(msgs), max_tokens=MAXTOK, temperature=temp, seed=seed)
    code = extract_code(gen)
    return code, solved(code, p)


def main():
    t0 = time.time()
    repo = sys.argv[1] if len(sys.argv) > 1 else "mlx-community/Qwen2.5-Coder-7B-Instruct-4bit"
    diffs = sys.argv[2].split(",") if len(sys.argv) > 2 else ["easy", "medium"]
    n = int(sys.argv[3]) if len(sys.argv) > 3 else 8
    K = int(sys.argv[4]) if len(sys.argv) > 4 else 10
    temp = float(sys.argv[5]) if len(sys.argv) > 5 else 0.8

    print(f"[load] {repo}", flush=True)
    mw = MLXModel(repo)
    problems = []
    for d in diffs:
        problems += load_problems(difficulties=(d,), after="2024-12", limit=n)
    print(f"[model] {mw.n_layers}L | {len(problems)} problems {diffs} | K={K} temp={temp} | {time.time()-t0:.0f}s", flush=True)

    rows = []
    for p in problems:
        gcode, gsolved = one_sample(mw, p, 0.0, 0)
        samples = [one_sample(mw, p, temp, 1000 + k) for k in range(K)]
        nsolve = sum(int(sv) for _, sv in samples)
        passk = nsolve > 0
        rows.append({"id": p["id"], "difficulty": p["difficulty"], "greedy": bool(gsolved),
                     "passk": passk, "n_solve": nsolve, "K": K,
                     "correct": [c for c, sv in samples if sv][:4],
                     "incorrect": [c for c, sv in samples if not sv][:4]})
        print(f"   {p['id']:11} {p['difficulty']:6} greedy={int(gsolved)} pass@{K}={int(passk)} "
              f"({nsolve}/{K} solve) | {time.time()-t0:.0f}s", flush=True)

    gf = [r for r in rows if not r["greedy"]]
    reach = [r for r in gf if r["passk"]]
    print(f"\n[reachability] greedy-failures: {len(gf)}/{len(rows)} | of those, "
          f"{len(reach)} are solved within {K} samples (REACHABLE = steerable target)", flush=True)
    for d in diffs:
        gfd = [r for r in gf if r["difficulty"] == d]
        rd = [r for r in gfd if r["passk"]]
        print(f"   {d:7}: {len(rd)}/{len(gfd)} greedy-failures reachable within {K} samples", flush=True)

    safe = repo.replace("/", "_").replace(":", "_")
    OUT.mkdir(exist_ok=True)
    (OUT / f"lcb_passk_{safe}.json").write_text(json.dumps({
        "repo": repo, "K": K, "temp": temp, "n_problems": len(rows),
        "n_greedy_fail": len(gf), "n_reachable": len(reach), "rows": rows}, indent=2))
    print(f"[done] wrote lcb_passk_{safe}.json | {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
