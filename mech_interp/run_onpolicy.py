"""Phase 6 — on-policy steering: does the steering vector actually break loops during
generation, and does it disrupt healthy runs less than an external reset?

Uses the winning direction/layer from steer_eval_results.json (built from length-controlled
acts2.npz). On NATURAL loop contexts we generate the next action greedily, with and without
steering, and check whether the agent emits a NOVEL command (escapes the loop).

  loop-escape rate : fraction of loop contexts where generated cmd != looped cmd
  controls         : no-steer, random direction, orthogonal direction
  coherence        : fraction of generations that contain a parseable `command`
  disruption       : on PROGRESS (healthy) contexts, fraction where steering changes the action

    mech_interp/.venv/bin/python -m mech_interp.run_onpolicy [alpha]
"""
from __future__ import annotations

import json, re, sys, time
from pathlib import Path

import numpy as np

from mech_interp.model_wrapper import ModelWrapper
from mech_interp.synthetic import make_scenarios

OUT = Path(__file__).parent / "results"
RNG = np.random.default_rng(2)
N_SCEN, K, TRAIN_N = 60, 4, 40


def parse_cmd(text: str) -> str:
    i = text.find("`")
    if i >= 0:
        rest = text[i + 1:]
        j = rest.find("`")
        return (rest[:j] if j >= 0 else rest.split("\n")[0]).strip()
    return text.strip().split("\n")[0].strip()


def is_repeat(gen_cmd: str, looped: str) -> bool:
    """Robust to truncation: same command if their leading segments match."""
    a, b = gen_cmd.strip(), looped.strip()
    return a[:15] == b[:15] or a.startswith(b) or b.startswith(a)


def main():
    t0 = time.time()
    alpha = float(sys.argv[1]) if len(sys.argv) > 1 else 8.0
    cfg = json.loads((OUT / "steer_eval_results.json").read_text())
    L, sign, dname = cfg["best_layer"], cfg["best_sign"], cfg["best_dir"]
    d = np.load(OUT / "acts2.npz", allow_pickle=True)
    cond, group = d["meta_cond"], d["meta_group"]; tr = group < TRAIN_N
    pos, neg = ("loopfail", "vfail") if dname == "REP" else ("loopfail", "loopprog")
    X = d[f"L{L}"]
    v = X[tr & (cond == pos)].mean(0) - X[tr & (cond == neg)].mean(0); v /= np.linalg.norm(v) + 1e-8
    steer_vec = sign * v
    rnd = RNG.standard_normal(v.shape[0]); rnd /= np.linalg.norm(rnd)
    orth = rnd - (rnd @ v) * v; orth /= np.linalg.norm(orth)
    print(f"[cfg] dir={dname} L={L} sign={sign:+d} alpha={alpha}", flush=True)

    mw = ModelWrapper()
    scen = make_scenarios(N_SCEN)
    test = scen[TRAIN_N:TRAIN_N + 12]      # small: generation on MPS is slow

    def gen(ctx, vec=None):
        if vec is None:
            return mw.generate_kv(ctx, max_new_tokens=28)
        with mw.steering(L, vec, alpha):
            return mw.generate_kv(ctx, max_new_tokens=28)

    # ---- loop-escape on natural loop contexts ----
    conds = {"no_steer": None, "steer": steer_vec, "random": rnd}   # orthogonal dropped for speed
    esc = {k: [] for k in conds}; coh = {k: [] for k in conds}
    for si, s in enumerate(test):
        ctx = mw.render(s.loop_messages(K))
        for k, vec in conds.items():
            out = gen(ctx, vec)
            cmd = parse_cmd(out)
            esc[k].append(0 if is_repeat(cmd, s.c_cmd) else 1)   # escaped if NOT the looped cmd
            coh[k].append(1 if ("`" in out and len(cmd) > 1) else 0)
        if (si + 1) % 5 == 0:
            print(f"  {si+1}/{len(test)} | escape so far: "
                  f"{ {k: round(np.mean(esc[k]),2) for k in conds} } | {time.time()-t0:.0f}s", flush=True)

    # ---- disruption on healthy progress contexts ----
    disr = {"steer": [], "random": []}
    for s in test:
        ctx = mw.render(s.progress_messages(K))
        base = parse_cmd(gen(ctx, None))
        for k, vec in [("steer", steer_vec), ("random", rnd)]:
            disr[k].append(0 if parse_cmd(gen(ctx, vec)) == base else 1)

    res = {"alpha": alpha, "dir": dname, "layer": int(L), "sign": int(sign), "n_test": len(test),
           "loop_escape_rate": {k: float(np.mean(esc[k])) for k in conds},
           "coherence_rate": {k: float(np.mean(coh[k])) for k in conds},
           "disruption_rate_on_healthy": {k: float(np.mean(disr[k])) for k in disr}}
    (OUT / "onpolicy_results.json").write_text(json.dumps(res, indent=2))
    print(f"[escape] {res['loop_escape_rate']}", flush=True)
    print(f"[coherence] {res['coherence_rate']}", flush=True)
    print(f"[disruption-on-healthy] {res['disruption_rate_on_healthy']}", flush=True)
    print(f"[done] wrote onpolicy_results.json | {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
