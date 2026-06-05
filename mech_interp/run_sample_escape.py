"""Phase 6b — on-policy escape under SAMPLING (realistic: agents sample, not greedy).

Greedy escape was 0% (the loop is a strong attractor). But steering shifts the repeat
log-odds by ~0.23 nats, which under temperature sampling should move escape *probability*.
We sample N generations per loop context with/without steering and measure escape fraction.

    mech_interp/.venv/bin/python -m mech_interp.run_sample_escape [alpha]
"""
from __future__ import annotations

import json, sys, time
from pathlib import Path

import numpy as np

from mech_interp.model_wrapper import ModelWrapper
from mech_interp.synthetic import make_scenarios
from mech_interp.run_onpolicy import parse_cmd, is_repeat

OUT = Path(__file__).parent / "results"
RNG = np.random.default_rng(3)
N_SCEN, K, TRAIN_N = 60, 4, 40
N_TEST, N_SAMP, TEMP = 6, 4, 0.8


def main():
    t0 = time.time()
    alpha = float(sys.argv[1]) if len(sys.argv) > 1 else 12.0
    cfg = json.loads((OUT / "steer_eval_results.json").read_text())
    L, sign, dname = cfg["best_layer"], cfg["best_sign"], cfg["best_dir"]
    d = np.load(OUT / "acts2.npz", allow_pickle=True)
    cond, group = d["meta_cond"], d["meta_group"]; tr = group < TRAIN_N
    pos, neg = ("loopfail", "vfail") if dname == "REP" else ("loopfail", "loopprog")
    X = d[f"L{L}"]
    v = X[tr & (cond == pos)].mean(0) - X[tr & (cond == neg)].mean(0); v /= np.linalg.norm(v) + 1e-8
    steer_vec = sign * v
    rnd = RNG.standard_normal(v.shape[0]); rnd /= np.linalg.norm(rnd)
    print(f"[cfg] dir={dname} L={L} sign={sign:+d} alpha={alpha} temp={TEMP} "
          f"n_test={N_TEST} n_samp={N_SAMP}", flush=True)

    mw = ModelWrapper()
    test = make_scenarios(N_SCEN)[TRAIN_N:TRAIN_N + N_TEST]
    conds = {"no_steer": None, "steer": steer_vec, "random": rnd}
    esc = {k: [] for k in conds}; examples = {k: [] for k in conds}

    for si, s in enumerate(test):
        ctx = mw.render(s.loop_messages(K))
        for k, vec in conds.items():
            for j in range(N_SAMP):
                if vec is None:
                    out = mw.generate_kv(ctx, max_new_tokens=28, temperature=TEMP, seed=1000 * si + j)
                else:
                    with mw.steering(L, vec, alpha):
                        out = mw.generate_kv(ctx, max_new_tokens=28, temperature=TEMP, seed=1000 * si + j)
                cmd = parse_cmd(out)
                esc[k].append(0 if is_repeat(cmd, s.c_cmd) else 1)
                if len(examples[k]) < 6 and not is_repeat(cmd, s.c_cmd):
                    examples[k].append(cmd[:50])
        print(f"  {si+1}/{len(test)} escape={ {k: round(np.mean(esc[k]),2) for k in conds} } | {time.time()-t0:.0f}s", flush=True)

    res = {"alpha": alpha, "temp": TEMP, "n_test": N_TEST, "n_samp": N_SAMP,
           "escape_rate": {k: float(np.mean(esc[k])) for k in conds},
           "escape_examples": examples}
    (OUT / "sample_escape_results.json").write_text(json.dumps(res, indent=2))
    print(f"[escape@temp{TEMP}] {res['escape_rate']}", flush=True)
    print(f"[examples steer] {examples['steer'][:4]}", flush=True)
    print(f"[done] | {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
