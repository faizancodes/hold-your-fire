"""Phase 4 (causal necessity) + Phase 5 (steering calibration) — the CAUSAL test.

Directions are built from the LENGTH-CONTROLLED activations (acts2.npz) so they are
confound-free; the steering test runs on NATURAL (unpadded) loop contexts so it is realistic.

  REP   = mean(resid|loopfail) - mean(resid|vfail)      "I'm repeating the same command"
  STUCK = mean(resid|loopfail) - mean(resid|loopprog)   "my repetition is failing"

repeat-preference on a loop context = logp(repeat C) - logp(best novel alt).  Lower = less repeat.

Stage 1: scan {REP,STUCK} x layers x sign at fixed alpha -> find the direction that most
         reduces repeat-pref.  Stage 2: alpha-sweep at the winner + random/orthogonal controls
         + coherence.  Plus directional ablation (necessity).

    mech_interp/.venv/bin/python -m mech_interp.run_steer_eval
"""
from __future__ import annotations

import json, time
from pathlib import Path

import numpy as np

from mech_interp.model_wrapper import ModelWrapper
from mech_interp.synthetic import make_scenarios

OUT = Path(__file__).parent / "results"
RNG = np.random.default_rng(1)
N_SCEN, K = 60, 4
SCAN_LAYERS = [4, 8, 12, 16, 20, 24]
TRAIN_N = 40                      # scenarios 0..39 build directions; 40..59 are held-out test
ALPHAS = [2.0, 4.0, 6.0, 8.0, 12.0]


def boot_ci(v, n=2000):
    v = np.asarray(v, float)
    bs = [np.mean(RNG.choice(v, len(v), replace=True)) for _ in range(n)]
    return float(np.mean(v)), float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))


def main():
    t0 = time.time()
    d = np.load(OUT / "acts2.npz", allow_pickle=True)
    cond, group = d["meta_cond"], d["meta_group"]
    tr = group < TRAIN_N

    def direction(layer, pos, neg):
        X = d[f"L{layer}"]
        v = X[tr & (cond == pos)].mean(0) - X[tr & (cond == neg)].mean(0)
        return v / (np.linalg.norm(v) + 1e-8)

    mw = ModelWrapper()
    scen = make_scenarios(N_SCEN)
    test = scen[TRAIN_N:]

    # natural (unpadded) loop contexts + candidate continuations + baselines
    rows = []
    for s in test:
        ctx = mw.render(s.loop_messages(K))
        rc = f" `{s.c_cmd}`"
        alts = [f" `{a}`" for a in s.novel_alts]
        alp = [mw.continuation_logprob(ctx, a) for a in alts]
        nc = alts[int(np.argmax(alp))]
        base = mw.continuation_logprob(ctx, rc) - max(alp)
        rows.append(dict(ctx=ctx, rc=rc, nc=nc, base=base))
    base_mean = float(np.mean([r["base"] for r in rows]))
    print(f"[base] natural-loop repeat-pref={base_mean:+.3f} (n={len(rows)}) | {time.time()-t0:.0f}s", flush=True)

    def dpref(direction_vec, layer, alpha, subset):
        out = []
        for r in subset:
            with mw.steering(layer, direction_vec, alpha):
                lp_r = mw.continuation_logprob(r["ctx"], r["rc"])
                lp_n = mw.continuation_logprob(r["ctx"], r["nc"])
            out.append((lp_r - lp_n) - r["base"])
        return out

    # ---- Stage 1: scan dir x layer x sign at alpha=8 on a 10-scenario subset ----
    print("[scan] Δrepeat-pref @α=8 (neg = steering reduces repeating):", flush=True)
    sub = rows[:10]
    dirs_def = {"REP": ("loopfail", "vfail"), "STUCK": ("loopfail", "loopprog")}
    scan = {}
    best = None
    for dname, (p, n) in dirs_def.items():
        for L in SCAN_LAYERS:
            v = direction(L, p, n)
            for sign in (+1, -1):
                m = float(np.mean(dpref(sign * v, L, 8.0, sub)))
                scan[f"{dname}_L{L}_s{sign:+d}"] = m
                if best is None or m < best[0]:
                    best = (m, dname, L, sign, v)
        print(f"  {dname}: " + " ".join(f"L{L}:{scan[f'{dname}_L{L}_s-1']:+.2f}/{scan[f'{dname}_L{L}_s+1']:+.2f}"
                                        for L in SCAN_LAYERS) + "   (−sign/+sign)", flush=True)
    _, bname, bL, bsign, bv = best
    print(f"[scan] best = {bname} L={bL} sign={bsign:+d}  (Δpref={best[0]:+.3f})", flush=True)

    # ---- Stage 2: α-sweep at winner + random/orthogonal controls + coherence ----
    rnd = RNG.standard_normal(bv.shape[0]); rnd /= np.linalg.norm(rnd)
    orth = rnd - (rnd @ bv) * bv; orth /= np.linalg.norm(orth)
    steer = {}
    for label, vec in [("best", bsign * bv), ("random", rnd), ("orthogonal", orth)]:
        steer[label] = {"alpha": ALPHAS, "delta_pref": [], "ci_lo": [], "ci_hi": [], "novel_logp": []}
        for a in ALPHAS:
            dp, nlp = [], []
            for r in rows:
                with mw.steering(bL, vec, a):
                    lp_r = mw.continuation_logprob(r["ctx"], r["rc"])
                    lp_n = mw.continuation_logprob(r["ctx"], r["nc"])
                dp.append((lp_r - lp_n) - r["base"]); nlp.append(lp_n)
            m, lo, hi = boot_ci(dp)
            steer[label]["delta_pref"].append(m); steer[label]["ci_lo"].append(lo)
            steer[label]["ci_hi"].append(hi); steer[label]["novel_logp"].append(float(np.mean(nlp)))
        print(f"  {label:11s} Δpref={[round(x,3) for x in steer[label]['delta_pref']]}", flush=True)

    # ---- Phase 4 ablation: project off the winning direction at its layer ----
    abl = []
    for r in rows:
        with mw.ablate(bL, bv):
            p = mw.continuation_logprob(r["ctx"], r["rc"]) - mw.continuation_logprob(r["ctx"], r["nc"])
        abl.append(p - r["base"])
    am, alo, ahi = boot_ci(abl)
    print(f"[ablation] project off {bname}@L{bL}: Δpref={am:+.3f} [{alo:+.3f},{ahi:+.3f}]", flush=True)

    res = {"baseline_pref": base_mean, "best_dir": bname, "best_layer": int(bL), "best_sign": int(bsign),
           "scan": scan, "steering": steer, "ablation": {"mean": am, "ci": [alo, ahi]}}
    (OUT / "steer_eval_results.json").write_text(json.dumps(res, indent=2))
    print(f"[done] wrote steer_eval_results.json | {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
