"""MVP (de-risking milestone): does Qwen-1.5B internally represent 'I'm in a loop',
and can a diff-of-means steering vector reduce its preference for repeating?

Runs Phases 2-5 in miniature on synthetic data and writes real numbers + figures.

    mech_interp/.venv/bin/python -m mech_interp.run_mvp
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

from mech_interp.model_wrapper import ModelWrapper
from mech_interp.synthetic import make_scenarios

OUT = Path(__file__).parent / "results"
OUT.mkdir(exist_ok=True)
RNG = np.random.default_rng(0)

N_SCEN = 72
K_MAIN = 4
ALPHAS = [2.0, 4.0, 6.0, 8.0, 12.0]
DOSE_K = [1, 2, 3, 4, 5, 6]


def grouped_diffmeans_auc(X, y, groups, k=5):
    """1-D AUC of the diff-of-means direction, group-CV (the steering direction's separability)."""
    y = np.asarray(y); groups = np.asarray(groups)
    aucs = []
    for tr, te in GroupKFold(k).split(X, y, groups):
        mu1 = X[tr][y[tr] == 1].mean(0); mu0 = X[tr][y[tr] == 0].mean(0)
        v = mu1 - mu0
        s = X[te] @ v
        if len(np.unique(y[te])) == 2:
            aucs.append(roc_auc_score(y[te], s))
    return float(np.mean(aucs)), float(np.std(aucs))


def grouped_lr_auc(X, y, groups, k=5):
    y = np.asarray(y); groups = np.asarray(groups)
    aucs = []
    for tr, te in GroupKFold(k).split(X, y, groups):
        sc = StandardScaler().fit(X[tr])
        clf = LogisticRegression(C=0.05, max_iter=2000).fit(sc.transform(X[tr]), y[tr])
        p = clf.predict_proba(sc.transform(X[te]))[:, 1]
        if len(np.unique(y[te])) == 2:
            aucs.append(roc_auc_score(y[te], p))
    return float(np.mean(aucs)), float(np.std(aucs))


def boot_ci(vals, n=2000):
    vals = np.asarray(vals, float)
    bs = [np.mean(RNG.choice(vals, len(vals), replace=True)) for _ in range(n)]
    return float(np.mean(vals)), float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))


def main():
    t0 = time.time()
    mw = ModelWrapper()
    scen = make_scenarios(N_SCEN)
    print(f"[setup] {mw.model_name} | {len(scen)} scenarios | {time.time()-t0:.1f}s", flush=True)

    # ---- capture decision-point activations: LOOP vs PROGRESS (length-matched) ----
    layers = list(range(mw.n_layers))
    Xl = {L: [] for L in layers}            # per-layer feature matrix
    y, groups = [], []
    for si, s in enumerate(scen):
        for cond, msgs in [("loop", s.loop_messages(K_MAIN)), ("prog", s.progress_messages(K_MAIN))]:
            text = mw.render(msgs)
            acts = mw.capture_resid(text)
            for L in layers:
                Xl[L].append(acts[L])
            y.append(1 if cond == "loop" else 0)
            groups.append(si)
        if (si + 1) % 20 == 0:
            print(f"  captured {si+1}/{len(scen)} scenarios | {time.time()-t0:.0f}s", flush=True)
    y = np.array(y); groups = np.array(groups)
    Xl = {L: np.array(v) for L, v in Xl.items()}

    # ---- Phase 3: layer-wise probe AUC (group-CV) ----
    print("[probe] layer-wise loop-vs-progress AUC (scenario-grouped CV)", flush=True)
    probe = {}
    for L in layers:
        dm, dms = grouped_diffmeans_auc(Xl[L], y, groups)
        probe[L] = {"diffmeans_auc": dm, "diffmeans_sd": dms}
    peak = max(layers, key=lambda L: probe[L]["diffmeans_auc"])
    lr_dm, lr_sd = grouped_lr_auc(Xl[peak], y, groups)
    probe[peak]["lr_auc"] = lr_dm; probe[peak]["lr_sd"] = lr_sd
    print(f"  peak layer L={peak}: diff-means AUC={probe[peak]['diffmeans_auc']:.3f}"
          f" (±{probe[peak]['diffmeans_sd']:.3f}) | LR AUC={lr_dm:.3f}", flush=True)
    print(f"  H1/H2: {'H2 (model REPRESENTS the loop)' if probe[peak]['diffmeans_auc']>0.65 else 'H1 (not decodable)'}", flush=True)

    # ---- build steering vector v on a TRAIN split; evaluate steering on TEST split ----
    n_test = 24
    test_ids = set(RNG.choice(len(scen), n_test, replace=False).tolist())
    tr_mask = np.array([g not in test_ids for g in groups])
    Xp = Xl[peak]
    v = Xp[tr_mask & (y == 0)].mean(0) - Xp[tr_mask & (y == 1)].mean(0)   # progress - loop (push loop->progress)
    v = v / (np.linalg.norm(v) + 1e-8)
    # orthogonal control: random vector projected off v
    rnd = RNG.standard_normal(mw.d_model); rnd /= np.linalg.norm(rnd)
    orth = rnd - (rnd @ v) * v; orth /= np.linalg.norm(orth)
    dirs = {"steer_v": v, "random": rnd, "orthogonal": orth}

    # ---- Phase 4/5: steering effect on repeat-preference, held-out LOOP contexts ----
    print(f"[steer] effect on repeat-preference over {n_test} held-out loop contexts", flush=True)
    test_scen = [scen[i] for i in sorted(test_ids)]
    # baseline preference & pick best novel alternative per scenario
    base_pref, repeat_cont, novel_cont, loop_ctx = [], [], [], []
    for s in test_scen:
        ctx = mw.render(s.loop_messages(K_MAIN))
        rc = f" `{s.c_cmd}`"
        alts = [f" `{a}`" for a in s.novel_alts]
        alt_lp = [mw.continuation_logprob(ctx, a) for a in alts]
        nc = alts[int(np.argmax(alt_lp))]
        base_pref.append(mw.continuation_logprob(ctx, rc) - max(alt_lp))
        repeat_cont.append(rc); novel_cont.append(nc); loop_ctx.append(ctx)
    base_pref = np.array(base_pref)
    print(f"  baseline repeat-pref (logp repeat - logp novel): mean={base_pref.mean():+.3f}"
          f"  (>0 ⇒ model favors repeating)", flush=True)

    steer = {dname: {"alpha": [], "delta_pref_mean": [], "ci_lo": [], "ci_hi": [],
                     "novel_logp_mean": []} for dname in dirs}
    for dname, d in dirs.items():
        for a in ALPHAS:
            dpref, novel_lp = [], []
            for i, s in enumerate(test_scen):
                with mw.steering(peak, d, a):
                    lp_r = mw.continuation_logprob(loop_ctx[i], repeat_cont[i])
                    lp_n = mw.continuation_logprob(loop_ctx[i], novel_cont[i])
                dpref.append((lp_r - lp_n) - base_pref[i])   # change in repeat-preference
                novel_lp.append(lp_n)
            m, lo, hi = boot_ci(dpref)
            steer[dname]["alpha"].append(a)
            steer[dname]["delta_pref_mean"].append(m)
            steer[dname]["ci_lo"].append(lo); steer[dname]["ci_hi"].append(hi)
            steer[dname]["novel_logp_mean"].append(float(np.mean(novel_lp)))
        print(f"  {dname:11s} Δpref @α={ALPHAS}: "
              f"{[round(x,3) for x in steer[dname]['delta_pref_mean']]}", flush=True)

    # ---- dose-response: does the loop projection grow with K? ----
    print("[dose] projection of resid onto v as a function of repeat count K", flush=True)
    dose = {"K": DOSE_K, "loop_proj": [], "prog_proj": []}
    sub = scen[:30]
    for K in DOSE_K:
        lp, pp = [], []
        for s in sub:
            lp.append(mw.capture_resid(mw.render(s.loop_messages(K)))[peak] @ (-v))  # -v = loop direction
            pp.append(mw.capture_resid(mw.render(s.progress_messages(K)))[peak] @ (-v))
        dose["loop_proj"].append(float(np.mean(lp)))
        dose["prog_proj"].append(float(np.mean(pp)))
    print(f"  loop_proj(K): {[round(x,2) for x in dose['loop_proj']]}", flush=True)
    print(f"  prog_proj(K): {[round(x,2) for x in dose['prog_proj']]}", flush=True)

    res = {"model": mw.model_name, "n_scenarios": N_SCEN, "K_main": K_MAIN,
           "peak_layer": int(peak), "probe": {str(L): probe[L] for L in layers},
           "baseline_repeat_pref_mean": float(base_pref.mean()),
           "steering": steer, "dose_response": dose,
           "resid_norm_peak": float(np.linalg.norm(Xp[0])),
           "runtime_s": round(time.time() - t0, 1)}
    (OUT / "mvp_results.json").write_text(json.dumps(res, indent=2))
    print(f"[done] wrote {OUT/'mvp_results.json'} | {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
