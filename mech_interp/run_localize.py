"""Phase 2 + 3 (rigorous): localize the repetition representation with confound controls.

Three length/outcome-matched conditions at K turns:
  A loop   : same command C x K, same failing observation
  B vfail  : K DIFFERENT commands, all failing the same way  (controls 'failing repeatedly')
  C prog   : K distinct productive commands                  (healthy baseline)

Key contrast = A-vs-B (isolates *identical-command* repetition from mere repeated failure,
with matched turn-count). For every layer & contrast we report:
  - diff-of-means AUC (scenario-grouped CV)        [the signal]
  - length-only AUC                                [trivial cue baseline]
  - |corr(projection, context length)|             [length-confound check]
so we never mistake a position/length artifact for a 'loop-awareness' representation.

Saves activations to results/acts.npz for reuse in Phases 4-5.

    mech_interp/.venv/bin/python -m mech_interp.run_localize
"""
from __future__ import annotations

import json, time
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold

from mech_interp.model_wrapper import ModelWrapper
from mech_interp.synthetic import make_scenarios

OUT = Path(__file__).parent / "results"; OUT.mkdir(exist_ok=True)
RNG = np.random.default_rng(0)
N_SCEN, K = 72, 4
CONDS = ["loop", "vfail", "prog"]


def diffmeans_cv(X, y, groups, k=5):
    """grouped-CV: returns (mean AUC, mean |corr(proj,?)| handled by caller, per-fold scores)."""
    y = np.asarray(y); groups = np.asarray(groups); aucs = []; proj = np.full(len(y), np.nan)
    for tr, te in GroupKFold(k).split(X, y, groups):
        v = X[tr][y[tr] == 1].mean(0) - X[tr][y[tr] == 0].mean(0)
        s = X[te] @ v
        proj[te] = s
        if len(np.unique(y[te])) == 2:
            aucs.append(roc_auc_score(y[te], s))
    return float(np.mean(aucs)), float(np.std(aucs)), proj


def contrast(Xl, meta, pos_cond, neg_cond):
    m = np.isin(meta["cond"], [pos_cond, neg_cond])
    y = (meta["cond"][m] == pos_cond).astype(int)
    groups = meta["group"][m]
    lengths = meta["len"][m]
    len_auc = roc_auc_score(y, lengths) if len(np.unique(y)) == 2 else 0.5
    len_auc = max(len_auc, 1 - len_auc)              # length cue strength (symmetric)
    res = {}
    for L in sorted(Xl):
        auc, sd, proj = diffmeans_cv(Xl[L][m], y, groups)
        corr = abs(np.corrcoef(proj, lengths)[0, 1]) if np.std(proj) > 0 else 0.0
        res[L] = {"auc": auc, "sd": sd, "len_corr": float(corr)}
    return {"length_only_auc": float(len_auc), "by_layer": res}


def main():
    t0 = time.time()
    mw = ModelWrapper()
    scen = make_scenarios(N_SCEN)
    print(f"[setup] {len(scen)} scenarios x {len(CONDS)} conds, K={K} | {time.time()-t0:.0f}s", flush=True)

    layers = list(range(mw.n_layers))
    Xl = {L: [] for L in layers}
    cond, group, length = [], [], []
    builders = {"loop": lambda s: s.loop_messages(K),
                "vfail": lambda s: s.varied_fail_messages(K),
                "prog": lambda s: s.progress_messages(K)}
    for si, s in enumerate(scen):
        for c in CONDS:
            text = mw.render(builders[c](s))
            ntok = mw.tok(text, return_tensors="pt").input_ids.shape[1]
            acts = mw.capture_resid(text)
            for L in layers:
                Xl[L].append(acts[L])
            cond.append(c); group.append(si); length.append(ntok)
        if (si + 1) % 20 == 0:
            print(f"  captured {si+1}/{len(scen)} | {time.time()-t0:.0f}s", flush=True)
    Xl = {L: np.asarray(v, dtype=np.float32) for L, v in Xl.items()}
    meta = {"cond": np.array(cond), "group": np.array(group), "len": np.array(length)}
    np.savez_compressed(OUT / "acts.npz", meta_cond=meta["cond"], meta_group=meta["group"],
                        meta_len=meta["len"], **{f"L{L}": Xl[L] for L in layers})
    print(f"[save] acts.npz | len(loop)={meta['len'][meta['cond']=='loop'].mean():.0f} "
          f"vfail={meta['len'][meta['cond']=='vfail'].mean():.0f} "
          f"prog={meta['len'][meta['cond']=='prog'].mean():.0f} tokens", flush=True)

    out = {"model": mw.model_name, "n_scen": N_SCEN, "K": K, "contrasts": {}}
    for pos, neg in [("loop", "prog"), ("loop", "vfail"), ("vfail", "prog")]:
        name = f"{pos}_vs_{neg}"
        c = contrast(Xl, meta, pos, neg)
        out["contrasts"][name] = c
        peakL = max(c["by_layer"], key=lambda L: c["by_layer"][L]["auc"])
        pk = c["by_layer"][peakL]
        print(f"[{name}] length-only AUC={c['length_only_auc']:.3f} | "
              f"peak L={peakL} AUC={pk['auc']:.3f}(±{pk['sd']:.3f}) len-corr={pk['len_corr']:.2f}", flush=True)
        # also report the best *length-decorrelated* layer (|corr|<0.5)
        clean = {L: v for L, v in c["by_layer"].items() if v["len_corr"] < 0.5}
        if clean:
            bL = max(clean, key=lambda L: clean[L]["auc"])
            print(f"   best layer with low length-confound (|corr|<0.5): L={bL} "
                  f"AUC={clean[bL]['auc']:.3f} len-corr={clean[bL]['len_corr']:.2f}", flush=True)
            out["contrasts"][name]["clean_peak_layer"] = int(bL)
        out["contrasts"][name]["peak_layer"] = int(peakL)

    # dose-response on the A-vs-B (genuine repetition) direction
    print("[dose] projection onto loop-vs-vfail direction vs K", flush=True)
    ab = out["contrasts"]["loop_vs_vfail"]
    L = ab.get("clean_peak_layer", ab["peak_layer"])
    m = np.isin(meta["cond"], ["loop", "vfail"]); y = (meta["cond"][m] == "loop").astype(int)
    v = Xl[L][m][y == 1].mean(0) - Xl[L][m][y == 0].mean(0); v /= np.linalg.norm(v) + 1e-8
    dose = {"K": [1, 2, 3, 4, 5, 6], "dir_layer": int(L), "loop": [], "vfail": [], "prog": []}
    sub = scen[:10]
    for kk in dose["K"]:
        for c, key in [("loop", "loop"), ("vfail", "vfail"), ("prog", "prog")]:
            vals = [mw.capture_resid(mw.render(builders[c](s)))[L] @ v for s in sub]
            dose[key].append(float(np.mean(vals)))
    out["dose_response"] = dose
    print(f"  loop:  {[round(x,2) for x in dose['loop']]}", flush=True)
    print(f"  vfail: {[round(x,2) for x in dose['vfail']]}", flush=True)
    print(f"  prog:  {[round(x,2) for x in dose['prog']]}", flush=True)

    (OUT / "localize_results.json").write_text(json.dumps(out, indent=2))
    print(f"[done] wrote localize_results.json | {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
