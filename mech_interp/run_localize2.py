"""Phase 2/3 v2 — EXACT length control + a token-repetition control.

Every command is padded to T_CMD tokens and every observation to T_OBS tokens, with a fixed
thought, so all conditions have identical context length. Conditions (K turns):
  loopfail : command C x K,  observation = FAILING (constant)
  vfail    : K different cmds, observation = FAILING (constant)
  loopprog : command C x K,  observation = IMPROVING (3->2->1->0 failed)

Contrasts:
  REP   = loopfail vs vfail      same-cmd repeat vs varied (length-matched) -> 'repetition encoded?'
  STUCK = loopfail vs loopprog   SAME repeated cmd, failing vs improving obs -> genuine
          'unproductive repetition / stuck' signal, with TOKEN-REPETITION HELD IDENTICAL.

    mech_interp/.venv/bin/python -m mech_interp.run_localize2
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
N_SCEN, K = 60, 4
T_CMD, T_OBS = 32, 16
THOUGHT = "Let me try this."


def pad_tok(tok, text, T):
    out = text + " #"
    while len(tok(out, add_special_tokens=False).input_ids) < T:
        out += " x"
    return out


def build(mw, s, cond):
    msgs = [{"role": "system", "content": s.system}, {"role": "user", "content": s.task}]
    if cond == "loopfail":
        cmds = [s.c_cmd] * K; obss = ["1 failed - assertion error"] * K
    elif cond == "vfail":
        cmds = [s.vfail[i % len(s.vfail)][1] for i in range(K)]
        obss = ["1 failed - assertion error"] * K
    else:  # loopprog
        cmds = [s.c_cmd] * K
        obss = [f"{max(3 - i, 0)} failed - assertion error" for i in range(K)]
    for cmd, obs in zip(cmds, obss):
        msgs.append({"role": "assistant", "content": f"{THOUGHT}\n`{pad_tok(mw.tok, cmd, T_CMD)}`"})
        msgs.append({"role": "user", "content": pad_tok(mw.tok, obs, T_OBS)})
    return msgs


def diffmeans_cv(X, y, groups, k=5):
    y = np.asarray(y); groups = np.asarray(groups); aucs = []; proj = np.full(len(y), np.nan)
    for tr, te in GroupKFold(k).split(X, y, groups):
        v = X[tr][y[tr] == 1].mean(0) - X[tr][y[tr] == 0].mean(0)
        proj[te] = X[te] @ v
        if len(np.unique(y[te])) == 2:
            aucs.append(roc_auc_score(y[te], proj[te]))
    return float(np.mean(aucs)), float(np.std(aucs)), proj


def main():
    t0 = time.time()
    mw = ModelWrapper()
    scen = make_scenarios(N_SCEN)
    conds = ["loopfail", "vfail", "loopprog"]
    layers = list(range(mw.n_layers))
    Xl = {L: [] for L in layers}; cond, group, length = [], [], []
    # length check on first scenario
    lens = {c: mw.tok(mw.render(build(mw, scen[0], c)), return_tensors="pt").input_ids.shape[1] for c in conds}
    print(f"[length-match] scenario0 token lengths: {lens}  (should be ~equal)", flush=True)
    for si, s in enumerate(scen):
        for c in conds:
            text = mw.render(build(mw, s, c))
            ntok = mw.tok(text, return_tensors="pt").input_ids.shape[1]
            acts = mw.capture_resid(text)
            for L in layers:
                Xl[L].append(acts[L])
            cond.append(c); group.append(si); length.append(ntok)
        if (si + 1) % 20 == 0:
            print(f"  captured {si+1}/{len(scen)} | {time.time()-t0:.0f}s", flush=True)
    Xl = {L: np.asarray(v, np.float32) for L, v in Xl.items()}
    cond = np.array(cond); group = np.array(group); length = np.array(length)
    np.savez_compressed(OUT / "acts2.npz", meta_cond=cond, meta_group=group, meta_len=length,
                        **{f"L{L}": Xl[L] for L in layers})
    for c in conds:
        print(f"  mean len[{c}]={length[cond==c].mean():.1f}", flush=True)

    out = {"model": mw.model_name, "n_scen": N_SCEN, "K": K, "contrasts": {}}
    for name, pos, neg in [("REP_loopfail_vs_vfail", "loopfail", "vfail"),
                           ("STUCK_loopfail_vs_loopprog", "loopfail", "loopprog")]:
        m = np.isin(cond, [pos, neg]); y = (cond[m] == pos).astype(int)
        gl = group[m]; ln = length[m]
        len_auc = max(roc_auc_score(y, ln), 1 - roc_auc_score(y, ln)) if len(np.unique(y)) == 2 else 0.5
        by = {}
        for L in layers:
            auc, sd, proj = diffmeans_cv(Xl[L][m], y, gl)
            corr = abs(np.corrcoef(proj, ln)[0, 1]) if np.std(proj) > 0 else 0.0
            by[L] = {"auc": auc, "sd": sd, "len_corr": float(corr)}
        peak = max(by, key=lambda L: by[L]["auc"])
        out["contrasts"][name] = {"length_only_auc": float(len_auc), "peak_layer": int(peak),
                                  "by_layer": {str(L): by[L] for L in layers}}
        print(f"[{name}] length-only AUC={len_auc:.3f} | peak L={peak} "
              f"AUC={by[peak]['auc']:.3f}(±{by[peak]['sd']:.3f}) len-corr={by[peak]['len_corr']:.2f}", flush=True)

    (OUT / "localize2_results.json").write_text(json.dumps(out, indent=2))
    print(f"[done] wrote localize2_results.json | {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
