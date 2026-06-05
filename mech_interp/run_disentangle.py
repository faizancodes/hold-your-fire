"""T4 — disentangle "stuck-awareness" from "reading the test output".

Pure analysis of cached acts2.npz (no model). The three conditions share features pairwise:
  loopfail : same command,      FAILING obs
  vfail    : DIFFERENT commands, FAILING obs   (shares OBS with loopfail, differs in command)
  loopprog : same command,      IMPROVING obs  (shares COMMAND with loopfail, differs in obs)

So projecting all three onto each axis reveals what the axis reads:
  STUCK = mean(loopfail) - mean(loopprog)   [command fixed, obs varies]  -> "is my output improving?"
  REP   = mean(loopfail) - mean(vfail)      [obs fixed, command varies]  -> "am I repeating the command?"

If STUCK is pure obs-reading, vfail (same obs as loopfail) sits with loopfail on the STUCK axis,
and loopprog sits apart. If REP is pure command-reading, loopprog (same command) sits with
loopfail on the REP axis, and vfail apart. Their cosine says whether they're one signal or two.

    mech_interp/.venv/bin/python -m mech_interp.run_disentangle
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold

OUT = Path(__file__).parent / "results"
LAYER = 8


def cv_auc(X, y, g, k=5):
    y = np.asarray(y); g = np.asarray(g); a = []
    for tr, te in GroupKFold(k).split(X, y, g):
        v = X[tr][y[tr] == 1].mean(0) - X[tr][y[tr] == 0].mean(0)
        if len(np.unique(y[te])) == 2:
            a.append(roc_auc_score(y[te], X[te] @ v))
    return float(np.mean(a))


def main():
    d = np.load(OUT / "acts2.npz", allow_pickle=True)
    cond, group = d["meta_cond"], d["meta_group"]
    res = {"layer": LAYER, "by_layer": {}}
    for L in [4, 8, 12, 16]:
        X = d[f"L{L}"]
        mu = {c: X[cond == c].mean(0) for c in ["loopfail", "vfail", "loopprog"]}
        STUCK = mu["loopfail"] - mu["loopprog"]; STUCK /= np.linalg.norm(STUCK)
        REP = mu["loopfail"] - mu["vfail"]; REP /= np.linalg.norm(REP)
        cos = float(abs(STUCK @ REP))
        # mean projection of each condition onto each axis (centered on loopfail=0)
        projS = {c: float((X[cond == c] @ STUCK).mean() - (X[cond == "loopfail"] @ STUCK).mean())
                 for c in ["loopfail", "vfail", "loopprog"]}
        projR = {c: float((X[cond == c] @ REP).mean() - (X[cond == "loopfail"] @ REP).mean())
                 for c in ["loopfail", "vfail", "loopprog"]}
        res["by_layer"][str(L)] = {"cos_STUCK_REP": cos, "proj_on_STUCK": projS, "proj_on_REP": projR}
        if L == LAYER:
            m_rep = np.isin(cond, ["loopfail", "vfail"])
            m_stk = np.isin(cond, ["loopfail", "loopprog"])
            res["REP_auc_obs_fixed"] = cv_auc(X[m_rep], (cond[m_rep] == "loopfail").astype(int), group[m_rep])
            res["STUCK_auc_cmd_fixed"] = cv_auc(X[m_stk], (cond[m_stk] == "loopfail").astype(int), group[m_stk])

    print(f"=== Disentangle @ L{LAYER} ===", flush=True)
    b = res["by_layer"][str(LAYER)]
    print(f"cos(STUCK,REP) = {b['cos_STUCK_REP']:.3f}  (low ⇒ two distinct signals)")
    print(f"REP probe (command varies, OBS FIXED)  AUC = {res['REP_auc_obs_fixed']:.3f}")
    print(f"STUCK probe (obs varies, COMMAND FIXED) AUC = {res['STUCK_auc_cmd_fixed']:.3f}")
    print("projection on STUCK axis (loopfail=0):", {k: round(v, 2) for k, v in b["proj_on_STUCK"].items()})
    print("   -> if vfail≈loopfail and loopprog far ⇒ STUCK reads the OBSERVATION (test output)")
    print("projection on REP axis (loopfail=0):", {k: round(v, 2) for k, v in b["proj_on_REP"].items()})
    print("   -> if loopprog≈loopfail and vfail far ⇒ REP reads the COMMAND history")
    print("cos by layer:", {L: round(res['by_layer'][L]['cos_STUCK_REP'], 2) for L in res['by_layer']})
    (OUT / "disentangle_results.json").write_text(json.dumps(res, indent=2))
    print("[done] wrote disentangle_results.json")


if __name__ == "__main__":
    main()
