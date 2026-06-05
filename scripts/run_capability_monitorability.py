#!/usr/bin/env python3
"""Is monitorability driven by agent CAPABILITY or by SCAFFOLD? (generalization analysis)

Tests the hypothesis "more capable agents are harder to monitor". We measure each agent's
in-domain monitorability (grouped-CV AUC of the prefix-feature monitor) at a MATCHED instance
budget (controls the data-size confound) and plot it against the agent's success rate
(capability). Within-Nebius (8b/70b/405b) is the controlled comparison (same shell scaffold,
overlapping tasks); OpenHands/CodeAct is a different-scaffold point.

Finding (refutes the hypothesis): within a scaffold, capability does NOT reduce monitorability
— it slightly increases it; the real driver is the SCAFFOLD (shell vs CodeAct action space).

  python scripts/run_capability_monitorability.py
"""

from __future__ import annotations

import numpy as np
import pandas as pd

import _bootstrap  # noqa: F401
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import GroupKFold

from localguard.evaluate import roc_auc_metric
from localguard.train import MonitorModel
from localguard.utils import REPO_ROOT, RESULTS_FIGURES, RESULTS_OFFLINE, load_config, write_json

BUDGET, REPS = 60, 20  # matched instance budget (controls data-size confound) + subsamples


def _matched_auc(df, cols, rng):
    X = df[cols].apply(pd.to_numeric, errors="coerce").fillna(0).to_numpy()
    y = df["y_fail"].to_numpy(int); g = df["instance_id"].to_numpy()
    insts = np.unique(g); aucs = []
    for _ in range(REPS):
        keep = set(rng.choice(insts, min(BUDGET, len(insts)), replace=False))
        m = np.array([x in keep for x in g])
        Xs, ys, gs = X[m], y[m], g[m]
        if len(np.unique(ys)) < 2:
            continue
        oof = np.zeros(len(ys))
        for tr, te in GroupKFold(min(5, len(np.unique(gs)))).split(Xs, ys, groups=gs):
            mdl = HistGradientBoostingClassifier(random_state=0).fit(Xs[tr], ys[tr])
            oof[te] = mdl.predict_proba(Xs[te])[:, 1]
        aucs.append(roc_auc_metric(ys, oof))
    return float(np.mean(aucs)), float(np.std(aucs))


def main() -> None:
    cfg = load_config("configs/offline_full.yaml").data
    cols = MonitorModel.load(REPO_ROOT / cfg["models_dir"] / "hist_gradient_boosting.joblib").numeric_cols
    neb = pd.read_parquet(REPO_ROOT / cfg["dataset"])
    oh = pd.read_parquet(REPO_ROOT / "data/processed/prefix_openhands.parquet")
    rng = np.random.default_rng(0)

    agents = [
        ("swe-agent-llama-8b", "shell", neb[neb.model_name == "swe-agent-llama-8b"]),
        ("swe-agent-llama-70b", "shell", neb[neb.model_name == "swe-agent-llama-70b"]),
        ("swe-agent-llama-405b", "shell", neb[neb.model_name == "swe-agent-llama-405b"]),
        ("openhands-coderforge-32b", "CodeAct", oh),
    ]
    rows = []
    for name, scaffold, df in agents:
        succ = 1 - df.groupby("trajectory_id")["y_fail"].first().mean()
        auc, sd = _matched_auc(df, cols, rng)
        rows.append({"agent": name, "scaffold": scaffold, "success_rate": round(float(succ), 3),
                     "n_instances": int(df["instance_id"].nunique()),
                     "monitorability_auc_matched60": round(auc, 4), "auc_sd": round(sd, 4)})
        print(f"  {name:26s} [{scaffold:7s}] success={succ:.3f} n_inst={df['instance_id'].nunique():5d} "
              f"AUC@{BUDGET}={auc:.3f}±{sd:.3f}")

    out = {"budget_instances": BUDGET, "reps": REPS,
           "note": "matched-budget AUC controls data-size; within-shell is the controlled capability test",
           "agents": rows}
    write_json(RESULTS_OFFLINE / "full" / "capability_monitorability.json", out)

    # figure: monitorability vs capability, by scaffold
    fig, ax = plt.subplots(figsize=(7, 4.6))
    for scaf, color, mk in [("shell", "tab:blue", "o"), ("CodeAct", "tab:red", "X")]:
        pts = [r for r in rows if r["scaffold"] == scaf]
        ax.errorbar([p["success_rate"] for p in pts], [p["monitorability_auc_matched60"] for p in pts],
                    yerr=[p["auc_sd"] for p in pts], fmt=mk, ms=11, color=color, label=scaf, capsize=4, ls="none")
    sh = sorted([r for r in rows if r["scaffold"] == "shell"], key=lambda r: r["success_rate"])
    ax.plot([p["success_rate"] for p in sh], [p["monitorability_auc_matched60"] for p in sh],
            color="tab:blue", alpha=0.5, ls="--")
    for r in rows:
        ax.annotate(r["agent"].replace("swe-agent-llama-", "").replace("openhands-coderforge-", "OH-"),
                    (r["success_rate"], r["monitorability_auc_matched60"]), fontsize=7,
                    xytext=(4, 5), textcoords="offset points")
    ax.axhline(0.5, color="k", alpha=0.3, ls=":")
    ax.set_xlabel("agent capability (trajectory success rate)")
    ax.set_ylabel(f"in-domain monitorability (AUC @ {BUDGET} inst)")
    ax.set_title("Monitorability is scaffold-driven, not capability-driven\n"
                 "(within shell, capability HELPS; CodeAct is low regardless)", fontsize=10)
    ax.legend(title="scaffold"); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(RESULTS_FIGURES / "fig10_capability_monitorability.png", dpi=130); plt.close()
    print(f"\n[cap] wrote capability_monitorability.json + fig10_capability_monitorability.png")


if __name__ == "__main__":
    main()
