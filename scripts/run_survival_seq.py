#!/usr/bin/env python3
"""Cleaner-label study, part 2: discrete-time survival / hazard model (deeper #3).

A causal GRU emits a per-step HAZARD; risk accumulates MONOTONICALLY
(risk_t = 1 - prod_{s<=t}(1-hazard_s)). Unlike per-prefix binary classification,
the monotone-survival structure does not FORCE risk->1 on early prefixes of doomed
runs, so in principle it can use early features more cleanly. We test whether this
lifts the EARLY/MID stratified AUC vs the plain HGB baseline. Same per-step
features and forward-only GRU as run_seq_model.py ⇒ no future leak.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import _bootstrap  # noqa: F401
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_seq_model import CAP, step_features  # noqa: E402

from localguard.evaluate import paired_bootstrap_auc_delta, roc_auc_metric  # noqa: E402
from localguard.schemas import NormalizedTrajectory  # noqa: E402
from localguard.utils import (  # noqa: E402
    DEFAULT_SEED,
    REPO_ROOT,
    RESULTS_OFFLINE,
    read_json,
    read_jsonl,
    write_json,
)


def _strata_auc(pos, y, p):
    out = {}
    for nm, m in [("early", pos <= 0.33), ("mid", (pos > 0.33) & (pos <= 0.66)), ("late", pos > 0.66),
                  ("overall", np.ones(len(pos), bool))]:
        if m.sum() > 50 and len(np.unique(y[m])) > 1:
            out[nm] = round(roc_auc_metric(y[m], p[m]), 4)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--normalized", default="data/interim/normalized_prefix_offline_full.jsonl")
    ap.add_argument("--split", default="results/offline/full/split_assignment.parquet")
    ap.add_argument("--v1", default="data/processed/prefix_offline_full.parquet")
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--hidden", type=int, default=64)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--n-boot", type=int, default=800)
    args = ap.parse_args()

    import torch
    import torch.nn as nn

    torch.manual_seed(DEFAULT_SEED); np.random.seed(DEFAULT_SEED)

    folds = pd.read_parquet(REPO_ROOT / args.split)
    v1 = pd.read_parquet(REPO_ROOT / args.v1).merge(folds, on="prefix_id")
    inst_fold = dict(zip(v1["instance_id"], v1["fold"]))

    print("[surv] loading normalized trajectories ...")
    trajs = [NormalizedTrajectory(**r) for r in read_jsonl(REPO_ROOT / args.normalized)]
    feats = {t.trajectory_id: step_features(t) for t in trajs}
    label = {t.trajectory_id: int(not t.target) for t in trajs}
    fold_of = {t.trajectory_id: inst_fold.get(t.instance_id, "train") for t in trajs}
    F = next(iter(feats.values())).shape[1]
    train_ids = [t for t in feats if fold_of[t] == "train"]
    stacked = np.concatenate([feats[t] for t in train_ids], axis=0)
    mu, sd = stacked.mean(0), stacked.std(0); sd[sd < 1e-6] = 1.0
    for t in feats:
        feats[t] = (feats[t] - mu) / sd

    class HazardGRU(nn.Module):
        def __init__(self, f, h):
            super().__init__()
            self.gru = nn.GRU(f, h, batch_first=True)
            self.head = nn.Linear(h, 1)

        def forward(self, x):
            out, _ = self.gru(x)
            hz_logit = self.head(out).squeeze(-1)            # (B,T)
            log_surv_step = -nn.functional.softplus(hz_logit)  # log(1-hazard)
            cum_log_surv = torch.cumsum(log_surv_step, dim=1)
            risk = 1.0 - torch.exp(cum_log_surv)             # monotone non-decreasing
            return risk.clamp(1e-6, 1 - 1e-6)

    model = HazardGRU(F, args.hidden)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
    bce = nn.BCELoss(reduction="none")

    def batch(ids):
        mats = [feats[t] for t in ids]; lens = [m.shape[0] for m in mats]; T = max(lens)
        x = np.zeros((len(ids), T, F), np.float32); m = np.zeros((len(ids), T), np.float32)
        y = np.zeros((len(ids),), np.float32)
        for i, mt in enumerate(mats):
            x[i, : lens[i]] = mt; m[i, : lens[i]] = 1.0; y[i] = label[ids[i]]
        return torch.from_numpy(x), torch.from_numpy(m), torch.from_numpy(y)

    rng = np.random.default_rng(DEFAULT_SEED)
    for ep in range(args.epochs):
        rng.shuffle(train_ids); model.train(); tot = 0.0
        for i in range(0, len(train_ids), args.batch):
            ids = train_ids[i: i + args.batch]
            x, m, y = batch(ids)
            risk = model(x)
            yb = y.unsqueeze(1).expand_as(risk)
            loss = (bce(risk, yb) * m).sum() / m.sum()
            opt.zero_grad(); loss.backward(); opt.step()
            tot += float(loss) * len(ids)
        print(f"[surv] epoch {ep+1}/{args.epochs} loss={tot/len(train_ids):.4f}")

    # predict at exact v1 prefix positions
    base = pd.read_parquet(RESULTS_OFFLINE / "full" / "auc_lift_baseline.parquet").sort_values("prefix_id").reset_index(drop=True)
    te = v1[v1.fold == "test"].sort_values("prefix_id").reset_index(drop=True)
    assert (te["prefix_id"].values == base["prefix_id"].values).all()
    model.eval()
    cache, p = {}, np.zeros(len(te))
    with torch.no_grad():
        for j, (tid, k) in enumerate(zip(te["trajectory_id"], te["prefix_step"])):
            if tid not in cache:
                cache[tid] = model(torch.from_numpy(feats[tid]).unsqueeze(0)).squeeze(0).numpy() if tid in feats else np.array([0.5])
            arr = cache[tid]; p[j] = arr[min(int(k) - 1, len(arr) - 1)]

    yt = te["y_fail"].to_numpy(int)
    pos = (te["prefix_step"] / te["n_total_steps"].clip(lower=1)).to_numpy()
    pt_base = base["p_base"].to_numpy()
    print(f"\n[surv] survival-GRU stratified AUC: {_strata_auc(pos, yt, p)}")
    print(f"[surv] baseline HGB    stratified AUC: {_strata_auc(pos, yt, pt_base)}")
    r = paired_bootstrap_auc_delta(te["instance_id"].to_numpy(), yt, pt_base, p, n_boot=args.n_boot)
    print(f"[surv] overall paired vs HGB baseline: Δ={r['delta']:+.4f} CI[{r['delta_lo']:+.4f},{r['delta_hi']:+.4f}] sig={r['significant']}")
    # per-stratum paired
    strat = {}
    for nm, m in [("early", pos <= 0.33), ("mid", (pos > 0.33) & (pos <= 0.66)), ("late", pos > 0.66)]:
        rr = paired_bootstrap_auc_delta(te["instance_id"].to_numpy()[m], yt[m], pt_base[m], p[m], n_boot=400)
        strat[nm] = rr
        print(f"   {nm:5s}: HGB={rr['auc_base']:.4f} -> survival={rr['auc_new']:.4f} Δ={rr['delta']:+.4f} "
              f"CI[{rr['delta_lo']:+.4f},{rr['delta_hi']:+.4f}] sig={rr['significant']}")

    out = RESULTS_OFFLINE / "full" / "cleaner_label.json"
    d = read_json(out) if out.exists() else {}
    d["survival_gru"] = {"strata_survival": _strata_auc(pos, yt, p),
                         "strata_baseline": _strata_auc(pos, yt, pt_base),
                         "overall_paired": r, "stratified_paired": strat}
    write_json(out, d)
    print(f"\n[surv] wrote {out}")


if __name__ == "__main__":
    main()
