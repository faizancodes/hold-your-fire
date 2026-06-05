#!/usr/bin/env python3
"""AUC-lift experiment #2: a small GRU over per-step features (CPU).

Processes each trajectory step-by-step with a forward (causal) GRU and emits a
risk at every position; trained on the same terminal labels. Predictions are read
at the exact v1 test prefix positions and paired-bootstrapped against the same
v1-HGB baseline. Forward-only GRU + per-step (prefix-visible) features ⇒ no future
leak.

  python scripts/run_seq_model.py --epochs 8 --hidden 64
"""

from __future__ import annotations

import argparse

import _bootstrap  # noqa: F401
import numpy as np
import pandas as pd

from localguard.evaluate import paired_bootstrap_auc_delta, roc_auc_metric
from localguard.schemas import NormalizedTrajectory
from localguard.utils import (
    INTERIM_DIR,
    REPO_ROOT,
    RESULTS_OFFLINE,
    DEFAULT_SEED,
    read_json,
    read_jsonl,
    write_json,
)

ACTION_TYPES = ["read", "search", "edit", "test", "git", "install", "submit", "environment", "other"]
_AT_IDX = {t: i for i, t in enumerate(ACTION_TYPES)}
CAP = 160  # max steps processed per trajectory (covers ~p99; longer prefixes clamp)


def step_features(traj: NormalizedTrajectory) -> np.ndarray:
    seen: set[str] = set()
    rows = []
    for s in traj.steps[:CAP]:
        v = [0.0] * len(ACTION_TYPES)
        v[_AT_IDX.get(s.action_type, _AT_IDX["other"])] = 1.0
        new_file = 0.0
        if s.action_type == "read":
            for p in s.file_paths:
                if p not in seen:
                    new_file = 1.0
            seen.update(s.file_paths)
        v += [
            1.0 if s.contains_traceback else 0.0,
            1.0 if s.contains_exception else 0.0,
            np.log1p(len(s.observation_text or "")),
            float(s.test_fail_count) if s.test_fail_count is not None else 0.0,
            1.0 if s.test_fail_count is not None else 0.0,
            np.log1p(len(s.file_paths)),
            new_file,
            np.log1p(len(s.thought_text or "")),
            np.log1p(len(s.action_text or "")),
            1.0 if (s.returncode not in (None, 0)) else 0.0,
        ]
        rows.append(v)
    if not rows:
        rows = [[0.0] * (len(ACTION_TYPES) + 10)]
    return np.asarray(rows, dtype=np.float32)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--normalized", default="data/interim/normalized_prefix_offline_full.jsonl")
    ap.add_argument("--v1", default="data/processed/prefix_offline_full.parquet")
    ap.add_argument("--split", default="results/offline/full/split_assignment.parquet")
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--hidden", type=int, default=64)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--n-boot", type=int, default=1000)
    args = ap.parse_args()

    import torch
    import torch.nn as nn

    torch.manual_seed(DEFAULT_SEED)
    np.random.seed(DEFAULT_SEED)

    folds = pd.read_parquet(REPO_ROOT / args.split)
    v1 = pd.read_parquet(REPO_ROOT / args.v1).merge(folds, on="prefix_id")
    inst_fold = dict(zip(v1["instance_id"], v1["fold"]))

    print("[seq] loading normalized trajectories ...")
    trajs = [NormalizedTrajectory(**r) for r in read_jsonl(REPO_ROOT / args.normalized)]
    feats = {t.trajectory_id: step_features(t) for t in trajs}
    label = {t.trajectory_id: int(not t.target) for t in trajs}
    fold_of = {t.trajectory_id: inst_fold.get(t.instance_id, "train") for t in trajs}
    F = next(iter(feats.values())).shape[1]
    print(f"[seq] {len(trajs)} trajectories, {F} per-step features, cap={CAP}")

    # standardize features on TRAIN positions
    train_ids = [tid for tid in feats if fold_of[tid] == "train"]
    stacked = np.concatenate([feats[t] for t in train_ids], axis=0)
    mu = stacked.mean(0); sd = stacked.std(0); sd[sd < 1e-6] = 1.0
    for tid in feats:
        feats[tid] = (feats[tid] - mu) / sd

    class GRUNet(nn.Module):
        def __init__(self, f, h):
            super().__init__()
            self.gru = nn.GRU(f, h, batch_first=True)
            self.head = nn.Linear(h, 1)

        def forward(self, x):
            out, _ = self.gru(x)
            return self.head(out).squeeze(-1)  # (B, T) logits

    model = GRUNet(F, args.hidden)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
    lossfn = nn.BCEWithLogitsLoss(reduction="none")

    def make_batch(ids):
        mats = [feats[t] for t in ids]
        lens = [m.shape[0] for m in mats]
        T = max(lens)
        x = np.zeros((len(ids), T, F), dtype=np.float32)
        m = np.zeros((len(ids), T), dtype=np.float32)
        y = np.zeros((len(ids),), dtype=np.float32)
        for i, mat in enumerate(mats):
            x[i, : lens[i]] = mat
            m[i, : lens[i]] = 1.0
            y[i] = label[ids[i]]
        return torch.from_numpy(x), torch.from_numpy(m), torch.from_numpy(y)

    rng = np.random.default_rng(DEFAULT_SEED)
    for ep in range(args.epochs):
        rng.shuffle(train_ids)
        model.train()
        tot = 0.0
        for i in range(0, len(train_ids), args.batch):
            ids = train_ids[i : i + args.batch]
            x, m, y = make_batch(ids)
            logit = model(x)                      # (B,T)
            yb = y.unsqueeze(1).expand_as(logit)  # terminal label at every position
            loss = (lossfn(logit, yb) * m).sum() / m.sum()
            opt.zero_grad(); loss.backward(); opt.step()
            tot += float(loss) * len(ids)
        # quick val AUC at schedule positions
        vauc = _eval_split(model, feats, label, fold_of, v1, "val")
        print(f"[seq] epoch {ep+1}/{args.epochs} loss={tot/len(train_ids):.4f} val_auc={vauc:.4f}")

    # ---- test: read GRU risk at the exact v1 test prefix positions -------
    base = pd.read_parquet(RESULTS_OFFLINE / "full" / "auc_lift_baseline.parquet").sort_values("prefix_id").reset_index(drop=True)
    te = v1[v1["fold"] == "test"].sort_values("prefix_id").reset_index(drop=True)
    assert (te["prefix_id"].values == base["prefix_id"].values).all()
    p_seq = _predict_prefixes(model, feats, te)
    yt = te["y_fail"].to_numpy(int)
    r = paired_bootstrap_auc_delta(te["instance_id"].to_numpy(), yt, base["p_base"].to_numpy(), p_seq, n_boot=args.n_boot)
    print(f"[seq] GRU TEST {r}")

    out = RESULTS_OFFLINE / "full" / "auc_lift_results.json"
    existing = read_json(out) if out.exists() else {}
    existing["seq_model_gru"] = {"hidden": args.hidden, "epochs": args.epochs, "n_features": F, **r}
    write_json(out, existing)
    print(f"[seq] wrote {out}")


def _predict_prefixes(model, feats, prefix_df):
    import torch

    model.eval()
    # cache per-trajectory position logits
    cache: dict[str, np.ndarray] = {}
    out = np.zeros(len(prefix_df), dtype=float)
    with torch.no_grad():
        for j, (tid, k) in enumerate(zip(prefix_df["trajectory_id"], prefix_df["prefix_step"])):
            if tid not in cache:
                if tid in feats:
                    x = torch.from_numpy(feats[tid]).unsqueeze(0)
                    cache[tid] = torch.sigmoid(model(x)).squeeze(0).numpy()
                else:
                    cache[tid] = np.array([0.5])
            arr = cache[tid]
            out[j] = arr[min(int(k) - 1, len(arr) - 1)]
    return out


def _eval_split(model, feats, label, fold_of, v1, split):
    sub = v1[v1["fold"] == split]
    # sample up to 4000 prefixes for a quick val AUC
    sub = sub.sample(min(4000, len(sub)), random_state=0)
    p = _predict_prefixes(model, feats, sub)
    y = sub["y_fail"].to_numpy(int)
    return roc_auc_metric(y, p) if len(np.unique(y)) > 1 else float("nan")


if __name__ == "__main__":
    main()
