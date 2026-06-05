"""Scale test: does the loop / vfail / prog geometry hold for BIGGER Qwen2.5-Coder models?

Captures the SAME length-matched battery (loop = same command x K; vfail = K different commands,
all failing; prog = K productive commands) on each 4-bit MLX model, and reports per layer:
  - loop-vs-vfail diff-of-means grouped-CV AUC  (genuine identical-command repetition, length-matched)
  - length-only AUC  (trivial-cue baseline; should sit near 0.5 since lengths are matched)
  - |corr(projection, context length)| at the peak  (length-confound control)
  - 3-class PCA separation  (the README-figure metric)
Everything is 4-bit MLX so quantisation is constant and only model size varies.

  mech_interp/.venv/bin/python -u -m mech_interp.run_localize_scale
"""
from __future__ import annotations

import gc
import itertools
import json
import time
from pathlib import Path

import numpy as np

from mech_interp.mlx_wrapper import MLXModel
from mech_interp.synthetic import make_scenarios

OUT = Path(__file__).parent / "results"
OUT.mkdir(exist_ok=True)
RESULT = OUT / "scale_localize.json"

MODELS = [
    "mlx-community/Qwen2.5-Coder-0.5B-Instruct-4bit",
    "mlx-community/Qwen2.5-Coder-1.5B-Instruct-4bit",
    "mlx-community/Qwen2.5-Coder-3B-Instruct-4bit",
    "mlx-community/Qwen2.5-Coder-7B-Instruct-4bit",
    "mlx-community/Qwen2.5-Coder-14B-Instruct-4bit",
    "mlx-community/Qwen2.5-Coder-32B-Instruct-4bit",   # stretch; may OOM on 32 GB
]
N_SCEN, K = 72, 4
CONDS = ["loop", "vfail", "prog"]


def auc(y, s):
    y = np.asarray(y); s = np.asarray(s, dtype=np.float64)
    pos, neg = s[y == 1], s[y == 0]
    if len(pos) == 0 or len(neg) == 0:
        return 0.5
    gt = (pos[:, None] > neg[None, :]).sum()
    eq = (pos[:, None] == neg[None, :]).sum()
    return float((gt + 0.5 * eq) / (len(pos) * len(neg)))


def grouped_cv_diffmeans(X, y, groups, k=5):
    """Diff-of-means direction trained on train groups, scored on held-out groups (scenario-grouped)."""
    y = np.asarray(y); groups = np.asarray(groups)
    ug = np.random.default_rng(0).permutation(np.unique(groups))
    proj = np.full(len(y), np.nan); aucs = []
    for f in np.array_split(ug, k):
        te = np.isin(groups, f); tr = ~te
        if len(np.unique(y[tr])) < 2:
            continue
        v = X[tr][y[tr] == 1].mean(0) - X[tr][y[tr] == 0].mean(0)
        s = X[te] @ v; proj[te] = s
        if len(np.unique(y[te])) == 2:
            aucs.append(auc(y[te], s))
    return (float(np.mean(aucs)) if aucs else 0.5), proj


def pca_sep(X, cidx):
    """3-class separation in the top-2 PCs (mean pairwise centroid dist / mean within-class spread)."""
    X = X.astype(np.float64); X = (X - X.mean(0)) / (X.std(0) + 1e-6); Xc = X - X.mean(0)
    _, _, Vt = np.linalg.svd(Xc, full_matrices=False); Y = Xc @ Vt[:2].T
    c = np.array([Y[cidx == j].mean(0) for j in range(3)])
    pd = np.mean([np.linalg.norm(c[a] - c[b]) for a, b in itertools.combinations(range(3), 2)])
    wi = np.mean([np.linalg.norm(Y[cidx == j] - c[j], axis=1).mean() for j in range(3)])
    return float(pd / (wi + 1e-9))


def capture(repo, scen):
    mw = MLXModel(repo)
    nL = mw.n_layers; layers = list(range(nL))
    builders = {"loop": lambda s: s.loop_messages(K),
                "vfail": lambda s: s.varied_fail_messages(K),
                "prog": lambda s: s.progress_messages(K)}
    Xl = {L: [] for L in layers}; cond = []; group = []; length = []
    t0 = time.time()
    for si, s in enumerate(scen):
        for c in CONDS:
            ids = mw.render(builders[c](s))
            acts = mw.capture_resid(ids, layers)
            for L in layers:
                Xl[L].append(acts[L])
            cond.append(c); group.append(si); length.append(len(ids))
        if (si + 1) % 24 == 0:
            print(f"    captured {si+1}/{len(scen)} | {time.time()-t0:.0f}s", flush=True)
    Xl = {L: np.asarray(v, dtype=np.float32) for L, v in Xl.items()}
    meta = {"cond": np.array(cond), "group": np.array(group), "len": np.array(length)}
    d = Xl[0].shape[1]
    del mw; gc.collect()
    try:
        import mlx.core as mx
        (getattr(mx, "clear_cache", None) or getattr(mx.metal, "clear_cache", lambda: None))()
    except Exception:
        pass
    return Xl, meta, nL, d


def analyze(Xl, meta, nL):
    cidx = np.array([CONDS.index(c) for c in meta["cond"]])
    m = np.isin(meta["cond"], ["loop", "vfail"])
    y = (meta["cond"][m] == "loop").astype(int)
    g, ln = meta["group"][m], meta["len"][m]
    len_only = max(auc(y, ln), 1 - auc(y, ln))
    lv_auc, lv_corr, sep = {}, {}, {}
    for L in range(nL):
        a, proj = grouped_cv_diffmeans(Xl[L][m], y, g)
        lv_auc[L] = a
        lv_corr[L] = float(abs(np.corrcoef(proj, ln)[0, 1])) if np.std(proj) > 0 else 0.0
        sep[L] = pca_sep(Xl[L], cidx)
    peakL = max(lv_auc, key=lambda L: lv_auc[L])
    clean = {L: lv_auc[L] for L in range(nL) if lv_corr[L] < 0.5}
    cleanL = max(clean, key=lambda L: clean[L]) if clean else peakL
    sepL = max(sep, key=lambda L: sep[L])
    den = max(nL - 1, 1)
    return {
        "n_layers": nL,
        "length_only_auc": round(len_only, 3),
        "loopvfail_peak_auc": round(lv_auc[peakL], 3),
        "peak_layer": peakL, "peak_depth_frac": round(peakL / den, 2),
        "len_corr_at_peak": round(lv_corr[peakL], 2),
        "clean_peak_auc": round(lv_auc[cleanL], 3), "clean_peak_layer": cleanL,
        "clean_peak_depth_frac": round(cleanL / den, 2),
        "pca_sep_peak": round(sep[sepL], 2), "pca_sep_peak_depth_frac": round(sepL / den, 2),
        "auc_profile": [round(lv_auc[L], 3) for L in range(nL)],
        "sep_profile": [round(sep[L], 2) for L in range(nL)],
    }


def main():
    t0 = time.time()
    scen = make_scenarios(N_SCEN)
    results = json.loads(RESULT.read_text()) if RESULT.exists() else []
    done = {r["model"] for r in results}
    for repo in MODELS:
        if repo in done:
            print(f"[skip] {repo} (already done)", flush=True)
            continue
        size = repo.split("Coder-")[1].split("-Instruct")[0]
        print(f"\n[load] {repo}  ({size}) | {time.time()-t0:.0f}s", flush=True)
        try:
            Xl, meta, nL, d = capture(repo, scen)
            rec = {"model": repo, "size": size, "d_model": d, **analyze(Xl, meta, nL)}
            results.append(rec)
            RESULT.write_text(json.dumps(results, indent=2))
            print(f"[{size}] layers={nL} d={d} | loop-vs-vfail peak AUC={rec['loopvfail_peak_auc']} "
                  f"@L{rec['peak_layer']} ({rec['peak_depth_frac']} depth, len-corr {rec['len_corr_at_peak']}) | "
                  f"clean-peak AUC={rec['clean_peak_auc']} @{rec['clean_peak_depth_frac']} | "
                  f"len-only AUC={rec['length_only_auc']} | PCA-sep peak={rec['pca_sep_peak']} "
                  f"@{rec['pca_sep_peak_depth_frac']} | {time.time()-t0:.0f}s", flush=True)
            del Xl, meta; gc.collect()
        except Exception as e:
            print(f"[skip] {repo}: {type(e).__name__}: {str(e)[:160]}", flush=True)
    print(f"\n[done] {len(results)} models | wrote {RESULT.name} | {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
