"""Capture 2D PCA coordinates of the loop / vfail / prog battery at each model's sharpest layer,
for the Qwen2.5-Coder scale series, so we can draw an intuitive "three clusters, every size"
contact sheet. Saves a small committable JSON of coordinates (no raw activations).

  mech_interp/.venv/bin/python -u -m mech_interp.capture_scale_coords
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

OUT = Path(__file__).parent.parent / "assets" / "scale_scatter.json"

MODELS = [
    "mlx-community/Qwen2.5-Coder-0.5B-Instruct-4bit",
    "mlx-community/Qwen2.5-Coder-1.5B-Instruct-4bit",
    "mlx-community/Qwen2.5-Coder-3B-Instruct-4bit",
    "mlx-community/Qwen2.5-Coder-7B-Instruct-4bit",
    "mlx-community/Qwen2.5-Coder-14B-Instruct-4bit",
]
N_SCEN, K = 72, 4
CONDS = ["loop", "vfail", "prog"]


def standardize(X):
    X = X.astype(np.float64)
    X = (X - X.mean(0)) / (X.std(0) + 1e-6)
    return X - X.mean(0)


def pca2(X):
    Xc = standardize(X)
    _, _, Vt = np.linalg.svd(Xc, full_matrices=False)
    return Xc @ Vt[:2].T


def sep(Y, cidx):
    c = np.array([Y[cidx == j].mean(0) for j in range(3)])
    pd = np.mean([np.linalg.norm(c[a] - c[b]) for a, b in itertools.combinations(range(3), 2)])
    wi = np.mean([np.linalg.norm(Y[cidx == j] - c[j], axis=1).mean() for j in range(3)])
    return float(pd / (wi + 1e-9))


def capture(repo, scen):
    mw = MLXModel(repo)
    nL = mw.n_layers
    layers = list(range(nL))
    builders = {"loop": lambda s: s.loop_messages(K),
                "vfail": lambda s: s.varied_fail_messages(K),
                "prog": lambda s: s.progress_messages(K)}
    Xl = {L: [] for L in layers}
    cond = []
    for s in scen:
        for c in CONDS:
            ids = mw.render(builders[c](s))
            acts = mw.capture_resid(ids, layers)
            for L in layers:
                Xl[L].append(acts[L])
            cond.append(c)
    Xl = {L: np.asarray(v, dtype=np.float32) for L, v in Xl.items()}
    cidx = np.array([CONDS.index(c) for c in cond])
    del mw
    gc.collect()
    try:
        import mlx.core as mx
        (getattr(mx, "clear_cache", None) or getattr(mx.metal, "clear_cache", lambda: None))()
    except Exception:
        pass
    return Xl, cidx, nL


def main():
    t0 = time.time()
    scen = make_scenarios(N_SCEN)
    data = json.loads(OUT.read_text()) if OUT.exists() else {"conds": CONDS, "models": []}
    done = {m["model"] for m in data["models"]}
    for repo in MODELS:
        if repo in done:
            print(f"[skip] {repo}", flush=True)
            continue
        size = repo.split("Coder-")[1].split("-Instruct")[0]
        print(f"\n[load] {repo} ({size}) | {time.time()-t0:.0f}s", flush=True)
        try:
            Xl, cidx, nL = capture(repo, scen)
            # sharpest-separation layer in 2D
            seps = {L: sep(pca2(Xl[L]), cidx) for L in range(nL)}
            peakL = max(seps, key=lambda L: seps[L])
            Y = pca2(Xl[peakL])
            pts = [[round(float(Y[i, 0]), 3), round(float(Y[i, 1]), 3), int(cidx[i])]
                   for i in range(len(cidx))]
            rec = {"model": repo, "size": size, "n_layers": nL, "layer": peakL,
                   "depth_frac": round(peakL / max(nL - 1, 1), 2),
                   "sep": round(seps[peakL], 2), "points": pts}
            data["models"].append(rec)
            OUT.write_text(json.dumps(data, indent=2))
            print(f"[{size}] layers={nL} peakL={peakL} ({rec['depth_frac']} depth) sep={rec['sep']} "
                  f"| n={len(pts)} | {time.time()-t0:.0f}s", flush=True)
            del Xl
            gc.collect()
        except Exception as e:
            print(f"[skip] {repo}: {type(e).__name__}: {str(e)[:160]}", flush=True)
    print(f"\n[done] {len(data['models'])} models | wrote {OUT.name} | {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
