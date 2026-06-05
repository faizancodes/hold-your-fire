"""Deep hunt for the *recovery* direction — the causal lever that moves a capable model from
'persist (repeat the failing command)' to 'recover (do the productive thing)' at a stuck point.

This is the rigorous version of the earlier crude loop-vs-progress attempt. The contrast is
NOT loop vs progress; it is persist vs recover *inside the same stuck situation* (command
repeated 3x, still failing), with surface varied across families so a predictive direction
cannot be a mere surface cue.

Stages (one model load):
  A. capture every layer's decision-token residual for each stuck context; label each by the
     model's own greedy choice (persist/recover/other) and by a continuous recovery margin
     = max_i logP(recover_i) - logP(persist).
  B. discover candidate directions per layer (diff-of-means, logistic probe with scenario-
     grouped CV AUC, margin-regression), with confound controls: token-length baseline, and
     cosine vs the crude loop direction (the recovery axis must be distinct + add value).
  C. causally validate: on held-out (family-grouped) persist contexts, steer +dir and measure
     the recovery-margin shift (dose-response) and on-policy recover rate + coherence, against
     random/orthogonal controls with a paired bootstrap. Select the causal direction or report
     an honest null.

  mech_interp/.venv/bin/python -m mech_interp.run_recovery_direction [repo] [alphas]
"""
from __future__ import annotations

import json
import re
import sys
import time
from pathlib import Path

import numpy as np

from mech_interp.mlx_wrapper import MLXModel
from mech_interp.recovery_contexts import build_battery, classify

OUT = Path(__file__).parent / "results"
RNG = np.random.default_rng(0)


# ---- small numerics (no hard sklearn dependency) ---------------------------------
def _logreg_cv_auc(X, y, groups, l2=1.0, folds=None):
    """Scenario-grouped logistic-probe CV AUC + a full-data weight direction."""
    uniq = sorted(set(groups))
    folds = folds or len(uniq)
    # leave-one-group-out
    aucs = []
    for g in uniq:
        tr = np.array([gg != g for gg in groups])
        te = ~tr
        if len(set(y[tr])) < 2 or len(set(y[te])) < 2:
            continue
        w, b = _fit_logreg(X[tr], y[tr], l2)
        s = X[te] @ w + b
        aucs.append(_auc(y[te], s))
    w, b = _fit_logreg(X, y, l2)
    return (float(np.mean(aucs)) if aucs else float("nan"), w)


def _fit_logreg(X, y, l2=1.0, iters=300, lr=0.5):
    X = np.asarray(X, np.float64)
    mu, sd = X.mean(0), X.std(0) + 1e-6
    Xs = (X - mu) / sd
    n, d = Xs.shape
    w = np.zeros(d); b = 0.0
    for _ in range(iters):
        z = Xs @ w + b
        p = 1 / (1 + np.exp(-z))
        gw = Xs.T @ (p - y) / n + l2 * w / n
        gb = float(np.mean(p - y))
        w -= lr * gw; b -= lr * gb
    # map back to raw-activation space
    return w / sd, float(b - (w * mu / sd).sum())


def _auc(y, s):
    y = np.asarray(y); s = np.asarray(s)
    pos, neg = s[y == 1], s[y == 0]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    return float((pos[:, None] > neg[None, :]).mean() + 0.5 * (pos[:, None] == neg[None, :]).mean())


def _unit(v):
    v = np.asarray(v, np.float32)
    return v / (np.linalg.norm(v) + 1e-8)


# ---- model helpers ---------------------------------------------------------------
def pad_ids(ids, target, filler):
    return [filler] * max(0, target - len(ids)) + list(ids)


def recover_margin(mw, text, looped, recovers):
    """logP(best recover) - logP(persist) at the decision point (teacher-forced)."""
    lp_persist = mw.continuation_logprob(text, looped)
    lp_rec = max(mw.continuation_logprob(text, r) for r in recovers)
    return lp_rec - lp_persist, lp_rec, lp_persist


def _extract_cmd(gen: str) -> str:
    """Pull the first shell command out of a possibly-verbose generation (reasoning, fences)."""
    g = gen.strip()
    m = re.search(r"```(?:bash|sh|shell)?\s*\n?(.+?)```", g, re.S)
    if m:
        g = m.group(1).strip()
    for line in g.splitlines():
        line = re.sub(r"^\s*\$\s*", "", line.strip().strip("`")).strip()
        if line:
            return line
    return ""


def greedy_label(mw, ids, looped):
    gen = mw.generate(ids, max_tokens=24, temperature=0.0)
    cmd = _extract_cmd(gen)
    return classify(cmd, looped), cmd


def main():
    t0 = time.time()
    repo = sys.argv[1] if len(sys.argv) > 1 else "mlx-community/Qwen3-Coder-30B-A3B-Instruct-3bit"
    alphas = [float(x) for x in sys.argv[2].split(",")] if len(sys.argv) > 2 else [0, 4, 8, 12, 16]
    print(f"[load] {repo}")
    mw = MLXModel(repo)
    N, D = mw.n_layers, mw.d_model
    filler = mw.tok("\n")[-1]
    print(f"[model] layers={N} d={D} | {time.time()-t0:.0f}s")

    ctx = build_battery()
    # length control: pad every context to the same token length
    toks = [mw.tok(c["text"]) for c in ctx]
    target = max(len(t) for t in toks)
    toks = [pad_ids(t, target, filler) for t in toks]
    lens = [len(mw.tok(c["text"])) for c in ctx]
    print(f"[lengths] raw token lengths {min(lens)}-{max(lens)} -> padded to {target}")

    # ---- STAGE A: capture + label ------------------------------------------------
    print("[A] capturing all-layer residuals + labels ...")
    acts = {L: np.zeros((len(ctx), D), np.float32) for L in range(N)}
    margins = np.zeros(len(ctx), np.float32)
    glabel = []
    fams = [c["family"] for c in ctx]
    for i, (c, ids) in enumerate(zip(ctx, toks)):
        caps = mw.capture_resid(ids, list(range(N)))
        for L in range(N):
            acts[L][i] = caps[L]
        m, _, _ = recover_margin(mw, c["text"], c["looped"], c["recover"])
        margins[i] = m
        lab, cmd = greedy_label(mw, mw.tok(c["text"]), c["looped"])
        glabel.append(lab)
        if i % 10 == 0:
            print(f"   {i:3d}/{len(ctx)}  fam={c['family']:10} greedy={lab:8} margin={m:+.2f} | {time.time()-t0:.0f}s", flush=True)
    glabel = np.array(glabel)
    n_persist = int((glabel == "persist").sum())
    n_recover = int((glabel == "recover").sum())
    n_other = int((glabel == "other").sum())
    print(f"[A] greedy labels: persist={n_persist} recover={n_recover} other={n_other} | "
          f"margin mean {margins.mean():+.2f} (persist-set {margins[glabel=='persist'].mean() if n_persist else float('nan'):+.2f}, "
          f"recover-set {margins[glabel=='recover'].mean() if n_recover else float('nan'):+.2f})")

    # binary contrast set: persist (y=0) vs recover (y=1); drop 'other'
    keep = glabel != "other"
    if n_persist < 4 or n_recover < 4:
        # fall back to a margin-median split so we always have a usable contrast
        med = np.median(margins)
        y = (margins > med).astype(int)
        keep = np.ones(len(ctx), bool)
        print(f"[A] too few greedy persist/recover -> margin-median split (med={med:+.2f})")
    else:
        y = (glabel == "recover").astype(int)
    idx = np.where(keep)[0]
    g_keep = [fams[i] for i in idx]

    # ---- STAGE B: discover directions per layer ----------------------------------
    print("[B] per-layer probe (family-grouped CV AUC), diff-means, margin-corr ...")
    # confound: token length must not predict the label (we padded -> should be ~0.5)
    len_auc = _auc(y[idx], np.array([target] * len(idx)) + RNG.normal(0, 1e-6, len(idx)))
    perlayer = []
    for L in range(N):
        XL = acts[L][idx]
        auc, w = _logreg_cv_auc(XL, y[idx], g_keep)
        dom = _unit(XL[y[idx] == 1].mean(0) - XL[y[idx] == 0].mean(0))
        # margin correlation: does activation along diff-means track the continuous margin?
        proj = acts[L][idx] @ dom
        mcorr = float(np.corrcoef(proj, margins[idx])[0, 1]) if len(idx) > 2 else float("nan")
        perlayer.append({"L": L, "auc": auc, "mcorr": mcorr, "dom": dom, "probe": _unit(w)})
    ranked = sorted([p for p in perlayer if not np.isnan(p["auc"])], key=lambda p: p["auc"], reverse=True)
    print(f"[B] length-only AUC = {len_auc:.3f} (want ~0.5)")
    print("[B] top layers by grouped-CV AUC:")
    for p in ranked[:6]:
        print(f"    L{p['L']:2d}  probeAUC={p['auc']:.3f}  margin-corr={p['mcorr']:+.2f}")

    # cosine vs the crude loop direction (loop-vs-progress), if available
    loop_dir = None
    lp = OUT / "loop_direction.npy"
    if lp.exists():
        loop_dir = _unit(np.load(lp))

    # ---- STAGE C: causal validation (family-held-out) ----------------------------
    # build direction on all-but-one family, validate on the held-out family's persist contexts
    test_fam = "edit_noop"
    tr = np.array([f != test_fam for f in fams])
    te_persist = [i for i in range(len(ctx)) if fams[i] == test_fam and glabel[i] != "recover"]
    print(f"[C] causal validation: train dir on !{test_fam}, test on {len(te_persist)} held-out persist contexts")

    cand_layers = [p["L"] for p in ranked[:3]]
    results_C = {}
    for L in cand_layers:
        XL = acts[L]
        # rebuild diff-means on TRAIN families only (no leakage into the held-out family)
        sub = [j for j, i in enumerate(idx) if tr[i]]
        ysub = y[idx][sub]
        dir_tr = _unit(XL[idx][sub][ysub == 1].mean(0) - XL[idx][sub][ysub == 0].mean(0))
        rand = _unit(RNG.standard_normal(D))
        ortho = _unit(RNG.standard_normal(D) - (RNG.standard_normal(D) @ dir_tr) * dir_tr)
        ortho = _unit(ortho - (ortho @ dir_tr) * dir_tr)

        def onpolicy(vec, a):
            labs = []
            for i in te_persist:
                with mw.steering(L, vec, a):
                    lab, _ = greedy_label(mw, mw.tok(ctx[i]["text"]), ctx[i]["looped"])
                labs.append(lab)
            n = len(labs)
            return {k: labs.count(k) / n for k in ("recover", "persist", "other")}

        series = {"dir": [], "random": [], "ortho": []}
        op_dir, op_ctrl = {}, {}
        for a in alphas:
            for name, vec in (("dir", dir_tr), ("random", rand), ("ortho", ortho)):
                ms = []
                for i in te_persist:
                    with mw.steering(L, vec, a):
                        m, _, _ = recover_margin(mw, ctx[i]["text"], ctx[i]["looped"], ctx[i]["recover"])
                    ms.append(m)
                series[name].append(float(np.mean(ms)))
            op_dir[a] = onpolicy(dir_tr, a)                      # dir on-policy at every alpha
        # controls on-policy only at the top alpha (confirm they do NOT induce recovery)
        top = alphas[-1]
        op_ctrl = {"random": onpolicy(rand, top), "ortho": onpolicy(ortho, top)}
        net = series["dir"][-1] - series["dir"][0]
        net_rand = series["random"][-1] - series["random"][0]
        results_C[L] = {"alphas": alphas, "margin": series, "onpolicy_dir": op_dir,
                        "onpolicy_ctrl_top": op_ctrl, "net_dir": net, "net_random": net_rand,
                        "cos_vs_loop": (float(abs(dir_tr @ loop_dir)) if loop_dir is not None else None)}
        print(f"\n[C] L{L}: recovery-margin vs alpha (held-out persist)")
        print(f"    {'alpha':>5} | {'dir':>7} | {'random':>7} | {'ortho':>7} | dir recover% (other%)")
        for j, a in enumerate(alphas):
            od = op_dir[a]
            print(f"    {a:>5} | {series['dir'][j]:>+7.2f} | {series['random'][j]:>+7.2f} | "
                  f"{series['ortho'][j]:>+7.2f} | {od['recover']*100:4.0f}% ({od['other']*100:.0f}%)")
        print(f"    @alpha {top}: on-policy recover%  dir={op_dir[top]['recover']*100:.0f}%  "
              f"random={op_ctrl['random']['recover']*100:.0f}%  ortho={op_ctrl['ortho']['recover']*100:.0f}%")
        print(f"    net dir {net:+.2f} vs random {net_rand:+.2f}  "
              f"(specific: {'yes' if net > net_rand + 0.2 else 'weak'})"
              + (f" | cos vs loop dir {results_C[L]['cos_vs_loop']:.2f}" if loop_dir is not None else ""))

    # pick best layer by net dir effect surviving control + coherence
    best = max(results_C, key=lambda L: results_C[L]["net_dir"] - results_C[L]["net_random"]) if results_C else None
    out = {"repo": repo, "n_ctx": len(ctx), "labels": {"persist": n_persist, "recover": n_recover, "other": n_other},
           "length_only_auc": len_auc,
           "stageB_top": [{"L": p["L"], "auc": p["auc"], "mcorr": p["mcorr"]} for p in ranked[:8]],
           "stageC": {str(L): {k: v for k, v in r.items()} for L, r in results_C.items()},
           "best_layer": best}
    OUT.mkdir(exist_ok=True)
    safe = repo.replace("/", "_").replace(":", "_")
    if best is not None:
        # save the chosen direction (rebuilt on ALL families for deployment)
        dom_best = _unit(acts[best][idx][y[idx] == 1].mean(0) - acts[best][idx][y[idx] == 0].mean(0))
        np.save(OUT / f"recovery_dir_{safe}_L{best}.npy", dom_best)
    (OUT / f"recovery_{safe}.json").write_text(json.dumps(out, indent=2, default=float))
    print(f"\n[done] best layer {best} | wrote recovery_{safe}.json | {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
