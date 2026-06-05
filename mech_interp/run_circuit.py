"""T2 — find the loop CIRCUIT (causal heads), not just a direction.

On synthetic loop contexts (known repeated command), we causally localize the attention heads
that drive the re-run decision:
  Stage 1  ablate each layer's whole attention block -> Δ repeat-pref  (which layers matter)
  Stage 2  ablate each head in the top layers       -> Δ repeat-pref  (which heads matter)
  Stage 3  for the top heads, measure attention from the decision token back to the prior
           occurrences of the repeated command (induction signature).

Δ repeat-pref < 0  ==  ablating that component REDUCES the urge to repeat (it drives the loop).

    mech_interp/.venv/bin/python -m mech_interp.run_circuit
"""
from __future__ import annotations

import json, time
from pathlib import Path

import numpy as np
import torch

from mech_interp.model_wrapper import ModelWrapper
from mech_interp.run_real_loops import real_loops, build_ctx, sys_prompt, NOVEL_ALTS

OUT = Path(__file__).parent / "results"
N_CTX = 5


def find_occurrences(ctx_ids, cmd_ids):
    """token positions covered by any occurrence of cmd_ids inside ctx_ids."""
    pos = set()
    n, m = len(ctx_ids), len(cmd_ids)
    for i in range(n - m + 1):
        if ctx_ids[i:i + m] == cmd_ids:
            pos.update(range(i, i + m))
    return sorted(pos)


def main():
    t0 = time.time()
    mw = ModelWrapper()
    nL, nH = mw.n_layers, mw.model.config.num_attention_heads
    loops = real_loops(maxn=N_CTX); sysp = sys_prompt()
    print(f"[setup] {nL} layers x {nH} heads | {len(loops)} REAL loop contexts | {time.time()-t0:.0f}s", flush=True)

    # precompute contexts, repeat/novel continuations, baseline repeat-pref (real loops)
    ctxs = []
    for lp in loops:
        ctx = build_ctx(mw, lp, sysp)
        rc = f" `{lp['cmd']}`"
        alts = [f" `{a}`" for a in NOVEL_ALTS]
        alp = [mw.continuation_logprob(ctx, a) for a in alts]
        nc = alts[int(np.argmax(alp))]
        base = mw.continuation_logprob(ctx, rc) - max(alp)
        ctxs.append({"ctx": ctx, "rc": rc, "nc": nc, "base": base, "cmd": lp["cmd"]})
    base_mean = float(np.mean([c["base"] for c in ctxs]))
    print(f"[base] repeat-pref={base_mean:+.3f}", flush=True)

    def delta_pref(layer, heads):
        ds = []
        for c in ctxs:
            with mw.ablate_heads(layer, heads):
                p = mw.continuation_logprob(c["ctx"], c["rc"]) - mw.continuation_logprob(c["ctx"], c["nc"])
            ds.append(p - c["base"])
        return float(np.mean(ds))

    # ---- Stage 1: whole-attention ablation per layer ----
    layer_eff = {}
    for L in range(nL):
        layer_eff[L] = delta_pref(L, list(range(nH)))
    top_layers = sorted(layer_eff, key=lambda L: layer_eff[L])[:3]   # most negative = drives repeat
    print("[stage1] layers most reducing repeat when attn ablated:",
          [(L, round(layer_eff[L], 3)) for L in top_layers], flush=True)

    # ---- Stage 2: per-head ablation in the top layers ----
    head_eff = {}
    for L in top_layers:
        for h in range(nH):
            head_eff[f"L{L}H{h}"] = delta_pref(L, [h])
    top_heads = sorted(head_eff, key=lambda k: head_eff[k])[:6]
    print("[stage2] heads most reducing repeat when ablated:",
          [(k, round(head_eff[k], 3)) for k in top_heads], flush=True)

    # ---- Stage 3: induction signature of the top heads ----
    induction = {}
    c = ctxs[0]
    ids = mw.tok(c["ctx"], return_tensors="pt").to(mw.device)
    cmd_ids = mw.tok(f"`{c['cmd']}`", add_special_tokens=False).input_ids
    occ = find_occurrences(ids.input_ids[0].tolist(), cmd_ids)
    with torch.no_grad():
        att = mw.model(**ids, output_attentions=True).attentions   # tuple[L] (1,H,S,S)
    for k in top_heads:
        L = int(k[1:k.index("H")]); h = int(k[k.index("H") + 1:])
        row = att[L][0, h, -1, :].float().cpu().numpy()             # last-token attention
        induction[k] = {"attn_to_prior_command": float(row[occ].sum()) if occ else 0.0,
                        "head_ablation_delta_pref": head_eff[k]}
    print(f"[stage3] prior-command tokens located: {len(occ)} | induction scores:", flush=True)
    for k in top_heads:
        print(f"   {k}: attn→prior-cmd={induction[k]['attn_to_prior_command']:.2f} "
              f"Δpref(ablate)={head_eff[k]:+.3f}", flush=True)

    res = {"baseline_pref": base_mean, "n_heads": nH,
           "layer_effect": {str(L): layer_eff[L] for L in layer_eff},
           "top_layers": top_layers, "head_effect": head_eff, "top_heads": top_heads,
           "induction": induction}
    (OUT / "circuit_results.json").write_text(json.dumps(res, indent=2))
    print(f"[done] wrote circuit_results.json | {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
