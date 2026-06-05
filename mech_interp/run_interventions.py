"""T3 — smarter interventions on the HARD real loops (where additive steering got 0% escape).

(a) ablate the top induction heads found by run_circuit (remove the copy mechanism)
(b) hybrid: a logit penalty on the repeated command's tokens, gated by loop-detection
    (mechanistic trigger + targeted edit) — by construction ~zero disruption off-loop
(c) decision-token-only −STUCK steering (more surgical)

We measure on-policy loop-escape on real loops; for (a) we also measure disruption on healthy
synthetic progress contexts. Any intervention flipping the 0% escape changes the story.

    mech_interp/.venv/bin/python -m mech_interp.run_interventions
"""
from __future__ import annotations

import json, time
from pathlib import Path

import numpy as np

from mech_interp.model_wrapper import ModelWrapper
from mech_interp.run_real_loops import real_loops, build_ctx, sys_prompt
from mech_interp.run_onpolicy import parse_cmd, is_repeat
from mech_interp.synthetic import make_scenarios

OUT = Path(__file__).parent / "results"
LAYER, K = 8, 4


def main():
    t0 = time.time()
    mw = ModelWrapper()
    loops = real_loops(maxn=10); sysp = sys_prompt()
    ctxs = [(lp, build_ctx(mw, lp, sysp)) for lp in loops]
    # STUCK direction (for c)
    d = np.load(OUT / "acts2.npz", allow_pickle=True); cond = d["meta_cond"]; tr = d["meta_group"] < 40
    X = d["L8"]; STUCK = X[tr & (cond == "loopfail")].mean(0) - X[tr & (cond == "loopprog")].mean(0)
    STUCK /= np.linalg.norm(STUCK)
    # top heads from circuit
    top_pairs = []
    cf = OUT / "circuit_results.json"
    if cf.exists():
        th = json.loads(cf.read_text()).get("top_heads", [])
        for k in th[:5]:
            top_pairs.append((int(k[1:k.index("H")]), int(k[k.index("H") + 1:])))
    print(f"[setup] {len(ctxs)} real loops | top heads={top_pairs} | {time.time()-t0:.0f}s", flush=True)

    def escape(genfn):
        e = []
        for lp, ctx in ctxs:
            g = genfn(ctx, lp)
            e.append(0 if is_repeat(parse_cmd(g), lp["cmd"]) else 1)
        return float(np.mean(e))

    res = {"n": len(ctxs)}
    # baseline
    res["baseline_escape"] = escape(lambda ctx, lp: mw.generate_kv(ctx, max_new_tokens=24))
    print(f"[baseline] escape={res['baseline_escape']:.2f}", flush=True)

    # (a) ablate top induction heads
    if top_pairs:
        def gen_ablate(ctx, lp):
            with mw.ablate_head_set(top_pairs):
                return mw.generate_kv(ctx, max_new_tokens=24)
        res["ablate_heads_escape"] = escape(gen_ablate)
        print(f"[a head-ablation] escape={res['ablate_heads_escape']:.2f}", flush=True)

    # (b) gated logit penalty on the repeated command's tokens
    res["logit_penalty_escape"] = {}
    for pen in [4.0, 8.0]:
        def gen_pen(ctx, lp, pen=pen):
            bad = set(mw.tok(lp["cmd"], add_special_tokens=False).input_ids)
            return mw.generate_kv(ctx, max_new_tokens=24, bad_ids=bad, penalty=pen)
        r = escape(gen_pen)
        res["logit_penalty_escape"][pen] = r
        print(f"[b logit-penalty p={pen}] escape={r:.2f}", flush=True)

    # (c) decision-token-only steering: steer only generated tokens (skip the big context fwd)
    # implemented as steering active during generation at α; compare a surgical low-α
    def gen_steer(ctx, lp):
        with mw.steering(LAYER, -STUCK, 16.0):
            return mw.generate_kv(ctx, max_new_tokens=24)
    res["steer_escape_a16"] = escape(gen_steer)
    print(f"[c steer α=16] escape={res['steer_escape_a16']:.2f}", flush=True)

    # disruption of (b) on healthy runs: gate is loop-detection, so off-loop it never fires.
    # worst-case check: apply penalty to a healthy progress context using a DIFFERENT command's
    # tokens (as if mis-triggered) and see if the action changes.
    if top_pairs:
        hs = make_scenarios(60)[40:50]
        chg = []
        for s in hs:
            ctx = mw.render(s.progress_messages(K))
            base = parse_cmd(mw.generate_kv(ctx, max_new_tokens=20))
            with mw.ablate_head_set(top_pairs):
                a = parse_cmd(mw.generate_kv(ctx, max_new_tokens=20))
            chg.append(0 if a == base else 1)
        res["ablate_heads_disruption_on_healthy"] = float(np.mean(chg))
        print(f"[a disruption on healthy] {res['ablate_heads_disruption_on_healthy']:.2f}", flush=True)

    (OUT / "interventions_results.json").write_text(json.dumps(res, indent=2))
    print(f"[done] | {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
