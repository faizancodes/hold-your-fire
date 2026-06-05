"""T1 — test on REAL loop trajectories (LocalGuard's audited tight loops), not synthetic.

Real loops come from results/audits/audit_offline_full_true_positive.jsonl (rows whose recent
actions are a genuine repeated command, e.g. `find_file "SecretStr"` x3 -> "No matches found").
We rebuild a chat context ending at the loop decision point, feed it to Qwen, and test whether
(a) Qwen also wants to repeat (is the real attractor weaker than synthetic?), (b) the SYNTHETIC
−STUCK@L8 direction reduces that, (c) it breaks the loop on-policy. If real loops are weaker and
steering works here, the synthetic 0%-escape was an artifact.

    mech_interp/.venv/bin/python -m mech_interp.run_real_loops [alpha]
"""
from __future__ import annotations

import json, re, sys, time
from pathlib import Path

import numpy as np

from mech_interp.model_wrapper import ModelWrapper
from mech_interp.run_onpolicy import parse_cmd, is_repeat

OUT = Path(__file__).parent / "results"
ROOT = Path(__file__).parent.parent
LAYER, K = 8, 4
NOVEL_ALTS = ["open setup.py", "ls -la", "search_dir 'def '", "cat README.md"]


def strip_type(a: str) -> str:
    return re.sub(r"^\(\w+\)\s*", "", str(a)).strip()


def real_loops(maxn=24):
    rows = [json.loads(l) for l in open(ROOT / "results/audits/audit_offline_full_true_positive.jsonl")]
    out = []
    for r in rows:
        la = [strip_type(a) for a in (r.get("last_actions") or [])]
        lo = [str(o) for o in (r.get("last_observations") or [])]
        if len(la) >= 2 and la[-1] == la[-2] and len(la[-1]) > 3:
            out.append({"inst": r["instance_id"], "cmd": la[-1], "actions": la, "obs": lo})
    return out[:maxn]


def sys_prompt():
    line = open(ROOT / "data/raw/nebius_sample.jsonl").readline()
    try:
        traj = json.loads(json.loads(line)["trajectory"])
        sp = traj[0].get("system_prompt") or traj[0].get("content") or ""
        return sp[:500]
    except Exception:
        return "You are an autonomous programmer working in a command-line interface. Respond with one command."


def build_ctx(mw, loop, sysp):
    """system + brief task + last K (action, obs) turns, ending at the decision point."""
    acts, obs = loop["actions"], loop["obs"]
    turns = max(K, 2)
    a_tail = acts[-turns:]
    o_tail = (obs + [""] * turns)[-turns:]
    msgs = [{"role": "system", "content": sysp},
            {"role": "user", "content": f"Fix the issue in repo {loop['inst']}. Recent session below; continue."}]
    for a, o in zip(a_tail, o_tail):
        msgs.append({"role": "assistant", "content": f"`{a}`"})
        msgs.append({"role": "user", "content": o[:180]})
    return mw.render(msgs)


def main():
    t0 = time.time()
    alpha = float(sys.argv[1]) if len(sys.argv) > 1 else 12.0
    loops = real_loops()
    sysp = sys_prompt()
    d = np.load(OUT / "acts2.npz", allow_pickle=True)
    cond = d["meta_cond"]; tr = d["meta_group"] < 40
    X = d["L8"]
    STUCK = X[tr & (cond == "loopfail")].mean(0) - X[tr & (cond == "loopprog")].mean(0)
    STUCK /= np.linalg.norm(STUCK)
    RNG = np.random.default_rng(7); rnd = RNG.standard_normal(X.shape[1]); rnd /= np.linalg.norm(rnd)
    print(f"[setup] {len(loops)} real tight loops | α={alpha} | {time.time()-t0:.0f}s", flush=True)

    mw = ModelWrapper()

    # ---- baseline repeat-pref + steered repeat-pref on real loop contexts ----
    base, steered = [], []
    proj_stuck = []
    for lp in loops:
        ctx = build_ctx(mw, lp, sysp)
        rc = f" `{lp['cmd']}`"
        nlp = [mw.continuation_logprob(ctx, f" `{a}`") for a in NOVEL_ALTS]
        nc = f" `{NOVEL_ALTS[int(np.argmax(nlp))]}`"
        b = mw.continuation_logprob(ctx, rc) - max(nlp)
        base.append(b)
        with mw.steering(LAYER, -STUCK, alpha):
            s = mw.continuation_logprob(ctx, rc) - mw.continuation_logprob(ctx, nc)
        steered.append(s - b)
        proj_stuck.append(float(mw.capture_resid(ctx)[LAYER] @ STUCK))
    base = np.array(base)
    print(f"[real repeat-pref] mean={base.mean():+.3f} median={np.median(base):+.3f} "
          f"frac>0={np.mean(base>0):.2f}  (synthetic was +0.35, frac>0≈1.0)", flush=True)
    print(f"[real steering Δpref @α={alpha}] mean={np.mean(steered):+.3f}  (neg = less repeat)", flush=True)

    # ---- on-policy escape on a subset (generation slow) ----
    sub = loops[:10]
    esc = {"no_steer": [], "steer": [], "random": []}
    for lp in sub:
        ctx = build_ctx(mw, lp, sysp)
        for k, vec in [("no_steer", None), ("steer", -STUCK), ("random", rnd)]:
            if vec is None:
                g = mw.generate_kv(ctx, max_new_tokens=24)
            else:
                with mw.steering(LAYER, vec, alpha):
                    g = mw.generate_kv(ctx, max_new_tokens=24)
            esc[k].append(0 if is_repeat(parse_cmd(g), lp["cmd"]) else 1)
    res = {"alpha": alpha, "n_loops": len(loops), "n_onpolicy": len(sub),
           "real_repeat_pref_mean": float(base.mean()), "real_repeat_pref_frac_pos": float(np.mean(base > 0)),
           "real_steering_delta_pref": float(np.mean(steered)),
           "on_policy_escape": {k: float(np.mean(v)) for k, v in esc.items()},
           "proj_stuck_mean": float(np.mean(proj_stuck))}
    (OUT / "real_loops_results.json").write_text(json.dumps(res, indent=2))
    print(f"[real on-policy escape] {res['on_policy_escape']}", flush=True)
    print(f"[done] | {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
