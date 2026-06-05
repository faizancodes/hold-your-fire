"""Efficacy–disruption frontier: is the loop-breaking "win" the penalty, or the GATING?

Efficacy   = loop-escape on REAL loops (unproductive repetition; want HIGH).
Disruption = action-change vs no-intervention on PRODUCTIVE repetition (loopprog: same command
             re-run while the fail count drops 3→2→1→0; re-running is legitimate, so a change
             is disruption; want LOW).

Gate (a realistic monitor signal) = commands repeat AND observations not improving.
  real loops -> fires;  productive repetition -> does NOT fire.

Interventions: none; gated targeted penalty; always-on targeted penalty (gate removed);
always-on repetition_penalty; always-on no_repeat_ngram(3); −STUCK steering.
Claim under test: always-on penalties break loops BUT disrupt productive repetition; gating
removes the disruption -> the MONITOR is the contribution, not the penalty.

    mech_interp/.venv/bin/python -m mech_interp.run_frontier
"""
from __future__ import annotations

import json, re, time
from pathlib import Path

import numpy as np
import torch

from mech_interp.model_wrapper import ModelWrapper
from mech_interp.synthetic import make_scenarios
from mech_interp.run_real_loops import real_loops, build_ctx, sys_prompt
from mech_interp.run_onpolicy import parse_cmd, is_repeat

OUT = Path(__file__).parent / "results"
LAYER, K, PEN, REP_R = 8, 4, 8.0, 2.0
N_LOOP, N_PROD = 10, 12


def first_int(s):
    m = re.search(r"\d+", s or "")
    return int(m.group()) if m else None


def obs_improving(obs_list):
    xs = [first_int(o) for o in obs_list[-3:]]
    xs = [x for x in xs if x is not None]
    return len(xs) >= 2 and all(b < a for a, b in zip(xs, xs[1:]))


def gate_fires(cmds, obs):
    repeat = len(cmds) >= 2 and cmds[-1] == cmds[-2]
    return bool(repeat and not obs_improving(obs))


def rep_penalty(r):
    def f(logits, seq):
        idx = torch.tensor(list(set(seq)), device=logits.device)
        l = logits.clone(); sel = l[idx]
        l[idx] = torch.where(sel > 0, sel / r, sel * r)
        return l
    return f


def no_repeat(n):
    def f(logits, seq):
        if len(seq) < n:
            return logits
        prefix = tuple(seq[-(n - 1):]); banned = set()
        for i in range(len(seq) - (n - 1)):
            if tuple(seq[i:i + n - 1]) == prefix:
                banned.add(seq[i + n - 1])
        if not banned:
            return logits
        l = logits.clone()
        for b in banned:
            l[b] = -1e9
        return l
    return f


def main():
    t0 = time.time()
    mw = ModelWrapper()
    d = np.load(OUT / "acts2.npz", allow_pickle=True); cond = d["meta_cond"]; tr = d["meta_group"] < 40
    X = d["L8"]; STUCK = X[tr & (cond == "loopfail")].mean(0) - X[tr & (cond == "loopprog")].mean(0)
    STUCK /= np.linalg.norm(STUCK)

    # efficacy set: real loops
    sysp = sys_prompt()
    loops = []
    for lp in real_loops(N_LOOP):
        loops.append({"ctx": build_ctx(mw, lp, sysp), "rep": lp["cmd"],
                      "gate": gate_fires(lp["actions"], lp["obs"]), "kind": "loop"})
    # disruption set: productive repetition (same command, improving obs)
    prods = []
    for s in make_scenarios(60)[40:40 + N_PROD]:
        obs = [f"{max(3 - i, 0)} failed - assertion error" for i in range(K)]
        msgs = [{"role": "system", "content": s.system}, {"role": "user", "content": s.task}]
        for o in obs:
            msgs.append({"role": "assistant", "content": f"{s.c_thought}\n`{s.c_cmd}`"})
            msgs.append({"role": "user", "content": o})
        prods.append({"ctx": mw.render(msgs), "rep": s.c_cmd,
                      "gate": gate_fires([s.c_cmd] * K, obs), "kind": "prod"})
    print(f"[setup] loops={len(loops)} (gate-fires {sum(c['gate'] for c in loops)}) | "
          f"prod-repeat={len(prods)} (gate-fires {sum(c['gate'] for c in prods)}) | {time.time()-t0:.0f}s", flush=True)

    def cmd_ids(c):
        return set(mw.tok(c["rep"], add_special_tokens=False).input_ids)

    INTERV = {
        "none": lambda c: mw.generate_kv(c["ctx"], 20),
        "gated_targeted": lambda c: mw.generate_kv(c["ctx"], 20, bad_ids=cmd_ids(c),
                                                   penalty=PEN if c["gate"] else 0.0),
        "alwayson_targeted": lambda c: mw.generate_kv(c["ctx"], 20, bad_ids=cmd_ids(c), penalty=PEN),
        "alwayson_rep_pen": lambda c: mw.generate_kv(c["ctx"], 20, logits_fn=rep_penalty(REP_R)),
        "alwayson_norepeat3": lambda c: mw.generate_kv(c["ctx"], 20, logits_fn=no_repeat(3)),
    }

    def steer_gen(c):
        with mw.steering(LAYER, -STUCK, 16.0):
            return mw.generate_kv(c["ctx"], 20)
    INTERV["steering"] = steer_gen

    # baseline actions on prod-repeat (for disruption)
    base_prod = [parse_cmd(INTERV["none"](c)) for c in prods]

    res = {"interventions": {}}
    for name, fn in INTERV.items():
        esc = float(np.mean([0 if is_repeat(parse_cmd(fn(c)), c["rep"]) else 1 for c in loops]))
        disr = float(np.mean([parse_cmd(fn(c)) != base_prod[i] for i, c in enumerate(prods)]))
        res["interventions"][name] = {"efficacy_escape": esc, "disruption_prodrepeat": disr}
        print(f"  {name:18s} escape={esc:.2f}  disruption={disr:.2f}", flush=True)

    (OUT / "frontier_results.json").write_text(json.dumps(res, indent=2))
    print(f"[done] wrote frontier_results.json | {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
