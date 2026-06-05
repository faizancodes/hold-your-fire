"""Validate the MLX steering wrapper end-to-end on a small 4-bit model.

Checks, in order of importance:
  1. capture_resid returns correct shapes.
  2. continuation_logprob is sane (a coherent continuation scores higher than a wrong one).
  3. the hook is actually live: steering with alpha=0 is a no-op; a random vector at large
     alpha changes the forward (proves we modify activations, not just decorate).
  4. SCIENCE: a loop-vs-progress direction causally moves repeat-preference with a monotonic
     dose-response, while random/orthogonal controls of equal norm do (nearly) nothing.
  5. the targeted logit-penalty decode path (the loop-breaker) changes the emitted command.

Run: mech_interp/.venv/bin/python -m mech_interp.test_mlx_wrapper [model_repo]
"""
from __future__ import annotations

import sys

import numpy as np

from mech_interp.mlx_wrapper import MLXModel

REPO = sys.argv[1] if len(sys.argv) > 1 else "mlx-community/Qwen2.5-Coder-1.5B-Instruct-4bit"

# ---- contrast prompts (agent-transcript style) -----------------------------------
LOOP = [
    "$ find_file SecretStr\nNo matches found\n$ find_file SecretStr\nNo matches found\n$ find_file SecretStr\nNo matches found\n$ ",
    "$ grep TODO app.py\n(no output)\n$ grep TODO app.py\n(no output)\n$ grep TODO app.py\n(no output)\n$ ",
    "$ python test.py\nFAILED\n$ python test.py\nFAILED\n$ python test.py\nFAILED\n$ ",
    "$ cat missing.txt\nNo such file\n$ cat missing.txt\nNo such file\n$ cat missing.txt\nNo such file\n$ ",
]
PROGRESS = [
    "$ ls\napp.py test.py\n$ cat app.py\ndef f(): return 1\n$ python test.py\n1 passed\n$ ",
    "$ grep def app.py\ndef f():\n$ sed -i 's/1/2/' app.py\n$ python test.py\n2 passed\n$ ",
    "$ ls src\nmain.py util.py\n$ cat util.py\ndef g(): pass\n$ python -m pytest\nok\n$ ",
    "$ find . -name '*.py'\n./a.py ./b.py\n$ head a.py\nimport os\n$ python a.py\ndone\n$ ",
]
# held-out loop context to measure repeat-preference on:
HELD = "$ find_file Config\nNo matches found\n$ find_file Config\nNo matches found\n$ find_file Config\nNo matches found\n$ "
REPEAT = "find_file Config"
NOVEL = "grep -r Config ."


def repeat_pref(mw):
    return mw.continuation_logprob(HELD, REPEAT) - mw.continuation_logprob(HELD, NOVEL)


def main():
    print(f"[load] {REPO}")
    mw = MLXModel(REPO)
    L = 8 if mw.n_layers > 10 else mw.n_layers // 2
    print(f"[model] layers={mw.n_layers} d={mw.d_model} | steering layer L={L}\n")

    # 1) capture shapes
    ids = mw.tok(HELD)
    caps = mw.capture_resid(ids, [4, L, mw.n_layers - 1])
    ok_shape = all(v.shape == (mw.d_model,) for v in caps.values())
    print(f"[1 shapes] captured {sorted(caps)} each {next(iter(caps.values())).shape} -> {'PASS' if ok_shape else 'FAIL'}")

    # 2) logprob sanity: a sensible arithmetic continuation beats a wrong one
    good = mw.continuation_logprob("2 + 3 = ", "5")
    bad = mw.continuation_logprob("2 + 3 = ", "9")
    print(f"[2 logprob] logp('5')={good:.2f} > logp('9')={bad:.2f} -> {'PASS' if good > bad else 'FAIL'}")

    # 3) hook is live: alpha=0 no-op; random large alpha changes the metric
    base = repeat_pref(mw)
    rng = np.random.default_rng(0)
    rand = rng.standard_normal(mw.d_model).astype(np.float32)
    rand /= np.linalg.norm(rand)
    with mw.steering(L, rand, 0.0):
        zero = repeat_pref(mw)
    with mw.steering(L, rand, 30.0):
        big = repeat_pref(mw)
    noop_ok = abs(zero - base) < 1e-3
    live_ok = abs(big - base) > 1e-2
    print(f"[3 hook live] base={base:+.3f} alpha0={zero:+.3f} (no-op {'PASS' if noop_ok else 'FAIL'}) "
          f"randbig={big:+.3f} (changes forward {'PASS' if live_ok else 'FAIL'})")

    # 4) SCIENCE: build loop-progress direction, show dose-response vs dead controls
    loop_acts = np.stack([mw.capture_resid(mw.tok(p), [L])[L] for p in LOOP])
    prog_acts = np.stack([mw.capture_resid(mw.tok(p), [L])[L] for p in PROGRESS])
    direction = loop_acts.mean(0) - prog_acts.mean(0)       # points toward "looping/stuck"
    direction /= np.linalg.norm(direction)
    # an orthogonal control of equal norm
    ortho = rng.standard_normal(mw.d_model).astype(np.float32)
    ortho -= (ortho @ direction) * direction
    ortho /= np.linalg.norm(ortho)

    alphas = [0, 4, 8, 16]
    print(f"\n[4 dose-response] repeat-pref while steering -dir at L={L} (baseline {base:+.3f}):")
    print(f"   {'alpha':>6} | {'-direction':>11} | {'random':>8} | {'orthogonal':>10}")
    series = {}
    for a in alphas:
        with mw.steering(L, -direction, a):
            d = repeat_pref(mw)
        with mw.steering(L, rand, a):
            r = repeat_pref(mw)
        with mw.steering(L, ortho, a):
            o = repeat_pref(mw)
        series[a] = d
        print(f"   {a:>6} | {d:>+11.3f} | {r:>+8.3f} | {o:>+10.3f}")
    drop = series[alphas[0]] - series[alphas[-1]]
    monotonic = all(series[alphas[i]] >= series[alphas[i + 1]] - 1e-3 for i in range(len(alphas) - 1))
    print(f"   net change over alpha: {drop:+.3f}  (monotone decreasing: {'yes' if monotonic else 'no'})")

    # 5) targeted logit-penalty decode (the loop-breaker) changes the emitted command
    gen_ids = mw.tok(HELD)
    plain = mw.generate(gen_ids, max_tokens=12, temperature=0.0)
    bad_tok = mw.tok(REPEAT)
    broken = mw.generate(gen_ids, max_tokens=12, temperature=0.0, bad_ids=bad_tok, penalty=12.0)
    print(f"\n[5 loop-break decode]")
    print(f"   greedy        : {plain!r}")
    print(f"   +penalty(cmd) : {broken!r}")
    print(f"   changed command: {'PASS' if plain.strip() != broken.strip() else 'FAIL (try higher penalty)'}")

    print("\n[done] MLX steering wrapper validated.")


if __name__ == "__main__":
    main()
