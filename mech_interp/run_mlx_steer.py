"""Residual-stream steering eval for a *quantized* model via MLX -- works on the small dev
model AND the big 4-bit MoE (30B/35B) with identical code (the MoE only lives inside the
block's MLP; we steer the block's dense fp16 output).

What it does, reusing the validated mlx_wrapper:
  1. builds a loop-vs-progress direction at each candidate layer (diff-of-means);
  2. picks the layer whose -direction most reduces repeat-preference at a probe alpha;
  3. dose-response sweep at that layer: -direction vs random vs orthogonal controls;
  4. coherence check (novel-action logprob must not collapse);
  5. targeted logit-penalty decode (the loop-breaker) greedy vs +penalty.

Writes results/mlx_steer_<model>.json.

Memory guard: refuses to load a model larger than available RAM unless --force, so it never
swap-thrashes a busy laptop.

  mech_interp/.venv/bin/python -m mech_interp.run_mlx_steer [repo] [--force]
"""
from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path

import numpy as np

from mech_interp.mlx_wrapper import MLXModel

OUT = Path(__file__).parent / "results"

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
HELD = "$ find_file Config\nNo matches found\n$ find_file Config\nNo matches found\n$ find_file Config\nNo matches found\n$ "
REPEAT, NOVEL = "find_file Config", "grep -r Config ."


def available_gb() -> float:
    """Reclaimable RAM (free + inactive + speculative + purgeable). Fails *safe* (0.0) so
    the guard aborts rather than thrashing if it cannot read memory."""
    def _find(pat: str, text: str) -> int:
        m = re.search(pat, text)
        return int(m.group(1)) if m else 0
    try:
        ps = int(subprocess.check_output(["sysctl", "-n", "hw.pagesize"]).split()[0])
        out = subprocess.check_output(["vm_stat"]).decode()
        pages = sum(_find(rf"Pages {k}:\s+(\d+)", out)
                    for k in ("free", "inactive", "speculative", "purgeable"))
        return pages * ps / 1e9 if pages else 0.0
    except Exception:
        return 0.0


def est_model_gb(repo: str) -> float:
    try:
        from huggingface_hub import model_info
        mi = model_info(repo, files_metadata=True)
        sib = mi.siblings or []
        return sum((f.size or 0) for f in sib if f.rfilename.endswith(".safetensors")) / 1e9
    except Exception:
        return 0.0


def rp(mw):
    return mw.continuation_logprob(HELD, REPEAT) - mw.continuation_logprob(HELD, NOVEL)


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    force = "--force" in sys.argv
    repo = args[0] if args else "mlx-community/Qwen3-Coder-30B-A3B-Instruct-4bit"

    size, avail = est_model_gb(repo), available_gb()
    print(f"[mem] model ~{size:.1f}GB on disk | available ~{avail:.1f}GB RAM")
    # margin 1.1: weights dominate; our forwards are short so KV/activation memory is small.
    if size and avail < size * 1.10 and not force:
        print(f"[abort] not enough free RAM to load {repo} without swap-thrashing.\n"
              f"        free ~{size*1.10:.0f}GB (quit a couple apps) or pass --force to override.")
        sys.exit(3)

    print(f"[load] {repo}")
    mw = MLXModel(repo)
    N = mw.n_layers
    cands = sorted({max(1, N // 6), N // 4, N // 3, N // 2, 2 * N // 3})
    print(f"[model] layers={N} d={mw.d_model} | candidate layers {cands}")

    base = rp(mw)
    rng = np.random.default_rng(0)
    rand = rng.standard_normal(mw.d_model).astype(np.float32); rand /= np.linalg.norm(rand)

    # build direction at every candidate layer once (one capture per prompt, all layers)
    loop_caps = [mw.capture_resid(mw.tok(p), cands) for p in LOOP]
    prog_caps = [mw.capture_resid(mw.tok(p), cands) for p in PROGRESS]
    dirs = {}
    for L in cands:
        d = np.mean([c[L] for c in loop_caps], 0) - np.mean([c[L] for c in prog_caps], 0)
        dirs[L] = d / (np.linalg.norm(d) + 1e-8)

    # pick the layer whose -direction most reduces repeat-pref at a probe alpha *without
    # collapsing coherence* (early layers often wreck the output distribution; a steer that
    # only "works" by breaking the model is not a real result).
    probe = 8.0
    base_nov = mw.continuation_logprob(HELD, NOVEL)
    pick = {}
    for L in cands:
        with mw.steering(L, -dirs[L], probe):
            pick[L] = (rp(mw), mw.continuation_logprob(HELD, NOVEL))
    coherent = [L for L in cands if pick[L][1] > base_nov - 5.0]   # novel-logp within 5 nats
    pool = coherent or cands
    best = min(pool, key=lambda L: pick[L][0])
    print(f"[layer pick] repeat-pref @a={probe} (novel-logp): "
          + ", ".join(f"L{L}={pick[L][0]:+.1f}({pick[L][1]:+.0f})" for L in cands))
    print(f"   coherence-preserving layers: {coherent or 'NONE -> falling back to all'}  -> best L={best}")

    direction = dirs[best]
    ortho = rng.standard_normal(mw.d_model).astype(np.float32)
    ortho -= (ortho @ direction) * direction; ortho /= np.linalg.norm(ortho)

    alphas = [0, 4, 8, 12, 16]
    rows = []
    print(f"\n[dose-response] layer L={best}, baseline repeat-pref {base:+.3f}")
    print(f"   {'alpha':>5} | {'-dir':>8} | {'random':>8} | {'ortho':>8} | {'novel_logp(-dir)':>16}")
    for a in alphas:
        with mw.steering(best, -direction, a):
            d = rp(mw); nov = mw.continuation_logprob(HELD, NOVEL)
        with mw.steering(best, rand, a):
            r = rp(mw)
        with mw.steering(best, ortho, a):
            o = rp(mw)
        rows.append({"alpha": a, "dir": d, "random": r, "ortho": o, "novel_logp": nov})
        print(f"   {a:>5} | {d:>+8.3f} | {r:>+8.3f} | {o:>+8.3f} | {nov:>+16.3f}")

    # loop-break decode
    gids = mw.tok(HELD)
    plain = mw.generate(gids, max_tokens=12, temperature=0.0)
    broken = mw.generate(gids, max_tokens=12, temperature=0.0, bad_ids=mw.tok(REPEAT), penalty=12.0)

    drop = rows[0]["dir"] - rows[-1]["dir"]
    rand_drop = rows[0]["random"] - rows[-1]["random"]
    res = {"repo": repo, "n_layers": N, "d_model": mw.d_model, "best_layer": best,
           "baseline_repeat_pref": base, "alphas": alphas, "rows": rows,
           "dir_net_drop": drop, "random_net_drop": rand_drop,
           "coherent_novel_logp_preserved": bool(rows[-1]["novel_logp"] > -15),
           "decode_greedy": plain, "decode_penalized": broken,
           "decode_changed": plain.strip() != broken.strip()}
    OUT.mkdir(exist_ok=True)
    safe = re.sub(r"[^A-Za-z0-9]+", "_", repo)
    (OUT / f"mlx_steer_{safe}.json").write_text(json.dumps(res, indent=2))
    print(f"\n[result] -dir net drop {drop:+.3f} vs random {rand_drop:+.3f} "
          f"(specific: {'yes' if drop > rand_drop + 0.05 else 'weak'})")
    print(f"[decode] greedy {plain!r}\n         +penalty {broken!r}  changed={res['decode_changed']}")
    print(f"[done] wrote mlx_steer_{safe}.json")


if __name__ == "__main__":
    main()
