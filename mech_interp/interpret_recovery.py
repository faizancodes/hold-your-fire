"""Interpret a discovered recovery direction via the logit lens.

Given the residual-space recovery direction `v` (from run_recovery_direction.py), project it
through the model's unembedding to see which tokens it *promotes* and *suppresses*. If the
direction is a genuine 'recover / investigate' lever (not just 'stop repeating'), the promoted
tokens should be evidence-gathering moves (cat, ls, grep, read, open ...) and the suppressed
tokens should include the repeated-command tokens.

  mech_interp/.venv/bin/python -m mech_interp.interpret_recovery <repo> <dir.npy> [topk]
"""
from __future__ import annotations

import sys

import mlx.core as mx
import numpy as np

from mech_interp.mlx_wrapper import MLXModel


def main():
    repo = sys.argv[1]
    dpath = sys.argv[2]
    topk = int(sys.argv[3]) if len(sys.argv) > 3 else 30
    v = np.load(dpath).astype(np.float32)
    v = v / (np.linalg.norm(v) + 1e-8)

    mw = MLXModel(repo)
    emb = mw.model.model.embed_tokens          # tied unembedding in qwen3/qwen2
    scores = emb.as_linear(mx.array(v))        # (vocab,) logit contribution per token
    scores = np.array(scores.astype(mx.float32))

    order = np.argsort(scores)
    promoted = order[::-1][:topk]
    suppressed = order[:topk]

    def show(ids):
        out = []
        for t in ids:
            try:
                s = mw.tokenizer.decode([int(t)])
            except Exception:
                s = f"<{t}>"
            out.append(repr(s))
        return out

    print(f"[interpret] {repo}\n  direction: {dpath}")
    print(f"\n  TOP PROMOTED tokens (the direction pushes the model toward these):")
    print("   " + "  ".join(show(promoted)))
    print(f"\n  TOP SUPPRESSED tokens (pushed away from):")
    print("   " + "  ".join(show(suppressed)))


if __name__ == "__main__":
    main()
