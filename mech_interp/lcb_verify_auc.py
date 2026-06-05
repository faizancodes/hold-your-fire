"""Decisive cheap test: does the model's EXPLICIT self-verification (full forward pass) distinguish
correct from incorrect code, where the linear activation probe could not (fresh AUC 0.45)?

The generation-verification gap says models verify better than they generate. If the 7B can rank its
own correct solutions above its incorrect ones via an explicit YES/NO judgement, then verification-
guided selection is a real 'way' to lift pass@1. We test this on Phase 1's ALREADY-SAVED correct vs
incorrect code samples (no new generation -> fast).

  mech_interp/.venv/bin/python -u -m mech_interp.lcb_verify_auc <repo>
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

from mech_interp.mlx_wrapper import MLXModel
from mech_interp.lcb_data import load_problems
from mech_interp.run_recovery_direction import _auc

OUT = Path(__file__).parent / "results"
JUDGE = "You are a meticulous competitive-programming judge. Think about edge cases and constraints."


def _verify_msgs(problem, code):
    return [{"role": "system", "content": JUDGE},
            {"role": "user", "content":
                f"{problem['statement'][:1600]}\n\nCandidate Python solution:\n```python\n{code[:1600]}\n```\n\n"
                "Will this solution pass ALL hidden tests, including tricky edge cases and the stated "
                "constraints? Answer with exactly one word: YES or NO."}]


def verify_score(mw, problem, code):
    """P(YES) - P(NO) that the solution passes all tests, under the model's own judgement."""
    prompt = mw.tokenizer.apply_chat_template(_verify_msgs(problem, code), add_generation_prompt=True, tokenize=False)
    return max(mw.continuation_logprob(prompt, c) for c in ("YES", " YES")) - \
        max(mw.continuation_logprob(prompt, c) for c in ("NO", " NO"))


_YESNO = {}


def _yes_no_ids(mw):
    k = id(mw.tokenizer)
    if k not in _YESNO:
        def first(ws):
            s = set()
            for w in ws:
                t = mw.tokenizer.encode(w, add_special_tokens=False)
                if t:
                    s.add(t[0])
            return list(s)
        _YESNO[k] = (first(["YES", " YES", "Yes", " Yes", "yes", " yes"]),
                     first(["NO", " NO", "No", " No", "no", " no"]))
    return _YESNO[k]


def verify_score_fast(mw, problem, code):
    """Same judgement in ONE forward: read the next-token logits and compare YES vs NO mass."""
    ids = mw.render(_verify_msgs(problem, code))           # ends at the assistant generation point
    row = mw._logits(ids)[-1]                               # next-token logits (V,); softmax-norm cancels
    yes, no = _yes_no_ids(mw)
    return max(float(row[t]) for t in yes) - max(float(row[t]) for t in no)


def main():
    t0 = time.time()
    repo = sys.argv[1] if len(sys.argv) > 1 else "mlx-community/Qwen2.5-Coder-7B-Instruct-4bit"
    safe = repo.replace("/", "_").replace(":", "_")
    pk = json.loads((OUT / f"lcb_passk_{safe}.json").read_text())
    statements = {}
    for d in ("easy", "medium", "hard"):
        for p in load_problems(difficulties=(d,), after="2024-12"):
            statements[p["id"]] = p
    usable = [r for r in pk["rows"] if r["correct"] and r["incorrect"] and r["id"] in statements]
    print(f"[setup] explicit self-verification on {len(usable)} problems' saved correct/incorrect codes", flush=True)

    print(f"[load] {repo}", flush=True)
    mw = MLXModel(repo)

    labels, scores, perprob = [], [], []
    for r in usable:
        p = statements[r["id"]]
        ll, ss = [], []
        for c in r["correct"][:4]:
            ss.append(verify_score(mw, p, c)); ll.append(1)
        for c in r["incorrect"][:4]:
            ss.append(verify_score(mw, p, c)); ll.append(0)
        labels += ll; scores += ss
        cmean = np.mean([s for s, l in zip(ss, ll) if l == 1])
        imean = np.mean([s for s, l in zip(ss, ll) if l == 0])
        perprob.append({"id": r["id"], "greedy": r["greedy"], "corr_score": float(cmean), "incorr_score": float(imean)})
        print(f"   {r['id']:11} mean verify-score correct={cmean:+.2f} incorrect={imean:+.2f} "
              f"(sep {cmean-imean:+.2f}) | {time.time()-t0:.0f}s", flush=True)

    labels, scores = np.array(labels), np.array(scores)
    auc = _auc(labels, scores)
    held = [r for r in usable if not r["greedy"]]
    hl, hs = [], []
    for r in held:
        p = statements[r["id"]]
        for c in r["correct"][:4]:
            hs.append(verify_score(mw, p, c)); hl.append(1)
        for c in r["incorrect"][:4]:
            hs.append(verify_score(mw, p, c)); hl.append(0)
    held_auc = _auc(np.array(hl), np.array(hs)) if hl else float("nan")
    print(f"\n[result] explicit self-verification AUC (correct vs incorrect code):", flush=True)
    print(f"   ALL usable: {auc:.2f}   | greedy-fail held-out only: {held_auc:.2f}", flush=True)
    print(f"   (compare: linear activation probe was 0.75 teacher-forced but 0.45 on fresh samples)", flush=True)
    print(f"   >>> {'VERIFICATION HAS SIGNAL (>0.6) -> selection is a viable way' if auc > 0.6 else 'weak (<=0.6) -> 7B cannot reliably verify its own code'}", flush=True)
    (OUT / f"lcb_verify_auc_{safe}.json").write_text(json.dumps(
        {"repo": repo, "auc_all": auc, "auc_held": held_auc, "perprob": perprob}, indent=2, default=float))
    print(f"[done] | {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
