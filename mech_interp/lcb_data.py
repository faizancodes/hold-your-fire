"""LiveCodeBench loader — real, hard, *contamination-free* competitive-programming problems.

We use `livecodebench/code_generation_lite` (test6.jsonl: 175 problems, all 2025-01..2025-04,
i.e. AFTER the training cutoff of Qwen2.5-Coder-7B and Qwen3-Coder-30B, so the model cannot have
memorized solutions). We keep the AtCoder **stdin** problems (a self-contained program that reads
stdin and writes stdout), which run locally with no Docker. Each problem ships public examples +
decoded hidden ("private") test cases for a solid solve/fail label.

This is the substrate the whole recovery study was missing: tasks hard enough that a capable model
genuinely struggles — and, tuned to its ~50% frontier, sometimes recovers and sometimes stays stuck.
"""
from __future__ import annotations

import base64
import json
import zlib
from functools import lru_cache

from huggingface_hub import hf_hub_download

REPO = "livecodebench/code_generation_lite"


def _decode_private(s: str) -> list[dict]:
    raw = zlib.decompress(base64.b64decode(s))
    try:
        return json.loads(raw)
    except Exception:
        import pickle
        return json.loads(pickle.loads(raw))


@lru_cache(maxsize=8)
def _rows(fn: str) -> tuple:
    path = hf_hub_download(REPO, fn, repo_type="dataset")
    return tuple(json.loads(l) for l in open(path))


def load_problems(difficulties=("easy",), after="2024-12", testtype="stdin",
                  files=("test6.jsonl",), limit=None) -> list[dict]:
    """Contamination-free problems matching difficulty/date/testtype filters."""
    out = []
    for fn in files:
        for r in _rows(fn):
            if r["difficulty"] not in difficulties:
                continue
            if r["contest_date"][:7] < after:
                continue
            pub = json.loads(r["public_test_cases"])
            if not pub or pub[0]["testtype"] != testtype:
                continue
            try:
                priv = _decode_private(r["private_test_cases"])
            except Exception:
                priv = []
            out.append({
                "id": r["question_id"], "title": r["question_title"],
                "statement": r["question_content"], "difficulty": r["difficulty"],
                "date": r["contest_date"][:10], "platform": r["platform"],
                "starter": r["starter_code"], "public": pub, "private": priv,
            })
    out.sort(key=lambda p: p["id"])
    return out[:limit] if limit else out


if __name__ == "__main__":
    import collections
    for diff in ("easy", "medium", "hard"):
        ps = load_problems(difficulties=(diff,), after="2024-12")
        ntests = [len(p["public"]) + len(p["private"]) for p in ps]
        print(f"{diff:7} stdin problems: {len(ps):3d} | avg tests {sum(ntests)/max(1,len(ntests)):.0f} "
              f"| dates {collections.Counter(p['date'][:7] for p in ps)}")
    ex = load_problems(difficulties=("easy",))[0]
    print(f"\nexample {ex['id']} ({ex['difficulty']}, {ex['date']}): {ex['title']}")
    print("statement[:200]:", ex["statement"][:200].replace("\n", " "))
    print("public[0]:", ex["public"][0])
