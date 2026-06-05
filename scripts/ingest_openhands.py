#!/usr/bin/env python3
"""Ingest a SECOND, independent trajectory source for zero-shot generalization testing.

Source: `togethercomputer/CoderForge-Preview-32B-SWE-Bench-Verified-Evaluation-trajectories`
— the **OpenHands** agent scaffold (CodeAct: execute_bash + str_replace_editor + finish),
a genuinely different agent from the SWE-agent corpus the monitor was trained on. We map
OpenHands tool-calls into the project's raw-row schema so they flow through the *same*
normalize/feature pipeline and the frozen monitor scores them with zero retraining.

  python scripts/ingest_openhands.py --max-traj 1500 --max-runs-per-instance 3

Output: data/raw/openhands_coderforge.jsonl  (raw rows; trajectory as JSON string)
"""

from __future__ import annotations

import argparse
import json
from collections import Counter

import _bootstrap  # noqa: F401

from localguard.utils import RAW_DIR, ensure_dirs, write_jsonl

DATASET = "togethercomputer/CoderForge-Preview-32B-SWE-Bench-Verified-Evaluation-trajectories"


def _reconstruct_action(fn_name: str, args: dict) -> str:
    """Map an OpenHands tool-call to action text the SHARED classify_action understands."""
    if fn_name == "execute_bash":
        return str(args.get("command", ""))
    if fn_name == "str_replace_editor":
        sub, path = args.get("command", ""), args.get("path", "")
        return {
            "view": f"open {path}", "create": f"create {path}",
            "str_replace": f"str_replace {path}", "undo_edit": f"edit 1:1 {path}",
        }.get(sub, f"insert {args.get('insert_line', 1)} {path}" if sub == "insert" else f"{sub} {path}")
    if fn_name == "finish":
        return "submit"
    return ""  # think / unknown -> folded into thought


def _row_to_turns(messages: list[dict]) -> list[dict]:
    """Pair each substantive assistant tool-call with its observation (by tool_call_id)."""
    # map tool_call_id -> observation text
    obs_by_id = {m.get("tool_call_id"): str(m.get("content") or "")
                 for m in messages if m.get("role") == "tool" and m.get("tool_call_id")}
    turns, pending_thought = [], ""
    for m in messages:
        if m.get("role") != "assistant":
            continue
        thought = str(m.get("content") or "")
        tcs = m.get("tool_calls") or []
        substantive = [tc for tc in tcs if ((tc.get("function") or {}).get("name")) != "think"]
        if not substantive:  # pure-think turn -> fold into next action's thought
            pending_thought = (pending_thought + " " + thought).strip()
            continue
        tc = substantive[0]
        fn = tc.get("function") or {}
        try:
            args = json.loads(fn.get("arguments") or "{}")
        except Exception:
            args = {}
        turns.append({
            "thought": (pending_thought + " " + thought).strip(),
            "action": _reconstruct_action(fn.get("name", ""), args),
            "observation": obs_by_id.get(tc.get("id"), ""),
        })
        pending_thought = ""
    return turns


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--max-traj", type=int, default=1500)
    ap.add_argument("--max-runs-per-instance", type=int, default=3)
    args = ap.parse_args()

    from datasets import load_dataset
    d = load_dataset(DATASET, split="train", streaming=True)

    rows, per_inst, kinds = [], Counter(), Counter()
    for row in d:
        try:
            ds = json.loads(row["ds"]); inst = ds.get("instance_id")
            if not inst or per_inst[inst] >= args.max_runs_per_instance:
                continue
            turns = _row_to_turns(json.loads(row["messages"]))
            if len(turns) < 3:
                continue
            per_inst[inst] += 1
            kinds[bool(float(row["reward"]) >= 0.5)] += 1
            rows.append({
                "instance_id": inst,
                "model_name": "openhands-coderforge-32b",
                "target": bool(float(row["reward"]) >= 0.5),
                "trajectory": json.dumps(turns, ensure_ascii=False),  # JSON string (flat schema)
            })
            if len(rows) >= args.max_traj:
                break
        except Exception:
            continue

    ensure_dirs(RAW_DIR)
    out = RAW_DIR / "openhands_coderforge.jsonl"
    write_jsonl(out, rows)
    print(f"[ingest] wrote {len(rows)} OpenHands trajectories over {len(per_inst)} instances -> {out}")
    print(f"[ingest] label balance (target=resolved): {dict(kinds)} "
          f"(fail rate {kinds[False]/max(1,len(rows)):.2f})")


if __name__ == "__main__":
    main()
