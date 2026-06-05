"""Capable-model outcome experiment: does breaking the loop FLIP outcomes once the model
is strong enough to *recover*?  (Ollama backend.)

The 1.5B Tier-2 result (run_outcome.py) was an honest null: the targeted loop-break broke
the loop live every time but task success stayed 0/6, because a 1.5B model can't recover
once un-stuck. The open question was therefore "the model must be capable of recovery once
un-stuck." This script tests exactly that with a capable local model.

Same sandboxed agent_env as the 1.5B run (apples-to-apples), but:
  * harder, recovery-requiring tasks (harder_tasks.HARD_TASKS) so a capable model actually
    loops on something rather than one-shotting everything;
  * an Ollama backend (e.g. qwen3.6:35b-a3b, qwen3-coder:30b) instead of white-box MPS;
  * treatment = LocalGuard's *deployable* loop_break (the message form: name + forbid the
    repeated command). Ollama has no per-token logit_bias, so the white-box token penalty
    isn't available here -- the message form is the realistic deployment and is the exact
    intervention wired into src/localguard (interventions.loop_break_message).

  control   : let it loop to the step budget
  treatment : on loop-detection, inject the loop_break message before the next action

Metric: REAL task success (test passes), escape rate, steps, loop rate, and the paired
recovery count (control looped+failed -> did treatment then solve?).

  python3 mech_interp/run_outcome_ollama.py [model] [temp] [seeds] [modes] [taskset]
    model   : ollama tag (default qwen3.6:35b-a3b-q4_K_M)
    temp    : sampling temperature (default 0.0 -> greedy, paired)
    seeds   : comma list (default 0)
    modes   : control,treatment (default both)
    taskset : hard | easy | all (default hard)
"""
from __future__ import annotations

import json
import os
import re
import sys
import time
from collections import Counter
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from localguard.interventions import loop_break_message  # the shipped, integrated intervention

from mech_interp.agent_env import TASKS as EASY_TASKS, setup_task, run_cmd, check_solved, cleanup
from mech_interp.harder_tasks import HARD_TASKS

OUT = Path(__file__).parent / "results"
OLLAMA = "http://localhost:11434"
BUDGET = 10
SYS = ("You are fixing a bug in sol.py so that test.py passes. Each turn reply with EXACTLY ONE "
       "shell command in backticks, nothing else. You can use: `cat sol.py` to read the source, "
       "`sed -i 's/old/new/' sol.py` to edit it, and `python test.py` to run the test. "
       "Read the SOURCE (sol.py); do not edit test.py. /no_think")


def is_test_runner(c: str) -> bool:
    c = c.strip()
    return c.startswith("python test") or c.startswith("python3 test") or c.startswith("pytest")


def parse_cmd(text: str) -> str:
    """First backticked command (tolerant of <think> blocks / chatter)."""
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.S)
    m = re.search(r"`([^`]+)`", text)
    if m:
        return m.group(1).strip().splitlines()[0].strip()
    for ln in text.strip().splitlines():
        ln = ln.strip().strip("`").strip()
        if ln and not ln.lower().startswith(("here", "i ", "the ", "to ")):
            return ln
    return text.strip().splitlines()[0].strip() if text.strip() else ""


def ollama_chat(model: str, msgs: list[dict], temperature: float,
                num_predict: int = 160, num_ctx: int | None = None, timeout: int = 300) -> str:
    # num_ctx via env (OLLAMA_NUM_CTX) so a tight-on-memory model (e.g. 23GB 35B on a 32GB
    # box) can shrink the KV cache without code changes; default 8192.
    if num_ctx is None:
        num_ctx = int(os.environ.get("OLLAMA_NUM_CTX", "8192"))
    body = {"model": model, "messages": msgs, "stream": False,
            "options": {"temperature": temperature, "num_predict": num_predict, "num_ctx": num_ctx}}
    for attempt in ("think", "plain"):
        b = dict(body)
        if attempt == "think":
            b["think"] = False  # disable reasoning on thinking models (faster, deterministic)
        r = requests.post(f"{OLLAMA}/api/chat", json=b, timeout=timeout)
        if r.status_code == 400 and attempt == "think":
            continue  # model doesn't accept `think` -> retry plain
        r.raise_for_status()
        return r.json()["message"]["content"]
    return ""


def model_available(model: str) -> bool:
    try:
        tags = requests.get(f"{OLLAMA}/api/tags", timeout=10).json().get("models", [])
        names = {m.get("name", "") for m in tags} | {m.get("model", "") for m in tags}
        return model in names or any(n.split(":")[0] == model.split(":")[0] and n == model for n in names)
    except Exception:
        return False


def episode(model, task, treatment, temp, seed):
    d = setup_task(task)
    init = run_cmd("python test.py", d)
    msgs = [{"role": "system", "content": SYS},
            {"role": "user", "content": f"The test fails:\n{init[:240]}\nFix sol.py so test.py passes."}]
    hist, trace = [], []
    solved = False
    n_loops = n_interv = n_escape = 0
    for step in range(BUDGET):
        # loop = a non-test command repeated within the recent window while still failing
        recent = Counter(hist[-5:])
        looped = [c for c, n in recent.items() if n >= 2 and c and not is_test_runner(c)]
        is_loop = bool(looped) and not solved
        n_loops += int(is_loop)
        if is_loop and treatment:
            top = max(looped, key=lambda c: recent[c])
            msgs.append({"role": "user", "content": loop_break_message(top, recent[top])})
            n_interv += 1
        gen = ollama_chat(model, msgs, temperature=temp)
        cmd = parse_cmd(gen)
        if is_loop and treatment and cmd and cmd not in looped:
            n_escape += 1
        obs = run_cmd(cmd, d)
        solved = check_solved(task, d)
        msgs.append({"role": "assistant", "content": gen[:160]})
        msgs.append({"role": "user", "content": obs[:240]})
        hist.append(cmd)
        trace.append((cmd[:46], obs[:46]))
        if solved:
            break
    cleanup(d)
    return {"solved": solved, "steps": len(hist), "n_loops": n_loops,
            "n_interventions": n_interv, "n_escape": n_escape, "trace": trace}


def main():
    t0 = time.time()
    model = sys.argv[1] if len(sys.argv) > 1 else "qwen3.6:35b-a3b-q4_K_M"
    temp = float(sys.argv[2]) if len(sys.argv) > 2 else 0.0
    seeds = [int(x) for x in sys.argv[3].split(",")] if len(sys.argv) > 3 else [0]
    modes = sys.argv[4].split(",") if len(sys.argv) > 4 else ["control", "treatment"]
    which = sys.argv[5] if len(sys.argv) > 5 else "hard"
    tasks = {"hard": HARD_TASKS, "easy": EASY_TASKS, "all": EASY_TASKS + HARD_TASKS}[which]

    if not model_available(model):
        print(f"[error] model '{model}' not found in Ollama. `ollama pull {model}` first, or pass a present tag.")
        print("        present:", ", ".join(sorted({m.get('name','') for m in requests.get(f'{OLLAMA}/api/tags').json().get('models', [])})))
        sys.exit(2)

    print(f"[setup] model={model} tasks={which}({len(tasks)}) modes={modes} seeds={seeds} temp={temp} "
          f"budget={BUDGET} | {time.time()-t0:.0f}s", flush=True)

    rows = []
    for task in tasks:
        for seed in seeds:
            for mode in modes:
                r = episode(model, task, mode != "control", temp, seed * 100 + 1)
                r.update(task=task["name"], seed=seed, mode=mode)
                rows.append(r)
                print(f"  {task['name']:14s} s{seed} {mode:9s} solved={int(r['solved'])} "
                      f"loops={r['n_loops']} interv={r['n_interventions']} esc={r['n_escape']} "
                      f"steps={r['steps']} | {time.time()-t0:.0f}s", flush=True)

    import numpy as np

    def agg(mode):
        rs = [r for r in rows if r["mode"] == mode]
        return {"solve_rate": float(np.mean([r["solved"] for r in rs])),
                "loop_rate": float(np.mean([r["n_loops"] > 0 for r in rs])),
                "mean_steps": float(np.mean([r["steps"] for r in rs])),
                "n": len(rs)}

    pairs = {}
    for r in rows:
        pairs.setdefault((r["task"], r["seed"]), {})[r["mode"]] = r
    looped_fail = [(k, p) for k, p in pairs.items()
                   if "control" in p and "treatment" in p
                   and p["control"]["n_loops"] > 0 and not p["control"]["solved"]]
    recovered = sum(1 for _, p in looped_fail if p["treatment"]["solved"])

    res = {"model": model, "taskset": which, "temp": temp, "seeds": seeds, "budget": BUDGET,
           "control": agg("control"), "treatment": agg("treatment"),
           "n_control_loop_and_fail": len(looped_fail), "n_recovered_by_treatment": recovered,
           "rows": rows}
    safe = model.replace("/", "_").replace(":", "_")
    (OUT / f"outcome_ollama_{safe}.json").write_text(json.dumps(res, indent=2))
    print(f"\n[control]   {res['control']}", flush=True)
    print(f"[treatment] {res['treatment']}", flush=True)
    print(f"[recovery]  control looped+failed on {len(looped_fail)} runs; treatment then solved {recovered}", flush=True)
    print(f"[done] wrote outcome_ollama_{safe}.json | {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
