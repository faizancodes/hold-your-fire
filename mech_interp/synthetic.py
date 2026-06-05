"""Synthetic coding-agent transcripts for the looping-circuit study (Phase 2 MVP).

Each scenario is a bug-fix task. We can render it in two length-matched conditions:
  LOOP     : the agent re-issues the SAME command C, K times, each returning the SAME
             failing observation -> decision point (a "tight loop").
  PROGRESS : the agent issues K DISTINCT productive commands -> decision point.
The only systematic difference at K>=2 is *repetition*, not context length, so a probe
that separates LOOP from PROGRESS is detecting "I've been here before", not length.

Surface details (files, symbols, error types, commands) vary across scenarios so a
probe/steering-vector can't latch onto specific tokens.
"""
from __future__ import annotations

import random

# slot pools -------------------------------------------------------------------
MODULES = ["client", "parser", "cache", "auth", "router", "scheduler", "session",
           "loader", "encoder", "worker", "config", "registry", "stream", "pool"]
ERRORS = [
    ("TimeoutError", "connection timed out"),
    ("AssertionError", "assert 3 == 4"),
    ("KeyError", "KeyError: 'token'"),
    ("ValueError", "invalid literal for int()"),
    ("AttributeError", "'NoneType' object has no attribute 'send'"),
    ("IndexError", "list index out of range"),
]
RUN_TMPL = [
    "pytest tests/test_{m}.py",
    "python -m pytest tests/test_{m}.py -q",
    "python -m pytest tests/test_{m}.py::test_main",
    "python run_{m}.py",
]
THOUGHTS_LOOP = [
    "Let me run the tests again.",
    "Re-running to check.",
    "Let me try running it once more.",
    "Running the test suite.",
]
# distinct productive actions for the PROGRESS condition (and as novel alternatives)
def productive_rounds(m, sym, rng):
    pool = [
        (f"Let me read the source.", f"cat src/{m}.py", f"def {sym}(self):\n    ...  # {m} logic"),
        (f"Search for the symbol.", f"grep -rn {sym} src/", f"src/{m}.py:42: def {sym}(self):"),
        (f"Look at the test.", f"cat tests/test_{m}.py", f"def test_main():\n    assert {m}.{sym}() == 4"),
        (f"Check the imports.", f"head -20 src/{m}.py", f"import os, sys\nfrom .util import helper"),
        (f"Inspect the helper.", f"grep -rn helper src/", f"src/util.py:8: def helper(x):"),
        (f"Apply a fix.", f"sed -i 's/return 3/return 4/' src/{m}.py", "edited src/{m}.py".format(m=m)),
        (f"List the directory.", f"ls src/", f"{m}.py  util.py  __init__.py"),
        (f"Show git diff.", f"git diff src/{m}.py", "+    return 4\n-    return 3"),
    ]
    rng.shuffle(pool)
    return pool


# distinct *failing* fix attempts for the VARIED-FAIL control (different command each
# turn, but the SAME failing observation -> "failing repeatedly without repeating").
def varied_fail_rounds(m, rng):
    pool = [
        ("Try bumping the value.", f"sed -i 's/=5/=10/' src/{m}.py"),
        ("Try another value.", f"sed -i 's/=10/=20/' src/{m}.py"),
        ("Reset and retry import.", f"git checkout src/{m}.py && python -c 'import {m}'"),
        ("Add a guard.", f"sed -i 's/return/if x: return/' src/{m}.py"),
        ("Try a cast.", f"sed -i 's/int(x)/int(x or 0)/' src/{m}.py"),
        ("Patch the helper.", f"sed -i 's/helper()/helper(0)/' src/util.py"),
        ("Wrap in try.", f"sed -i 's/^/try: /' src/{m}.py"),
        ("Flip a flag.", f"sed -i 's/False/True/' src/{m}.py"),
    ]
    rng.shuffle(pool)
    return pool


class Scenario:
    def __init__(self, sid, system, task, c_thought, c_cmd, obs, novel_alts, prog, vfail):
        self.sid = sid
        self.system = system
        self.task = task
        self.c_thought = c_thought          # thought used in the loop
        self.c_cmd = c_cmd                  # the looped command (no backticks)
        self.obs = obs                      # the repeated failing observation
        self.novel_alts = novel_alts        # productive alternative commands (no backticks)
        self.prog = prog                    # list[(thought, cmd, obs)] for PROGRESS
        self.vfail = vfail                  # list[(thought, cmd)] for VARIED-FAIL (same obs)

    # ---- message builders ----
    def _assistant(self, thought, cmd):
        return {"role": "assistant", "content": f"{thought}\n`{cmd}`"}

    def loop_messages(self, k: int):
        msgs = [{"role": "system", "content": self.system},
                {"role": "user", "content": self.task}]
        for _ in range(k):
            msgs.append(self._assistant(self.c_thought, self.c_cmd))
            msgs.append({"role": "user", "content": self.obs})
        return msgs

    def progress_messages(self, k: int):
        msgs = [{"role": "system", "content": self.system},
                {"role": "user", "content": self.task}]
        for i in range(k):
            th, cmd, obs = self.prog[i % len(self.prog)]
            msgs.append(self._assistant(th, cmd))
            msgs.append({"role": "user", "content": obs})
        return msgs

    def varied_fail_messages(self, k: int):
        """K DIFFERENT commands, each returning the SAME failing observation as the loop."""
        msgs = [{"role": "system", "content": self.system},
                {"role": "user", "content": self.task}]
        for i in range(k):
            th, cmd = self.vfail[i % len(self.vfail)]
            msgs.append(self._assistant(th, cmd))
            msgs.append({"role": "user", "content": self.obs})
        return msgs


SYSTEMS = [
    "You are a coding agent fixing a failing test. Each turn output a short thought and one shell command in backticks.",
    "You are an autonomous software engineer. Respond with a brief plan and a single shell command in backticks.",
    "You fix bugs by running commands. Each step: one line of reasoning, then one command in backticks.",
]


def make_scenarios(n: int, seed: int = 0) -> list[Scenario]:
    rng = random.Random(seed)
    out = []
    for i in range(n):
        m = MODULES[i % len(MODULES)]
        err, errmsg = rng.choice(ERRORS)
        runcmd = rng.choice(RUN_TMPL).format(m=m)
        sym = rng.choice(["compute", "connect", "resolve", "parse", "load", "encode", "fetch"])
        system = rng.choice(SYSTEMS)
        task = (f"The test tests/test_{m}.py fails with {err}: {errmsg}. "
                f"Find and fix the bug in src/{m}.py.")
        c_thought = rng.choice(THOUGHTS_LOOP)
        obs = f"1 failed in 0.3s - {err}: {errmsg}"
        prog = productive_rounds(m, sym, rng)
        vfail = varied_fail_rounds(m, rng)
        novel_alts = [f"cat src/{m}.py", f"grep -rn {sym} src/", f"cat tests/test_{m}.py",
                      f"sed -i 's/return 3/return 4/' src/{m}.py"]
        out.append(Scenario(f"s{i:03d}_{m}", system, task, c_thought, runcmd, obs, novel_alts, prog, vfail))
    return out
