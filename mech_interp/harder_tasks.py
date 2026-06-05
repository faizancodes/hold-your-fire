"""Harder, *recovery-requiring* bug-fix tasks for the capable-model outcome experiment.

The 1.5B Tier-2 tasks (agent_env.TASKS) are single-token edits a capable model solves
outright — so they never loop, and can't test "does breaking the loop help a model that
CAN recover?". These tasks are still solvable with `cat` + `sed` in the same sandbox, but
each one:
  * has the bug in a *non-obvious* place (a helper, a loop bound, a base case), so the fix
    requires READING the source rather than editing the test call (the exact failure the
    1.5B couldn't escape);
  * has a tempting-but-wrong first move (edit the test, guess a sed string that no-ops,
    fix only one of two bugs) that can send even a capable model into a loop.

Same dict schema as agent_env.TASKS: {name, sol, test}. Success = `python test.py` exits 0
with PASS. Every task is verified solvable below.
"""
from __future__ import annotations

HARD_TASKS = [
    # bug is in the helper, not the obvious function -> must read source
    {"name": "helper_bug",
     "sol": "def _scale(x):\n    return x * 2\ndef triple(x):\n    return _scale(x)\n",
     "test": "from sol import triple\nassert triple(4) == 12, triple(4)\nprint('PASS')\n"},

    # off-by-one loop bound; editing the test is the tempting wrong move
    {"name": "off_by_one",
     "sol": "def sum_to(n):\n    s = 0\n    for i in range(1, n):\n        s += i\n    return s\n",
     "test": "from sol import sum_to\nassert sum_to(5) == 15, sum_to(5)\nprint('PASS')\n"},

    # two independent bugs: fixing only one leaves the test failing (loop pressure)
    {"name": "two_bugs",
     "sol": "def mean(xs):\n    total = 0\n    for x in xs:\n        total = x\n    return total / 2\n",
     "test": "from sol import mean\nassert mean([2, 4, 6]) == 4, mean([2, 4, 6])\nprint('PASS')\n"},

    # KeyError symptom; fix is .get(item, 0), not adding the obvious-looking key
    {"name": "dict_missing",
     "sol": ("def price(item):\n    table = {'apple': 3, 'pear': 2}\n    return table[item]\n"
             "def basket(items):\n    return sum(price(i) for i in items)\n"),
     "test": ("from sol import basket\nassert basket(['apple', 'pear', 'plum']) == 5, "
              "basket(['apple', 'pear', 'plum'])\nprint('PASS')\n")},

    # subtle wrong recursive step (n-3 vs n-2)
    {"name": "recursion_step",
     "sol": "def fib(n):\n    if n < 2:\n        return n\n    return fib(n - 1) + fib(n - 3)\n",
     "test": "from sol import fib\nassert fib(7) == 13, fib(7)\nprint('PASS')\n"},

    # wrong index into the second word; reads fine at a glance
    {"name": "initials",
     "sol": "def initials(name):\n    parts = name.split()\n    return parts[0][0] + parts[0][1]\n",
     "test": "from sol import initials\nassert initials('John Smith') == 'JS', initials('John Smith')\nprint('PASS')\n"},

    # name says evens, code returns odds (== 1 should be == 0)
    {"name": "evens_filter",
     "sol": "def evens(xs):\n    return [x for x in xs if x % 2 == 1]\n",
     "test": "from sol import evens\nassert evens([1, 2, 3, 4]) == [2, 4], evens([1, 2, 3, 4])\nprint('PASS')\n"},

    # swapped ratio, unique to c_to_f (must target the right line)
    {"name": "temp_convert",
     "sol": "def c_to_f(c):\n    return c * 5 / 9 + 32\ndef f_to_c(f):\n    return (f - 32) * 5 / 9\n",
     "test": "from sol import c_to_f\nassert c_to_f(100) == 212, c_to_f(100)\nprint('PASS')\n"},
]


if __name__ == "__main__":
    # self-check: apply the intended fix to each and confirm the test passes.
    import subprocess, sys, tempfile
    from pathlib import Path
    fixes = {
        "helper_bug": ("x * 2", "x * 3"),
        "off_by_one": ("range(1, n)", "range(1, n + 1)"),
        "two_bugs": [("total = x", "total += x"), ("/ 2", "/ len(xs)")],
        "dict_missing": ("table[item]", "table.get(item, 0)"),
        "recursion_step": ("fib(n - 3)", "fib(n - 2)"),
        "initials": ("parts[0][1]", "parts[1][0]"),
        "evens_filter": ("% 2 == 1", "% 2 == 0"),
        "temp_convert": ("c * 5 / 9", "c * 9 / 5"),
    }
    ok = 0
    for t in HARD_TASKS:
        d = tempfile.mkdtemp()
        sol = t["sol"]
        edits = fixes[t["name"]]
        for old, new in (edits if isinstance(edits, list) else [edits]):
            assert old in sol, f"{t['name']}: fix string {old!r} not in sol"
            sol = sol.replace(old, new)
        (Path(d) / "sol.py").write_text(sol)
        (Path(d) / "test.py").write_text(t["test"])
        p = subprocess.run([sys.executable, "test.py"], cwd=d, capture_output=True, text=True)
        passed = p.returncode == 0 and "PASS" in (p.stdout + p.stderr)
        ok += passed
        print(f"{t['name']:16s} {'SOLVABLE' if passed else 'BROKEN: ' + (p.stdout + p.stderr)[:80]}")
    print(f"\n{ok}/{len(HARD_TASKS)} tasks verified solvable")
