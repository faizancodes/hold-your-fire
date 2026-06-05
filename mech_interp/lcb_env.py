"""Sandboxed executor for LiveCodeBench stdin solutions.

The model's program is written to a temp dir and run once per test case with the test input on
stdin; stdout is compared to the expected output after whitespace normalization. Safety: isolated
temp dir, hard wall-clock timeout, no shell (argv list), and we only ever run the model's own
competitive-programming code (standard practice for code-eval harnesses like HumanEval).
"""
from __future__ import annotations

import shutil
import subprocess
import sys
import tempfile
from pathlib import Path


def _norm(s: str) -> str:
    return "\n".join(line.rstrip() for line in (s or "").strip("\n").split("\n")).strip()


def run_tests(code: str, tests: list[dict], timeout: float = 8.0,
              stop_on_fail: bool = True, max_tests: int | None = None) -> dict:
    """Run `code` against `tests`. Returns {passed, total, n_ok, first_fail}.

    first_fail (when present) = {input, expected, got, err} for the first failing test — used as
    the agent's feedback message.
    """
    d = tempfile.mkdtemp(prefix="lcb_")
    (Path(d) / "sol.py").write_text(code)
    tests = tests[:max_tests] if max_tests else tests
    n_ok, first_fail = 0, None
    try:
        for t in tests:
            try:
                p = subprocess.run([sys.executable, "sol.py"], input=t["input"], cwd=d,
                                   timeout=timeout, capture_output=True, text=True)
                ok = p.returncode == 0 and _norm(p.stdout) == _norm(t["output"])
                got, err = p.stdout, ("" if p.returncode == 0 else (p.stderr or "")[:300])
            except subprocess.TimeoutExpired:
                ok, got, err = False, "", "(timeout)"
            except Exception as e:
                ok, got, err = False, "", str(e)[:160]
            if ok:
                n_ok += 1
            elif first_fail is None:
                first_fail = {"input": t["input"], "expected": t["output"], "got": got, "err": err}
                if stop_on_fail:
                    break
    finally:
        shutil.rmtree(d, ignore_errors=True)
    total = len(tests)
    return {"passed": n_ok == total and total > 0, "n_ok": n_ok, "total": total, "first_fail": first_fail}


def solved(code: str, problem: dict, n_private: int = 15) -> bool:
    """Solve label: passes ALL public tests AND a sample of private tests."""
    if not run_tests(code, problem["public"], stop_on_fail=True)["passed"]:
        return False
    priv = problem["private"][:n_private]
    return run_tests(code, priv, stop_on_fail=True)["passed"] if priv else True


if __name__ == "__main__":
    # self-test: a correct solution must pass, a wrong one must fail
    from mech_interp.lcb_data import load_problems
    p = load_problems(difficulties=("easy",))[0]
    print(f"self-test on {p['id']}: {p['title']}")
    print("public[0] input:", repr(p["public"][0]["input"]), "expected:", repr(p["public"][0]["output"]))
    # the first easy problem (abc387_b '9x9 Sum'): N -> sum over i,j in 1..9 of i*j where i*j != N
    correct = ("N=int(input())\n"
               "print(sum(i*j for i in range(1,10) for j in range(1,10) if i*j!=N))\n")
    wrong = "N=int(input())\nprint(0)\n"
    print("correct ->", run_tests(correct, p["public"]), "| solved:", solved(correct, p))
    print("wrong   ->", run_tests(wrong, p["public"]))
