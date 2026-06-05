"""Self-contained local toy bug-fix tasks for online experiments (Phase 14).

Each task materializes a tiny git repo with a buggy module + a failing pytest, a
natural-language issue prompt, and a verify command (pytest exit 0 == solved).
These cover the experiment ladder's Level 0/1 without needing Docker or
SWE-bench. SWE-bench Verified Mini integration is a separate (heavier) path.
"""

from __future__ import annotations

import subprocess
from dataclasses import dataclass
from pathlib import Path


@dataclass
class ToyTask:
    task_id: str
    prompt: str
    files: dict[str, str]          # relative path -> content (buggy state)
    verify_cmd: str = "python3 -m pytest -q"


def _git_init(repo: Path) -> None:
    subprocess.run(["git", "-C", str(repo), "init", "-q"], check=False)
    subprocess.run(["git", "-C", str(repo), "config", "user.email", "a@b.c"], check=False)
    subprocess.run(["git", "-C", str(repo), "config", "user.name", "localguard"], check=False)
    subprocess.run(["git", "-C", str(repo), "add", "-A"], check=False)
    subprocess.run(["git", "-C", str(repo), "commit", "-q", "-m", "initial buggy state",
                    "--no-verify"], check=False)


def materialize(task: ToyTask, base_dir: Path) -> Path:
    """Write task files into ``base_dir/<task_id>`` and git-init it."""
    repo = Path(base_dir) / task.task_id
    repo.mkdir(parents=True, exist_ok=True)
    for rel, content in task.files.items():
        p = repo / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content)
    _git_init(repo)
    return repo


TASKS: dict[str, ToyTask] = {
    "off_by_one": ToyTask(
        task_id="off_by_one",
        prompt=(
            "The function `running_max` in mathutils.py is supposed to return a list where "
            "each element i is the maximum of the input list up to and including index i "
            "(a running/prefix maximum). It currently returns wrong values. Fix mathutils.py "
            "so the tests in test_mathutils.py pass."
        ),
        files={
            "mathutils.py": (
                "def running_max(xs):\n"
                "    out = []\n"
                "    m = 0  # BUG: should start from the first element, not 0\n"
                "    for x in xs:\n"
                "        if x > m:\n"
                "            m = x\n"
                "        out.append(m)\n"
                "    return out\n"
            ),
            "test_mathutils.py": (
                "from mathutils import running_max\n\n"
                "def test_positive():\n"
                "    assert running_max([1, 3, 2, 5, 4]) == [1, 3, 3, 5, 5]\n\n"
                "def test_with_negatives():\n"
                "    assert running_max([-5, -3, -4, -1]) == [-5, -3, -3, -1]\n"
            ),
        },
    ),
    "strip_prefix": ToyTask(
        task_id="strip_prefix",
        prompt=(
            "`strip_prefix(s, prefix)` in textutils.py should remove `prefix` from the start of "
            "`s` only if `s` actually starts with `prefix`, otherwise return `s` unchanged. "
            "It is buggy. Fix textutils.py so test_textutils.py passes."
        ),
        files={
            "textutils.py": (
                "def strip_prefix(s, prefix):\n"
                "    # BUG: unconditionally removes len(prefix) chars\n"
                "    return s[len(prefix):]\n"
            ),
            "test_textutils.py": (
                "from textutils import strip_prefix\n\n"
                "def test_has_prefix():\n"
                "    assert strip_prefix('foobar', 'foo') == 'bar'\n\n"
                "def test_no_prefix():\n"
                "    assert strip_prefix('foobar', 'xyz') == 'foobar'\n\n"
                "def test_empty_prefix():\n"
                "    assert strip_prefix('abc', '') == 'abc'\n"
            ),
        },
    ),
    "dedup_order": ToyTask(
        task_id="dedup_order",
        prompt=(
            "`dedup(items)` in listutils.py should remove duplicates while preserving the order "
            "of first appearance. It currently loses ordering. Fix listutils.py so "
            "test_listutils.py passes."
        ),
        files={
            "listutils.py": (
                "def dedup(items):\n"
                "    # BUG: set() does not preserve order\n"
                "    return list(set(items))\n"
            ),
            "test_listutils.py": (
                "from listutils import dedup\n\n"
                "def test_order_preserved():\n"
                "    assert dedup([3, 1, 3, 2, 1]) == [3, 1, 2]\n\n"
                "def test_strings():\n"
                "    assert dedup(['b', 'a', 'b', 'c']) == ['b', 'a', 'c']\n"
            ),
        },
    ),
}


def _simple_task(task_id, prompt, module, modcode, testcode):
    return ToyTask(task_id=task_id, prompt=prompt,
                   files={f"{module}.py": modcode, f"test_{module}.py": testcode})


# Additional small bug-fix tasks (easier on average, so the 7B model produces a
# mix of successful and failed runs — needed to measure disruption on healthy runs).
TASKS.update({
    "sum_list": _simple_task(
        "sum_list",
        "`total(xs)` in nums.py should return the sum of the list, but it's off. Fix nums.py.",
        "nums",
        "def total(xs):\n    s = 1  # BUG: should start at 0\n    for x in xs:\n        s += x\n    return s\n",
        "from nums import total\n\ndef test_basic():\n    assert total([1,2,3]) == 6\n\ndef test_empty():\n    assert total([]) == 0\n",
    ),
    "max_of_three": _simple_task(
        "max_of_three",
        "`largest(a,b,c)` in cmp.py should return the largest of three numbers. It misses a case. Fix cmp.py.",
        "cmp",
        "def largest(a, b, c):\n    if a > b:\n        return a  # BUG: ignores c\n    return b\n",
        "from cmp import largest\n\ndef test_a():\n    assert largest(3,1,2) == 3\n\ndef test_c():\n    assert largest(1,2,5) == 5\n",
    ),
    "count_char": _simple_task(
        "count_char",
        "`count(s, ch)` in counts.py should count occurrences of `ch` in `s`, case-insensitively. Fix counts.py.",
        "counts",
        "def count(s, ch):\n    # BUG: case-sensitive\n    return s.count(ch)\n",
        "from counts import count\n\ndef test_ci():\n    assert count('Banana', 'a') == 3\n\ndef test_upper():\n    assert count('AaAa', 'a') == 4\n",
    ),
    "celsius": _simple_task(
        "celsius",
        "`to_f(c)` in temp.py converts Celsius to Fahrenheit. The formula is wrong. Fix temp.py.",
        "temp",
        "def to_f(c):\n    return c * 9 / 5  # BUG: missing + 32\n",
        "from temp import to_f\n\ndef test_zero():\n    assert to_f(0) == 32\n\ndef test_100():\n    assert to_f(100) == 212\n",
    ),
    "last_word": _simple_task(
        "last_word",
        "`last_word(s)` in words.py should return the last whitespace-separated word. It returns the first. Fix words.py.",
        "words",
        "def last_word(s):\n    return s.split()[0]  # BUG: should be [-1]\n",
        "from words import last_word\n\ndef test_two():\n    assert last_word('hello world') == 'world'\n\ndef test_many():\n    assert last_word('a b c d') == 'd'\n",
    ),
    "clamp": _simple_task(
        "clamp",
        "`clamp(x, lo, hi)` in clampmod.py should restrict x to [lo, hi]. The bounds are swapped. Fix clampmod.py.",
        "clampmod",
        "def clamp(x, lo, hi):\n    return max(hi, min(lo, x))  # BUG: lo/hi swapped\n",
        "from clampmod import clamp\n\ndef test_high():\n    assert clamp(10, 0, 5) == 5\n\ndef test_low():\n    assert clamp(-3, 0, 5) == 0\n\ndef test_mid():\n    assert clamp(3, 0, 5) == 3\n",
    ),
    "sum_evens": _simple_task(
        "sum_evens",
        "`sum_evens(xs)` in evens.py should sum only the even numbers. It includes odds. Fix evens.py.",
        "evens",
        "def sum_evens(xs):\n    return sum(x for x in xs if x % 2 == 1)  # BUG: keeps odds\n",
        "from evens import sum_evens\n\ndef test_mixed():\n    assert sum_evens([1,2,3,4]) == 6\n\ndef test_none():\n    assert sum_evens([1,3,5]) == 0\n",
    ),
})

def _t(tid, prompt, mod, code, test):
    return _simple_task(tid, prompt, mod, code, test)


# 15 more small bug-fix tasks for a larger online sample (Level 3).
TASKS.update({
    "is_even": _t("is_even", "`is_even(n)` in parity.py should return True iff n is even. It's inverted. Fix parity.py.",
        "parity", "def is_even(n):\n    return n % 2 == 1  # BUG: that's odd\n",
        "from parity import is_even\n\ndef test_e():\n    assert is_even(4)\n\ndef test_o():\n    assert not is_even(3)\n"),
    "average": _t("average", "`average(xs)` in avg.py should return the mean. It's off. Fix avg.py.",
        "avg", "def average(xs):\n    return sum(xs) // len(xs)  # BUG: integer division\n",
        "from avg import average\n\ndef test_mean():\n    assert average([1,2]) == 1.5\n\ndef test_int():\n    assert average([2,4,6]) == 4\n"),
    "reverse_str": _t("reverse_str", "`rev(s)` in rev.py should reverse a string. It drops a char. Fix rev.py.",
        "rev", "def rev(s):\n    return s[len(s)-1:0:-1]  # BUG: drops first char\n",
        "from rev import rev\n\ndef test_r():\n    assert rev('abc') == 'cba'\n\ndef test_one():\n    assert rev('x') == 'x'\n"),
    "factorial": _t("factorial", "`fact(n)` in fact.py computes n!. The base case is wrong. Fix fact.py.",
        "fact", "def fact(n):\n    if n <= 1:\n        return 0  # BUG: 0! and 1! are 1\n    return n * fact(n-1)\n",
        "from fact import fact\n\ndef test_z():\n    assert fact(0) == 1\n\ndef test_5():\n    assert fact(5) == 120\n"),
    "min_list": _t("min_list", "`smallest(xs)` in mn.py should return the minimum. It returns the max. Fix mn.py.",
        "mn", "def smallest(xs):\n    m = xs[0]\n    for x in xs:\n        if x > m:  # BUG: wrong comparison\n            m = x\n    return m\n",
        "from mn import smallest\n\ndef test_m():\n    assert smallest([3,1,2]) == 1\n\ndef test_neg():\n    assert smallest([-1,-5,0]) == -5\n"),
    "count_vowels": _t("count_vowels", "`vowels(s)` in vw.py should count vowels (a,e,i,o,u), case-insensitive. It misses uppercase. Fix vw.py.",
        "vw", "def vowels(s):\n    return sum(1 for c in s if c in 'aeiou')  # BUG: case-sensitive\n",
        "from vw import vowels\n\ndef test_l():\n    assert vowels('hello') == 2\n\ndef test_u():\n    assert vowels('AEIou') == 5\n"),
    "remove_spaces": _t("remove_spaces", "`despace(s)` in ds.py should remove ALL spaces. It removes only one. Fix ds.py.",
        "ds", "def despace(s):\n    return s.replace(' ', '', 1)  # BUG: only first\n",
        "from ds import despace\n\ndef test_a():\n    assert despace('a b c') == 'abc'\n\ndef test_n():\n    assert despace('xy') == 'xy'\n"),
    "is_sorted": _t("is_sorted", "`is_sorted(xs)` in srt.py should return True iff non-decreasing. Comparison is wrong. Fix srt.py.",
        "srt", "def is_sorted(xs):\n    return all(xs[i] > xs[i+1] for i in range(len(xs)-1))  # BUG\n",
        "from srt import is_sorted\n\ndef test_y():\n    assert is_sorted([1,2,2,3])\n\ndef test_n():\n    assert not is_sorted([3,1])\n"),
    "square_list": _t("square_list", "`squares(xs)` in sq.py should square each element. It doesn't. Fix sq.py.",
        "sq", "def squares(xs):\n    return [x for x in xs]  # BUG: not squared\n",
        "from sq import squares\n\ndef test_s():\n    assert squares([1,2,3]) == [1,4,9]\n\ndef test_e():\n    assert squares([]) == []\n"),
    "is_prime": _t("is_prime", "`is_prime(n)` in pr.py should return True for primes. It wrongly calls 1 prime. Fix pr.py.",
        "pr", "def is_prime(n):\n    if n < 2:\n        return True  # BUG: 0 and 1 are not prime\n    for d in range(2, n):\n        if n % d == 0:\n            return False\n    return True\n",
        "from pr import is_prime\n\ndef test_one():\n    assert not is_prime(1)\n\ndef test_seven():\n    assert is_prime(7)\n"),
    "swap_case": _t("swap_case", "`swapc(s)` in sc.py should swap case of each letter. It uppercases instead. Fix sc.py.",
        "sc", "def swapc(s):\n    return s.upper()  # BUG: should swap case\n",
        "from sc import swapc\n\ndef test_m():\n    assert swapc('aB') == 'Ab'\n\ndef test_w():\n    assert swapc('Hello') == 'hELLO'\n"),
    "find_first": _t("find_first", "`first_index(xs, t)` in fi.py should return the index of the first t, or -1. Off-by-one. Fix fi.py.",
        "fi", "def first_index(xs, t):\n    for i in range(len(xs)):\n        if xs[i] == t:\n            return i + 1  # BUG: off by one\n    return -1\n",
        "from fi import first_index\n\ndef test_f():\n    assert first_index([5,6,7], 6) == 1\n\ndef test_m():\n    assert first_index([1,2], 9) == -1\n"),
    "title_case": _t("title_case", "`titlecase(s)` in tc.py should capitalize the first letter of each word. It lowercases. Fix tc.py.",
        "tc", "def titlecase(s):\n    return s.lower()  # BUG\n",
        "from tc import titlecase\n\ndef test_t():\n    assert titlecase('hello world') == 'Hello World'\n\ndef test_o():\n    assert titlecase('a b') == 'A B'\n"),
    "gcd": _t("gcd", "`gcd(a,b)` in g.py should return the greatest common divisor. Base case wrong. Fix g.py.",
        "g", "def gcd(a, b):\n    while b:\n        a, b = b, a % b\n    return b  # BUG: should return a\n",
        "from g import gcd\n\ndef test_a():\n    assert gcd(12, 8) == 4\n\ndef test_b():\n    assert gcd(7, 1) == 1\n"),
    "list_sum_pos": _t("list_sum_pos", "`sum_pos(xs)` in spz.py should sum only positive numbers. It includes negatives. Fix spz.py.",
        "spz", "def sum_pos(xs):\n    return sum(x for x in xs if x != 0)  # BUG: keeps negatives\n",
        "from spz import sum_pos\n\ndef test_m():\n    assert sum_pos([1,-2,3]) == 4\n\ndef test_n():\n    assert sum_pos([-1,-2]) == 0\n"),
})

_L2 = ["off_by_one", "strip_prefix", "dedup_order", "sum_list", "max_of_three",
       "count_char", "celsius", "last_word", "clamp", "sum_evens"]
_L3_EXTRA = ["is_even", "average", "reverse_str", "factorial", "min_list", "count_vowels",
             "remove_spaces", "is_sorted", "square_list", "is_prime", "swap_case",
             "find_first", "title_case", "gcd", "list_sum_pos"]

LEVELS: dict[int, list[str]] = {
    0: ["off_by_one"],
    1: ["off_by_one", "strip_prefix", "dedup_order"],
    2: _L2,
    3: _L2 + _L3_EXTRA,  # 25 tasks for the larger online sample
}


def tasks_for_level(level: int) -> list[ToyTask]:
    ids = LEVELS.get(level, LEVELS[1])
    return [TASKS[i] for i in ids]
