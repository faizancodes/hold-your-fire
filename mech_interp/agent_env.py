"""A tiny, real, sandboxed shell environment + single-edit bug-fix tasks for the
recoverable-loop outcome experiment.

Safety: commands run in a per-task temp dir, first token must be whitelisted, dangerous
substrings are blocked, and each command has a wall-clock timeout. `sed -i 's/OLD/NEW/[g]'`
is executed PORTABLY in-process (macOS BSD sed silently fails on GNU syntax), so the
environment is never itself the cause of a loop.
"""
from __future__ import annotations

import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

PYTHON = sys.executable  # the real interpreter (the subprocess /bin/sh has no `python`)


def _route_python(cmd: str) -> str:
    tok = cmd.split()
    if tok and tok[0] in ("python", "python3"):
        return PYTHON + cmd[len(tok[0]):]
    return cmd

ALLOWED = {"cat", "ls", "grep", "sed", "python", "python3", "head", "tail", "pwd", "wc", "diff", "find", "echo"}
BLOCK = ["rm ", "rm-", "sudo", "curl", "wget", "dd ", "mkfs", "chmod", "chown", ">>", ">", "&&", ";", "|", "`", "$(", "..", "/etc", " ~", "rmdir", "mv "]

# ---- tasks: each = single-edit bug; success = `python test.py` exits 0 -------------
TASKS = [
    {"name": "add_op",
     "sol": "def add(a, b):\n    return a - b\n",
     "test": "from sol import add\nassert add(2, 3) == 5, add(2, 3)\nprint('PASS')\n"},
    {"name": "double",
     "sol": "def double(x):\n    return x\n",
     "test": "from sol import double\nassert double(4) == 8, double(4)\nprint('PASS')\n"},
    {"name": "greet_typo",
     "sol": "def greet(name):\n    return 'Hi ' + nm\n",
     "test": "from sol import greet\nassert greet('Bob') == 'Hi Bob', greet('Bob')\nprint('PASS')\n"},
    {"name": "last_index",
     "sol": "def last(xs):\n    return xs[len(xs)]\n",
     "test": "from sol import last\nassert last([1, 2, 3]) == 3, last([1, 2, 3])\nprint('PASS')\n"},
    {"name": "is_pos",
     "sol": "def is_pos(x):\n    return x < 0\n",
     "test": "from sol import is_pos\nassert is_pos(5) and not is_pos(-1)\nprint('PASS')\n"},
    {"name": "factorial",
     "sol": "def fact(n):\n    r = 1\n    for i in range(1, n):\n        r *= i\n    return r\n",
     "test": "from sol import fact\nassert fact(4) == 24, fact(4)\nprint('PASS')\n"},
]


def setup_task(task) -> str:
    d = tempfile.mkdtemp(prefix=f"looptask_{task['name']}_")
    (Path(d) / "sol.py").write_text(task["sol"])
    (Path(d) / "test.py").write_text(task["test"])
    return d


def cleanup(d):
    shutil.rmtree(d, ignore_errors=True)


def _portable_sed(cmd: str, cwd: str) -> str | None:
    """Execute `sed -i [''] 's<d>OLD<d>NEW<d>[g]' FILE` in Python (literal replace). None if not this form."""
    m = re.match(r"""\s*sed\s+-i\s+(?:''\s+|""\s+)?(['"])s(.)(.*?)\2(.*?)\2(g?)\1\s+(\S+)\s*$""", cmd)
    if not m:
        return None
    _, _delim, old, new, flag, fname = m.groups()
    p = Path(cwd) / fname
    if not p.exists():
        return f"sed: {fname}: No such file"
    text = p.read_text()
    if flag == "g":
        out = text.replace(old, new)
    else:  # first occurrence per line (BRE-less approximation)
        out = "\n".join(ln.replace(old, new, 1) for ln in text.split("\n"))
    p.write_text(out)
    return ""  # sed is silent on success


def run_cmd(cmd: str, cwd: str, timeout: int = 8) -> str:
    cmd = cmd.strip()
    if not cmd:
        return "(no command)"
    # handle sed -i portably FIRST (pure in-process string replace -> safe, and the s/// body
    # may legitimately contain < > etc. that the shell-block list would reject)
    if cmd.startswith("sed -i"):
        r = _portable_sed(cmd, cwd)
        if r is not None:
            return r or "(edited)"
    low = cmd.lower()
    if any(b in low for b in BLOCK):
        return "command not allowed"
    if cmd.split()[0] not in ALLOWED:
        return f"command not allowed: {cmd.split()[0]}"
    try:
        p = subprocess.run(_route_python(cmd), shell=True, cwd=cwd, timeout=timeout,
                           capture_output=True, text=True)
        out = (p.stdout + p.stderr).strip()
        return out[:400] if out else "(no output)"
    except subprocess.TimeoutExpired:
        return "(timed out)"
    except Exception as e:
        return f"(error: {e})"


def check_solved(task, cwd: str) -> bool:
    try:
        p = subprocess.run([PYTHON, "test.py"], cwd=cwd, timeout=8,
                           capture_output=True, text=True)
        return p.returncode == 0 and "PASS" in (p.stdout + p.stderr)
    except Exception:
        return False
