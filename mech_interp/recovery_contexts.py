"""A battery of *matched* 'stuck' decision points for hunting the recovery direction.

Each context is a coding-agent transcript in a bug-fix task where one command has just been
repeated K times with an identical failure, ending at the decision prompt `$ `. The situation
is held structurally constant (looped + still failing) while the SURFACE varies (command
family, file names, search terms, error text), so a direction that predicts whether the model
*recovers* (does something productive) vs *persists* (repeats) cannot be just a surface cue.

Every context carries:
  - `looped`  : the repeated command (the persist action),
  - `recover` : productive next moves that read evidence / change strategy (the right thing),
  - `family`  : for scenario-grouped CV (no family leaks across train/test),
  - `text`    : the transcript ending at `$ `.

`classify(emitted, looped)` -> 'persist' | 'recover' | 'other' labels a generated command.

This is deliberately NOT the old loop-vs-progress contrast: persist vs recover are *both*
inside the same stuck situation, differing only by the decision. That is the contrast whose
causal axis is the recovery lever.
"""
from __future__ import annotations

import re

K_REPEAT = 3
PRODUCTIVE = {"cat", "ls", "grep", "head", "tail", "open", "python", "python3",
              "sed", "find", "find_file", "diff", "wc", "less", "nl"}


def _norm(c: str) -> str:
    return re.sub(r"\s+", " ", (c or "").strip())


def classify(emitted: str, looped: str) -> str:
    """persist = repeats the failing command; recover = a productive *different* command;
    other = empty / non-productive / gibberish."""
    e, l = _norm(emitted), _norm(looped)
    if not e:
        return "other"
    # strip a leading "$ " or backticks the model may echo
    e = re.sub(r"^\$\s*", "", e).strip("`").strip()
    if e == l or e.startswith(l):
        return "persist"
    verb = e.split()[0] if e.split() else ""
    if verb in PRODUCTIVE:
        return "recover"
    return "other"


def _stuck(intro: str, cmd: str, fail: str, k: int = K_REPEAT) -> str:
    body = "".join(f"$ {cmd}\n{fail}\n" for _ in range(k))
    return f"{intro}\n{body}$ "


# surface variants: (file, sol, test, term, missing) tuples drive diverse names
_VARIANTS = [
    {"file": "app.py", "term": "SecretStr", "missing": "config.yaml", "pat": "TODO", "detail": "expected 5 got 6"},
    {"file": "sol.py", "term": "parse_date", "missing": "utils.py", "pat": "import os", "detail": "expected 'JS' got 'Jo'"},
    {"file": "models.py", "term": "UserSchema", "missing": "schema.json", "pat": "def save", "detail": "KeyError: 'plum'"},
    {"file": "main.py", "term": "load_cfg", "missing": "settings.ini", "pat": "raise ", "detail": "expected 15 got 10"},
    {"file": "core.py", "term": "Handler", "missing": "routes.py", "pat": "return None", "detail": "expected 12 got 6"},
    {"file": "lib.py", "term": "to_celsius", "missing": "data.csv", "pat": "class ", "detail": "expected 212 got 32"},
    {"file": "service.py", "term": "retry", "missing": "client.py", "pat": "async def", "detail": "expected 13 got 5"},
    {"file": "db.py", "term": "connect", "missing": "pool.py", "pat": "yield", "detail": "expected [2, 4] got [1, 3]"},
    {"file": "api.py", "term": "validate", "missing": "auth.py", "pat": "assert ", "detail": "expected True got False"},
    {"file": "graph.py", "term": "bfs", "missing": "node.py", "pat": "while ", "detail": "expected 24 got 6"},
    {"file": "parse.py", "term": "tokenize", "missing": "lexer.py", "pat": "for ", "detail": "expected 3 got 2"},
    {"file": "cache.py", "term": "evict", "missing": "store.py", "pat": "lru", "detail": "IndexError: list index"},
]


def build_battery() -> list[dict]:
    ctx = []
    for i, v in enumerate(_VARIANTS):
        intro = (f"# You are a shell coding agent fixing a bug in {v['file']} so test.py passes. "
                 f"At each turn reply with EXACTLY ONE shell command and nothing else (no prose).")

        # family: failed search
        ctx.append({"family": "search", "id": f"search_{i}",
                    "looped": f"find_file {v['term']}",
                    "recover": [f"cat {v['file']}", "ls", f"grep -rn {v['term']} ."],
                    "text": _stuck(intro, f"find_file {v['term']}", "No matches found")})

        # family: failed grep (no output)
        ctx.append({"family": "grep", "id": f"grep_{i}",
                    "looped": f"grep {v['pat']} {v['file']}",
                    "recover": [f"cat {v['file']}", f"grep -rn {v['pat']} .", "ls"],
                    "text": _stuck(intro, f"grep {v['pat']} {v['file']}", "(no output)")})

        # family: test rerun, identical failure
        ctx.append({"family": "test", "id": f"test_{i}",
                    "looped": "python test.py",
                    "recover": [f"cat {v['file']}", "cat test.py", "ls"],
                    "text": _stuck(intro, "python test.py", f"AssertionError: {v['detail']}")})

        # family: cat a missing file
        ctx.append({"family": "cat_missing", "id": f"cat_{i}",
                    "looped": f"cat {v['missing']}",
                    "recover": ["ls", f"find . -name '{v['missing']}'", f"cat {v['file']}"],
                    "text": _stuck(intro, f"cat {v['missing']}", f"cat: {v['missing']}: No such file or directory")})

        # family: a no-op edit that never fixes the test
        ctx.append({"family": "edit_noop", "id": f"edit_{i}",
                    "looped": f"sed -i 's/{v['term']}/{v['pat']}/' {v['file']}",
                    "recover": [f"cat {v['file']}", "python test.py", f"grep {v['term']} {v['file']}"],
                    "text": _stuck(intro, f"sed -i 's/{v['term']}/{v['pat']}/' {v['file']}",
                                   "(no change to test result; still failing)")})
    return ctx


if __name__ == "__main__":
    b = build_battery()
    fams = {}
    for c in b:
        fams[c["family"]] = fams.get(c["family"], 0) + 1
    print(f"{len(b)} contexts across families: {fams}\n")
    ex = b[0]
    print("--- example context ---")
    print(ex["text"])
    print(f"\nlooped(persist): {ex['looped']!r}   recover: {ex['recover']}")
    # classifier sanity
    cases = [("find_file SecretStr", "persist"), ("cat app.py", "recover"),
             ("ls", "recover"), ("", "other"), ("blah blah", "other"),
             ("`find_file SecretStr`", "persist")]
    print("\n--- classifier sanity ---")
    for emit, want in cases:
        got = classify(emit, "find_file SecretStr")
        print(f"  {emit!r:28} -> {got:8} {'OK' if got == want else 'WANT ' + want}")
