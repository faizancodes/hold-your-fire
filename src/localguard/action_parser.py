"""Deterministic, regex-based classification of agent actions (Phase 3).

Everything here is intentionally rule-based (no ML, no LLM) so that feature
extraction is fast, reproducible, and auditable. The classifier inspects only the
*action* text (the command the agent ran) and the *observation* text (what came
back); it never sees the task target or final outcome.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

# ---------------------------------------------------------------------------
# Command-family patterns. Order of the FAMILIES list below encodes priority
# when a command matches more than one family (e.g. ``python -m pytest`` is a
# test, not an "other" python invocation; ``sed -i`` edits while ``sed -n`` reads).
# ---------------------------------------------------------------------------
SUBMIT_PATTERNS = [
    r"COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT",
    r"\bsubmit\b",
    r"\bend_task\b",
]
TEST_PATTERNS = [
    r"\bpython\s+-m\s+pytest\b",
    r"\bpytest\b",
    r"\bpy\.test\b",
    r"\bunittest\b",
    r"\btox\b",
    r"\bnox\b",
    r"\bmake\s+test\b",
    # SWE-agent reproduction / verification scripts: running them IS the agent's
    # behavioral test loop. Match repro/test-style script names only (avoid bare
    # words like "ok"/"fail" that would catch unrelated files such as tokens.py).
    r"\bpython3?\s+(?:[\w./-]*/)?(?:reproduce|repro|reproduce_bug|run_tests?|"
    r"verify\w*|check_\w*|mre|minimal_?repro)[\w./-]*\.py\b",
    r"\bpython3?\s+(?:[\w./-]*/)?test_[\w./-]*\.py\b",
]
EDIT_PATTERNS = [
    r"\bapply_patch\b",
    r"\bedit\b\s+\d+:\d+",          # swe-agent style `edit <start>:<end>`
    r"\bcreate\s+[\w./-]+\.\w+",    # swe-agent `create <newfile>`
    r"\bstr_replace\b",
    r"\binsert\b\s+\d+",
    r"\bpyupgrade\b",               # in-place source rewriter
    r"cat\s+<<\s*['\"]?\w+['\"]?\s*>\s*[\w./-]+",  # heredoc into a file
    r">\s*[\w./-]+\.(?:py|js|ts|txt|cfg|toml|yaml|yml|json|md|rst)\b",  # redirect to file
    r"\bsed\s+-i\b",
    r"\bpython\b[^\n]*open\([^)]*['\"][wa]",  # open(..., 'w'/'a')
    r"\btee\b",
]
INSTALL_PATTERNS = [
    r"\bpip\s+install\b",
    r"\bpip3\s+install\b",
    r"\bpython\s+-m\s+pip\s+install\b",
    r"\bconda\s+install\b",
    r"\bapt-get\b",
    r"\bapt\s+install\b",
    r"\bpoetry\s+(?:add|install)\b",
    r"\bsetup\.py\s+install\b",
]
GIT_PATTERNS = [
    r"\bgit\s+diff\b",
    r"\bgit\s+status\b",
    r"\bgit\s+checkout\b",
    r"\bgit\s+reset\b",
    r"\bgit\s+stash\b",
    r"\bgit\s+log\b",
    r"\bgit\s+add\b",
    r"\bgit\s+commit\b",
    r"\bgit\s+restore\b",
]
SEARCH_PATTERNS = [
    r"\brg\b",
    r"\bgrep\b",
    r"\bfind\b",
    r"\back\b",
    r"\bag\b",
    r"\bgit\s+grep\b",
    r"\bsearch_dir\b",   # swe-agent search commands
    r"\bsearch_file\b",
    r"\bfind_file\b",
]
READ_PATTERNS = [
    r"\bcat\b",
    r"\bsed\s+-n\b",
    r"\bnl\s+-ba\b",
    r"\bhead\b",
    r"\btail\b",
    r"\bless\b",
    r"\bmore\b",
    r"\bopen\b\s+[\w./-]+",          # swe-agent `open <file>`
    r"\bgoto\b\s+\d+",               # swe-agent `goto <line>`
    r"\bscroll_(?:up|down)\b",
]
ENV_PATTERNS = [
    r"\bcd\b",
    r"\bls\b",
    r"\bpwd\b",
    r"\bexport\b",
    r"\bsource\b",
    r"\bconda\s+activate\b",
    r"\bmkdir\b",
    r"\bwhich\b",
    r"\becho\b",
    r"\benv\b",
    r"\brm\b",
    r"\bmv\b",
    r"\bcp\b",
    r"\btouch\b",
    r"\bchmod\b",
]

# (action_type, attribute-name, compiled pattern list) in priority order.
FAMILIES: list[tuple[str, str | None, list[re.Pattern]]] = [
    ("submit", "is_submit_command", [re.compile(p) for p in SUBMIT_PATTERNS]),
    ("test", "is_test_command", [re.compile(p) for p in TEST_PATTERNS]),
    ("edit", "is_edit_command", [re.compile(p) for p in EDIT_PATTERNS]),
    ("install", "is_install_command", [re.compile(p) for p in INSTALL_PATTERNS]),
    ("git", "is_git_command", [re.compile(p) for p in GIT_PATTERNS]),
    ("search", "is_search_command", [re.compile(p) for p in SEARCH_PATTERNS]),
    ("read", "is_read_command", [re.compile(p) for p in READ_PATTERNS]),
    ("environment", None, [re.compile(p) for p in ENV_PATTERNS]),
]

FILE_RE = re.compile(
    r"(?P<path>(?:[\w.-]+/)*[\w.-]+\.(?:py|pyi|js|ts|tsx|jsx|java|go|rs|cpp|cc|"
    r"c|h|hpp|rb|php|sh|yaml|yml|toml|cfg|ini|json|md|rst|txt))"
)

# Test-result signals in observation text.
_PASSED_RE = re.compile(r"(\d+)\s+passed", re.IGNORECASE)
_FAILED_RE = re.compile(r"(\d+)\s+failed", re.IGNORECASE)
_ERROR_RE = re.compile(r"(\d+)\s+errors?", re.IGNORECASE)
_FAILED_LINE_RE = re.compile(r"(?m)^(?:FAILED|ERROR)\b")
_PASSED_LINE_RE = re.compile(r"(?m)^PASSED\b")
_TRACEBACK_RE = re.compile(r"Traceback \(most recent call last\)|^Traceback\b", re.MULTILINE)
_EXCEPTION_RE = re.compile(r"\b[A-Za-z_]\w*(?:Error|Exception)\b")


@dataclass
class ParsedAction:
    action_type: str = "unknown"
    command: str | None = None
    file_paths: list[str] = field(default_factory=list)
    is_test_command: bool = False
    is_search_command: bool = False
    is_read_command: bool = False
    is_edit_command: bool = False
    is_git_command: bool = False
    is_install_command: bool = False
    is_submit_command: bool = False


def _first_command_line(action_text: str) -> str | None:
    """Best-effort extraction of the primary command string for logging."""
    for line in action_text.splitlines():
        s = line.strip().strip("`")
        if s and not s.startswith("#"):
            return s[:300]
    return None


def classify_action(action_text: str) -> ParsedAction:
    """Classify an action string into a command family + boolean flags.

    The *primary* ``action_type`` is the highest-priority matching family, but
    every matching family flag is set independently (a single action line can,
    for example, both edit and reference a file).
    """
    text = action_text or ""
    pa = ParsedAction(command=_first_command_line(text))

    matched_primary = False
    for action_type, attr, patterns in FAMILIES:
        hit = any(p.search(text) for p in patterns)
        if attr is not None and hit:
            setattr(pa, attr, True)
        if hit and not matched_primary:
            pa.action_type = action_type
            matched_primary = True

    if not matched_primary and text.strip():
        pa.action_type = "other"

    pa.file_paths = extract_file_paths(text)
    return pa


def extract_file_paths(text: str) -> list[str]:
    """Conservatively extract source/test file paths, de-duplicated, in order."""
    seen: dict[str, None] = {}
    for m in FILE_RE.finditer(text or ""):
        seen.setdefault(m.group("path"), None)
    return list(seen.keys())


def is_test_path(path: str) -> bool:
    """Heuristic: does this path look like a test file?"""
    low = path.lower()
    base = low.rsplit("/", 1)[-1]
    return (
        "/test" in low
        or low.startswith("test")
        or base.startswith("test_")
        or base.endswith("_test.py")
        or "/tests/" in low
        or low.startswith("tests/")
    )


def parse_test_counts(observation: str) -> tuple[int | None, int | None]:
    """Parse (pass_count, fail_count) from a test-runner observation.

    Returns (None, None) when no recognizable test signal is present so that
    "ran no tests" is distinguishable from "ran tests, 0 failed".
    """
    text = observation or ""
    passed = sum(int(m.group(1)) for m in _PASSED_RE.finditer(text))
    failed = sum(int(m.group(1)) for m in _FAILED_RE.finditer(text))
    errors = sum(int(m.group(1)) for m in _ERROR_RE.finditer(text))
    fail_total = failed + errors

    has_summary = bool(
        _PASSED_RE.search(text) or _FAILED_RE.search(text) or _ERROR_RE.search(text)
    )
    if has_summary:
        return (passed, fail_total)

    # Fallback: count per-test result lines.
    n_failed_lines = len(_FAILED_LINE_RE.findall(text))
    n_passed_lines = len(_PASSED_LINE_RE.findall(text))
    if n_failed_lines or n_passed_lines:
        return (n_passed_lines or None, n_failed_lines)

    return (None, None)


def detect_traceback(observation: str) -> bool:
    return bool(_TRACEBACK_RE.search(observation or ""))


def detect_exception(observation: str) -> bool:
    return bool(_EXCEPTION_RE.search(observation or ""))
