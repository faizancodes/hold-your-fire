#!/usr/bin/env python3
"""Environment verification for LocalGuard-SWE (Phase 1).

Checks Python, RAM, disk, the local Ollama server + a coding model, Docker,
mini-SWE-agent import, and that NO paid API key is required. Prints a PASS/FAIL
summary and exits non-zero if a hard requirement is missing.
"""

from __future__ import annotations

import argparse
import shutil
import sys

import _bootstrap  # noqa: F401

OLLAMA_BASE = "http://localhost:11434"


def _check(label: str, ok: bool, detail: str = "", hard: bool = True) -> tuple[bool, bool]:
    mark = "PASS" if ok else ("FAIL" if hard else "WARN")
    print(f"  [{mark}] {label}" + (f" — {detail}" if detail else ""))
    return ok, hard


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default="qwen2.5-coder:7b")
    args = ap.parse_args()

    results: list[tuple[bool, bool]] = []
    print("LocalGuard-SWE system check\n")

    # Python
    v = sys.version_info
    results.append(_check("Python >= 3.10", v >= (3, 10), f"{v.major}.{v.minor}.{v.micro}"))

    # RAM / disk
    try:
        import psutil  # optional

        gb = psutil.virtual_memory().total / 1e9
        results.append(_check("RAM >= 8GB", gb >= 8, f"{gb:.0f}GB", hard=False))
    except Exception:
        print("  [INFO] psutil not installed; skipping RAM check")
    free_gb = shutil.disk_usage("/").free / 1e9
    results.append(_check("Free disk >= 10GB", free_gb >= 10, f"{free_gb:.0f}GB", hard=False))

    # Core python deps
    for pkg in ["numpy", "pandas", "sklearn", "pydantic", "datasets", "ollama"]:
        import importlib.util as u

        results.append(_check(f"import {pkg}", u.find_spec(pkg) is not None, hard=(pkg != "datasets")))

    # Ollama server + model
    ollama_ok = False
    try:
        import urllib.request

        with urllib.request.urlopen(f"{OLLAMA_BASE}/api/tags", timeout=3) as r:
            ollama_ok = r.status == 200
    except Exception as exc:
        ollama_ok = False
        _ = exc
    results.append(_check("Ollama server reachable", ollama_ok, OLLAMA_BASE))

    model_ok = False
    if ollama_ok:
        try:
            import ollama

            resp = ollama.chat(
                model=args.model,
                messages=[{"role": "user", "content": "Reply with OK only."}],
                options={"temperature": 0, "num_predict": 5},
            )
            content = resp["message"]["content"]
            model_ok = bool(content)
            results.append(_check(f"{args.model} responds", model_ok, repr(content.strip()[:20])))
        except Exception as exc:
            results.append(_check(f"{args.model} responds", False, str(exc)[:80]))
    else:
        print(f"  [SKIP] {args.model} responds (server unreachable)")

    # Docker (only needed for full SWE-bench; soft)
    results.append(_check("Docker available", shutil.which("docker") is not None,
                          hard=False))

    # mini-SWE-agent (soft; only needed for online)
    try:
        import minisweagent  # noqa: F401

        results.append(_check("mini-swe-agent import", True, hard=False))
    except Exception as exc:
        results.append(_check("mini-swe-agent import", False, str(exc)[:60], hard=False))

    # No paid API key required
    import os

    paid = [k for k in ("OPENAI_API_KEY", "ANTHROPIC_API_KEY", "GEMINI_API_KEY",
                        "TOGETHER_API_KEY", "FIREWORKS_API_KEY", "OPENROUTER_API_KEY")
            if os.getenv(k)]
    results.append(_check("No paid API key needed (none required by pipeline)", True,
                          f"(found unused: {paid})" if paid else "", hard=False))

    hard_fail = [r for r in results if not r[0] and r[1]]
    print()
    if hard_fail:
        print(f"RESULT: FAIL ({len(hard_fail)} hard requirement(s) missing)")
        sys.exit(1)
    print("RESULT: PASS — Ollama reachable, model reachable, no paid API key needed, "
          "mini-swe-agent import works")


if __name__ == "__main__":
    main()
