"""SWE-bench Verified Mini loader + Docker-gated runner scaffold (Phase 14).

`MariusHobbhahn/swe-bench-verified-mini` is a 50-task subset of SWE-bench Verified
(~5GB vs ~130GB; django + sphinx only) with a similar difficulty distribution.

IMPORTANT: official SWE-bench evaluation is resource-intensive and x86_64-oriented
(the upstream harness recommends ≥120GB free, 16GB RAM, 8 cores). On Apple Silicon
this path is **gated behind a smoke test**: we load the task list and verify
prerequisites, but the per-task containerized agent run + official grading is left
to mini-SWE-agent's own `run/benchmarks/swebench.py` harness. The toy-task path
(toy_tasks.py / online_runner.py) is the default for local online experiments.
"""

from __future__ import annotations

import shutil
from dataclasses import dataclass

DATASET = "MariusHobbhahn/swe-bench-verified-mini"


@dataclass
class SweTask:
    instance_id: str
    repo: str
    base_commit: str
    problem_statement: str
    fail_to_pass: str
    pass_to_pass: str


def load_tasks(n: int | None = None, split: str = "test") -> list[SweTask]:
    """Load the SWE-bench Verified Mini task list (metadata only, no images)."""
    from datasets import load_dataset

    ds = load_dataset(DATASET, split=split)
    tasks: list[SweTask] = []
    for i, r in enumerate(ds):
        if n is not None and i >= n:
            break
        tasks.append(SweTask(
            instance_id=r.get("instance_id", f"task{i}"),
            repo=r.get("repo", ""),
            base_commit=r.get("base_commit", ""),
            problem_statement=r.get("problem_statement", ""),
            fail_to_pass=str(r.get("FAIL_TO_PASS", "")),
            pass_to_pass=str(r.get("PASS_TO_PASS", "")),
        ))
    return tasks


def prerequisites() -> dict[str, object]:
    """Check whether the heavy SWE-bench path can run locally."""
    free_gb = shutil.disk_usage("/").free / 1e9
    return {
        "docker_available": shutil.which("docker") is not None,
        "free_disk_gb": round(free_gb, 1),
        "free_disk_ok": free_gb >= 120,
        "note": (
            "Run containerized agent + official grading via mini-SWE-agent's "
            "`python -m minisweagent.run.benchmarks.swebench` using "
            f"config/benchmarks/swebench.yaml against the {DATASET} subset. "
            "Apple Silicon may lack prebuilt x86 images for some instances."
        ),
    }


def gate_or_explain() -> bool:
    """Return True only if local prerequisites for the heavy path are met."""
    p = prerequisites()
    ok = bool(p["docker_available"]) and bool(p["free_disk_ok"])
    return ok
