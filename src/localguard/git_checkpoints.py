"""Git-based checkpoints and patch/churn features for online runs (Phase 5G/12).

Public offline trajectories rarely expose per-step diffs, so the patch/churn
feature family (G) is collected live during mini-SWE-agent runs by shelling out to
git in the task workspace after each step. Checkpoints back the (suggested or, in
later experiments, forced) rollback intervention.
"""

from __future__ import annotations

import subprocess
from dataclasses import dataclass
from pathlib import Path


def _git(repo: Path, *args: str, check: bool = False) -> str:
    try:
        out = subprocess.run(
            ["git", "-C", str(repo), *args],
            capture_output=True, text=True, timeout=30, check=check,
        )
        return out.stdout
    except Exception:
        return ""


@dataclass
class PatchStats:
    files_changed: int = 0
    lines_added: int = 0
    lines_deleted: int = 0
    working_tree_dirty: bool = False

    @property
    def total_changed(self) -> int:
        return self.lines_added + self.lines_deleted


def diff_stats(repo: Path, base_ref: str | None = None) -> PatchStats:
    """Parse ``git diff --numstat`` (vs base_ref or working tree) into a PatchStats."""
    args = ["diff", "--numstat"]
    if base_ref:
        args.append(base_ref)
    numstat = _git(repo, *args)
    files = added = deleted = 0
    for line in numstat.splitlines():
        parts = line.split("\t")
        if len(parts) >= 3:
            files += 1
            try:
                added += int(parts[0]) if parts[0] != "-" else 0
                deleted += int(parts[1]) if parts[1] != "-" else 0
            except ValueError:
                pass
    dirty = bool(_git(repo, "status", "--short").strip())
    return PatchStats(files_changed=files, lines_added=added, lines_deleted=deleted,
                      working_tree_dirty=dirty)


def patch_features(
    repo: Path,
    prev: PatchStats | None,
    test_improved: bool,
    base_ref: str | None = None,
) -> tuple[dict[str, float | int | bool], PatchStats]:
    """Compute Family-G patch/churn features and the new running PatchStats."""
    cur = diff_stats(repo, base_ref)
    growth = cur.total_changed - (prev.total_changed if prev else 0)
    feats: dict[str, float | int | bool] = {
        "diff_files_changed": cur.files_changed,
        "diff_lines_added": cur.lines_added,
        "diff_lines_deleted": cur.lines_deleted,
        "working_tree_dirty": cur.working_tree_dirty,
        "patch_growth_since_last_step": max(0, growth),
        "patch_growth_without_test_improvement": max(0, growth) if not test_improved else 0,
    }
    return feats, cur


def make_checkpoint(repo: Path, tag: str) -> str | None:
    """Create a lightweight checkpoint commit; return its SHA (or None)."""
    if not (repo / ".git").exists():
        return None
    _git(repo, "add", "-A")
    _git(repo, "commit", "--no-verify", "-q", "-m", f"localguard-checkpoint:{tag}", "--allow-empty")
    sha = _git(repo, "rev-parse", "HEAD").strip()
    return sha or None


def rollback_to(repo: Path, sha: str) -> bool:
    """Hard-reset the working tree to a prior checkpoint SHA."""
    if not sha:
        return False
    _git(repo, "reset", "--hard", sha)
    return True


def has_git(repo: Path) -> bool:
    return (Path(repo) / ".git").exists()
