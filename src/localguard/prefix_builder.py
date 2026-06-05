"""Build prefix datasets from normalized trajectories (Phase 4).

For each trajectory we emit several :class:`PrefixExample` rows, one per prefix
length in a configurable schedule. The label ``y_fail`` is the *terminal* outcome
(did the whole run fail) attached to a partial prefix — an intentionally noisy
"will this eventually fail" target, not a per-step correctness label.
"""

from __future__ import annotations

from typing import Any, Iterator, Sequence

from .features import extract_features
from .schemas import NormalizedTrajectory, PrefixExample
from .utils import assert_no_leakage, stable_hash


def prefix_schedule(n_steps: int, mode: str = "default") -> list[int]:
    """Return the prefix lengths to materialize for a trajectory of ``n_steps``.

    default : 1..5, then every 5th step, plus the final step (the plan schedule).
    dense   : every step up to a 60-step cap (then every 5th) — for small studies.
    sparse  : 1, 3, 5, then every 10th — for full-corpus scale.
    """
    if n_steps <= 0:
        return []
    steps: set[int] = set()
    if mode == "dense":
        cap = min(n_steps, 60)
        steps.update(range(1, cap + 1))
        steps.update(range(cap, n_steps + 1, 5))
    elif mode == "sparse":
        steps.update([1, 3, 5])
        steps.update(range(10, n_steps + 1, 10))
    else:  # default
        for i in range(1, min(n_steps, 5) + 1):
            steps.add(i)
        for i in range(10, n_steps + 1, 5):
            steps.add(i)
    steps.add(n_steps)
    steps.discard(0)
    return sorted(s for s in steps if 1 <= s <= n_steps)


def build_prefix_examples(
    traj: NormalizedTrajectory,
    schedule_mode: str = "default",
    check_leakage: bool = True,
    extractor=extract_features,
) -> list[PrefixExample]:
    """Materialize prefix examples for a single normalized trajectory.

    ``extractor`` lets callers swap in a richer feature function (e.g. the v2
    advanced extractor) while keeping identical ``prefix_id``s, so the same
    train/val/test split can be reused for a controlled comparison.
    """
    y_fail = int(not traj.target)
    examples: list[PrefixExample] = []
    for k in prefix_schedule(traj.n_steps, schedule_mode):
        prefix_steps = traj.steps[:k]
        feats = extractor(prefix_steps)
        if check_leakage:
            assert_no_leakage(feats)
        examples.append(
            PrefixExample(
                prefix_id=stable_hash(traj.trajectory_id, k),
                trajectory_id=traj.trajectory_id,
                instance_id=traj.instance_id,
                model_name=traj.model_name,
                prefix_step=k,
                n_total_steps=traj.n_steps,
                y_fail=y_fail,
                feature_dict=feats,
            )
        )
    return examples


def prefix_example_to_row(ex: PrefixExample) -> dict[str, Any]:
    """Flatten a PrefixExample to a single dict row (metadata + features)."""
    row: dict[str, Any] = {
        "prefix_id": ex.prefix_id,
        "trajectory_id": ex.trajectory_id,
        "instance_id": ex.instance_id,
        "model_name": ex.model_name,
        "prefix_step": ex.prefix_step,
        "n_total_steps": ex.n_total_steps,
        "y_fail": ex.y_fail,
    }
    # feature columns are namespaced with f__ so they never collide with metadata
    for k, v in ex.feature_dict.items():
        row[f"f__{k}"] = v
    return row


def trajectories_to_rows(
    trajs: Sequence[NormalizedTrajectory],
    schedule_mode: str = "default",
    extractor=extract_features,
) -> Iterator[dict[str, Any]]:
    for traj in trajs:
        for ex in build_prefix_examples(traj, schedule_mode, extractor=extractor):
            yield prefix_example_to_row(ex)


# Convenience constants for the rest of the pipeline.
META_COLUMNS = (
    "prefix_id", "trajectory_id", "instance_id", "model_name",
    "prefix_step", "n_total_steps", "y_fail",
)
FEATURE_PREFIX = "f__"
TEXT_COLUMN = "f__text_blob"
