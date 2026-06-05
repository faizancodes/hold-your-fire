"""Shared test fixtures and helpers (self-contained; no downloaded data needed)."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

_SRC = Path(__file__).resolve().parent.parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from localguard.schemas import NormalizedTrajectory, StepEvent  # noqa: E402


def make_step(idx: int, action: str, observation: str = "", **kw) -> StepEvent:
    """Build a StepEvent through the real classifier path."""
    from localguard.normalize import _finalize_step

    return _finalize_step(
        "traj1", "acme__widget-1", "m", idx, "assistant",
        action, "", action, observation, **kw,
    )


def make_trajectory(target: bool, actions: list[tuple[str, str]], instance_id="acme__widget-1") -> NormalizedTrajectory:
    steps = [make_step(i, a, o) for i, (a, o) in enumerate(actions)]
    for s in steps:
        s.instance_id = instance_id
    return NormalizedTrajectory(
        trajectory_id="traj_" + instance_id, instance_id=instance_id,
        model_name="m", target=target, steps=steps, n_steps=len(steps),
    )


@pytest.fixture
def fixtures_rows():
    from localguard.ingest_nebius import load_raw_rows

    return load_raw_rows("fixtures")


def make_synthetic_prefix_df(n_traj: int = 120, n_feats: int = 8, signal: float = 1.0,
                             seed: int = 0, n_instances: int = 40):
    """A synthetic prefix DataFrame with f__ columns + a known label signal.

    The label follows a logistic model with many *small* weights plus label noise
    (mirroring real tabular data: no single dominant feature), so the
    shuffled-label AUC is tightly concentrated near 0.5 rather than spiking on a
    lucky permutation. ``signal`` scales the weight vector.
    """
    import pandas as pd

    rng = np.random.default_rng(seed)
    # small, distributed weights -> moderate but non-trivial separability
    w = rng.normal(0, 0.45 * signal, size=n_feats)
    rows = []
    for t in range(n_traj):
        x_traj = rng.normal(size=n_feats)  # per-trajectory latent features
        logit = float(x_traj @ w)
        p = 1.0 / (1.0 + np.exp(-logit))
        y = int(rng.random() < p)
        n_steps = int(rng.integers(4, 20))
        for k in (1, n_steps // 2, n_steps):
            feats: dict[str, object] = {
                f"f__feat{i}": float(x_traj[i] + rng.normal(0, 0.25)) for i in range(n_feats)
            }
            feats["f__text_blob"] = "edit test pytest " if y else "read search cat "
            rows.append({
                "prefix_id": f"{t}_{k}", "trajectory_id": f"tr{t}",
                "instance_id": f"inst{t % n_instances}", "model_name": "m",
                "prefix_step": int(k), "n_total_steps": int(n_steps),
                "y_fail": y, **feats,
            })
    return pd.DataFrame(rows)
