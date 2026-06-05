"""Weak-supervision labels to de-noise the early-prefix target (cleaner label).

The terminal label marks EVERY prefix of a failed run as "fail", including early
prefixes where the agent has not gone wrong yet — pure label noise. We instead
derive cleaner *training* targets from prefix-visible "trouble" evidence:

  * trouble_indicator(df): is the agent observably in trouble at this prefix?
    (persistent test failures, loops, repeated errors, blind editing).
  * W1 trouble-gated relabel: a failed prefix counts as positive ONLY once trouble
    is visible; healthy-early failed prefixes become 0.
  * W2 down-weight: keep terminal labels but down-weight the noisy positives
    (failed + no observable trouble).
  * trouble_onset_step: first trouble step per trajectory — the "event time" for
    the survival/hazard model.

These shape only the TRAINING target/weights; evaluation always uses the original
terminal label on held-out instances, and the model only sees prefix features.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

FP = "f__"


def _g(df: pd.DataFrame, name: str) -> np.ndarray:
    col = FP + name
    if col in df.columns:
        return pd.to_numeric(df[col], errors="coerce").fillna(0).to_numpy(float)
    return np.zeros(len(df), dtype=float)


def trouble_indicator(df: pd.DataFrame) -> np.ndarray:
    """Prefix-visible 'the agent is in trouble right now' (boolean array)."""
    loop = (
        (_g(df, "max_command_repeat_count") >= 3)
        | (_g(df, "same_action_type_streak") >= 5)
        | (_g(df, "repeated_exact_command_last_3") > 0)
    )
    test_fail = (_g(df, "n_test_runs") >= 2) & (
        (_g(df, "tests_worsening") > 0) | (_g(df, "last_test_fail_count") > 0)
    )
    errors = _g(df, "n_tracebacks_seen") >= 2
    blind_edit = (_g(df, "edited_file_never_read_count") >= 1) & (_g(df, "n_edit") >= 2)
    return (loop | test_fail | errors | blind_edit)


def relabel_trouble_gated(df: pd.DataFrame, y_col: str = "y_fail") -> np.ndarray:
    """W1: positive iff terminally failed AND observably in trouble at this prefix.

    Healthy-early prefixes of doomed runs -> 0 (de-noised). Successful prefixes -> 0
    even if they show transient trouble (they recovered)."""
    y = df[y_col].to_numpy(int)
    return ((y == 1) & trouble_indicator(df)).astype(int)


def weights_downweight_noisy(df: pd.DataFrame, y_col: str = "y_fail",
                             low: float = 0.3) -> np.ndarray:
    """W2: keep terminal labels but down-weight failed prefixes with no observable
    trouble (the noisy positives)."""
    y = df[y_col].to_numpy(int)
    noisy_pos = (y == 1) & (~trouble_indicator(df))
    return np.where(noisy_pos, low, 1.0)


def trouble_onset_step(group: pd.DataFrame) -> int | None:
    """First prefix_step in a trajectory where trouble is observable (the survival
    'event time'). Returns None if trouble never appears."""
    g = group.sort_values("prefix_step")
    trb = trouble_indicator(g)
    steps = g["prefix_step"].to_numpy()
    hit = np.where(trb)[0]
    return int(steps[hit[0]]) if len(hit) else None
