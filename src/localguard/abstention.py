"""Selective prediction / abstention for the monitor (cleaner-label direction).

The terminal-outcome label is noise on early prefixes (AUC ~0.63) but informative
late (~0.81). Rather than report a cherry-picked late-prefix number (which would
need the UNKNOWN total length), the monitor *abstains* on prefixes it cannot judge
and commits a risk only when a PREFIX-VISIBLE gate is met:

    commit  iff  prefix_step >= step_floor  AND  |calibrated_risk - 0.5| >= conf_floor

We then report a risk-COVERAGE tradeoff (AUC/precision at a given coverage) instead
of one unconditional number. The step floor is the principled "determinability"
gate; the confidence floor is classic selective prediction layered on top.
"""

from __future__ import annotations

import numpy as np

from .evaluate import roc_auc_metric


def confidence(p: np.ndarray) -> np.ndarray:
    """Distance from maximal uncertainty (p=0.5). Larger = more confident."""
    return np.abs(np.asarray(p, dtype=float) - 0.5)


def is_committed(prefix_step: int, calibrated_risk: float,
                 step_floor: int, conf_floor: float) -> bool:
    """Deployable commit/abstain decision (prefix-visible inputs only)."""
    return (prefix_step >= step_floor) and (abs(calibrated_risk - 0.5) >= conf_floor)


def selective_curve(y: np.ndarray, p: np.ndarray, conf: np.ndarray | None = None,
                    coverages: np.ndarray | None = None) -> list[dict]:
    """Risk-coverage curve: keep the top-`coverage` most-confident, report AUC.

    Confidence-ranked selective prediction. AUC mechanically rises as coverage
    falls; compare against ``random_curve`` to see the *real* gain.
    """
    y = np.asarray(y, dtype=int)
    p = np.asarray(p, dtype=float)
    conf = confidence(p) if conf is None else np.asarray(conf, dtype=float)
    if coverages is None:
        coverages = np.linspace(0.1, 1.0, 19)
    order = np.argsort(-conf)  # most confident first
    out = []
    for cov in coverages:
        k = max(1, int(round(cov * len(y))))
        idx = order[:k]
        yy = y[idx]
        if len(np.unique(yy)) < 2:
            continue
        out.append({"coverage": round(float(k / len(y)), 4),
                    "auc": round(roc_auc_metric(yy, p[idx]), 4),
                    "n": int(k)})
    return out


def random_curve(y: np.ndarray, p: np.ndarray, coverages: np.ndarray | None = None,
                 seed: int = 42, repeats: int = 5) -> list[dict]:
    """Baseline: abstain at RANDOM. AUC should stay ~flat — proves selective
    prediction's gain is real, not a mechanical artifact of dropping rows."""
    y = np.asarray(y, dtype=int)
    p = np.asarray(p, dtype=float)
    if coverages is None:
        coverages = np.linspace(0.1, 1.0, 19)
    rng = np.random.default_rng(seed)
    out = []
    for cov in coverages:
        k = max(1, int(round(cov * len(y))))
        aucs = []
        for _ in range(repeats):
            idx = rng.choice(len(y), size=k, replace=False)
            if len(np.unique(y[idx])) == 2:
                aucs.append(roc_auc_metric(y[idx], p[idx]))
        if aucs:
            out.append({"coverage": round(float(k / len(y)), 4),
                        "auc": round(float(np.mean(aucs)), 4), "n": int(k)})
    return out


def step_gate_curve(y: np.ndarray, p: np.ndarray, prefix_step: np.ndarray,
                    step_floors: list[int]) -> list[dict]:
    """Risk-coverage by the principled deployable STEP gate (determinability)."""
    y = np.asarray(y, dtype=int)
    p = np.asarray(p, dtype=float)
    ps = np.asarray(prefix_step)
    out = []
    for S in step_floors:
        m = ps >= S
        if m.sum() < 50 or len(np.unique(y[m])) < 2:
            continue
        out.append({"step_floor": int(S),
                    "coverage": round(float(m.sum() / len(y)), 4),
                    "auc": round(roc_auc_metric(y[m], p[m]), 4),
                    "n": int(m.sum())})
    return out
