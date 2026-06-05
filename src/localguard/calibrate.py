"""Probability calibration for risk scores (Phase 9).

A ranking-good monitor is not enough: the intervention policy thresholds a
*probability*, so risk=0.8 should mean ~80% of such prefixes eventually fail.
Calibrators are always fit on validation predictions and never on test.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


@dataclass
class Calibrator:
    method: str  # platt | isotonic | identity
    model: Any = None

    def transform(self, p: np.ndarray) -> np.ndarray:
        p = np.asarray(p, dtype=float)
        if self.method == "identity" or self.model is None:
            return np.clip(p, 0.0, 1.0)
        if self.method == "platt":
            z = self.model.predict_proba(p.reshape(-1, 1))
            return z[:, 1]
        # isotonic
        return np.clip(self.model.predict(p), 0.0, 1.0)

    def save(self, path: Path) -> Path:
        import joblib

        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)
        return path

    @staticmethod
    def load(path: Path) -> "Calibrator":
        import joblib

        return joblib.load(path)


def fit_calibrator(method: str, p_val: np.ndarray, y_val: np.ndarray) -> Calibrator:
    """Fit a calibrator on validation (score, label) pairs."""
    p_val = np.asarray(p_val, dtype=float)
    y_val = np.asarray(y_val, dtype=int)

    if method == "none" or method == "identity":
        return Calibrator(method="identity")
    if len(np.unique(y_val)) < 2:
        # cannot calibrate without both classes; fall back to identity
        return Calibrator(method="identity")

    if method == "platt":
        from sklearn.linear_model import LogisticRegression

        lr = LogisticRegression(max_iter=1000)
        lr.fit(p_val.reshape(-1, 1), y_val)
        return Calibrator(method="platt", model=lr)

    if method == "isotonic":
        from sklearn.isotonic import IsotonicRegression

        iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
        iso.fit(p_val, y_val)
        return Calibrator(method="isotonic", model=iso)

    raise ValueError(f"unknown calibration method: {method}")


def reliability_curve(
    y: np.ndarray, p: np.ndarray, n_bins: int = 10
) -> list[dict[str, float]]:
    """Binned (mean predicted, observed frequency) points for a calibration plot."""
    y = np.asarray(y, dtype=float)
    p = np.asarray(p, dtype=float)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    idx = np.clip(np.digitize(p, bins) - 1, 0, n_bins - 1)
    out = []
    for b in range(n_bins):
        mask = idx == b
        if not mask.any():
            continue
        out.append({
            "bin": b,
            "mean_predicted": float(p[mask].mean()),
            "observed_freq": float(y[mask].mean()),
            "count": int(mask.sum()),
        })
    return out
