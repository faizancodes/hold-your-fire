"""Offline prediction metrics with group-aware bootstrap CIs (Phase 8).

Positive class is FAILURE (y_fail == 1). All confidence intervals resample whole
``instance_id`` groups, never individual prefix rows, so correlated prefixes from
one task cannot inflate apparent precision.
"""

from __future__ import annotations

from typing import Callable

import numpy as np
import pandas as pd

from .utils import DEFAULT_SEED


def expected_calibration_error(y: np.ndarray, p: np.ndarray, n_bins: int = 10) -> float:
    """Equal-width-binned ECE: mean |confidence - accuracy| weighted by bin size."""
    y = np.asarray(y, dtype=float)
    p = np.asarray(p, dtype=float)
    if len(y) == 0:
        return float("nan")
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    idx = np.clip(np.digitize(p, bins) - 1, 0, n_bins - 1)
    ece = 0.0
    for b in range(n_bins):
        mask = idx == b
        if not mask.any():
            continue
        conf = p[mask].mean()
        acc = y[mask].mean()
        ece += (mask.sum() / len(y)) * abs(conf - acc)
    return float(ece)


def prefix_metrics(y: np.ndarray, p: np.ndarray) -> dict[str, float]:
    """ROC AUC, AUPRC, Brier, ECE for failure-positive predictions."""
    from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

    y = np.asarray(y, dtype=int)
    p = np.asarray(p, dtype=float)
    out: dict[str, float] = {"n": int(len(y)), "pos_rate": float(y.mean()) if len(y) else float("nan")}
    if len(np.unique(y)) < 2:
        out.update(roc_auc=float("nan"), auprc=float("nan"),
                   brier=float(np.mean((p - y) ** 2)) if len(y) else float("nan"),
                   ece=expected_calibration_error(y, p))
        return out
    out["roc_auc"] = float(roc_auc_score(y, p))
    out["auprc"] = float(average_precision_score(y, p))
    out["brier"] = float(brier_score_loss(y, p))
    out["ece"] = expected_calibration_error(y, p)
    return out


def precision_recall_at_far(y: np.ndarray, p: np.ndarray, far: float) -> dict[str, float]:
    """Precision & recall at the prefix-level false-alarm rate (FPR) == ``far``."""
    from sklearn.metrics import roc_curve

    y = np.asarray(y, dtype=int)
    p = np.asarray(p, dtype=float)
    if len(np.unique(y)) < 2:
        return {"far": far, "threshold": float("nan"), "precision": float("nan"), "recall": float("nan")}
    fpr, tpr, thr = roc_curve(y, p)
    # largest threshold whose FPR <= far (most precise op point within budget)
    ok = np.where(fpr <= far)[0]
    j = ok[-1] if len(ok) else 0
    t = float(thr[j])
    pred = (p >= t).astype(int)
    tp = int(((pred == 1) & (y == 1)).sum())
    fp = int(((pred == 1) & (y == 0)).sum())
    fn = int(((pred == 0) & (y == 1)).sum())
    precision = tp / max(1, tp + fp)
    recall = tp / max(1, tp + fn)
    return {"far": far, "threshold": t, "precision": float(precision),
            "recall": float(recall), "fpr_actual": float(fpr[j])}


def bootstrap_ci(
    df: pd.DataFrame,
    metric_fn: Callable[[np.ndarray, np.ndarray], float],
    y_col: str = "y_fail",
    p_col: str = "risk",
    group_col: str = "instance_id",
    n_boot: int = 500,
    seed: int = DEFAULT_SEED,
    alpha: float = 0.05,
) -> dict[str, float]:
    """Group-bootstrap CI for a metric: resample whole ``group_col`` units."""
    rng = np.random.default_rng(seed)
    groups = df[group_col].astype(str).to_numpy()
    uniq = np.unique(groups)
    # index lookup per group for fast resampling
    by_group = {g: np.where(groups == g)[0] for g in uniq}
    y_all = df[y_col].to_numpy(dtype=int)
    p_all = df[p_col].to_numpy(dtype=float)

    point = float(metric_fn(y_all, p_all))
    stats: list[float] = []
    for _ in range(n_boot):
        chosen = rng.choice(uniq, size=len(uniq), replace=True)
        idx = np.concatenate([by_group[g] for g in chosen])
        yb, pb = y_all[idx], p_all[idx]
        if len(np.unique(yb)) < 2:
            continue
        try:
            stats.append(float(metric_fn(yb, pb)))
        except Exception:
            continue
    if not stats:
        return {"point": point, "lo": float("nan"), "hi": float("nan"), "n_boot": 0}
    lo = float(np.quantile(stats, alpha / 2))
    hi = float(np.quantile(stats, 1 - alpha / 2))
    return {"point": point, "lo": lo, "hi": hi, "n_boot": len(stats)}


def roc_auc_metric(y: np.ndarray, p: np.ndarray) -> float:
    from sklearn.metrics import roc_auc_score

    return float(roc_auc_score(y, p))


def paired_bootstrap_auc_delta(
    instance_ids: np.ndarray,
    y: np.ndarray,
    p_base: np.ndarray,
    p_new: np.ndarray,
    n_boot: int = 1000,
    seed: int = DEFAULT_SEED,
    alpha: float = 0.05,
) -> dict[str, float]:
    """Paired instance-bootstrap of AUC(p_new) - AUC(p_base) on the SAME rows.

    Resamples whole instances (groups), recomputing both AUCs on each resample so
    the comparison is paired — far more powerful than comparing two independent
    CIs. Returns point AUCs, the delta, its CI, and the fraction of resamples in
    which p_new wins.
    """
    rng = np.random.default_rng(seed)
    groups = np.asarray(instance_ids).astype(str)
    y = np.asarray(y, dtype=int)
    p_base = np.asarray(p_base, dtype=float)
    p_new = np.asarray(p_new, dtype=float)
    uniq = np.unique(groups)
    by = {g: np.where(groups == g)[0] for g in uniq}

    base = roc_auc_metric(y, p_base)
    new = roc_auc_metric(y, p_new)
    deltas, wins = [], 0
    for _ in range(n_boot):
        chosen = rng.choice(uniq, size=len(uniq), replace=True)
        idx = np.concatenate([by[g] for g in chosen])
        yb = y[idx]
        if len(np.unique(yb)) < 2:
            continue
        a_b = roc_auc_metric(yb, p_base[idx])
        a_n = roc_auc_metric(yb, p_new[idx])
        deltas.append(a_n - a_b)
        wins += int(a_n > a_b)
    deltas_arr = np.array(deltas) if deltas else np.array([0.0])
    return {
        "auc_base": round(base, 4),
        "auc_new": round(new, 4),
        "delta": round(new - base, 4),
        "delta_lo": round(float(np.quantile(deltas_arr, alpha / 2)), 4),
        "delta_hi": round(float(np.quantile(deltas_arr, 1 - alpha / 2)), 4),
        "frac_new_better": round(wins / max(1, len(deltas)), 3),
        "significant": bool(float(np.quantile(deltas_arr, alpha / 2)) > 0),
        "n_boot": len(deltas),
    }


def auprc_metric(y: np.ndarray, p: np.ndarray) -> float:
    from sklearn.metrics import average_precision_score

    return float(average_precision_score(y, p))


def risk_by_normalized_position(
    df: pd.DataFrame, risk_col: str = "risk", n_bins: int = 10
) -> pd.DataFrame:
    """Mean risk vs normalized prefix position, split by outcome (for Figure 1).

    Uses prefix_step / n_total_steps — a diagnostic that legitimately uses total
    length because it is only ever plotted, never fed to a model.
    """
    d = df.copy()
    d["pos"] = (d["prefix_step"] / d["n_total_steps"].clip(lower=1)).clip(0, 1)
    d["bin"] = np.clip((d["pos"] * n_bins).astype(int), 0, n_bins - 1)
    rows = []
    for (outcome, b), g in d.groupby(["y_fail", "bin"]):
        rows.append({
            "y_fail": int(outcome),
            "bin": int(b),
            "pos_center": (b + 0.5) / n_bins,
            "mean_risk": float(g[risk_col].mean()),
            "n": len(g),
        })
    return pd.DataFrame(rows).sort_values(["y_fail", "bin"])
