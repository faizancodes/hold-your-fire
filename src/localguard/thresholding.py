"""Threshold policies and first-alert (trajectory-level) evaluation (Phase 8).

The monitor is judged not on prefix-row accuracy alone but on *trajectory-level*
behaviour: for a failed run, how many steps of warning did the first alert give;
for a successful run, did we ever falsely alert. Thresholds are always selected
on validation and only then applied to test.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

RISK_COL = "risk"


@dataclass
class FirstAlertMetrics:
    threshold: float
    n_failed: int
    n_success: int
    failed_alerted: int
    success_false_alarmed: int
    failed_coverage: float          # recall on failed trajectories
    success_false_alarm_rate: float  # FAR on successful trajectories
    median_lead_steps: float
    median_lead_fraction: float

    def as_row(self) -> dict[str, float]:
        return {
            "threshold": round(self.threshold, 4),
            "n_failed": self.n_failed,
            "n_success": self.n_success,
            "failed_coverage": round(self.failed_coverage, 4),
            "success_false_alarm_rate": round(self.success_false_alarm_rate, 4),
            "median_lead_steps": round(self.median_lead_steps, 2),
            "median_lead_fraction": round(self.median_lead_fraction, 4),
        }


def first_alert_per_trajectory(
    df: pd.DataFrame, threshold: float, risk_col: str = RISK_COL
) -> pd.DataFrame:
    """One row per trajectory: first prefix step whose risk >= threshold.

    Expects columns: trajectory_id, n_total_steps, y_fail, prefix_step, <risk_col>.
    """
    rows = []
    for tid, g in df.groupby("trajectory_id"):
        g = g.sort_values("prefix_step")
        alarms = g[g[risk_col] >= threshold]
        first_step = int(alarms["prefix_step"].iloc[0]) if len(alarms) else None
        n_total = int(g["n_total_steps"].iloc[0])
        y_fail = int(g["y_fail"].iloc[0])
        lead = (n_total - first_step) if first_step is not None else None
        rows.append({
            "trajectory_id": tid,
            "y_fail": y_fail,
            "n_total_steps": n_total,
            "first_alarm_step": first_step,
            "alerted": first_step is not None,
            "lead_steps": lead,
            "lead_fraction": (lead / n_total) if (lead is not None and n_total) else None,
        })
    return pd.DataFrame(rows)


def first_alert_metrics(
    df: pd.DataFrame, threshold: float, risk_col: str = RISK_COL
) -> FirstAlertMetrics:
    per = first_alert_per_trajectory(df, threshold, risk_col)
    failed = per[per["y_fail"] == 1]
    success = per[per["y_fail"] == 0]

    failed_alerted = int(failed["alerted"].sum())
    success_fa = int(success["alerted"].sum())
    lead_steps = failed.loc[failed["alerted"], "lead_steps"].dropna()
    lead_frac = failed.loc[failed["alerted"], "lead_fraction"].dropna()

    return FirstAlertMetrics(
        threshold=float(threshold),
        n_failed=len(failed),
        n_success=len(success),
        failed_alerted=failed_alerted,
        success_false_alarmed=success_fa,
        failed_coverage=failed_alerted / max(1, len(failed)),
        success_false_alarm_rate=success_fa / max(1, len(success)),
        median_lead_steps=float(lead_steps.median()) if len(lead_steps) else 0.0,
        median_lead_fraction=float(lead_frac.median()) if len(lead_frac) else 0.0,
    )


def _candidate_thresholds(p: np.ndarray, n_grid: int = 200) -> np.ndarray:
    uniq = np.unique(p)
    if len(uniq) <= n_grid:
        grid = uniq
    else:
        grid = np.quantile(p, np.linspace(0, 1, n_grid))
    return np.unique(np.clip(grid, 0.0, 1.0))


def threshold_maximize_f1(y: np.ndarray, p: np.ndarray) -> float:
    """Prefix-level F1-optimal threshold (policy T1)."""
    from sklearn.metrics import f1_score

    best_t, best_f1 = 0.5, -1.0
    for t in _candidate_thresholds(p):
        f1 = f1_score(y, (p >= t).astype(int), zero_division=0)
        if f1 > best_f1:
            best_f1, best_t = f1, float(t)
    return best_t


def threshold_for_success_false_alarm(
    val_df: pd.DataFrame, max_far: float, risk_col: str = RISK_COL
) -> float:
    """Lowest threshold whose *trajectory-level* success FAR <= ``max_far``.

    Lower threshold => more sensitive but more false alarms, so the lowest
    threshold meeting the cap maximizes failure coverage within the budget.
    """
    p = val_df[risk_col].to_numpy(dtype=float)
    candidates = _candidate_thresholds(p)
    chosen = 1.0
    for t in candidates:  # ascending
        m = first_alert_metrics(val_df, float(t), risk_col)
        if m.success_false_alarm_rate <= max_far:
            chosen = float(t)
            break
    return chosen


# Policy registry: name -> callable(val_df, val_y, val_p) -> threshold
THRESHOLD_POLICIES = {
    "T1_max_f1": lambda df, y, p: threshold_maximize_f1(y, p),
    "T2_success_far_lte_20pct": lambda df, y, p: threshold_for_success_false_alarm(df, 0.20),
    "T3_success_far_lte_10pct": lambda df, y, p: threshold_for_success_false_alarm(df, 0.10),
    "T4_success_far_lte_5pct": lambda df, y, p: threshold_for_success_false_alarm(df, 0.05),
}

DEFAULT_DEPLOY_POLICY = "T3_success_far_lte_10pct"


def select_thresholds(val_df: pd.DataFrame, risk_col: str = RISK_COL) -> dict[str, float]:
    """Compute every policy's threshold on the validation set."""
    y = val_df["y_fail"].to_numpy(dtype=int)
    p = val_df[risk_col].to_numpy(dtype=float)
    return {name: fn(val_df, y, p) for name, fn in THRESHOLD_POLICIES.items()}
