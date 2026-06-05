"""Group-aware train/val/test splitting (Phase 6).

Prefix rows from one trajectory (and from one task instance) are highly
correlated. Splitting rows randomly leaks: the model would see prefixes of the
same task in train and test. We therefore split by GROUP — ``instance_id`` by
default — so an entire task lands wholly in one fold.

Three split regimes:
  * instance : standard, group by instance_id (primary).
  * repo     : harder, group by repository parsed from instance_id.
  * model    : stress, hold out entire ``model_name`` values for test.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

from .utils import DEFAULT_SEED


def parse_repo(instance_id: str) -> str:
    """Parse the repository from a SWE-bench-style ``{org}__{repo}-{number}`` id."""
    if not instance_id:
        return "unknown"
    head = instance_id.rsplit("-", 1)[0]  # drop trailing PR/issue number
    return head or instance_id


@dataclass
class Splits:
    train: pd.DataFrame
    val: pd.DataFrame
    test: pd.DataFrame
    group_col: str

    def summary(self) -> dict[str, object]:
        def fr(df: pd.DataFrame) -> float:
            return round(float(df["y_fail"].mean()), 4) if len(df) else 0.0

        return {
            "group_col": self.group_col,
            "n_train_rows": len(self.train),
            "n_val_rows": len(self.val),
            "n_test_rows": len(self.test),
            "n_train_groups": int(self.train[self.group_col].nunique()),
            "n_val_groups": int(self.val[self.group_col].nunique()),
            "n_test_groups": int(self.test[self.group_col].nunique()),
            "fail_rate_train": fr(self.train),
            "fail_rate_val": fr(self.val),
            "fail_rate_test": fr(self.test),
        }


def _group_split(
    df: pd.DataFrame, group_col: str, test_size: float, seed: int
) -> tuple[pd.DataFrame, pd.DataFrame]:
    groups = df[group_col].astype(str).values
    gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    a_idx, b_idx = next(gss.split(df, df["y_fail"].values, groups=groups))
    return df.iloc[a_idx].copy(), df.iloc[b_idx].copy()


def make_splits(
    df: pd.DataFrame,
    regime: str = "instance",
    test_size: float = 0.2,
    val_size: float = 0.2,
    seed: int = DEFAULT_SEED,
    holdout_models: list[str] | None = None,
) -> Splits:
    """Build disjoint grouped train/val/test splits.

    val_size is relative to the post-test remainder, so the nominal fractions are
    test=test_size, val=val_size*(1-test_size), train=the rest.
    """
    df = df.copy()
    if "repo" not in df.columns:
        df["repo"] = df["instance_id"].astype(str).map(parse_repo)

    if regime == "model":
        return _model_holdout_splits(df, holdout_models, val_size, seed)

    group_col = "instance_id" if regime == "instance" else "repo"
    train_val, test = _group_split(df, group_col, test_size, seed)
    train, val = _group_split(train_val, group_col, val_size, seed)
    return Splits(train=train, val=val, test=test, group_col=group_col)


def _model_holdout_splits(
    df: pd.DataFrame, holdout_models: list[str] | None, val_size: float, seed: int
) -> Splits:
    """Stress split: entire model_name families go to test."""
    models = sorted(df["model_name"].dropna().astype(str).unique())
    if not holdout_models:
        # hold out ~1/3 of models for test by default
        rng = np.random.default_rng(seed)
        k = max(1, len(models) // 3)
        holdout_models = list(rng.choice(models, size=k, replace=False)) if models else []
    holdout = list(dict.fromkeys(holdout_models))
    is_holdout = df["model_name"].astype(str).isin(holdout)
    test = df[is_holdout].copy()
    train_val = df[~is_holdout].copy()
    # val grouped by instance within the remaining models
    if len(train_val):
        train, val = _group_split(train_val, "instance_id", val_size, seed)
    else:
        train, val = train_val, train_val
    return Splits(train=train, val=val, test=test, group_col="instance_id")


def verify_disjoint(splits: Splits, col: str | None = None) -> None:
    """Assert the three folds share no group values (the Phase 6 gate)."""
    col = col or splits.group_col
    tr = set(splits.train[col].astype(str))
    va = set(splits.val[col].astype(str))
    te = set(splits.test[col].astype(str))
    assert not (tr & va), f"train/val overlap on {col}: {sorted(tr & va)[:5]}"
    assert not (tr & te), f"train/test overlap on {col}: {sorted(tr & te)[:5]}"
    assert not (va & te), f"val/test overlap on {col}: {sorted(va & te)[:5]}"
