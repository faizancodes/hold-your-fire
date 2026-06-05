"""Train offline failure monitors (Phase 7).

All models expose one interface, :class:`MonitorModel`, whose
``predict_proba_fail`` returns P(y_fail=1) for each prefix row. This lets the
heuristic rule monitor, the trivial baselines, and the sklearn pipelines all be
evaluated and calibrated identically.

Model roster:
  baseline_majority            constant = training failure rate
  baseline_step_count_only     LR on (prefix_step, n_actions_seen) only
  heuristic_rule_monitor       hand-written drift rules, no fitting
  logistic_regression          StandardScaler + LR on structured numeric features
  random_forest                RF on structured numeric features
  hist_gradient_boosting       HGB on structured numeric features
  structured_plus_tfidf_logistic  numeric + TF-IDF(text) -> LR
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .prefix_builder import FEATURE_PREFIX, TEXT_COLUMN
from .utils import DEFAULT_SEED, MONITOR_DIR, ensure_dirs

# ---------------------------------------------------------------------------
# Feature families for the ablation study (Phase 16, Table 2). Names are the
# feature base names (without the ``f__`` prefix); missing ones are skipped.
# ---------------------------------------------------------------------------
FEATURE_FAMILIES: dict[str, list[str]] = {
    "step_count_only": ["prefix_step", "n_actions_seen"],
    "length_pace": [
        "prefix_step", "n_actions_seen", "n_model_tokens_approx",
        "n_observation_chars", "avg_observation_chars",
    ],
    "action_counts": [
        "n_read", "n_search", "n_edit", "n_test", "n_git", "n_install",
        "n_submit", "n_environment", "n_other",
        "edit_to_read_ratio", "test_to_edit_ratio", "search_to_edit_ratio",
    ],
    "context_before_edit": [
        "first_edit_step", "n_reads_before_first_edit",
        "n_searches_before_first_edit", "edited_before_any_read",
        "edited_before_any_search",
    ],
    "file_behavior": [
        "n_unique_files_seen", "n_unique_files_read", "n_unique_files_edited",
        "n_unique_dirs_edited", "n_test_files_touched", "n_src_files_touched",
        "edited_file_never_read_count", "same_file_edit_count_max",
    ],
    "testing_behavior": [
        "n_test_runs", "last_test_returncode", "last_test_fail_count",
        "last_test_pass_count", "test_fail_count_delta", "tests_improving",
        "tests_worsening", "same_test_command_repeated", "n_tracebacks_seen",
        "n_assertion_errors_seen",
    ],
    "loop_behavior": [
        "repeated_exact_command_last_3", "repeated_exact_command_last_5",
        "max_command_repeat_count", "same_action_type_streak",
        "edit_test_edit_test_loop_count", "read_same_file_repeatedly",
    ],
}


def numeric_columns(df: pd.DataFrame) -> list[str]:
    """All ``f__`` feature columns except the text column, in stable order."""
    return [
        c for c in df.columns
        if c.startswith(FEATURE_PREFIX) and c != TEXT_COLUMN
    ]


def family_columns(df: pd.DataFrame, families: list[str]) -> list[str]:
    """Resolve a list of feature families to present ``f__`` columns."""
    if "all_structured" in families or "all_structured_plus_text" in families:
        return numeric_columns(df)
    wanted: list[str] = []
    for fam in families:
        for base in FEATURE_FAMILIES.get(fam, []):
            col = FEATURE_PREFIX + base
            if col in df.columns and col not in wanted:
                wanted.append(col)
    return wanted


def _to_numeric_matrix(df: pd.DataFrame, cols: list[str]) -> np.ndarray:
    return df[cols].apply(pd.to_numeric, errors="coerce").fillna(0.0).to_numpy(dtype=float)


# ---------------------------------------------------------------------------
# Unified model wrapper
# ---------------------------------------------------------------------------
@dataclass
class MonitorModel:
    name: str
    kind: str  # majority | heuristic | sklearn_numeric | sklearn_text
    numeric_cols: list[str] = field(default_factory=list)
    estimator: Any = None
    constant: float = 0.0
    pos_index: int = 1
    use_text: bool = False
    meta: dict[str, Any] = field(default_factory=dict)

    def predict_proba_fail(self, df: pd.DataFrame) -> np.ndarray:
        if self.kind == "majority":
            return np.full(len(df), self.constant, dtype=float)
        if self.kind == "heuristic":
            return heuristic_risk(df)
        if self.kind == "sklearn_text":
            X = df[self.numeric_cols].copy()
            X[TEXT_COLUMN] = df[TEXT_COLUMN].fillna("").astype(str).values
            proba = self.estimator.predict_proba(X)
            return proba[:, self.pos_index]
        # sklearn_numeric
        X = _to_numeric_matrix(df, self.numeric_cols)
        proba = self.estimator.predict_proba(X)
        return proba[:, self.pos_index]

    @staticmethod
    def safe_filename(name: str) -> str:
        return name.replace("::", "__").replace("+", "_").replace("/", "_")

    def save(self, directory: Path | None = None) -> Path:
        import joblib

        directory = directory or MONITOR_DIR
        ensure_dirs(directory)
        path = Path(directory) / f"{self.safe_filename(self.name)}.joblib"
        joblib.dump(self, path)
        return path

    @staticmethod
    def load(path: Path) -> "MonitorModel":
        import joblib

        return joblib.load(path)


# ---------------------------------------------------------------------------
# Heuristic rule monitor (Phase 7 baseline) — no fitting required.
# ---------------------------------------------------------------------------
def heuristic_risk(df: pd.DataFrame) -> np.ndarray:
    """Hand-written drift score in [0,1] from prefix-visible signals.

    Each rule contributes one point; the score is normalized by the number of
    rules whose required columns are present, so it stays a probability-like
    value even when patch/churn features are unavailable (offline).
    """
    n = len(df)

    def col(name: str, default: float = 0.0) -> np.ndarray:
        c = FEATURE_PREFIX + name
        if c in df.columns:
            return pd.to_numeric(df[c], errors="coerce").fillna(default).to_numpy(float)
        return np.full(n, np.nan)

    rules: list[np.ndarray] = []
    edited_before_read = col("edited_before_any_read")
    rules.append(edited_before_read > 0)
    rules.append(col("repeated_exact_command_last_3") > 0)
    # patch growth without test improvement (online only; NaN -> rule skipped)
    pg = col("patch_growth_without_test_improvement")
    rules.append(pg > 0)
    n_test_runs = col("n_test_runs")
    tests_worsening = col("tests_worsening")
    rules.append((n_test_runs >= 2) & (tests_worsening > 0))
    rules.append(col("edited_file_never_read_count") > 0)

    score = np.zeros(n, dtype=float)
    denom = np.zeros(n, dtype=float)
    for r in rules:
        present = ~np.isnan(r.astype(float)) if r.dtype != bool else np.ones(n, bool)
        # boolean arrays from comparisons with NaN inputs yield False; treat
        # rules whose source column is entirely missing as "not evaluated".
        contrib = np.where(np.isnan(r) if r.dtype != bool else False, 0.0, r.astype(float))
        evaluated = present
        score += np.where(evaluated, contrib, 0.0)
        denom += evaluated.astype(float)
    denom = np.where(denom == 0, 1.0, denom)
    return np.clip(score / denom, 0.0, 1.0)


# ---------------------------------------------------------------------------
# Model builders
# ---------------------------------------------------------------------------
def _build_sklearn_numeric(name: str, seed: int):
    from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    if name in ("logistic_regression", "baseline_step_count_only"):
        return Pipeline([
            ("scale", StandardScaler()),
            ("clf", LogisticRegression(max_iter=2000, class_weight="balanced", C=1.0)),
        ])
    if name == "random_forest":
        return RandomForestClassifier(
            n_estimators=300, max_depth=None, min_samples_leaf=5,
            class_weight="balanced", n_jobs=-1, random_state=seed,
        )
    if name == "hist_gradient_boosting":
        return HistGradientBoostingClassifier(
            max_iter=300, learning_rate=0.08, max_depth=None,
            l2_regularization=1.0, random_state=seed,
        )
    raise ValueError(f"unknown numeric model: {name}")


def _build_sklearn_text(seed: int, max_features: int = 50000):
    from sklearn.compose import ColumnTransformer
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    def make(numeric_cols: list[str]) -> Pipeline:
        ct = ColumnTransformer([
            ("num", StandardScaler(), numeric_cols),
            ("text", TfidfVectorizer(
                max_features=max_features, ngram_range=(1, 2),
                min_df=2, sublinear_tf=True,
            ), TEXT_COLUMN),
        ])
        return Pipeline([
            ("features", ct),
            ("clf", LogisticRegression(max_iter=2000, class_weight="balanced", C=1.0)),
        ])

    return make


def _pos_index(estimator) -> int:
    classes = list(getattr(estimator, "classes_", [0, 1]))
    return classes.index(1) if 1 in classes else len(classes) - 1


def train_model(
    name: str,
    train_df: pd.DataFrame,
    seed: int = DEFAULT_SEED,
    shuffle_labels: bool = False,
    text_max_features: int = 50000,
) -> MonitorModel:
    """Fit one model by name and return a :class:`MonitorModel`."""
    y = train_df["y_fail"].to_numpy(dtype=int)
    if shuffle_labels:
        rng = np.random.default_rng(seed)
        y = rng.permutation(y)

    if name == "baseline_majority":
        return MonitorModel(name=name, kind="majority", constant=float(y.mean()))

    if name == "heuristic_rule_monitor":
        return MonitorModel(name=name, kind="heuristic")

    if name == "baseline_step_count_only":
        cols = family_columns(train_df, ["step_count_only"])
        est = _build_sklearn_numeric(name, seed)
        est.fit(_to_numeric_matrix(train_df, cols), y)
        return MonitorModel(
            name=name, kind="sklearn_numeric", numeric_cols=cols,
            estimator=est, pos_index=_pos_index(est[-1]),
        )

    if name in ("logistic_regression", "random_forest", "hist_gradient_boosting"):
        cols = numeric_columns(train_df)
        est = _build_sklearn_numeric(name, seed)
        est.fit(_to_numeric_matrix(train_df, cols), y)
        pos = _pos_index(est[-1] if hasattr(est, "__getitem__") else est)
        return MonitorModel(
            name=name, kind="sklearn_numeric", numeric_cols=cols,
            estimator=est, pos_index=pos,
        )

    if name == "structured_plus_tfidf_logistic":
        cols = numeric_columns(train_df)
        make = _build_sklearn_text(seed, text_max_features)
        est = make(cols)
        X = train_df[cols].copy()
        X[TEXT_COLUMN] = train_df[TEXT_COLUMN].fillna("").astype(str).values
        est.fit(X, y)
        return MonitorModel(
            name=name, kind="sklearn_text", numeric_cols=cols, estimator=est,
            pos_index=_pos_index(est[-1]), use_text=True,
        )

    raise ValueError(f"unknown model name: {name}")


def train_family_model(
    train_df: pd.DataFrame, families: list[str], seed: int = DEFAULT_SEED,
    model: str = "hist_gradient_boosting",
) -> MonitorModel:
    """Train a model restricted to specific feature families (for ablations)."""
    use_text = "all_structured_plus_text" in families
    y = train_df["y_fail"].to_numpy(dtype=int)
    cols = family_columns(train_df, families)
    label = "+".join(families)

    if use_text:
        make = _build_sklearn_text(seed)
        est = make(cols)
        X = train_df[cols].copy()
        X[TEXT_COLUMN] = train_df[TEXT_COLUMN].fillna("").astype(str).values
        est.fit(X, y)
        return MonitorModel(
            name=f"ablation::{label}", kind="sklearn_text", numeric_cols=cols,
            estimator=est, pos_index=_pos_index(est[-1]), use_text=True,
            meta={"families": families},
        )

    est = _build_sklearn_numeric(model, seed)
    est.fit(_to_numeric_matrix(train_df, cols), y)
    pos = _pos_index(est[-1] if hasattr(est, "__getitem__") else est)
    return MonitorModel(
        name=f"ablation::{label}", kind="sklearn_numeric", numeric_cols=cols,
        estimator=est, pos_index=pos, meta={"families": families},
    )


DEFAULT_MODELS = [
    "baseline_majority",
    "baseline_step_count_only",
    "heuristic_rule_monitor",
    "logistic_regression",
    "random_forest",
    "hist_gradient_boosting",
    "structured_plus_tfidf_logistic",
]
