"""Shared utilities: paths, config loading, JSONL IO, hashing, logging, seeding.

Kept dependency-light on purpose so that the data/feature path has no hidden
coupling to sklearn or matplotlib (those are imported only where used).
"""

from __future__ import annotations

import hashlib
import json
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Iterator

import yaml

# ---------------------------------------------------------------------------
# Repository paths. Resolved relative to this file so scripts work regardless of
# the current working directory.
# ---------------------------------------------------------------------------
PKG_DIR = Path(__file__).resolve().parent
REPO_ROOT = PKG_DIR.parent.parent

DATA_DIR = REPO_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
INTERIM_DIR = DATA_DIR / "interim"
PROCESSED_DIR = DATA_DIR / "processed"
SAMPLES_DIR = DATA_DIR / "samples"

MODELS_DIR = REPO_ROOT / "models"
MONITOR_DIR = MODELS_DIR / "monitor"
CALIBRATOR_DIR = MODELS_DIR / "calibrators"

RESULTS_DIR = REPO_ROOT / "results"
RESULTS_OFFLINE = RESULTS_DIR / "offline"
RESULTS_ONLINE = RESULTS_DIR / "online"
RESULTS_FIGURES = RESULTS_DIR / "figures"
RESULTS_TABLES = RESULTS_DIR / "tables"
RESULTS_AUDITS = RESULTS_DIR / "audits"

CONFIG_DIR = REPO_ROOT / "configs"

DEFAULT_SEED = 42


def ensure_dirs(*paths: Path) -> None:
    """Create each directory (and parents) if missing."""
    for p in paths:
        Path(p).mkdir(parents=True, exist_ok=True)


def set_global_seed(seed: int = DEFAULT_SEED) -> None:
    """Seed python + numpy RNGs for reproducibility. numpy is optional here."""
    random.seed(seed)
    os.environ.setdefault("PYTHONHASHSEED", str(seed))
    try:  # numpy may not be needed in pure-IO contexts
        import numpy as np

        np.random.seed(seed)
    except Exception:  # pragma: no cover - numpy always present in practice
        pass


def stable_hash(*parts: Any, length: int = 12) -> str:
    """Deterministic short hash from arbitrary string-able parts.

    Used to mint trajectory_id / prefix_id values that are stable across runs
    (unlike python's salted ``hash``).
    """
    h = hashlib.sha1("\x1f".join(str(p) for p in parts).encode("utf-8"))
    return h.hexdigest()[:length]


# ---------------------------------------------------------------------------
# JSONL IO
# ---------------------------------------------------------------------------
def write_jsonl(path: str | Path, rows: Iterable[dict[str, Any]]) -> int:
    """Write an iterable of dicts to a JSONL file. Returns the row count."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False))
            fh.write("\n")
            n += 1
    return n


def read_jsonl(path: str | Path) -> Iterator[dict[str, Any]]:
    """Yield dicts from a JSONL file, skipping blank lines."""
    path = Path(path)
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def append_jsonl(path: str | Path, row: dict[str, Any]) -> None:
    """Append a single record to a JSONL file (creating it if needed)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(row, ensure_ascii=False))
        fh.write("\n")


def write_json(path: str | Path, obj: Any, indent: int = 2) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        json.dump(obj, fh, ensure_ascii=False, indent=indent, default=str)


def read_json(path: str | Path) -> Any:
    with Path(path).open("r", encoding="utf-8") as fh:
        return json.load(fh)


# ---------------------------------------------------------------------------
# Config loading
# ---------------------------------------------------------------------------
@dataclass
class Config:
    """Thin dict-backed config wrapper with dotted-key access and defaults."""

    data: dict[str, Any]
    path: Path | None = None

    def get(self, dotted: str, default: Any = None) -> Any:
        cur: Any = self.data
        for key in dotted.split("."):
            if not isinstance(cur, dict) or key not in cur:
                return default
            cur = cur[key]
        return cur

    def __getitem__(self, key: str) -> Any:
        return self.data[key]

    def __contains__(self, key: str) -> bool:
        return key in self.data


def load_config(path: str | Path) -> Config:
    """Load a YAML config file into a :class:`Config`."""
    path = Path(path)
    if not path.is_absolute():
        # allow passing "configs/foo.yaml" from any cwd
        cand = REPO_ROOT / path
        path = cand if cand.exists() else path
    with path.open("r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or {}
    return Config(data=data, path=path)


# ---------------------------------------------------------------------------
# Leakage guard: column/field names that must never enter feature vectors.
# Centralized here so features.py, prefix_builder.py and tests agree.
# ---------------------------------------------------------------------------
LEAKAGE_FIELDS = frozenset(
    {
        "target",
        "y_fail",
        "eval_logs",
        "generated_patch",
        "exit_status",
        "resolved",
        "passed_final",
        "fail_to_pass",
        "pass_to_pass",
        "final_patch",
    }
)


def assert_no_leakage(feature_dict: dict[str, Any]) -> None:
    """Raise if any feature key collides with a forbidden outcome field.

    Substring match is intentional: a key like ``target_file_count`` would be a
    bug we want surfaced (it reads as if it carries the label).
    """
    bad = [
        k
        for k in feature_dict
        if any(forbidden in k.lower() for forbidden in LEAKAGE_FIELDS)
    ]
    if bad:
        raise ValueError(f"Leakage: feature keys reference outcome fields: {bad}")
