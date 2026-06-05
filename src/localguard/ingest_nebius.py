"""Ingest the Nebius SWE-agent-trajectories dataset (Phase 2).

Primary offline corpus: ``nebius/SWE-agent-trajectories`` (~80k SWE-agent-style
runs, ~1.1GB). We persist a tolerant subset of columns and always store the
``trajectory`` field as a JSON *string* so downstream parquet/JSONL schemas stay
flat and stable regardless of the original nested structure.

Modes:
  * sample : stream the first N rows (no full download) -> data/raw/<name>.jsonl
  * full   : download everything -> data/raw/<name>_full.parquet
  * fixtures: load committed synthetic rows -> data/samples/fixtures.jsonl

No paid APIs are touched here; Hugging Face hosts the data publicly.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

from .utils import RAW_DIR, SAMPLES_DIR, ensure_dirs, read_jsonl, write_jsonl

KEEP_COLUMNS = (
    "instance_id",
    "model_name",
    "target",
    "trajectory",
    "exit_status",
    "generated_patch",
    "eval_logs",
)

DEFAULT_DATASET = "nebius/SWE-agent-trajectories"
SAMPLE_BASENAME = "nebius_sample"
FULL_BASENAME = "nebius_full"


def _jsonify_trajectory(value: Any) -> str:
    """Store trajectory as a compact JSON string (flat parquet/JSONL schema)."""
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=False, default=str)


def _project_row(row: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for col in KEEP_COLUMNS:
        out[col] = row.get(col)
    out["trajectory"] = _jsonify_trajectory(out.get("trajectory"))
    # target may be bool/int/str in the source; keep as-is, coerced at normalize.
    return out


def download_sample(
    dataset: str = DEFAULT_DATASET,
    n: int = 1000,
    split: str = "train",
    out_path: Path | None = None,
) -> Path:
    """Stream the first ``n`` rows and save to JSONL. Avoids full download."""
    from datasets import load_dataset  # local import: heavy dependency

    ensure_dirs(RAW_DIR)
    out_path = out_path or (RAW_DIR / f"{SAMPLE_BASENAME}.jsonl")

    ds = load_dataset(dataset, split=split, streaming=True)
    rows: list[dict[str, Any]] = []
    for i, row in enumerate(ds):
        if i >= n:
            break
        rows.append(_project_row(row))
    write_jsonl(out_path, rows)
    return out_path


def download_full(
    dataset: str = DEFAULT_DATASET,
    split: str = "train",
    out_path: Path | None = None,
) -> Path:
    """Download the full dataset and save as a single parquet file."""
    import pandas as pd
    from datasets import load_dataset

    ensure_dirs(RAW_DIR)
    out_path = out_path or (RAW_DIR / f"{FULL_BASENAME}.parquet")

    ds = load_dataset(dataset, split=split)
    records = (_project_row(dict(r)) for r in ds)
    df = pd.DataFrame.from_records(list(records))
    df.to_parquet(out_path, index=False)
    return out_path


def load_sampled_full(
    max_instances: int | None = None,
    max_success_per_instance: int = 4,
    max_fail_per_instance: int = 4,
    seed: int = 42,
    path: Path | None = None,
    only_instances: set[str] | None = None,
) -> list[dict[str, Any]]:
    """Memory-bounded, instance-diverse sample from the full parquet.

    Per instance, keep up to ``max_success_per_instance`` successful and
    ``max_fail_per_instance`` failed rollouts (success-enriched, since successes
    are rare at ~17% and most informative for *disruption* analysis). Reads the
    full file in row-group batches so peak memory stays small.
    """
    import numpy as np
    import pyarrow.parquet as pq

    path = path or (RAW_DIR / f"{FULL_BASENAME}.parquet")
    light = pq.read_table(path, columns=["instance_id", "target"]).to_pandas()
    light["is_fail"] = ~light["target"].astype(bool)

    rng = np.random.default_rng(seed)
    instances = list(light.groupby("instance_id").groups.keys())
    if only_instances is not None:
        instances = [i for i in instances if i in only_instances]
    rng.shuffle(instances)
    if max_instances is not None:
        instances = instances[:max_instances]
    inst_set = set(instances)

    keep_positions: list[int] = []
    grouped = light[light["instance_id"].isin(inst_set)].groupby("instance_id")
    for _, idxs in grouped.groups.items():
        pos = np.asarray(idxs)
        sub = light.loc[pos]
        fails = pos[sub["is_fail"].to_numpy()]
        succ = pos[~sub["is_fail"].to_numpy()]
        if len(fails) > max_fail_per_instance:
            fails = rng.choice(fails, size=max_fail_per_instance, replace=False)
        if len(succ) > max_success_per_instance:
            succ = rng.choice(succ, size=max_success_per_instance, replace=False)
        keep_positions.extend(int(p) for p in fails)
        keep_positions.extend(int(p) for p in succ)

    keep_set = set(keep_positions)
    out: list[dict[str, Any]] = []
    pf = pq.ParquetFile(path)
    pos = 0
    for batch in pf.iter_batches(batch_size=1000):
        bdf = batch.to_pandas()
        n = len(bdf)
        local = [p - pos for p in range(pos, pos + n) if p in keep_set]
        if local:
            for _, r in bdf.iloc[local].iterrows():
                out.append(dict(r))
        pos += n
    perm = rng.permutation(len(out))
    return [out[i] for i in perm]


def load_raw_rows(input_kind: str, limit: int | None = None) -> list[dict[str, Any]]:
    """Load raw rows by kind: 'sample' | 'full' | 'fixtures' | <path>."""
    rows: Iterable[dict[str, Any]]
    if input_kind == "sample":
        rows = read_jsonl(RAW_DIR / f"{SAMPLE_BASENAME}.jsonl")
    elif input_kind == "full":
        import pandas as pd

        df = pd.read_parquet(RAW_DIR / f"{FULL_BASENAME}.parquet")
        rows = (dict(r) for r in df.to_dict("records"))
    elif input_kind == "fixtures":
        rows = read_jsonl(SAMPLES_DIR / "fixtures.jsonl")
    else:
        p = Path(input_kind)
        if p.suffix == ".parquet":
            import pandas as pd

            rows = (dict(r) for r in pd.read_parquet(p).to_dict("records"))
        else:
            rows = read_jsonl(p)

    out: list[dict[str, Any]] = []
    for i, r in enumerate(rows):
        if limit is not None and i >= limit:
            break
        out.append(dict(r))
    return out
