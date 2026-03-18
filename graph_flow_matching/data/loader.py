"""Dataset registry and loading utilities.

Each dataset entry maps a short name to its source and column metadata.
Actual download/caching logic will be filled in once we finalize the
19-benchmark suite; for now the registry serves as documentation.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from graph_flow_matching.baselines.base import ColumnSpec


@dataclass(frozen=True)
class DatasetInfo:
    """Metadata for a benchmark dataset."""

    name: str
    n_samples: int
    source: str  # "uci", "kaggle", "openml"
    task: str  # "classification", "regression"
    target_column: str
    url: str = ""


# Will be populated per-dataset as we add benchmark configs
DATASET_REGISTRY: dict[str, DatasetInfo] = {}


def load_dataset(
    name: str,
    data_dir: str | Path = "data/raw",
) -> tuple[pd.DataFrame, list[ColumnSpec]]:
    """Load a registered dataset and return (DataFrame, column_specs).

    Parameters
    ----------
    name:
        Short name matching a key in ``DATASET_REGISTRY``.
    data_dir:
        Root directory containing raw CSV files.

    Returns
    -------
    df:
        Full dataset as a pandas DataFrame.
    columns:
        List of :class:`ColumnSpec` describing each column.
    """
    if name not in DATASET_REGISTRY:
        available = ", ".join(sorted(DATASET_REGISTRY)) or "(none)"
        raise KeyError(f"Unknown dataset '{name}'. Available: {available}")

    csv_path = Path(data_dir) / f"{name}.csv"
    if not csv_path.exists():
        raise FileNotFoundError(
            f"Expected CSV at {csv_path}. Download or place the file first."
        )

    df = pd.read_csv(csv_path)
    columns = infer_column_specs(df)
    return df, columns


def infer_column_specs(
    df: pd.DataFrame,
    categorical_threshold: int = 20,
    ordinal_columns: dict[str, tuple[str, ...]] | None = None,
) -> list[ColumnSpec]:
    """Heuristically assign column types from a DataFrame.

    Parameters
    ----------
    df:
        Input data.
    categorical_threshold:
        Columns with fewer unique values than this are treated as categorical.
    ordinal_columns:
        Explicit mapping of column name -> ordered categories for ordinal
        columns. If not provided, no columns are marked ordinal.
    """
    ordinal_columns = ordinal_columns or {}
    specs: list[ColumnSpec] = []

    for col in df.columns:
        if col in ordinal_columns:
            specs.append(
                ColumnSpec(
                    name=col,
                    dtype="ordinal",
                    categories=ordinal_columns[col],
                )
            )
        elif df[col].dtype == object or df[col].nunique() < categorical_threshold:
            cats = tuple(sorted(df[col].dropna().unique().astype(str)))
            specs.append(ColumnSpec(name=col, dtype="categorical", categories=cats))
        else:
            specs.append(ColumnSpec(name=col, dtype="continuous"))

    return specs
