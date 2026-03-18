"""TabbyFlow baseline wrapper.

Upstream: https://github.com/andresguzco/ef-vfm
Setup:   git clone https://github.com/andresguzco/ef-vfm third_party/ef-vfm

Reference: Guzman-Nateras et al. (2024), "TabbyFlow: Exponential-Family
Variational Flow Matching for Tabular Data".
"""

from __future__ import annotations

import json
import logging
import sys
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

from graph_flow_matching.baselines.base import BaseGenerator, ColumnSpec
from graph_flow_matching.baselines.registry import register

logger = logging.getLogger(__name__)

_REPO_DIR = Path(__file__).resolve().parents[2] / "third_party" / "ef-vfm"


def _ensure_on_path() -> None:
    repo_str = str(_REPO_DIR)
    if repo_str not in sys.path:
        sys.path.insert(0, repo_str)


def _prepare_data_dir(
    df: pd.DataFrame,
    columns: list[ColumnSpec],
    work_dir: Path,
) -> tuple[list[str], list[str], dict[str, LabelEncoder]]:
    """Write DataFrame to the numpy-file format expected by TabbyFlow.

    TabbyFlow expects: X_num_train.npy, X_cat_train.npy, info.json.
    """
    num_cols = [s.name for s in columns if s.dtype == "continuous"]
    cat_cols = [s.name for s in columns if s.dtype in ("categorical", "ordinal")]
    label_encoders: dict[str, LabelEncoder] = {}

    if num_cols:
        X_num = df[num_cols].values.astype(np.float32)
        np.save(work_dir / "X_num_train.npy", X_num)

    if cat_cols:
        cat_arrays = []
        cardinalities = []
        for col in cat_cols:
            le = LabelEncoder()
            le.fit(df[col].astype(str))
            label_encoders[col] = le
            cat_arrays.append(le.transform(df[col].astype(str)))
            cardinalities.append(len(le.classes_))
        X_cat = np.column_stack(cat_arrays).astype(np.int64)
        np.save(work_dir / "X_cat_train.npy", X_cat)
    else:
        cardinalities = []

    y = np.zeros(len(df), dtype=np.int64)
    np.save(work_dir / "y_train.npy", y)

    info = {
        "task_type": "binclass",
        "n_num_features": len(num_cols),
        "n_cat_features": len(cat_cols),
        "cat_cardinalities": cardinalities,
        "train_size": len(df),
    }
    (work_dir / "info.json").write_text(json.dumps(info, indent=2))
    return num_cols, cat_cols, label_encoders


@register("tabbyflow")
class TabbyFlowGenerator(BaseGenerator):
    """Wrapper around the TabbyFlow (ef-vfm) codebase.

    TabbyFlow uses exponential-family variational flow matching with
    distribution-specific source distributions per feature type.
    The upstream code uses a ``Trainer`` class with numpy-file data;
    this wrapper handles DataFrame <-> numpy conversion.
    """

    def __init__(
        self,
        steps: int = 10000,
        batch_size: int = 256,
        lr: float = 1e-3,
        device: str = "cuda:0",
        **kwargs: Any,
    ) -> None:
        self._hparams = dict(
            steps=steps,
            batch_size=batch_size,
            lr=lr,
            device=device,
            **kwargs,
        )
        self._trainer: Any = None
        self._work_dir: Path | None = None
        self._columns: list[ColumnSpec] = []
        self._num_cols: list[str] = []
        self._cat_cols: list[str] = []
        self._label_encoders: dict[str, LabelEncoder] = {}
        self._tmpdir: tempfile.TemporaryDirectory | None = None

    def fit(
        self,
        df: pd.DataFrame,
        columns: list[ColumnSpec],
        **kwargs: Any,
    ) -> None:
        _ensure_on_path()
        self._columns = columns

        self._tmpdir = tempfile.TemporaryDirectory(prefix="tabbyflow_")
        self._work_dir = Path(self._tmpdir.name)

        self._num_cols, self._cat_cols, self._label_encoders = _prepare_data_dir(
            df, columns, self._work_dir
        )
        logger.info(
            "Fitting TabbyFlow (steps=%d, num=%d, cat=%d)",
            self._hparams["steps"],
            len(self._num_cols),
            len(self._cat_cols),
        )

        try:
            from src.data import Dataset
            from ef_vfm.models.flow_model import ExpVFM
            from ef_vfm.trainer import Trainer

            dataset = Dataset.from_dir(str(self._work_dir))
            train_loader = dataset.build_loader(
                batch_size=self._hparams["batch_size"], split="train"
            )

            flow = ExpVFM(
                num_dims_num=len(self._num_cols),
                num_dims_cat=len(self._cat_cols),
                categorical_cardinalities=dataset.cat_cardinalities,
                device=self._hparams["device"],
            )

            self._trainer = Trainer(
                flow=flow,
                train_loader=train_loader,
                val_loader=None,
                steps=self._hparams["steps"],
                lr=self._hparams["lr"],
            )
            self._trainer.run_loop()
        except ImportError:
            logger.warning(
                "TabbyFlow repo not found at %s. Clone it first:\n"
                "  git clone https://github.com/andresguzco/ef-vfm %s",
                _REPO_DIR, _REPO_DIR,
            )
            raise

    def sample(self, n: int) -> pd.DataFrame:
        if self._trainer is None:
            raise RuntimeError("Call fit() before sample()")
        return self._trainer.sample_synthetic(num_samples=n)

    @property
    def name(self) -> str:
        return "TabbyFlow"
