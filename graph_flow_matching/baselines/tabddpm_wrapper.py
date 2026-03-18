"""TabDDPM baseline wrapper.

Upstream: https://github.com/coksvictoria/TTVAE/tree/main/baselines/tabddpm
(originally from https://github.com/rotot0/tab-ddpm)
Setup:   git clone https://github.com/coksvictoria/TTVAE third_party/TTVAE
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
from sklearn.preprocessing import LabelEncoder, QuantileTransformer

from graph_flow_matching.baselines.base import BaseGenerator, ColumnSpec
from graph_flow_matching.baselines.registry import register

logger = logging.getLogger(__name__)

_REPO_DIR = Path(__file__).resolve().parents[2] / "third_party" / "TTVAE"
_BASELINE_DIR = _REPO_DIR / "baselines" / "tabddpm"


def _ensure_on_path() -> None:
    for d in [str(_REPO_DIR), str(_BASELINE_DIR)]:
        if d not in sys.path:
            sys.path.insert(0, d)


def _prepare_data_dir(
    df: pd.DataFrame,
    columns: list[ColumnSpec],
    work_dir: Path,
) -> tuple[list[str], list[str], dict[str, LabelEncoder]]:
    """Write DataFrame to the numpy-file format expected by TabDDPM."""
    num_cols = [s.name for s in columns if s.dtype == "continuous"]
    cat_cols = [s.name for s in columns if s.dtype in ("categorical", "ordinal")]

    label_encoders: dict[str, LabelEncoder] = {}

    # Numerical features
    if num_cols:
        X_num = df[num_cols].values.astype(np.float32)
        np.save(work_dir / "X_num_train.npy", X_num)

    # Categorical features (integer-encoded)
    if cat_cols:
        cat_arrays = []
        for col in cat_cols:
            le = LabelEncoder()
            le.fit(df[col].astype(str))
            label_encoders[col] = le
            cat_arrays.append(le.transform(df[col].astype(str)))
        X_cat = np.column_stack(cat_arrays).astype(np.int64)
        np.save(work_dir / "X_cat_train.npy", X_cat)

    # Dummy target (required by TabDDPM)
    y = np.zeros(len(df), dtype=np.int64)
    np.save(work_dir / "y_train.npy", y)

    # Metadata
    num_classes = [
        int(df[s.name].nunique()) if s.dtype in ("categorical", "ordinal") else None
        for s in columns
    ]
    info = {
        "task_type": "binclass",
        "n_num_features": len(num_cols),
        "n_cat_features": len(cat_cols),
        "num_classes": num_classes,
        "train_size": len(df),
    }
    (work_dir / "info.json").write_text(json.dumps(info, indent=2))

    return num_cols, cat_cols, label_encoders


@register("tabddpm")
class TabDDPMGenerator(BaseGenerator):
    """Wrapper around TabDDPM from the TTVAE baselines.

    TabDDPM applies Gaussian diffusion to continuous features and
    multinomial diffusion to categorical features independently.
    The upstream code uses a file-based interface; this wrapper
    handles DataFrame <-> numpy-file conversion transparently.
    """

    def __init__(
        self,
        steps: int = 1000,
        lr: float = 0.002,
        weight_decay: float = 1e-4,
        batch_size: int = 1024,
        num_timesteps: int = 1000,
        scheduler: str = "cosine",
        model_type: str = "mlp",
        device: str = "cuda:0",
        **kwargs: Any,
    ) -> None:
        self._hparams = dict(
            steps=steps,
            lr=lr,
            weight_decay=weight_decay,
            batch_size=batch_size,
            num_timesteps=num_timesteps,
            scheduler=scheduler,
            model_type=model_type,
            device=device,
            **kwargs,
        )
        self._work_dir: Path | None = None
        self._model_path: Path | None = None
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

        # Create working directory for numpy files
        self._tmpdir = tempfile.TemporaryDirectory(prefix="tabddpm_")
        self._work_dir = Path(self._tmpdir.name)
        self._model_path = self._work_dir / "model"
        self._model_path.mkdir()

        self._num_cols, self._cat_cols, self._label_encoders = _prepare_data_dir(
            df, columns, self._work_dir
        )
        logger.info(
            "Fitting TabDDPM (steps=%d, num=%d, cat=%d)",
            self._hparams["steps"],
            len(self._num_cols),
            len(self._cat_cols),
        )

        try:
            from baselines.tabddpm.train import train

            train(
                model_save_path=str(self._model_path),
                real_data_path=str(self._work_dir),
                **self._hparams,
                **kwargs,
            )
        except ImportError:
            logger.warning(
                "TabDDPM baseline not found at %s. "
                "Clone TTVAE repo first:\n"
                "  git clone https://github.com/coksvictoria/TTVAE %s",
                _BASELINE_DIR, _REPO_DIR,
            )
            raise

    def sample(self, n: int) -> pd.DataFrame:
        if self._model_path is None or self._work_dir is None:
            raise RuntimeError("Call fit() before sample()")

        from baselines.tabddpm.sample import sample

        sample_path = self._work_dir / "synthetic.csv"
        sample(
            model_save_path=str(self._model_path),
            sample_save_path=str(sample_path),
            real_data_path=str(self._work_dir),
            num_samples=n,
            batch_size=min(n, 2000),
            num_timesteps=self._hparams["num_timesteps"],
            scheduler=self._hparams["scheduler"],
            model_type=self._hparams["model_type"],
            device=self._hparams["device"],
        )

        if sample_path.exists():
            return pd.read_csv(sample_path)

        # Fallback: reconstruct from numpy outputs
        parts: dict[str, np.ndarray] = {}
        X_num_path = self._work_dir / "X_num_syn.npy"
        X_cat_path = self._work_dir / "X_cat_syn.npy"

        if X_num_path.exists():
            X_num = np.load(X_num_path)
            for i, col in enumerate(self._num_cols):
                parts[col] = X_num[:, i]

        if X_cat_path.exists():
            X_cat = np.load(X_cat_path)
            for i, col in enumerate(self._cat_cols):
                le = self._label_encoders[col]
                encoded = np.clip(X_cat[:, i].astype(int), 0, len(le.classes_) - 1)
                parts[col] = le.inverse_transform(encoded)

        col_order = [s.name for s in self._columns]
        return pd.DataFrame(parts)[[c for c in col_order if c in parts]]

    @property
    def name(self) -> str:
        return "TabDDPM"
