"""TTVAE baseline wrapper.

Upstream: https://github.com/coksvictoria/TTVAE
Setup:   git clone https://github.com/coksvictoria/TTVAE third_party/TTVAE

Reference: Wang & Nguyen (2025), "TTVAE: Transformer-based Generative
Modeling for Tabular Data Generation", Artificial Intelligence, Elsevier.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

from graph_flow_matching.baselines.base import BaseGenerator, ColumnSpec
from graph_flow_matching.baselines.registry import register

logger = logging.getLogger(__name__)

_REPO_DIR = Path(__file__).resolve().parents[2] / "third_party" / "TTVAE"


def _ensure_on_path() -> None:
    repo_str = str(_REPO_DIR)
    if repo_str not in sys.path:
        sys.path.insert(0, repo_str)


@register("ttvae")
class TTVAEGenerator(BaseGenerator):
    """Wrapper around the TTVAE codebase.

    Expects the repo cloned at ``third_party/TTVAE``.  The upstream
    model takes separate numpy arrays for numerical and categorical
    features; this wrapper handles the DataFrame <-> numpy conversion.
    """

    def __init__(
        self,
        latent_dim: int = 32,
        embedding_dim: int = 128,
        nhead: int = 8,
        dim_feedforward: int = 512,
        dropout: float = 0.1,
        epochs: int = 100,
        batch_size: int = 64,
        loss_factor: float = 1.0,
        device: str = "cuda:0",
        **kwargs: Any,
    ) -> None:
        self._hparams = dict(
            latent_dim=latent_dim,
            embedding_dim=embedding_dim,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            epochs=epochs,
            batch_size=batch_size,
            loss_factor=loss_factor,
            device=device,
            **kwargs,
        )
        self._model: Any = None
        self._columns: list[ColumnSpec] = []
        self._num_cols: list[str] = []
        self._cat_cols: list[str] = []
        self._label_encoders: dict[str, LabelEncoder] = {}

    def _df_to_numpy(
        self, df: pd.DataFrame
    ) -> tuple[np.ndarray | None, np.ndarray | None]:
        """Convert DataFrame to (X_num, X_cat) numpy arrays."""
        X_num = None
        X_cat = None

        if self._num_cols:
            X_num = df[self._num_cols].values.astype(np.float32)

        if self._cat_cols:
            cat_arrays = []
            for col in self._cat_cols:
                le = self._label_encoders[col]
                cat_arrays.append(le.transform(df[col].astype(str)))
            X_cat = np.column_stack(cat_arrays).astype(np.int64)

        return X_num, X_cat

    def _numpy_to_df(
        self, X_num: np.ndarray | None, X_cat: np.ndarray | None
    ) -> pd.DataFrame:
        """Convert (X_num, X_cat) numpy arrays back to DataFrame."""
        parts: dict[str, np.ndarray] = {}

        if X_num is not None:
            for i, col in enumerate(self._num_cols):
                parts[col] = X_num[:, i]

        if X_cat is not None:
            for i, col in enumerate(self._cat_cols):
                le = self._label_encoders[col]
                encoded = np.clip(X_cat[:, i].astype(int), 0, len(le.classes_) - 1)
                parts[col] = le.inverse_transform(encoded)

        # Restore original column order
        col_order = [s.name for s in self._columns]
        return pd.DataFrame(parts)[col_order]

    def fit(
        self,
        df: pd.DataFrame,
        columns: list[ColumnSpec],
        **kwargs: Any,
    ) -> None:
        _ensure_on_path()
        self._columns = columns
        self._num_cols = [s.name for s in columns if s.dtype == "continuous"]
        self._cat_cols = [
            s.name for s in columns if s.dtype in ("categorical", "ordinal")
        ]

        # Fit label encoders for categorical columns
        for col in self._cat_cols:
            le = LabelEncoder()
            le.fit(df[col].astype(str))
            self._label_encoders[col] = le

        X_num, X_cat = self._df_to_numpy(df)
        logger.info(
            "Fitting TTVAE (epochs=%d, num=%d, cat=%d)",
            self._hparams["epochs"],
            len(self._num_cols),
            len(self._cat_cols),
        )

        try:
            from ttvae.model import TTVAE

            self._model = TTVAE(
                latent_dim=self._hparams["latent_dim"],
                embedding_dim=self._hparams["embedding_dim"],
                nhead=self._hparams["nhead"],
                dim_feedforward=self._hparams["dim_feedforward"],
                dropout=self._hparams["dropout"],
                batch_size=self._hparams["batch_size"],
                epochs=self._hparams["epochs"],
                loss_factor=self._hparams["loss_factor"],
                device=self._hparams["device"],
            )
            self._model.fit(X_num, X_cat)
        except ImportError:
            logger.warning(
                "TTVAE repo not found at %s. Clone it first:\n"
                "  git clone https://github.com/coksvictoria/TTVAE %s",
                _REPO_DIR, _REPO_DIR,
            )
            raise

    def sample(self, n: int) -> pd.DataFrame:
        if self._model is None:
            raise RuntimeError("Call fit() before sample()")
        X_num, X_cat = self._model.sample(n)
        return self._numpy_to_df(X_num, X_cat)

    @property
    def name(self) -> str:
        return "TTVAE"
