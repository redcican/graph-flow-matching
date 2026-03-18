"""Product Manifold FM baseline wrapper.

Upstream: https://github.com/ccr-cheng/statistical-flow-matching
Setup:   git clone https://github.com/ccr-cheng/statistical-flow-matching \
             third_party/statistical-flow-matching

Reference: Cheng et al. (2024), "Categorical Flow Matching on
Statistical Manifolds".
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
import torch
from sklearn.preprocessing import LabelEncoder

from graph_flow_matching.baselines.base import BaseGenerator, ColumnSpec
from graph_flow_matching.baselines.registry import register

logger = logging.getLogger(__name__)

_REPO_DIR = (
    Path(__file__).resolve().parents[2]
    / "third_party"
    / "statistical-flow-matching"
)


def _ensure_on_path() -> None:
    repo_str = str(_REPO_DIR)
    if repo_str not in sys.path:
        sys.path.insert(0, repo_str)


@register("product_fm")
class ProductFMGenerator(BaseGenerator):
    """Wrapper around the Product Manifold FM codebase.

    This is the coordinate-wise product manifold flow matching baseline
    (without graph augmentation) that our method builds upon. It processes
    continuous features in R^d, categorical on the probability simplex
    with Fisher-Rao metric, and ordinal on S^1.

    The upstream code is config-driven (YAML) with a registry pattern;
    this wrapper builds configs programmatically from ColumnSpec.
    """

    def __init__(
        self,
        max_iter: int = 10000,
        batch_size: int = 256,
        lr: float = 1e-3,
        hidden_dim: int = 256,
        num_layers: int = 5,
        n_steps: int = 100,
        ot: bool = True,
        device: str = "cuda:0",
        **kwargs: Any,
    ) -> None:
        self._hparams = dict(
            max_iter=max_iter,
            batch_size=batch_size,
            lr=lr,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            n_steps=n_steps,
            ot=ot,
            device=device,
            **kwargs,
        )
        self._model: Any = None
        self._columns: list[ColumnSpec] = []
        self._num_cols: list[str] = []
        self._cat_cols: list[str] = []
        self._ord_cols: list[str] = []
        self._label_encoders: dict[str, LabelEncoder] = {}
        self._n_classes: list[int] = []

    def fit(
        self,
        df: pd.DataFrame,
        columns: list[ColumnSpec],
        **kwargs: Any,
    ) -> None:
        _ensure_on_path()
        self._columns = columns
        self._num_cols = [s.name for s in columns if s.dtype == "continuous"]
        self._cat_cols = [s.name for s in columns if s.dtype == "categorical"]
        self._ord_cols = [s.name for s in columns if s.dtype == "ordinal"]

        # Encode categoricals
        for col in self._cat_cols + self._ord_cols:
            le = LabelEncoder()
            le.fit(df[col].astype(str))
            self._label_encoders[col] = le

        self._n_classes = [
            len(self._label_encoders[col].classes_)
            for col in self._cat_cols + self._ord_cols
        ]

        logger.info(
            "Fitting ProductFM (max_iter=%d, num=%d, cat=%d, ord=%d)",
            self._hparams["max_iter"],
            len(self._num_cols),
            len(self._cat_cols),
            len(self._ord_cols),
        )

        try:
            from models import get_model
            from datasets import get_dataset

            # Build config dict matching upstream YAML format
            cfg = self._build_config(df)
            self._model, self._train_data = self._train_model(df, cfg)
        except ImportError:
            logger.warning(
                "Product Manifold FM repo not found at %s. Clone it first:\n"
                "  git clone https://github.com/ccr-cheng/"
                "statistical-flow-matching %s",
                _REPO_DIR, _REPO_DIR,
            )
            raise

    def _build_config(self, df: pd.DataFrame) -> dict:
        """Build a config dict from hyperparams and column info."""
        return {
            "train": {
                "max_iter": self._hparams["max_iter"],
                "batch_size": self._hparams["batch_size"],
                "optimizer": {"type": "adam", "lr": self._hparams["lr"]},
            },
            "model": {
                "type": "simplex",
                "n_class": self._n_classes,
                "ot": self._hparams["ot"],
                "encoder": {
                    "hidden_dim": self._hparams["hidden_dim"],
                    "num_layers": self._hparams["num_layers"],
                },
            },
            "sample": {
                "steps": self._hparams["n_steps"],
            },
        }

    def _train_model(self, df: pd.DataFrame, cfg: dict) -> tuple[Any, Any]:
        """Run the upstream training loop."""
        from models import get_model

        device = torch.device(self._hparams["device"])

        # Prepare data tensors
        X_parts = []
        if self._num_cols:
            X_num = torch.tensor(
                df[self._num_cols].values, dtype=torch.float32
            )
            X_parts.append(X_num)

        for col in self._cat_cols + self._ord_cols:
            le = self._label_encoders[col]
            encoded = le.transform(df[col].astype(str))
            X_parts.append(torch.tensor(encoded, dtype=torch.long).unsqueeze(1))

        train_data = X_parts
        model = get_model(cfg).to(device)

        optimizer = torch.optim.Adam(model.parameters(), lr=self._hparams["lr"])

        dataset = torch.utils.data.TensorDataset(*X_parts)
        loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=self._hparams["batch_size"],
            shuffle=True,
            drop_last=True,
        )

        model.train()
        for iteration in range(self._hparams["max_iter"]):
            for batch in loader:
                batch = [b.to(device) for b in batch]
                loss = model.get_loss(*batch)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        return model, train_data

    def sample(self, n: int) -> pd.DataFrame:
        if self._model is None:
            raise RuntimeError("Call fit() before sample()")

        device = torch.device(self._hparams["device"])
        self._model.eval()

        with torch.no_grad():
            samples = self._model.sample_euler(
                num_samples=n,
                steps=self._hparams["n_steps"],
            )

        # Convert tensors back to DataFrame
        parts: dict[str, np.ndarray] = {}

        if isinstance(samples, torch.Tensor):
            samples = samples.cpu().numpy()
            offset = 0
            for col in self._num_cols:
                parts[col] = samples[:, offset]
                offset += 1
            for col in self._cat_cols + self._ord_cols:
                le = self._label_encoders[col]
                encoded = np.clip(
                    samples[:, offset].astype(int), 0, len(le.classes_) - 1
                )
                parts[col] = le.inverse_transform(encoded)
                offset += 1
        elif isinstance(samples, (tuple, list)):
            idx = 0
            if self._num_cols:
                X_num = samples[idx].cpu().numpy()
                for i, col in enumerate(self._num_cols):
                    parts[col] = X_num[:, i]
                idx += 1
            for col in self._cat_cols + self._ord_cols:
                enc = samples[idx].cpu().numpy().flatten()
                le = self._label_encoders[col]
                enc = np.clip(enc.astype(int), 0, len(le.classes_) - 1)
                parts[col] = le.inverse_transform(enc)
                idx += 1

        col_order = [s.name for s in self._columns]
        return pd.DataFrame(parts)[[c for c in col_order if c in parts]]

    @property
    def name(self) -> str:
        return "ProductFM"
