"""CTGAN baseline wrapper.

Upstream: https://github.com/sdv-dev/CTGAN
Install:  pip install ctgan
"""

from __future__ import annotations

import logging
from typing import Any

import pandas as pd

from graph_flow_matching.baselines.base import BaseGenerator, ColumnSpec
from graph_flow_matching.baselines.registry import register

logger = logging.getLogger(__name__)


@register("ctgan")
class CTGANGenerator(BaseGenerator):
    """Thin wrapper around ``ctgan.CTGAN``."""

    def __init__(
        self,
        epochs: int = 300,
        batch_size: int = 500,
        generator_dim: tuple[int, ...] = (256, 256),
        discriminator_dim: tuple[int, ...] = (256, 256),
        generator_lr: float = 2e-4,
        discriminator_lr: float = 2e-4,
        pac: int = 10,
        cuda: bool = True,
        **kwargs: Any,
    ) -> None:
        try:
            from ctgan import CTGAN
        except ImportError as exc:
            raise ImportError(
                "CTGAN not installed. Run: pip install ctgan"
            ) from exc

        self._model = CTGAN(
            epochs=epochs,
            batch_size=batch_size,
            generator_dim=generator_dim,
            discriminator_dim=discriminator_dim,
            generator_lr=generator_lr,
            discriminator_lr=discriminator_lr,
            pac=pac,
            cuda=cuda,
            **kwargs,
        )
        self._columns: list[ColumnSpec] = []

    def fit(
        self,
        df: pd.DataFrame,
        columns: list[ColumnSpec],
        **kwargs: Any,
    ) -> None:
        self._columns = columns
        discrete_cols = [
            s.name for s in columns if s.dtype in ("categorical", "ordinal")
        ]
        logger.info("Fitting CTGAN (discrete_columns=%s)", discrete_cols)
        self._model.fit(df, discrete_columns=discrete_cols)

    def sample(self, n: int) -> pd.DataFrame:
        return self._model.sample(n)

    @property
    def name(self) -> str:
        return "CTGAN"
