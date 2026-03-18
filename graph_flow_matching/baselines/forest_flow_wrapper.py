"""Forest Flow (ForestDiffusion) baseline wrapper.

Upstream: https://github.com/SamsungSAILMontreal/ForestDiffusion
Install:  pip install ForestDiffusion
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd

from graph_flow_matching.baselines.base import BaseGenerator, ColumnSpec
from graph_flow_matching.baselines.registry import register

logger = logging.getLogger(__name__)


@register("forest_flow")
class ForestFlowGenerator(BaseGenerator):
    """Thin wrapper around ``ForestDiffusion.ForestDiffusionModel``."""

    def __init__(
        self,
        n_t: int = 50,
        diffusion_type: str = "flow",
        max_depth: int = 7,
        n_estimators: int = 100,
        duplicate_K: int = 100,
        seed: int = 42,
        **kwargs: Any,
    ) -> None:
        try:
            from ForestDiffusion import ForestDiffusionModel
        except ImportError as exc:
            raise ImportError(
                "ForestDiffusion not installed. Run: pip install ForestDiffusion"
            ) from exc

        self._cls = ForestDiffusionModel
        self._init_kwargs = dict(
            n_t=n_t,
            diffusion_type=diffusion_type,
            max_depth=max_depth,
            n_estimators=n_estimators,
            duplicate_K=duplicate_K,
            seed=seed,
            **kwargs,
        )
        self._model: Any = None
        self._columns: list[ColumnSpec] = []
        self._col_names: list[str] = []

    def fit(
        self,
        df: pd.DataFrame,
        columns: list[ColumnSpec],
        **kwargs: Any,
    ) -> None:
        self._columns = columns
        self._col_names = [s.name for s in columns]

        cat_indexes = [
            i for i, s in enumerate(columns)
            if s.dtype in ("categorical", "ordinal")
        ]

        X = df[self._col_names].values
        logger.info(
            "Fitting ForestFlow (n_t=%d, cat_indexes=%s)",
            self._init_kwargs["n_t"], cat_indexes,
        )
        self._model = self._cls(
            X=X,
            cat_indexes=cat_indexes,
            **self._init_kwargs,
        )

    def sample(self, n: int) -> pd.DataFrame:
        if self._model is None:
            raise RuntimeError("Call fit() before sample()")
        X_syn = self._model.generate(batch_size=n)
        return pd.DataFrame(X_syn, columns=self._col_names)

    @property
    def name(self) -> str:
        return "ForestFlow"
