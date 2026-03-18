"""Wrapper exposing Graph-Augmented Flow Matching via the BaseGenerator interface."""

from __future__ import annotations

from typing import Any, TYPE_CHECKING

import pandas as pd
import torch

from graph_flow_matching.baselines.base import BaseGenerator, ColumnSpec
from graph_flow_matching.baselines.registry import register

if TYPE_CHECKING:
    from graph_flow_matching.models.velocity_field import GAFMConfig
    from graph_flow_matching.training.sampler import GAFMSampler
    from graph_flow_matching.training.trainer import GAFMTrainer


@register("gafm")
class GAFMGenerator(BaseGenerator):
    """Graph-Augmented Flow Matching generator (our method)."""

    def __init__(
        self,
        config: Any = None,
        device: torch.device | None = None,
    ) -> None:
        from graph_flow_matching.models.velocity_field import GAFMConfig

        self._config = config or GAFMConfig()
        self._device = device
        self._trainer: GAFMTrainer | None = None
        self._sampler: GAFMSampler | None = None

    def fit(
        self,
        df: pd.DataFrame,
        columns: list[ColumnSpec],
        **kwargs: Any,
    ) -> None:
        from graph_flow_matching.training.trainer import GAFMTrainer
        from graph_flow_matching.training.sampler import GAFMSampler

        self._trainer = GAFMTrainer(columns, self._config, self._device)
        model = self._trainer.fit(df)
        self._sampler = GAFMSampler(model, self._trainer.preprocessor, self._config, self._device)

    def sample(self, n: int) -> pd.DataFrame:
        if self._sampler is None:
            raise RuntimeError("Call fit() before sample()")
        return self._sampler.sample(n)

    @property
    def name(self) -> str:
        return "GAFM"
