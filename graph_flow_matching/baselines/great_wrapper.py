"""GReaT (Generation of Realistic Tabular data) baseline wrapper.

Upstream: https://github.com/tabularis-ai/be_great
Install:  pip install be_great
"""

from __future__ import annotations

import logging
from typing import Any

import pandas as pd

from graph_flow_matching.baselines.base import BaseGenerator, ColumnSpec
from graph_flow_matching.baselines.registry import register

logger = logging.getLogger(__name__)


@register("great")
class GReaTGenerator(BaseGenerator):
    """Thin wrapper around ``be_great.GReaT``."""

    def __init__(
        self,
        llm: str = "distilgpt2",
        epochs: int = 100,
        batch_size: int = 32,
        fp16: bool = True,
        temperature: float = 0.7,
        experiment_dir: str = "trainer_great",
        **kwargs: Any,
    ) -> None:
        try:
            from be_great import GReaT
        except ImportError as exc:
            raise ImportError(
                "GReaT not installed. Run: pip install be_great"
            ) from exc

        self._cls = GReaT
        self._init_kwargs = dict(
            llm=llm,
            epochs=epochs,
            batch_size=batch_size,
            fp16=fp16,
            experiment_dir=experiment_dir,
            **kwargs,
        )
        self._temperature = temperature
        self._model: Any = None
        self._columns: list[ColumnSpec] = []

    def fit(
        self,
        df: pd.DataFrame,
        columns: list[ColumnSpec],
        **kwargs: Any,
    ) -> None:
        self._columns = columns
        logger.info("Fitting GReaT (llm=%s, epochs=%d)",
                     self._init_kwargs["llm"], self._init_kwargs["epochs"])
        self._model = self._cls(**self._init_kwargs)
        self._model.fit(df, **kwargs)

    def sample(self, n: int) -> pd.DataFrame:
        if self._model is None:
            raise RuntimeError("Call fit() before sample()")
        return self._model.sample(
            n_samples=n,
            temperature=self._temperature,
        )

    @property
    def name(self) -> str:
        return "GReaT"
