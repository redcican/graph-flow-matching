"""TabuLa baseline wrapper.

Upstream: https://github.com/zhao-zilong/Tabula
Setup:   git clone https://github.com/zhao-zilong/Tabula third_party/Tabula
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Any

import pandas as pd

from graph_flow_matching.baselines.base import BaseGenerator, ColumnSpec
from graph_flow_matching.baselines.registry import register

logger = logging.getLogger(__name__)

_REPO_DIR = Path(__file__).resolve().parents[2] / "third_party" / "Tabula"


def _ensure_on_path() -> None:
    repo_str = str(_REPO_DIR)
    if repo_str not in sys.path:
        sys.path.insert(0, repo_str)


@register("tabula")
class TabulaGenerator(BaseGenerator):
    """Wrapper around the Tabula codebase.

    TabuLa fine-tunes a pre-trained LLM (e.g. GPT-2) on serialized
    tabular rows and samples new rows via autoregressive generation.
    The upstream ``Tabula`` class accepts DataFrames directly.
    """

    def __init__(
        self,
        llm: str = "gpt2",
        epochs: int = 100,
        batch_size: int = 32,
        experiment_dir: str = "trainer_tabula",
        **kwargs: Any,
    ) -> None:
        self._hparams = dict(
            llm=llm,
            epochs=epochs,
            batch_size=batch_size,
            experiment_dir=experiment_dir,
            **kwargs,
        )
        self._model: Any = None
        self._columns: list[ColumnSpec] = []

    def fit(
        self,
        df: pd.DataFrame,
        columns: list[ColumnSpec],
        **kwargs: Any,
    ) -> None:
        _ensure_on_path()
        self._columns = columns

        cat_cols = [
            s.name for s in columns if s.dtype in ("categorical", "ordinal")
        ]
        logger.info(
            "Fitting TabuLa (llm=%s, epochs=%d)",
            self._hparams["llm"],
            self._hparams["epochs"],
        )

        try:
            from tabula import Tabula

            self._model = Tabula(
                llm=self._hparams["llm"],
                epochs=self._hparams["epochs"],
                batch_size=self._hparams["batch_size"],
                experiment_dir=self._hparams["experiment_dir"],
                categorical_columns=cat_cols,
                **{
                    k: v
                    for k, v in self._hparams.items()
                    if k not in ("llm", "epochs", "batch_size", "experiment_dir")
                },
            )
            self._model.fit(df, **kwargs)
        except ImportError:
            logger.warning(
                "Tabula repo not found at %s. Clone it first:\n"
                "  git clone https://github.com/zhao-zilong/Tabula %s",
                _REPO_DIR, _REPO_DIR,
            )
            raise

    def sample(self, n: int) -> pd.DataFrame:
        if self._model is None:
            raise RuntimeError("Call fit() before sample()")
        return self._model.sample(num_samples=n)

    @property
    def name(self) -> str:
        return "TabuLa"
