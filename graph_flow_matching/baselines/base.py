"""Base abstractions for tabular data generators."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Literal

import pandas as pd


@dataclass(frozen=True)
class ColumnSpec:
    """Metadata for a single column in a tabular dataset.

    Attributes:
        name: Column name matching the DataFrame header.
        dtype: One of "continuous", "categorical", or "ordinal".
        categories: Exhaustive tuple of category labels (required for
            categorical and ordinal columns; ``None`` for continuous).
    """

    name: str
    dtype: Literal["continuous", "categorical", "ordinal"]
    categories: tuple[str, ...] | None = None

    def __post_init__(self) -> None:
        if self.dtype in ("categorical", "ordinal") and self.categories is None:
            raise ValueError(
                f"Column '{self.name}' is {self.dtype} but categories is None"
            )


class BaseGenerator(ABC):
    """Abstract base class that every baseline wrapper must implement."""

    @abstractmethod
    def fit(
        self,
        df: pd.DataFrame,
        columns: list[ColumnSpec],
        **kwargs: Any,
    ) -> None:
        """Train the generator on *df* described by *columns*."""

    @abstractmethod
    def sample(self, n: int) -> pd.DataFrame:
        """Generate *n* synthetic rows. Must be called after :meth:`fit`."""

    @property
    def name(self) -> str:
        """Human-readable name used in result tables."""
        return self.__class__.__name__
