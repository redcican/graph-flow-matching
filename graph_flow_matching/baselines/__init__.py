"""Baseline generators for tabular data synthesis."""

from graph_flow_matching.baselines.base import BaseGenerator, ColumnSpec
from graph_flow_matching.baselines.registry import REGISTRY, create, register

# Import all wrappers to trigger registration
from graph_flow_matching.baselines import (  # noqa: F401
    ctgan_wrapper,
    forest_flow_wrapper,
    gafm_wrapper,
    great_wrapper,
    product_fm_wrapper,
    tabbyflow_wrapper,
    tabddpm_wrapper,
    tabsyn_wrapper,
    tabula_wrapper,
    ttvae_wrapper,
)

__all__ = [
    "BaseGenerator",
    "ColumnSpec",
    "REGISTRY",
    "create",
    "register",
]
