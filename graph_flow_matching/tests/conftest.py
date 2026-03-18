"""Shared fixtures for graph_flow_matching tests."""

import numpy as np
import pandas as pd
import pytest
import torch

from graph_flow_matching.baselines.base import ColumnSpec
from graph_flow_matching.models.velocity_field import GAFMConfig


# ---------------------------------------------------------------------------
# Deterministic seeding
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def _seed():
    torch.manual_seed(42)
    np.random.seed(42)


# ---------------------------------------------------------------------------
# Device
# ---------------------------------------------------------------------------

@pytest.fixture()
def device():
    return torch.device("cpu")


# ---------------------------------------------------------------------------
# Column specifications
# ---------------------------------------------------------------------------

@pytest.fixture()
def mixed_columns():
    """Columns with all three types: continuous, categorical, ordinal."""
    return [
        ColumnSpec("age", "continuous"),
        ColumnSpec("income", "continuous"),
        ColumnSpec("color", "categorical", ("red", "blue", "green")),
        ColumnSpec("size", "categorical", ("S", "M", "L", "XL")),
        ColumnSpec("education", "ordinal", ("low", "mid", "high")),
    ]


@pytest.fixture()
def cont_only_columns():
    return [
        ColumnSpec("x1", "continuous"),
        ColumnSpec("x2", "continuous"),
        ColumnSpec("x3", "continuous"),
    ]


@pytest.fixture()
def cat_only_columns():
    return [
        ColumnSpec("c1", "categorical", ("a", "b", "c")),
        ColumnSpec("c2", "categorical", ("x", "y")),
    ]


# ---------------------------------------------------------------------------
# Synthetic DataFrames
# ---------------------------------------------------------------------------

@pytest.fixture()
def mixed_df():
    n = 80
    return pd.DataFrame({
        "age": np.random.randn(n) * 10 + 50,
        "income": np.random.randn(n) * 20000 + 60000,
        "color": np.random.choice(["red", "blue", "green"], n),
        "size": np.random.choice(["S", "M", "L", "XL"], n),
        "education": np.random.choice(["low", "mid", "high"], n),
    })


@pytest.fixture()
def cont_df():
    n = 60
    return pd.DataFrame({
        "x1": np.random.randn(n),
        "x2": np.random.randn(n) * 2 + 1,
        "x3": np.random.randn(n) * 0.5,
    })


# ---------------------------------------------------------------------------
# Small GAFMConfig for fast tests
# ---------------------------------------------------------------------------

@pytest.fixture()
def small_config():
    return GAFMConfig(
        hidden_dim=32,
        num_mlp_layers=2,
        time_embed_dim=16,
        feat_gnn_layers=1,
        samp_gnn_layers=1,
        samp_gnn_hidden=16,
        k_neighbors=3,
        lr=1e-3,
        batch_size=16,
        num_epochs=4,
        sinkhorn_iterations=10,
        val_fraction=0.2,
        num_workers=0,
        pin_memory=False,
        max_steps_per_epoch=3,
        n_ode_steps=5,
    )


# ---------------------------------------------------------------------------
# Common tensor fixtures
# ---------------------------------------------------------------------------

B = 8  # default batch size for unit tests


@pytest.fixture()
def batch_continuous():
    return torch.randn(B, 3)


@pytest.fixture()
def batch_categorical():
    """List of two categorical distributions on the simplex."""
    return [
        torch.softmax(torch.randn(B, 3), dim=-1),
        torch.softmax(torch.randn(B, 4), dim=-1),
    ]


@pytest.fixture()
def batch_ordinal():
    """List of one ordinal angle tensor."""
    return [torch.rand(B) * 2 * torch.pi]


@pytest.fixture()
def time_batch():
    return torch.rand(B)
