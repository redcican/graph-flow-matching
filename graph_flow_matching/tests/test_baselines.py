"""Tests for the baselines module (registry, wrappers, base classes)."""

import numpy as np
import pandas as pd
import pytest
import torch

from graph_flow_matching.baselines.base import BaseGenerator, ColumnSpec
from graph_flow_matching.baselines.registry import REGISTRY, create, register


# ===================================================================
# ColumnSpec
# ===================================================================

class TestColumnSpec:
    def test_continuous(self):
        cs = ColumnSpec("x", "continuous")
        assert cs.name == "x"
        assert cs.dtype == "continuous"
        assert cs.categories is None

    def test_categorical_requires_categories(self):
        with pytest.raises((ValueError, TypeError)):
            ColumnSpec("c", "categorical")  # no categories

    def test_ordinal_requires_categories(self):
        with pytest.raises((ValueError, TypeError)):
            ColumnSpec("o", "ordinal")  # no categories

    def test_categorical_with_categories(self):
        cs = ColumnSpec("c", "categorical", ("a", "b", "c"))
        assert cs.categories == ("a", "b", "c")

    def test_frozen(self):
        cs = ColumnSpec("x", "continuous")
        with pytest.raises(AttributeError):
            cs.name = "y"


# ===================================================================
# Registry
# ===================================================================

class TestRegistry:
    def test_all_baselines_registered(self):
        expected = {
            "ctgan", "ttvae", "great", "tabula", "forest_flow",
            "tabddpm", "tabsyn", "tabbyflow", "product_fm", "gafm",
        }
        assert expected.issubset(set(REGISTRY.keys()))

    def test_create_gafm(self):
        gen = create("gafm")
        assert isinstance(gen, BaseGenerator)
        assert gen.name == "GAFM"

    def test_create_unknown_raises(self):
        with pytest.raises(KeyError):
            create("nonexistent_baseline")

    def test_register_custom(self):
        @register("test_dummy_baseline")
        class DummyGen(BaseGenerator):
            def fit(self, df, columns, **kwargs):
                pass
            def sample(self, n):
                return pd.DataFrame({"x": np.zeros(n)})
            @property
            def name(self):
                return "Dummy"

        gen = create("test_dummy_baseline")
        assert gen.name == "Dummy"
        df = gen.sample(5)
        assert len(df) == 5
        # Cleanup
        del REGISTRY["test_dummy_baseline"]


# ===================================================================
# GAFM Wrapper
# ===================================================================

class TestGAFMWrapper:
    def test_fit_and_sample(self, mixed_columns, mixed_df):
        from graph_flow_matching.models.velocity_field import GAFMConfig
        cfg = GAFMConfig(
            hidden_dim=32, num_mlp_layers=2, time_embed_dim=16,
            feat_gnn_layers=1, samp_gnn_layers=1, samp_gnn_hidden=16,
            k_neighbors=3, batch_size=16, num_epochs=2,
            sinkhorn_iterations=5, num_workers=0, pin_memory=False,
            max_steps_per_epoch=2, n_ode_steps=3,
        )
        gen = create("gafm", config=cfg, device=torch.device("cpu"))
        gen.fit(mixed_df, mixed_columns)
        syn = gen.sample(10)
        assert isinstance(syn, pd.DataFrame)
        assert len(syn) == 10
        assert set(syn.columns) == {c.name for c in mixed_columns}

    def test_sample_before_fit_raises(self):
        gen = create("gafm")
        with pytest.raises(RuntimeError, match="fit"):
            gen.sample(5)


# ===================================================================
# Baseline wrapper instantiation (no upstream deps required)
# ===================================================================

class TestBaselineInstantiation:
    @pytest.mark.parametrize("name", [
        "ctgan", "ttvae", "great", "tabula", "forest_flow",
        "tabddpm", "tabsyn", "tabbyflow", "product_fm",
    ])
    def test_can_instantiate(self, name):
        """All wrappers should be instantiable without upstream dependencies."""
        try:
            gen = create(name)
        except ImportError:
            pytest.skip(f"{name} upstream dependency not installed")
        assert isinstance(gen, BaseGenerator)
        assert isinstance(gen.name, str)
        assert len(gen.name) > 0
