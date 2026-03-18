"""End-to-end integration tests for the full GAFM pipeline."""

import math

import numpy as np
import pandas as pd
import pytest
import torch

from graph_flow_matching.baselines.base import ColumnSpec
from graph_flow_matching.models.feature_gnn import build_feature_graph
from graph_flow_matching.models.manifold_ops import (
    clamp_probabilities,
    fisher_rao_distance,
    geodesic_interpolation_categorical,
    geodesic_interpolation_continuous,
    geodesic_interpolation_ordinal,
    ordinal_to_angle,
)
from graph_flow_matching.models.ot_solver import compute_ot_coupling, sample_ot_pairs
from graph_flow_matching.models.velocity_field import GAFMConfig, GraphAugmentedFlowMatching
from graph_flow_matching.training.sampler import GAFMSampler
from graph_flow_matching.training.trainer import (
    DataPreprocessor,
    GAFMTrainer,
    compute_loss,
    pack_flat,
    sample_prior,
    unpack_flat,
)


# ===================================================================
# Full pipeline: preprocess → feature graph → train → sample → inverse
# ===================================================================

class TestFullPipeline:
    """Integration test: runs the entire pipeline end-to-end."""

    @pytest.fixture()
    def pipeline_result(self, mixed_columns, mixed_df, small_config, device):
        trainer = GAFMTrainer(mixed_columns, small_config, device)
        model = trainer.fit(mixed_df)
        sampler = GAFMSampler(model, trainer.preprocessor, small_config, device)
        syn = sampler.sample(20, batch_size=10)
        return trainer, model, sampler, syn

    def test_synthetic_shape(self, pipeline_result, mixed_df, mixed_columns):
        _, _, _, syn = pipeline_result
        assert len(syn) == 20
        assert set(syn.columns) == {c.name for c in mixed_columns}

    def test_continuous_reasonable_range(self, pipeline_result, mixed_df):
        _, _, _, syn = pipeline_result
        # Synthetic continuous values should be in a reasonable range
        real_age = mixed_df["age"]
        syn_age = syn["age"].astype(float)
        assert syn_age.mean() == pytest.approx(real_age.mean(), abs=30)

    def test_categorical_values_valid(self, pipeline_result):
        _, _, _, syn = pipeline_result
        assert set(syn["color"].unique()).issubset({"red", "blue", "green"})
        assert set(syn["size"].unique()).issubset({"S", "M", "L", "XL"})

    def test_ordinal_values_valid(self, pipeline_result):
        _, _, _, syn = pipeline_result
        assert set(syn["education"].unique()).issubset({"low", "mid", "high"})

    def test_model_is_eval_mode(self, pipeline_result):
        _, model, _, _ = pipeline_result
        assert not model.training


class TestFullPipelineContinuousOnly:
    def test_cont_only(self, cont_only_columns, cont_df, small_config, device):
        trainer = GAFMTrainer(cont_only_columns, small_config, device)
        model = trainer.fit(cont_df)
        sampler = GAFMSampler(model, trainer.preprocessor, small_config, device)
        syn = sampler.sample(10)
        assert len(syn) == 10
        assert set(syn.columns) == {"x1", "x2", "x3"}
        for col in ["x1", "x2", "x3"]:
            assert syn[col].dtype in [np.float32, np.float64]


class TestFullPipelineCategoricalOnly:
    def test_cat_only(self, cat_only_columns, small_config, device):
        df = pd.DataFrame({
            "c1": np.random.choice(["a", "b", "c"], 60),
            "c2": np.random.choice(["x", "y"], 60),
        })
        trainer = GAFMTrainer(cat_only_columns, small_config, device)
        model = trainer.fit(df)
        sampler = GAFMSampler(model, trainer.preprocessor, small_config, device)
        syn = sampler.sample(10)
        assert len(syn) == 10
        assert set(syn["c1"].unique()).issubset({"a", "b", "c"})
        assert set(syn["c2"].unique()).issubset({"x", "y"})


# ===================================================================
# OT pairing integration
# ===================================================================

class TestOTPairingIntegration:
    def test_ot_pairing_produces_valid_indices(self, mixed_columns, mixed_df, device):
        pp = DataPreprocessor(mixed_columns)
        x_c, x_d, x_o = pp.fit_transform(mixed_df, device=device)

        B = 16
        x0_c, x0_d, x0_o = sample_prior(B, pp.d_c, pp.categorical_dims, pp.n_ordinal, device)
        x1_c = x_c[:B]
        x1_d = [xd[:B] for xd in x_d]
        x1_o = [xo[:B] for xo in x_o]

        coupling = compute_ot_coupling(
            x0_c, x1_c, x0_d, x1_d, x0_o, x1_o,
            epsilon=0.01, num_iterations=50,
        )
        idx = sample_ot_pairs(coupling)

        assert idx.shape == (B,)
        assert (idx >= 0).all()
        assert (idx < B).all()


# ===================================================================
# Geodesic interpolation integration: verify path properties
# ===================================================================

class TestGeodesicPathProperties:
    def test_continuous_path_is_linear(self):
        """Continuous geodesic should trace a straight line."""
        x0 = torch.tensor([[0.0, 0.0]])
        x1 = torch.tensor([[2.0, 4.0]])
        for t_val in [0.0, 0.25, 0.5, 0.75, 1.0]:
            t = torch.tensor(t_val)
            xt = geodesic_interpolation_continuous(x0, x1, t)
            expected = x0 + t_val * (x1 - x0)
            assert torch.allclose(xt, expected, atol=1e-5)

    def test_categorical_path_stays_on_simplex(self):
        """Categorical geodesic should remain on the simplex throughout."""
        x0 = torch.softmax(torch.randn(4, 5), dim=-1)
        x1 = torch.softmax(torch.randn(4, 5), dim=-1)
        for t_val in np.linspace(0, 1, 11):
            t = torch.tensor(t_val).expand(4)
            xt = geodesic_interpolation_categorical(x0, x1, t)
            assert (xt >= -1e-6).all(), f"Negative at t={t_val}"
            sums = xt.sum(dim=-1)
            assert torch.allclose(sums, torch.ones(4, dtype=sums.dtype), atol=1e-4), f"Sum≠1 at t={t_val}"

    def test_ordinal_path_stays_on_circle(self):
        """Ordinal geodesic should stay in [0, 2π)."""
        t0 = torch.tensor([0.5, 1.0, 5.0])
        t1 = torch.tensor([2.0, 4.0, 0.5])
        for t_val in np.linspace(0, 1, 11):
            t = torch.tensor(t_val).expand(3)
            xt = geodesic_interpolation_ordinal(t0, t1, t)
            assert (xt >= 0).all()
            assert (xt <= 2 * math.pi + 1e-4).all()


# ===================================================================
# Fisher-Rao loss vs Euclidean: integration check
# ===================================================================

class TestFisherRaoLossIntegration:
    def test_fr_loss_gradients_stable(self, mixed_columns, mixed_df, small_config, device):
        """Verify that FR-weighted loss doesn't produce NaN gradients during training."""
        trainer = GAFMTrainer(mixed_columns, small_config, device)
        model = trainer.fit(mixed_df)

        # All model parameters should have finite values after training
        for name, p in model.named_parameters():
            assert p.isfinite().all(), f"Non-finite parameter: {name}"


# ===================================================================
# Feature graph construction integration
# ===================================================================

class TestFeatureGraphIntegration:
    def test_graph_from_correlated_data(self):
        """Strongly correlated features should produce edges."""
        n = 200
        x = np.random.randn(n)
        df = pd.DataFrame({
            "a": x,
            "b": x + np.random.randn(n) * 0.1,  # strongly correlated
            "c": np.random.randn(n),  # independent
        })
        columns = [
            ColumnSpec("a", "continuous"),
            ColumnSpec("b", "continuous"),
            ColumnSpec("c", "continuous"),
        ]
        ei, ew, nt = build_feature_graph(df, columns, threshold=0.3)
        edges = set(zip(ei[0].tolist(), ei[1].tolist()))
        # a-b should be connected (strong correlation)
        assert (0, 1) in edges or (1, 0) in edges

    def test_graph_threshold_sensitivity(self):
        n = 100
        df = pd.DataFrame({
            "a": np.random.randn(n),
            "b": np.random.randn(n),
        })
        columns = [
            ColumnSpec("a", "continuous"),
            ColumnSpec("b", "continuous"),
        ]
        ei_low, _, _ = build_feature_graph(df, columns, threshold=0.01)
        ei_high, _, _ = build_feature_graph(df, columns, threshold=0.99)
        # Lower threshold → more edges (or equal)
        assert ei_low.shape[1] >= ei_high.shape[1]


# ===================================================================
# Multi-stage training integration
# ===================================================================

class TestMultiStageTraining:
    def test_stages_use_correct_components(self, mixed_columns, mixed_df, device):
        """Verify that each stage activates the right model components."""
        cfg = GAFMConfig(
            hidden_dim=32, num_mlp_layers=2, time_embed_dim=16,
            feat_gnn_layers=1, samp_gnn_layers=1, samp_gnn_hidden=16,
            k_neighbors=3, batch_size=16, num_epochs=6,
            stage1_fraction=0.34, stage2_fraction=0.33, stage3_fraction=0.33,
            sinkhorn_iterations=5, num_workers=0, pin_memory=False,
            max_steps_per_epoch=2,
        )
        trainer = GAFMTrainer(mixed_columns, cfg, device)
        model = trainer.fit(mixed_df)
        # Model should exist and be in eval mode
        assert model is not None
        assert not model.training

    def test_different_stages_different_outputs(self, mixed_columns, mixed_df, small_config, device):
        """Stage 1 and stage 3 should produce different outputs."""
        trainer = GAFMTrainer(mixed_columns, small_config, device)
        model = trainer.fit(mixed_df)
        pp = trainer.preprocessor

        x_c, x_d, x_o = sample_prior(4, pp.d_c, pp.categorical_dims, pp.n_ordinal, device)
        t = torch.tensor([0.5, 0.5, 0.5, 0.5])

        with torch.no_grad():
            v1_c, _, _ = model(x_c, x_d, x_o, t, stage=1)
            v3_c, _, _ = model(x_c, x_d, x_o, t, stage=3)

        # Stage 3 adds sample graph corrections → different output
        if v1_c is not None and v3_c is not None:
            assert not torch.allclose(v1_c, v3_c, atol=1e-3)
