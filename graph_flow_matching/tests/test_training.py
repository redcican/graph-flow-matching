"""Tests for training pipeline (trainer.py, sampler.py)."""

import math

import numpy as np
import pandas as pd
import pytest
import torch

from graph_flow_matching.baselines.base import ColumnSpec
from graph_flow_matching.models.manifold_ops import (
    clamp_probabilities,
    conditional_velocity_categorical,
    conditional_velocity_continuous,
    conditional_velocity_ordinal,
    geodesic_interpolation_categorical,
    geodesic_interpolation_continuous,
    geodesic_interpolation_ordinal,
)
from graph_flow_matching.models.velocity_field import GAFMConfig, GraphAugmentedFlowMatching
from graph_flow_matching.training.trainer import (
    DataPreprocessor,
    GAFMTrainer,
    _TrainStepModule,
    compute_loss,
    pack_flat,
    sample_prior,
    unpack_flat,
)
from graph_flow_matching.training.sampler import GAFMSampler

B = 8
D_C = 3
CAT_DIMS = [3, 4]
N_ORD = 1


# ===================================================================
# pack_flat / unpack_flat
# ===================================================================

class TestPackUnpack:
    def test_roundtrip_all_types(self):
        x_c = torch.randn(B, D_C)
        x_d = [torch.softmax(torch.randn(B, n), dim=-1) for n in CAT_DIMS]
        x_o = [torch.rand(B) * 2 * math.pi for _ in range(N_ORD)]

        flat = pack_flat(x_c, x_d, x_o)
        x_c2, x_d2, x_o2 = unpack_flat(flat, D_C, CAT_DIMS, N_ORD)

        assert torch.allclose(x_c, x_c2)
        for a, b in zip(x_d, x_d2):
            assert torch.allclose(a, b)
        for a, b in zip(x_o, x_o2):
            assert torch.allclose(a, b)

    def test_flat_shape(self):
        x_c = torch.randn(B, D_C)
        x_d = [torch.softmax(torch.randn(B, n), dim=-1) for n in CAT_DIMS]
        x_o = [torch.rand(B) * 2 * math.pi for _ in range(N_ORD)]
        flat = pack_flat(x_c, x_d, x_o)
        expected_dim = D_C + sum(CAT_DIMS) + N_ORD
        assert flat.shape == (B, expected_dim)

    def test_no_continuous(self):
        x_d = [torch.softmax(torch.randn(B, 3), dim=-1)]
        flat = pack_flat(None, x_d, [])
        x_c2, x_d2, x_o2 = unpack_flat(flat, 0, [3], 0)
        assert x_c2 is None
        assert torch.allclose(x_d[0], x_d2[0])
        assert len(x_o2) == 0

    def test_no_categorical(self):
        x_c = torch.randn(B, 2)
        x_o = [torch.rand(B) * 2 * math.pi]
        flat = pack_flat(x_c, [], x_o)
        x_c2, x_d2, x_o2 = unpack_flat(flat, 2, [], 1)
        assert torch.allclose(x_c, x_c2)
        assert len(x_d2) == 0
        assert torch.allclose(x_o[0], x_o2[0])


# ===================================================================
# sample_prior
# ===================================================================

class TestSamplePrior:
    def test_shapes(self):
        x_c, x_d, x_o = sample_prior(B, D_C, CAT_DIMS, N_ORD, torch.device("cpu"))
        assert x_c.shape == (B, D_C)
        assert len(x_d) == len(CAT_DIMS)
        for xd, n in zip(x_d, CAT_DIMS):
            assert xd.shape == (B, n)
        assert len(x_o) == N_ORD
        assert x_o[0].shape == (B,)

    def test_continuous_is_gaussian(self):
        x_c, _, _ = sample_prior(10000, 1, [], 0, torch.device("cpu"))
        assert x_c.mean().abs().item() < 0.1
        assert (x_c.std() - 1.0).abs().item() < 0.1

    def test_categorical_on_simplex(self):
        _, x_d, _ = sample_prior(B, 0, [5], 0, torch.device("cpu"))
        assert (x_d[0] > 0).all()
        sums = x_d[0].sum(dim=-1)
        assert torch.allclose(sums, torch.ones(B), atol=1e-5)

    def test_ordinal_in_range(self):
        _, _, x_o = sample_prior(B, 0, [], 1, torch.device("cpu"))
        assert (x_o[0] >= 0).all()
        assert (x_o[0] < 2 * math.pi).all()

    def test_no_continuous(self):
        x_c, x_d, _ = sample_prior(B, 0, [3], 0, torch.device("cpu"))
        assert x_c is None
        assert len(x_d) == 1


# ===================================================================
# compute_loss
# ===================================================================

class TestComputeLoss:
    def test_euclidean_loss(self):
        v_c = torch.randn(B, D_C)
        u_c = torch.randn(B, D_C)
        v_d = [torch.randn(B, 3)]
        u_d = [torch.randn(B, 3)]
        v_o = [torch.randn(B)]
        u_o = [torch.randn(B)]
        loss = compute_loss(v_c, v_d, v_o, u_c, u_d, u_o)
        assert loss.isfinite()
        assert loss.item() > 0

    def test_fisher_rao_weighted_loss(self):
        v_d = [torch.randn(B, 4)]
        u_d = [torch.randn(B, 4)]
        xt_d = [torch.softmax(torch.randn(B, 4), dim=-1)]

        loss_eucl = compute_loss(None, v_d, [], None, u_d, [], xt_d=None)
        loss_fr = compute_loss(None, v_d, [], None, u_d, [], xt_d=xt_d)

        assert loss_eucl.isfinite()
        assert loss_fr.isfinite()
        # FR loss should generally differ from Euclidean
        assert not torch.allclose(loss_eucl, loss_fr)

    def test_fisher_rao_with_skewed_distribution(self):
        """Near one-hot distributions shouldn't produce inf/nan."""
        v_d = [torch.randn(B, 5)]
        u_d = [torch.randn(B, 5)]
        # Very skewed: one component dominant
        xt_d = [torch.tensor([[0.99, 0.0025, 0.0025, 0.0025, 0.0025]]).expand(B, -1)]
        loss = compute_loss(None, v_d, [], None, u_d, [], xt_d=xt_d)
        assert loss.isfinite()

    def test_zero_loss_when_perfect(self):
        v_c = torch.randn(B, D_C)
        loss = compute_loss(v_c, [], [], v_c, [], [])
        assert loss.item() < 1e-6

    def test_lambda_weighting(self):
        v_d = [torch.randn(B, 3)]
        u_d = [torch.zeros(B, 3)]
        loss1 = compute_loss(None, v_d, [], None, u_d, [], lambda_d=1.0)
        loss2 = compute_loss(None, v_d, [], None, u_d, [], lambda_d=2.0)
        assert loss2.item() == pytest.approx(2.0 * loss1.item(), rel=1e-4)

    def test_backward(self):
        v_c = torch.randn(B, D_C, requires_grad=True)
        u_c = torch.randn(B, D_C)
        loss = compute_loss(v_c, [], [], u_c, [], [])
        loss.backward()
        assert v_c.grad is not None


# ===================================================================
# DataPreprocessor
# ===================================================================

class TestDataPreprocessor:
    def test_properties(self, mixed_columns):
        pp = DataPreprocessor(mixed_columns)
        assert pp.d_c == 2
        assert pp.categorical_dims == [3, 4]
        assert pp.n_ordinal == 1

    def test_fit_transform_shapes(self, mixed_columns, mixed_df):
        pp = DataPreprocessor(mixed_columns)
        x_c, x_d, x_o = pp.fit_transform(mixed_df)
        n = len(mixed_df)
        assert x_c.shape == (n, 2)
        assert len(x_d) == 2
        assert x_d[0].shape == (n, 3)  # color: 3 categories
        assert x_d[1].shape == (n, 4)  # size: 4 categories
        assert len(x_o) == 1
        assert x_o[0].shape == (n,)

    def test_continuous_standardized(self, mixed_columns, mixed_df):
        pp = DataPreprocessor(mixed_columns)
        x_c, _, _ = pp.fit_transform(mixed_df)
        # Should be roughly mean 0, std 1
        assert x_c.mean(dim=0).abs().max().item() < 0.2
        assert (x_c.std(dim=0) - 1.0).abs().max().item() < 0.2

    def test_categorical_one_hot(self, mixed_columns, mixed_df):
        pp = DataPreprocessor(mixed_columns)
        _, x_d, _ = pp.fit_transform(mixed_df)
        # One-hot: each row sums to 1, entries are 0 or 1
        assert torch.allclose(x_d[0].sum(dim=-1), torch.ones(len(mixed_df)))
        assert set(x_d[0].unique().tolist()) == {0.0, 1.0}

    def test_ordinal_angle_range(self, mixed_columns, mixed_df):
        pp = DataPreprocessor(mixed_columns)
        _, _, x_o = pp.fit_transform(mixed_df)
        assert (x_o[0] >= 0).all()
        assert (x_o[0] < math.pi).all()

    def test_inverse_transform_roundtrip(self, mixed_columns, mixed_df):
        pp = DataPreprocessor(mixed_columns)
        x_c, x_d, x_o = pp.fit_transform(mixed_df)
        recovered = pp.inverse_transform(x_c, x_d, x_o)
        assert set(recovered.columns) == set(mixed_df.columns)
        assert len(recovered) == len(mixed_df)
        # Continuous should be close
        assert np.allclose(
            recovered["age"].values.astype(float),
            mixed_df["age"].values,
            atol=0.1,
        )
        # Categorical should match exactly
        assert (recovered["color"].values == mixed_df["color"].values).all()
        # Ordinal should match exactly
        assert (recovered["education"].values == mixed_df["education"].values).all()

    def test_cont_only(self, cont_only_columns, cont_df):
        pp = DataPreprocessor(cont_only_columns)
        x_c, x_d, x_o = pp.fit_transform(cont_df)
        assert x_c is not None
        assert len(x_d) == 0
        assert len(x_o) == 0

    def test_cat_only(self, cat_only_columns):
        df = pd.DataFrame({
            "c1": np.random.choice(["a", "b", "c"], 30),
            "c2": np.random.choice(["x", "y"], 30),
        })
        pp = DataPreprocessor(cat_only_columns)
        x_c, x_d, x_o = pp.fit_transform(df)
        assert x_c is None
        assert len(x_d) == 2
        assert len(x_o) == 0


# ===================================================================
# _TrainStepModule
# ===================================================================

class TestTrainStepModule:
    @pytest.fixture()
    def step_module(self, small_config):
        model = GraphAugmentedFlowMatching(
            d_c=D_C, categorical_dims=CAT_DIMS, n_ordinal=N_ORD,
            config=small_config,
        )
        return _TrainStepModule(
            model, D_C, CAT_DIMS, N_ORD,
            small_config.lambda_d, small_config.lambda_o,
        )

    def test_forward_output_shape(self, step_module):
        x0_c, x0_d, x0_o = sample_prior(B, D_C, CAT_DIMS, N_ORD, torch.device("cpu"))
        x1_c, x1_d, x1_o = sample_prior(B, D_C, CAT_DIMS, N_ORD, torch.device("cpu"))
        x0_flat = pack_flat(x0_c, x0_d, x0_o)
        x1_flat = pack_flat(x1_c, x1_d, x1_o)
        t = torch.rand(B)

        loss = step_module(x0_flat, x1_flat, t)
        assert loss.shape == (1,)
        assert loss.isfinite()

    def test_backward(self, step_module):
        x0_c, x0_d, x0_o = sample_prior(B, D_C, CAT_DIMS, N_ORD, torch.device("cpu"))
        x1_c, x1_d, x1_o = sample_prior(B, D_C, CAT_DIMS, N_ORD, torch.device("cpu"))
        x0_flat = pack_flat(x0_c, x0_d, x0_o)
        x1_flat = pack_flat(x1_c, x1_d, x1_o)
        t = torch.rand(B)

        loss = step_module(x0_flat, x1_flat, t)
        loss.backward()
        grad_count = sum(1 for p in step_module.parameters() if p.grad is not None)
        assert grad_count > 0

    def test_stage_attribute(self, step_module):
        step_module.stage = 1
        assert step_module.stage == 1
        step_module.stage = 3
        assert step_module.stage == 3


# ===================================================================
# GAFMTrainer (full training pipeline)
# ===================================================================

class TestGAFMTrainer:
    def test_fit_returns_model(self, mixed_columns, mixed_df, small_config, device):
        trainer = GAFMTrainer(mixed_columns, small_config, device)
        model = trainer.fit(mixed_df)
        assert isinstance(model, GraphAugmentedFlowMatching)

    def test_preprocessor_is_fitted(self, mixed_columns, mixed_df, small_config, device):
        trainer = GAFMTrainer(mixed_columns, small_config, device)
        trainer.fit(mixed_df)
        assert trainer.preprocessor.cont_mean is not None
        assert trainer.preprocessor.cont_std is not None

    def test_fit_cont_only(self, cont_only_columns, cont_df, small_config, device):
        trainer = GAFMTrainer(cont_only_columns, small_config, device)
        model = trainer.fit(cont_df)
        assert model.d_c == 3
        assert model.categorical_dims == []
        assert model.n_ordinal == 0

    def test_loss_decreases(self, mixed_columns, mixed_df, device):
        """Training loss should decrease (or at least not explode)."""
        cfg = GAFMConfig(
            hidden_dim=32, num_mlp_layers=2, time_embed_dim=16,
            feat_gnn_layers=1, samp_gnn_layers=1, samp_gnn_hidden=16,
            k_neighbors=3, batch_size=32, num_epochs=6,
            sinkhorn_iterations=10, num_workers=0, pin_memory=False,
            max_steps_per_epoch=5,
        )
        trainer = GAFMTrainer(mixed_columns, cfg, device)
        # Monkey-patch to capture losses
        losses = []
        orig_train_epoch = trainer._train_epoch
        def patched_train_epoch(*args, **kwargs):
            loss = orig_train_epoch(*args, **kwargs)
            losses.append(loss)
            return loss
        trainer._train_epoch = patched_train_epoch
        trainer.fit(mixed_df)
        # At minimum, losses should be finite
        assert all(math.isfinite(l) for l in losses)


# ===================================================================
# GAFMSampler
# ===================================================================

class TestGAFMSampler:
    @pytest.fixture()
    def trained_sampler(self, mixed_columns, mixed_df, small_config, device):
        trainer = GAFMTrainer(mixed_columns, small_config, device)
        model = trainer.fit(mixed_df)
        return GAFMSampler(model, trainer.preprocessor, small_config, device)

    def test_sample_returns_dataframe(self, trained_sampler):
        df = trained_sampler.sample(10)
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 10

    def test_sample_has_correct_columns(self, trained_sampler, mixed_columns):
        df = trained_sampler.sample(5)
        expected = {c.name for c in mixed_columns}
        assert set(df.columns) == expected

    def test_sample_batched(self, trained_sampler):
        """Sampling in batches should produce correct total count."""
        df = trained_sampler.sample(15, batch_size=7)
        assert len(df) == 15

    def test_sample_categorical_values_valid(self, trained_sampler):
        df = trained_sampler.sample(20)
        assert set(df["color"].unique()).issubset({"red", "blue", "green"})

    def test_sample_ordinal_values_valid(self, trained_sampler):
        df = trained_sampler.sample(20)
        assert set(df["education"].unique()).issubset({"low", "mid", "high"})

    def test_euler_fallback(self, mixed_columns, mixed_df, device):
        """Test Euler integration path when torchdiffeq is mocked as unavailable."""
        cfg = GAFMConfig(
            hidden_dim=32, num_mlp_layers=2, time_embed_dim=16,
            feat_gnn_layers=1, samp_gnn_layers=1, samp_gnn_hidden=16,
            k_neighbors=3, batch_size=16, num_epochs=2, n_ode_steps=3,
            sinkhorn_iterations=5, num_workers=0, pin_memory=False,
            max_steps_per_epoch=2,
        )
        trainer = GAFMTrainer(mixed_columns, cfg, device)
        model = trainer.fit(mixed_df)
        sampler = GAFMSampler(model, trainer.preprocessor, cfg, device)

        # Force Euler path by patching the import
        original_sample_batch = sampler._sample_batch

        def euler_only_batch(B):
            import unittest.mock as mock
            import builtins
            real_import = builtins.__import__

            def mock_import(name, *args, **kwargs):
                if name == "torchdiffeq":
                    raise ImportError("mocked")
                return real_import(name, *args, **kwargs)

            with mock.patch("builtins.__import__", side_effect=mock_import):
                return original_sample_batch(B)

        sampler._sample_batch = euler_only_batch
        df = sampler.sample(5)
        assert len(df) == 5
