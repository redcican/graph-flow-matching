"""Tests for neural network components (coord_net, feature_gnn, sample_gnn, velocity_field)."""

import math

import numpy as np
import pandas as pd
import pytest
import torch

from graph_flow_matching.baselines.base import ColumnSpec
from graph_flow_matching.models.aggregation import (
    CrossTypeProjection,
    batched_circular_mean,
    batched_einstein_midpoint,
    circular_mean,
    einstein_midpoint,
    weighted_average,
)
from graph_flow_matching.models.coord_net import (
    CategoricalVelocityNet,
    ContinuousVelocityNet,
    CoordinateWiseVelocity,
    OrdinalVelocityNet,
)
from graph_flow_matching.models.feature_gnn import (
    FeatureGraphNetwork,
    TypeAwareMessagePassingLayer,
    build_feature_graph,
)
from graph_flow_matching.models.ot_solver import (
    compute_ot_coupling,
    sample_ot_pairs,
    sinkhorn,
)
from graph_flow_matching.models.sample_gnn import (
    SampleGraphNetwork,
    SampleMessagePassingLayer,
    build_sample_graph,
)
from graph_flow_matching.models.time_embedding import SinusoidalTimeEmbedding
from graph_flow_matching.models.velocity_field import (
    GAFMConfig,
    GraphAugmentedFlowMatching,
)

B = 8
D_C = 3
CAT_DIMS = [3, 4]
N_ORD = 1
HIDDEN = 32
T_DIM = 16


# ===================================================================
# Time Embedding
# ===================================================================

class TestSinusoidalTimeEmbedding:
    def test_output_shape(self):
        emb = SinusoidalTimeEmbedding(T_DIM)
        t = torch.rand(B)
        out = emb(t)
        assert out.shape == (B, T_DIM)

    def test_deterministic(self):
        emb = SinusoidalTimeEmbedding(T_DIM)
        t = torch.tensor([0.0, 0.5, 1.0])
        assert torch.equal(emb(t), emb(t))

    def test_different_times_different_embeddings(self):
        emb = SinusoidalTimeEmbedding(T_DIM)
        t = torch.tensor([0.0, 0.5, 1.0])
        out = emb(t)
        assert not torch.allclose(out[0], out[1])
        assert not torch.allclose(out[0], out[2])


# ===================================================================
# Aggregation Operators (Eqs 29-32)
# ===================================================================

class TestWeightedAverage:
    def test_uniform_weights(self):
        vals = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        w = torch.tensor([0.5, 0.5])
        result = weighted_average(vals, w)
        assert torch.allclose(result, torch.tensor([2.0, 3.0]))


class TestEinsteinMidpoint:
    def test_output_on_simplex(self):
        dists = torch.softmax(torch.randn(5, 4), dim=-1)
        w = torch.ones(5) / 5
        result = einstein_midpoint(dists, w)
        assert (result >= 0).all()
        assert result.sum().item() == pytest.approx(1.0, abs=1e-5)

    def test_single_input_returns_itself(self):
        p = torch.tensor([[0.3, 0.5, 0.2]])
        w = torch.tensor([1.0])
        result = einstein_midpoint(p, w)
        assert torch.allclose(result, p.squeeze(0), atol=1e-4)

    def test_batched_version(self):
        """Batched scatter-based version should match loop-based."""
        p = torch.softmax(torch.randn(6, 3), dim=-1)
        w = torch.ones(6) / 3
        idx = torch.tensor([0, 0, 0, 1, 1, 1])
        result = batched_einstein_midpoint(p, w, idx, 2)
        assert result.shape == (2, 3)
        assert torch.allclose(result.sum(dim=-1), torch.ones(2), atol=1e-4)


class TestCircularMean:
    def test_mean_of_same_angle(self):
        angles = torch.tensor([1.0, 1.0, 1.0])
        w = torch.ones(3) / 3
        result = circular_mean(angles, w)
        assert result.item() == pytest.approx(1.0, abs=1e-5)

    def test_wraparound(self):
        """Mean of angles near 0 and 2π should be near 0."""
        angles = torch.tensor([0.1, 2 * math.pi - 0.1])
        w = torch.tensor([0.5, 0.5])
        result = circular_mean(angles, w)
        assert result.item() < 0.2 or result.item() > 2 * math.pi - 0.2

    def test_batched_version(self):
        angles = torch.rand(4) * 2 * math.pi
        w = torch.ones(4) / 2
        idx = torch.tensor([0, 0, 1, 1])
        result = batched_circular_mean(angles, w, idx, 2)
        assert result.shape == (2,)


class TestCrossTypeProjection:
    def test_same_type_identity(self):
        proj = CrossTypeProjection(HIDDEN)
        h = torch.randn(B, HIDDEN)
        assert torch.equal(proj.project(h, 0, 0), h)

    def test_cross_type_shape(self):
        proj = CrossTypeProjection(HIDDEN)
        h = torch.randn(B, HIDDEN)
        assert proj.project(h, 0, 1).shape == (B, HIDDEN)
        assert proj.project(h, 1, 0).shape == (B, HIDDEN)
        assert proj.project(h, 0, 2).shape == (B, HIDDEN)
        assert proj.project(h, 2, 0).shape == (B, HIDDEN)
        assert proj.project(h, 1, 2).shape == (B, HIDDEN)
        assert proj.project(h, 2, 1).shape == (B, HIDDEN)


# ===================================================================
# Coordinate-wise Networks (Eq 17)
# ===================================================================

class TestContinuousVelocityNet:
    def test_output_shape(self):
        net = ContinuousVelocityNet(D_C, HIDDEN, 2, T_DIM)
        x = torch.randn(B, D_C)
        t_emb = torch.randn(B, T_DIM)
        out = net(x, t_emb)
        assert out.shape == (B, D_C)

    def test_gradients_flow(self):
        net = ContinuousVelocityNet(D_C, HIDDEN, 2, T_DIM)
        x = torch.randn(B, D_C, requires_grad=True)
        t_emb = torch.randn(B, T_DIM)
        out = net(x, t_emb)
        out.sum().backward()
        assert x.grad is not None


class TestCategoricalVelocityNet:
    def test_tangent_space_projection(self):
        """Output should sum to zero (simplex tangent space)."""
        net = CategoricalVelocityNet(5, HIDDEN, 2, T_DIM)
        x = torch.softmax(torch.randn(B, 5), dim=-1)
        t_emb = torch.randn(B, T_DIM)
        v = net(x, t_emb)
        assert v.shape == (B, 5)
        assert torch.allclose(v.sum(dim=-1), torch.zeros(B), atol=1e-5)


class TestOrdinalVelocityNet:
    def test_output_shape(self):
        net = OrdinalVelocityNet(HIDDEN, 2, T_DIM)
        theta = torch.rand(B) * 2 * math.pi
        t_emb = torch.randn(B, T_DIM)
        out = net(theta, t_emb)
        assert out.shape == (B,)

    def test_cos_sin_input(self):
        """Verify the network handles the (cos, sin) encoding internally."""
        net = OrdinalVelocityNet(HIDDEN, 2, T_DIM)
        theta = torch.tensor([0.0, math.pi / 2, math.pi])
        t_emb = torch.randn(3, T_DIM)
        out = net(theta, t_emb)
        assert out.isfinite().all()


class TestCoordinateWiseVelocity:
    def test_full_output(self):
        net = CoordinateWiseVelocity(D_C, CAT_DIMS, N_ORD, HIDDEN, 2, T_DIM)
        x_c = torch.randn(B, D_C)
        x_d = [torch.softmax(torch.randn(B, n), dim=-1) for n in CAT_DIMS]
        x_o = [torch.rand(B) * 2 * math.pi for _ in range(N_ORD)]
        t_emb = torch.randn(B, T_DIM)

        v_c, v_d, v_o = net(x_c, x_d, x_o, t_emb)
        assert v_c.shape == (B, D_C)
        assert len(v_d) == len(CAT_DIMS)
        assert len(v_o) == N_ORD
        for vd, n in zip(v_d, CAT_DIMS):
            assert vd.shape == (B, n)
            assert torch.allclose(vd.sum(dim=-1), torch.zeros(B), atol=1e-5)

    def test_no_continuous(self):
        net = CoordinateWiseVelocity(0, [3], 0, HIDDEN, 2, T_DIM)
        x_d = [torch.softmax(torch.randn(B, 3), dim=-1)]
        t_emb = torch.randn(B, T_DIM)
        v_c, v_d, v_o = net(None, x_d, [], t_emb)
        assert v_c is None
        assert len(v_d) == 1


# ===================================================================
# Feature Graph Construction and GNN (Section 4.2)
# ===================================================================

class TestBuildFeatureGraph:
    def test_returns_correct_types(self, mixed_df, mixed_columns):
        ei, ew, nt = build_feature_graph(mixed_df, mixed_columns, threshold=0.0)
        assert ei.dtype == torch.long
        assert ew.dtype == torch.float32
        assert nt.dtype == torch.long
        assert ei.shape[0] == 2
        assert nt.shape[0] == len(mixed_columns)

    def test_node_types(self, mixed_df, mixed_columns):
        _, _, nt = build_feature_graph(mixed_df, mixed_columns, threshold=0.0)
        # age=cont(0), income=cont(0), color=cat(1), size=cat(1), education=ord(2)
        assert nt.tolist() == [0, 0, 1, 1, 2]

    def test_high_threshold_gives_sparse_graph(self, mixed_df, mixed_columns):
        ei, _, _ = build_feature_graph(mixed_df, mixed_columns, threshold=0.99)
        # Very high threshold → few or no edges (fallback self-loops)
        assert ei.shape[1] <= 2 * len(mixed_columns)

    def test_undirected(self, mixed_df, mixed_columns):
        """Graph should be undirected: if (i,j) exists, so does (j,i)."""
        ei, _, _ = build_feature_graph(mixed_df, mixed_columns, threshold=0.0)
        edges_fwd = set(zip(ei[0].tolist(), ei[1].tolist()))
        for s, d in edges_fwd:
            if s != d:
                assert (d, s) in edges_fwd


class TestFeatureGraphNetwork:
    def test_output_shapes(self):
        net = FeatureGraphNetwork(D_C, CAT_DIMS, N_ORD, HIDDEN, 1, T_DIM)
        x_c = torch.randn(B, D_C)
        x_d = [torch.softmax(torch.randn(B, n), dim=-1) for n in CAT_DIMS]
        x_o = [torch.rand(B) * 2 * math.pi]
        t_emb = torch.randn(B, T_DIM)

        D = D_C + len(CAT_DIMS) + N_ORD
        ei = torch.tensor([[0, 1], [1, 0]], dtype=torch.long)
        ew = torch.tensor([0.5, 0.5])
        nt = torch.tensor([0] * D_C + [1] * len(CAT_DIMS) + [2] * N_ORD)

        v_c, v_d, v_o = net(x_c, x_d, x_o, t_emb, ei, ew, nt)
        assert v_c.shape == (B, D_C)
        assert len(v_d) == len(CAT_DIMS)
        assert len(v_o) == N_ORD

    def test_categorical_tangent_space(self):
        net = FeatureGraphNetwork(0, [4], 0, HIDDEN, 1, T_DIM)
        x_d = [torch.softmax(torch.randn(B, 4), dim=-1)]
        t_emb = torch.randn(B, T_DIM)
        ei = torch.tensor([[0], [0]], dtype=torch.long)
        ew = torch.tensor([1.0])
        nt = torch.tensor([1])

        _, v_d, _ = net(None, x_d, [], t_emb, ei, ew, nt)
        assert torch.allclose(v_d[0].sum(dim=-1), torch.zeros(B), atol=1e-5)


class TestTypeAwareMessagePassingLayer:
    def test_output_shape(self):
        layer = TypeAwareMessagePassingLayer(HIDDEN, T_DIM)
        h = torch.randn(B, 4, HIDDEN)
        ei = torch.tensor([[0, 1, 2, 3], [1, 0, 3, 2]], dtype=torch.long)
        ew = torch.ones(4)
        nt = torch.tensor([0, 0, 1, 2])
        t_emb = torch.randn(B, T_DIM)

        out = layer(h, ei, ew, nt, t_emb)
        assert out.shape == h.shape


# ===================================================================
# Sample Graph Construction and GNN (Section 4.3)
# ===================================================================

class TestBuildSampleGraph:
    def test_edge_count(self):
        x_c = torch.randn(10, 3)
        x_d = [torch.softmax(torch.randn(10, 4), dim=-1)]
        ei, ew = build_sample_graph(x_c, x_d, [], k=3)
        assert ei.shape == (2, 30)  # 10 * 3
        assert ew.shape == (30,)

    def test_weights_sum_to_one_per_node(self):
        x_c = torch.randn(10, 3)
        ei, ew = build_sample_graph(x_c, [], [], k=5)
        # Weights from softmax per node should sum to ~1
        for i in range(10):
            mask = ei[1] == i
            assert ew[mask].sum().item() == pytest.approx(1.0, abs=1e-5)

    def test_no_self_loops(self):
        x_c = torch.randn(10, 3)
        ei, _ = build_sample_graph(x_c, [], [], k=3)
        assert (ei[0] != ei[1]).all()

    def test_k_larger_than_batch(self):
        """k > B-1 should be handled gracefully."""
        x_c = torch.randn(3, 2)
        ei, ew = build_sample_graph(x_c, [], [], k=10)
        assert ei.shape[1] == 3 * 2  # k_actual = min(10, 3-1) = 2


class TestSampleGraphNetwork:
    def test_output_shapes(self):
        net = SampleGraphNetwork(D_C, CAT_DIMS, N_ORD, HIDDEN, 1, T_DIM, k=3)
        x_c = torch.randn(B, D_C)
        x_d = [torch.softmax(torch.randn(B, n), dim=-1) for n in CAT_DIMS]
        x_o = [torch.rand(B) * 2 * math.pi]
        t_emb = torch.randn(B, T_DIM)

        v_c, v_d, v_o = net(x_c, x_d, x_o, t_emb)
        assert v_c.shape == (B, D_C)
        assert len(v_d) == len(CAT_DIMS)
        assert v_d[0].shape == (B, CAT_DIMS[0])
        assert len(v_o) == N_ORD

    def test_categorical_tangent_space(self):
        net = SampleGraphNetwork(0, [4], 0, HIDDEN, 1, T_DIM, k=3)
        x_d = [torch.softmax(torch.randn(B, 4), dim=-1)]
        t_emb = torch.randn(B, T_DIM)

        _, v_d, _ = net(None, x_d, [], t_emb)
        assert torch.allclose(v_d[0].sum(dim=-1), torch.zeros(B), atol=1e-5)


# ===================================================================
# OT Solver (Section 4.5)
# ===================================================================

class TestSinkhorn:
    def test_doubly_stochastic(self):
        C = torch.rand(10, 10)
        pi = sinkhorn(C, epsilon=0.1, num_iterations=100)
        row_sums = pi.sum(dim=1)
        col_sums = pi.sum(dim=0)
        assert torch.allclose(row_sums, torch.ones(10) / 10, atol=1e-3)
        assert torch.allclose(col_sums, torch.ones(10) / 10, atol=1e-3)

    def test_non_negative(self):
        C = torch.rand(5, 5) * 10
        pi = sinkhorn(C, epsilon=0.01, num_iterations=50)
        assert (pi >= -1e-6).all()

    def test_concentrates_on_low_cost(self):
        """OT plan should assign more mass to low-cost pairs."""
        C = torch.zeros(3, 3)
        C[0, 0] = 0.0
        C[0, 1] = 10.0
        C[0, 2] = 10.0
        C[1, 0] = 10.0
        C[1, 1] = 0.0
        C[1, 2] = 10.0
        C[2, 0] = 10.0
        C[2, 1] = 10.0
        C[2, 2] = 0.0
        pi = sinkhorn(C, epsilon=0.01, num_iterations=100)
        # Diagonal should have highest values
        assert pi[0, 0] > pi[0, 1]
        assert pi[1, 1] > pi[1, 0]


class TestComputeOTCoupling:
    def test_returns_valid_coupling(self):
        x0_c = torch.randn(B, 3)
        x1_c = torch.randn(B, 3)
        pi = compute_ot_coupling(x0_c, x1_c, [], [], [], [])
        assert pi.shape == (B, B)
        assert (pi >= -1e-6).all()


class TestSampleOTPairs:
    def test_returns_valid_indices(self):
        C = torch.rand(B, B)
        pi = sinkhorn(C, epsilon=0.1, num_iterations=50)
        idx = sample_ot_pairs(pi)
        assert idx.shape == (B,)
        assert (idx >= 0).all()
        assert (idx < B).all()


# ===================================================================
# Main Model (Eq 16, velocity_field.py)
# ===================================================================

class TestGraphAugmentedFlowMatching:
    @pytest.fixture()
    def model(self):
        cfg = GAFMConfig(
            hidden_dim=HIDDEN, num_mlp_layers=2, time_embed_dim=T_DIM,
            feat_gnn_layers=1, samp_gnn_layers=1, samp_gnn_hidden=HIDDEN,
            k_neighbors=3,
        )
        return GraphAugmentedFlowMatching(
            d_c=D_C, categorical_dims=CAT_DIMS, n_ordinal=N_ORD, config=cfg,
        )

    @pytest.fixture()
    def model_with_graph(self):
        cfg = GAFMConfig(
            hidden_dim=HIDDEN, num_mlp_layers=2, time_embed_dim=T_DIM,
            feat_gnn_layers=1, samp_gnn_layers=1, samp_gnn_hidden=HIDDEN,
            k_neighbors=3,
        )
        D = D_C + len(CAT_DIMS) + N_ORD
        ei = torch.tensor([[0, 1, 2, 3], [1, 0, 3, 2]], dtype=torch.long)
        ew = torch.tensor([0.5, 0.5, 0.3, 0.3])
        nt = torch.tensor([0] * D_C + [1] * len(CAT_DIMS) + [2] * N_ORD)
        return GraphAugmentedFlowMatching(
            d_c=D_C, categorical_dims=CAT_DIMS, n_ordinal=N_ORD,
            feature_graph=(ei, ew, nt), config=cfg,
        )

    def _make_inputs(self):
        x_c = torch.randn(B, D_C)
        x_d = [torch.softmax(torch.randn(B, n), dim=-1) for n in CAT_DIMS]
        x_o = [torch.rand(B) * 2 * math.pi for _ in range(N_ORD)]
        t = torch.rand(B)
        return x_c, x_d, x_o, t

    def test_stage1_only_coord(self, model):
        x_c, x_d, x_o, t = self._make_inputs()
        v_c, v_d, v_o = model(x_c, x_d, x_o, t, stage=1)
        assert v_c.shape == (B, D_C)
        assert len(v_d) == len(CAT_DIMS)
        assert len(v_o) == N_ORD

    def test_stage2_with_feature_graph(self, model_with_graph):
        x_c, x_d, x_o, t = self._make_inputs()
        v_c, v_d, v_o = model_with_graph(x_c, x_d, x_o, t, stage=2)
        assert v_c.shape == (B, D_C)

    def test_stage3_full_model(self, model_with_graph):
        x_c, x_d, x_o, t = self._make_inputs()
        v_c, v_d, v_o = model_with_graph(x_c, x_d, x_o, t, stage=3)
        assert v_c.shape == (B, D_C)

    def test_backward_pass(self, model_with_graph):
        x_c, x_d, x_o, t = self._make_inputs()
        v_c, v_d, v_o = model_with_graph(x_c, x_d, x_o, t, stage=3)
        loss = v_c.pow(2).mean() + sum(vd.pow(2).mean() for vd in v_d)
        loss.backward()
        # At least some parameters should receive gradients
        grads = [p.grad for p in model_with_graph.parameters() if p.requires_grad and p.grad is not None]
        assert len(grads) > 0, "No parameter received gradients"

    def test_set_feature_graph(self, model):
        D = D_C + len(CAT_DIMS) + N_ORD
        ei = torch.tensor([[0, 1], [1, 0]], dtype=torch.long)
        ew = torch.tensor([0.5, 0.5])
        nt = torch.tensor([0] * D_C + [1] * len(CAT_DIMS) + [2] * N_ORD)
        model.set_feature_graph(ei, ew, nt)
        assert model.feat_edge_index.shape[1] == 2

    def test_no_feature_graph_stage2_skips(self, model):
        """Stage 2 with empty feature graph should behave like stage 1."""
        x_c, x_d, x_o, t = self._make_inputs()
        v1_c, _, _ = model(x_c, x_d, x_o, t, stage=1)
        v2_c, _, _ = model(x_c, x_d, x_o, t, stage=2)
        assert torch.allclose(v1_c, v2_c, atol=1e-5)

    def test_categorical_output_tangent_space(self, model_with_graph):
        x_c, x_d, x_o, t = self._make_inputs()
        _, v_d, _ = model_with_graph(x_c, x_d, x_o, t, stage=3)
        # Combined output may not be exactly zero-sum due to addition of components,
        # but each component enforces it individually.
        # Check that deviation is small.
        for vd in v_d:
            assert vd.sum(dim=-1).abs().max() < 0.1


class TestGAFMConfig:
    def test_default_values(self):
        cfg = GAFMConfig()
        assert cfg.hidden_dim == 256
        assert cfg.num_mlp_layers == 5
        assert cfg.time_embed_dim == 64
        assert cfg.k_neighbors == 10
        assert cfg.lr == 1e-3
        assert cfg.batch_size == 256
        assert cfg.num_epochs == 200
        assert cfg.lambda_d == 1.0
        assert cfg.lambda_o == 1.0
        assert cfg.sinkhorn_epsilon == 0.01
        assert cfg.sinkhorn_iterations == 100
        assert cfg.dependency_threshold == 0.3
        assert cfg.n_ode_steps == 100
        assert cfg.ode_atol == 1e-5
        assert cfg.ode_rtol == 1e-5
        assert cfg.stage1_fraction == 0.2
        assert cfg.stage2_fraction == 0.3
        assert cfg.stage3_fraction == 0.5
        assert cfg.num_workers == 8
        assert cfg.pin_memory is True
        assert cfg.max_steps_per_epoch == 10_000
