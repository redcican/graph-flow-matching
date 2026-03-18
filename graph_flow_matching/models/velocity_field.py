"""Combined velocity field: v = v_coord + v_feat + v_samp. Eq 16."""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
import torch.nn as nn
from torch import Tensor

from graph_flow_matching.models.coord_net import CoordinateWiseVelocity
from graph_flow_matching.models.feature_gnn import FeatureGraphNetwork
from graph_flow_matching.models.sample_gnn import SampleGraphNetwork
from graph_flow_matching.models.time_embedding import SinusoidalTimeEmbedding


@dataclass
class GAFMConfig:
    """Configuration for Graph-Augmented Flow Matching (Section 5.1)."""

    # Architecture
    hidden_dim: int = 256
    num_mlp_layers: int = 5
    time_embed_dim: int = 64
    feat_gnn_layers: int = 3
    samp_gnn_layers: int = 2
    samp_gnn_hidden: int = 128
    k_neighbors: int = 10

    # Training
    lr: float = 1e-3
    batch_size: int = 256
    num_epochs: int = 200
    lambda_d: float = 1.0
    lambda_o: float = 1.0
    sinkhorn_epsilon: float = 0.01
    sinkhorn_iterations: int = 100
    val_fraction: float = 0.1

    # DataLoader / DataParallel
    num_workers: int = 8
    pin_memory: bool = True

    # Multi-stage fractions
    stage1_fraction: float = 0.2
    stage2_fraction: float = 0.3
    stage3_fraction: float = 0.5
    stage2_lr_factor: float = 0.1
    stage3_lr_factor: float = 0.05

    # Graph construction
    dependency_threshold: float = 0.3

    # Sampling
    n_ode_steps: int = 100
    ode_atol: float = 1e-5
    ode_rtol: float = 1e-5

    # Large-dataset iteration budget
    max_steps_per_epoch: int = 10_000


class GraphAugmentedFlowMatching(nn.Module):
    """Main model: velocity decomposition v = v_coord + v_feat + v_samp.

    Args:
        d_c: number of continuous features.
        categorical_dims: list of category counts per categorical feature.
        n_ordinal: number of ordinal features.
        feature_graph: (edge_index, edge_weight, node_types) or None.
        config: model/training configuration.
    """

    def __init__(
        self,
        d_c: int,
        categorical_dims: list[int],
        n_ordinal: int,
        feature_graph: tuple[Tensor, Tensor, Tensor] | None = None,
        config: GAFMConfig | None = None,
    ) -> None:
        super().__init__()
        self.config = config or GAFMConfig()
        self.d_c = d_c
        self.categorical_dims = categorical_dims
        self.n_ordinal = n_ordinal

        cfg = self.config

        # Time embedding
        self.time_embed = SinusoidalTimeEmbedding(cfg.time_embed_dim)

        # Component 1: coordinate-wise MLP velocity
        self.coord_net = CoordinateWiseVelocity(
            d_c=d_c,
            categorical_dims=categorical_dims,
            n_ordinal=n_ordinal,
            hidden_dim=cfg.hidden_dim,
            num_layers=cfg.num_mlp_layers,
            time_embed_dim=cfg.time_embed_dim,
        )

        # Component 2: feature graph network
        self.feat_gnn = FeatureGraphNetwork(
            d_c=d_c,
            categorical_dims=categorical_dims,
            n_ordinal=n_ordinal,
            hidden_dim=cfg.hidden_dim,
            num_layers=cfg.feat_gnn_layers,
            time_embed_dim=cfg.time_embed_dim,
        )

        # Component 3: sample graph network
        self.samp_gnn = SampleGraphNetwork(
            d_c=d_c,
            categorical_dims=categorical_dims,
            n_ordinal=n_ordinal,
            hidden_dim=cfg.samp_gnn_hidden,
            num_layers=cfg.samp_gnn_layers,
            time_embed_dim=cfg.time_embed_dim,
            k=cfg.k_neighbors,
        )

        # Store feature graph as buffers
        if feature_graph is not None:
            ei, ew, nt = feature_graph
            self.register_buffer("feat_edge_index", ei)
            self.register_buffer("feat_edge_weight", ew)
            self.register_buffer("feat_node_types", nt)
        else:
            self.register_buffer("feat_edge_index", torch.zeros(2, 0, dtype=torch.long))
            self.register_buffer("feat_edge_weight", torch.zeros(0))
            self.register_buffer("feat_node_types", torch.zeros(0, dtype=torch.long))

    def set_feature_graph(self, edge_index: Tensor, edge_weight: Tensor, node_types: Tensor) -> None:
        """Set or update the feature graph after construction."""
        self.feat_edge_index = edge_index.to(self.feat_edge_weight.device)
        self.feat_edge_weight = edge_weight.to(self.feat_edge_weight.device)
        self.feat_node_types = node_types.to(self.feat_edge_weight.device)

    def forward(
        self,
        x_c: Tensor | None,
        x_d_list: list[Tensor],
        x_o_list: list[Tensor],
        t: Tensor,
        stage: int = 3,
    ) -> tuple[Tensor | None, list[Tensor], list[Tensor]]:
        """Compute combined velocity field.

        Args:
            x_c: (B, d_c) continuous features or None.
            x_d_list: list of (B, n_j) categorical distributions.
            x_o_list: list of (B,) ordinal angles.
            t: (B,) time values in [0, 1].
            stage: training stage (1=coord only, 2=+feat, 3=+samp).
        """
        t_emb = self.time_embed(t)

        # Component 1: always active
        v_c, v_d, v_o = self.coord_net(x_c, x_d_list, x_o_list, t_emb)

        # Component 2: active in stages 2 and 3
        if stage >= 2 and self.feat_edge_index.shape[1] > 0:
            vf_c, vf_d, vf_o = self.feat_gnn(
                x_c, x_d_list, x_o_list, t_emb,
                self.feat_edge_index, self.feat_edge_weight, self.feat_node_types,
            )
            v_c = v_c + vf_c if v_c is not None and vf_c is not None else (v_c or vf_c)
            v_d = [a + b for a, b in zip(v_d, vf_d)]
            v_o = [a + b for a, b in zip(v_o, vf_o)]

        # Component 3: active in stage 3
        if stage >= 3:
            vs_c, vs_d, vs_o = self.samp_gnn(x_c, x_d_list, x_o_list, t_emb)
            v_c = v_c + vs_c if v_c is not None and vs_c is not None else (v_c or vs_c)
            v_d = [a + b for a, b in zip(v_d, vs_d)]
            v_o = [a + b for a, b in zip(v_o, vs_o)]

        return v_c, v_d, v_o
