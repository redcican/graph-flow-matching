"""Dynamic k-NN sample graph and lightweight MPNN for v_samp. Section 4.3, Eqs 24-28."""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor

from graph_flow_matching.models.manifold_ops import pairwise_sample_distance_l1


# ---------------------------------------------------------------------------
# Dynamic graph construction (Eqs 24-25)
# ---------------------------------------------------------------------------

def build_sample_graph(
    x_c: Tensor | None,
    x_d_list: list[Tensor],
    x_o_list: list[Tensor],
    k: int = 10,
) -> tuple[Tensor, Tensor]:
    """Construct k-NN graph over batch samples using product manifold distance.

    Args:
        x_c, x_d_list, x_o_list: current sample positions at time t.
        k: number of nearest neighbors.
    Returns:
        edge_index: (2, B*k) directed edges j -> i.
        edge_weight: (B*k,) softmax(-distance) proximity weights (Eq 27).
    """
    dist = pairwise_sample_distance_l1(
        x_c, x_c, x_d_list, x_d_list, x_o_list, x_o_list
    )  # (B, B) — L1 combination per Eq 25

    B = dist.shape[0]
    k_actual = min(k, B - 1)

    # Set self-distance to infinity
    dist.fill_diagonal_(float("inf"))

    # k nearest neighbors per sample
    knn_dist, knn_idx = dist.topk(k_actual, dim=1, largest=False)  # (B, k)

    # Build edge index
    src = knn_idx.reshape(-1)  # (B*k,)
    dst = torch.arange(B, device=dist.device).unsqueeze(1).expand(-1, k_actual).reshape(-1)
    edge_index = torch.stack([src, dst], dim=0)

    # Edge weights: softmax(-distance) per target node (Eq 27)
    weights = torch.softmax(-knn_dist, dim=1).reshape(-1)

    return edge_index, weights


# ---------------------------------------------------------------------------
# Sample-level MPNN (Eqs 26-28)
# ---------------------------------------------------------------------------

class SampleMessagePassingLayer(nn.Module):
    """Single layer of sample-level message passing. Eqs 26-27."""

    def __init__(self, hidden_dim: int, input_dim: int, time_embed_dim: int = 64) -> None:
        super().__init__()
        # MLP_msg: [s_j || s_i || (x_j - x_i) || t_emb] -> hidden_dim
        self.mlp_msg = nn.Sequential(
            nn.Linear(2 * hidden_dim + input_dim + time_embed_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        # MLP_upd: [s_i || m_tilde_i] -> hidden_dim
        self.mlp_upd = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(
        self,
        s: Tensor,            # (B, hidden_dim)
        x_flat: Tensor,       # (B, input_dim)
        edge_index: Tensor,   # (2, E)
        edge_weight: Tensor,  # (E,)
        t_emb: Tensor,        # (B, time_embed_dim)
    ) -> Tensor:
        src, dst = edge_index
        E = src.shape[0]
        B = s.shape[0]

        # Compute messages (Eq 26)
        s_j = s[src]                         # (E, hd)
        s_i = s[dst]                         # (E, hd)
        dx = x_flat[src] - x_flat[dst]       # (E, input_dim)
        t_exp = t_emb[dst]                   # (E, t_dim)

        msg = self.mlp_msg(torch.cat([s_j, s_i, dx, t_exp], dim=-1))  # (E, hd)

        # Weighted aggregation (Eq 27)
        weighted_msg = msg * edge_weight.unsqueeze(-1)
        agg = torch.zeros(B, s.shape[-1], device=s.device)
        agg.scatter_add_(0, dst.unsqueeze(-1).expand_as(weighted_msg), weighted_msg)

        # Update (Eq 27)
        return self.mlp_upd(torch.cat([s, agg], dim=-1))


class SampleGraphNetwork(nn.Module):
    """Lightweight MPNN over dynamic sample graph. Section 4.3."""

    def __init__(
        self,
        d_c: int,
        categorical_dims: list[int],
        n_ordinal: int,
        hidden_dim: int = 128,
        num_layers: int = 2,
        time_embed_dim: int = 64,
        k: int = 10,
    ) -> None:
        super().__init__()
        self.d_c = d_c
        self.categorical_dims = categorical_dims
        self.n_ordinal = n_ordinal
        self.k = k

        # Flat input dimension: d_c + sum(n_j) + K_o (angles as scalars)
        self.input_dim = d_c + sum(categorical_dims) + n_ordinal
        self.output_dim = d_c + sum(categorical_dims) + n_ordinal

        self.input_proj = nn.Linear(self.input_dim, hidden_dim)
        self.layers = nn.ModuleList([
            SampleMessagePassingLayer(hidden_dim, self.input_dim, time_embed_dim)
            for _ in range(num_layers)
        ])
        self.output_proj = nn.Linear(hidden_dim, self.output_dim)

    def _flatten(
        self,
        x_c: Tensor | None,
        x_d_list: list[Tensor],
        x_o_list: list[Tensor],
    ) -> Tensor:
        """Flatten all features to a single vector per sample."""
        parts: list[Tensor] = []
        if x_c is not None:
            parts.append(x_c)
        parts.extend(x_d_list)
        for x_o in x_o_list:
            parts.append(x_o.unsqueeze(-1))
        return torch.cat(parts, dim=-1)

    def _decompose(
        self, v_flat: Tensor
    ) -> tuple[Tensor | None, list[Tensor], list[Tensor]]:
        """Decompose flat velocity back into per-type components."""
        offset = 0
        v_c = None
        if self.d_c > 0:
            v_c = v_flat[:, offset:offset + self.d_c]
            offset += self.d_c

        v_d: list[Tensor] = []
        for n in self.categorical_dims:
            v = v_flat[:, offset:offset + n]
            v = v - v.mean(dim=-1, keepdim=True)  # tangent space
            v_d.append(v)
            offset += n

        v_o: list[Tensor] = []
        for _ in range(self.n_ordinal):
            v_o.append(v_flat[:, offset])
            offset += 1

        return v_c, v_d, v_o

    def forward(
        self,
        x_c: Tensor | None,
        x_d_list: list[Tensor],
        x_o_list: list[Tensor],
        t_emb: Tensor,
    ) -> tuple[Tensor | None, list[Tensor], list[Tensor]]:
        # Build dynamic k-NN graph
        edge_index, edge_weight = build_sample_graph(x_c, x_d_list, x_o_list, self.k)

        # Flatten features
        x_flat = self._flatten(x_c, x_d_list, x_o_list)

        # Embed
        s = self.input_proj(x_flat)

        # Message passing
        for layer in self.layers:
            s = s + layer(s, x_flat, edge_index, edge_weight, t_emb)

        # Output projection (Eq 28)
        v_flat = self.output_proj(s)
        return self._decompose(v_flat)
