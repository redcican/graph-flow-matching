"""Feature graph construction and heterogeneous GNN for v_feat. Section 4.2, Eqs 18-22."""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import mutual_info_score, normalized_mutual_info_score
from sklearn.feature_selection import mutual_info_classif, mutual_info_regression
from torch import Tensor

from graph_flow_matching.baselines.base import ColumnSpec

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Feature graph construction (Eqs 18-20)
# ---------------------------------------------------------------------------

def build_feature_graph(
    df: pd.DataFrame,
    columns: list[ColumnSpec],
    threshold: float = 0.3,
) -> tuple[Tensor, Tensor, Tensor]:
    """Construct static feature dependency graph from training data.

    Args:
        df: training DataFrame.
        columns: column specifications.
        threshold: tau for edge filtering.
    Returns:
        edge_index: (2, E) long tensor.
        edge_weight: (E,) float tensor.
        node_types: (D,) long tensor (0=cont, 1=cat, 2=ord).
    """
    D = len(columns)
    type_map = {"continuous": 0, "categorical": 1, "ordinal": 2}
    node_types = torch.tensor([type_map[c.dtype] for c in columns], dtype=torch.long)

    src_list: list[int] = []
    dst_list: list[int] = []
    weight_list: list[float] = []

    for i in range(D):
        for j in range(i + 1, D):
            w = _compute_dependency(df, columns[i], columns[j])
            if w > threshold:
                src_list.extend([i, j])
                dst_list.extend([j, i])
                weight_list.extend([w, w])

    if not src_list:
        # Fallback: connect each node to its nearest neighbor
        logger.warning("Empty feature graph; adding self-loops")
        src_list = list(range(D))
        dst_list = list(range(D))
        weight_list = [1.0] * D

    edge_index = torch.tensor([src_list, dst_list], dtype=torch.long)
    edge_weight = torch.tensor(weight_list, dtype=torch.float32)

    logger.info("Feature graph: %d nodes, %d edges (avg degree %.1f)",
                D, edge_index.shape[1], edge_index.shape[1] / D)
    return edge_index, edge_weight, node_types


def _compute_dependency(df: pd.DataFrame, spec_i: ColumnSpec, spec_j: ColumnSpec) -> float:
    """Compute pairwise dependency between two features."""
    xi = df[spec_i.name].dropna()
    xj = df[spec_j.name].dropna()
    common = xi.index.intersection(xj.index)
    if len(common) < 10:
        return 0.0
    xi, xj = xi.loc[common], xj.loc[common]

    ti, tj = spec_i.dtype, spec_j.dtype

    try:
        if ti == "continuous" and tj == "continuous":
            return abs(float(np.corrcoef(xi.values.astype(float), xj.values.astype(float))[0, 1]))

        if ti in ("categorical", "ordinal") and tj in ("categorical", "ordinal"):
            return float(normalized_mutual_info_score(xi.astype(str), xj.astype(str)))

        # Mixed: use sklearn mutual_info as MIC proxy (Eq 20)
        if ti == "continuous":
            cont, disc = xi.values.astype(float).reshape(-1, 1), xj.astype(str)
        else:
            cont, disc = xj.values.astype(float).reshape(-1, 1), xi.astype(str)

        from sklearn.preprocessing import LabelEncoder
        le = LabelEncoder()
        disc_encoded = le.fit_transform(disc)
        mi = mutual_info_classif(cont, disc_encoded, random_state=42, n_neighbors=5)
        # Normalize to [0, 1] range
        h_disc = max(float(np.log(len(le.classes_))), 1e-8)
        return min(float(mi[0]) / h_disc, 1.0)
    except Exception:
        return 0.0


# ---------------------------------------------------------------------------
# Heterogeneous GNN (Eqs 20-22)
# ---------------------------------------------------------------------------

class TypeAwareMessagePassingLayer(nn.Module):
    """Single layer of heterogeneous message passing. Eqs 20-21."""

    def __init__(self, hidden_dim: int, time_embed_dim: int = 64, num_types: int = 3) -> None:
        super().__init__()
        # Type-specific update: W_Ti [h_i || m_i || t_emb] + b_Ti
        self.update_nets = nn.ModuleList([
            nn.Sequential(
                nn.Linear(2 * hidden_dim + time_embed_dim, hidden_dim),
                nn.GELU(),
            )
            for _ in range(num_types)
        ])

    def forward(
        self,
        h: Tensor,            # (B, D, hidden_dim) or (D, hidden_dim)
        edge_index: Tensor,   # (2, E)
        edge_weight: Tensor,  # (E,)
        node_types: Tensor,   # (D,)
        t_emb: Tensor,        # (B, time_embed_dim) or (time_embed_dim,)
    ) -> Tensor:
        batched = h.dim() == 3
        if not batched:
            h = h.unsqueeze(0)
            t_emb = t_emb.unsqueeze(0)

        B, D, hd = h.shape
        src, dst = edge_index  # (E,)

        # Gather source embeddings and aggregate for each destination
        h_src = h[:, src, :]  # (B, E, hd)
        w = edge_weight.unsqueeze(0).unsqueeze(-1)  # (1, E, 1)
        weighted_msg = h_src * w  # (B, E, hd)

        # Scatter-add to destination nodes
        agg = torch.zeros(B, D, hd, device=h.device)
        dst_exp = dst.unsqueeze(0).unsqueeze(-1).expand(B, -1, hd)
        agg.scatter_add_(1, dst_exp, weighted_msg)

        # Normalize by degree
        degree = torch.zeros(D, device=h.device)
        degree.scatter_add_(0, dst, edge_weight)
        degree = degree.clamp(min=1e-8)
        agg = agg / degree.unsqueeze(0).unsqueeze(-1)

        # Type-specific update
        t_exp = t_emb.unsqueeze(1).expand(B, D, -1)  # (B, D, t_dim)
        cat_input = torch.cat([h, agg, t_exp], dim=-1)  # (B, D, 2*hd + t_dim)

        out = torch.zeros_like(h)
        for type_id in range(len(self.update_nets)):
            mask = (node_types == type_id)
            if mask.any():
                out[:, mask] = self.update_nets[type_id](cat_input[:, mask])

        if not batched:
            out = out.squeeze(0)
        return out


class FeatureGraphNetwork(nn.Module):
    """Heterogeneous GNN over the static feature graph. Section 4.2."""

    def __init__(
        self,
        d_c: int,
        categorical_dims: list[int],
        n_ordinal: int,
        hidden_dim: int = 256,
        num_layers: int = 3,
        time_embed_dim: int = 64,
    ) -> None:
        super().__init__()
        self.d_c = d_c
        self.categorical_dims = categorical_dims
        self.n_ordinal = n_ordinal
        self.hidden_dim = hidden_dim

        # Input projections: map each feature value to hidden_dim
        if d_c > 0:
            self.input_proj_cont = nn.Linear(1, hidden_dim)
        self.input_proj_cat = nn.ModuleList(
            [nn.Linear(n, hidden_dim) for n in categorical_dims]
        )
        if n_ordinal > 0:
            self.input_proj_ord = nn.Linear(2, hidden_dim)  # (cos, sin)

        # Message passing layers
        self.layers = nn.ModuleList([
            TypeAwareMessagePassingLayer(hidden_dim, time_embed_dim)
            for _ in range(num_layers)
        ])

        # Output projections: map embeddings back to velocity corrections
        if d_c > 0:
            self.output_proj_cont = nn.Linear(hidden_dim, 1)
        self.output_proj_cat = nn.ModuleList(
            [nn.Linear(hidden_dim, n) for n in categorical_dims]
        )
        if n_ordinal > 0:
            self.output_proj_ord = nn.Linear(hidden_dim, 1)

    def forward(
        self,
        x_c: Tensor | None,
        x_d_list: list[Tensor],
        x_o_list: list[Tensor],
        t_emb: Tensor,
        edge_index: Tensor,
        edge_weight: Tensor,
        node_types: Tensor,
    ) -> tuple[Tensor | None, list[Tensor], list[Tensor]]:
        B = t_emb.shape[0]
        embeddings: list[Tensor] = []

        # Project each feature to hidden_dim
        if x_c is not None and self.d_c > 0:
            for i in range(self.d_c):
                embeddings.append(self.input_proj_cont(x_c[:, i:i+1]))  # (B, hd)

        for proj, x_d in zip(self.input_proj_cat, x_d_list):
            embeddings.append(proj(x_d))  # (B, hd)

        for x_o in x_o_list:
            feat = torch.stack([torch.cos(x_o), torch.sin(x_o)], dim=-1)
            embeddings.append(self.input_proj_ord(feat))  # (B, hd)

        h = torch.stack(embeddings, dim=1)  # (B, D, hd)

        # Message passing
        for layer in self.layers:
            h = h + layer(h, edge_index, edge_weight, node_types, t_emb)

        # Output projections
        v_c = None
        idx = 0
        if x_c is not None and self.d_c > 0:
            parts = []
            for i in range(self.d_c):
                parts.append(self.output_proj_cont(h[:, idx]).squeeze(-1))
                idx += 1
            v_c = torch.stack(parts, dim=-1)  # (B, d_c)

        v_d: list[Tensor] = []
        for proj in self.output_proj_cat:
            v = proj(h[:, idx])  # (B, n_j)
            v = v - v.mean(dim=-1, keepdim=True)  # tangent space
            v_d.append(v)
            idx += 1

        v_o: list[Tensor] = []
        for _ in x_o_list:
            v_o.append(self.output_proj_ord(h[:, idx]).squeeze(-1))  # (B,)
            idx += 1

        return v_c, v_d, v_o
