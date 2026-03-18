"""Manifold-preserving aggregation: Einstein midpoint, circular mean, cross-type projections.

Implements Section 4.4, Eqs 29-32 and Definition 5.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor

from graph_flow_matching.models.manifold_ops import EPS, clamp_probabilities


# ---------------------------------------------------------------------------
# Within-type aggregation (Eqs 29-32)
# ---------------------------------------------------------------------------

def weighted_average(values: Tensor, weights: Tensor) -> Tensor:
    """Weighted average for continuous features (Eq 29).

    Args:
        values: (K, d) feature vectors.
        weights: (K,) non-negative, sum to 1.
    """
    return (weights.unsqueeze(-1) * values).sum(dim=0)


def einstein_midpoint(distributions: Tensor, weights: Tensor) -> Tensor:
    """Einstein midpoint for categorical features on the simplex (Eqs 30-31).

    Minimizes weighted Frechet variance under Fisher-Rao metric.

    Args:
        distributions: (K, n) probability distributions.
        weights: (K,) normalized weights.
    Returns:
        (n,) probability distribution on the simplex.
    """
    p = clamp_probabilities(distributions)
    sqrt_p = torch.sqrt(p)  # (K, n)
    weighted_sqrt = (weights.unsqueeze(-1) * sqrt_p).sum(dim=0)  # (n,)
    result = weighted_sqrt.pow(2)
    return result / result.sum().clamp(min=EPS)


def circular_mean(angles: Tensor, weights: Tensor) -> Tensor:
    """Circular mean for ordinal features on S^1 (Eq 32).

    Args:
        angles: (K,) angles in [0, 2*pi).
        weights: (K,) normalized weights.
    Returns:
        Scalar mean angle.
    """
    real = (weights * torch.cos(angles)).sum()
    imag = (weights * torch.sin(angles)).sum()
    return torch.atan2(imag, real) % (2.0 * torch.pi)


# ---------------------------------------------------------------------------
# Batched variants for GNN layers
# ---------------------------------------------------------------------------

def batched_einstein_midpoint(
    distributions: Tensor,
    weights: Tensor,
    index: Tensor,
    num_nodes: int,
) -> Tensor:
    """Scatter-based Einstein midpoint for batched message passing.

    Args:
        distributions: (E, n) probability distributions of source nodes.
        weights: (E,) edge weights.
        index: (E,) target node indices.
        num_nodes: total number of target nodes.
    Returns:
        (num_nodes, n) aggregated distributions.
    """
    p = clamp_probabilities(distributions)
    sqrt_p = torch.sqrt(p) * weights.unsqueeze(-1)

    agg = torch.zeros(num_nodes, p.shape[-1], device=p.device)
    agg.scatter_add_(0, index.unsqueeze(-1).expand_as(sqrt_p), sqrt_p)

    result = agg.pow(2)
    return result / result.sum(dim=-1, keepdim=True).clamp(min=EPS)


def batched_circular_mean(
    angles: Tensor,
    weights: Tensor,
    index: Tensor,
    num_nodes: int,
) -> Tensor:
    """Scatter-based circular mean for batched message passing.

    Args:
        angles: (E,) source angles.
        weights: (E,) edge weights.
        index: (E,) target node indices.
        num_nodes: total number of target nodes.
    Returns:
        (num_nodes,) aggregated angles.
    """
    real = torch.zeros(num_nodes, device=angles.device)
    imag = torch.zeros(num_nodes, device=angles.device)
    real.scatter_add_(0, index, weights * torch.cos(angles))
    imag.scatter_add_(0, index, weights * torch.sin(angles))
    return torch.atan2(imag, real) % (2.0 * torch.pi)


# ---------------------------------------------------------------------------
# Cross-type projection operators (Definition 5)
# ---------------------------------------------------------------------------

class CrossTypeProjection(nn.Module):
    """Learnable cross-type projection operators between manifold types."""

    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        self.cont_to_cat = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, hidden_dim)
        )
        self.cont_to_ord = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, hidden_dim)
        )
        self.cat_to_cont = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, hidden_dim)
        )
        self.ord_to_cont = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, hidden_dim)
        )

    def project(self, h: Tensor, src_type: int, tgt_type: int) -> Tensor:
        """Project embedding from source type to target type.

        Type codes: 0=continuous, 1=categorical, 2=ordinal.
        """
        if src_type == tgt_type:
            return h
        key = f"{src_type}_{tgt_type}"
        proj_map = {
            "0_1": self.cont_to_cat,
            "0_2": self.cont_to_ord,
            "1_0": self.cat_to_cont,
            "2_0": self.ord_to_cont,
            "1_2": nn.Sequential(self.cat_to_cont, self.cont_to_ord),
            "2_1": nn.Sequential(self.ord_to_cont, self.cont_to_cat),
        }
        return proj_map[key](h)
