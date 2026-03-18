"""Sinkhorn optimal transport solver with product manifold cost. Section 4.5, Eqs 33-35."""

from __future__ import annotations

import torch
from torch import Tensor

from graph_flow_matching.models.manifold_ops import pairwise_product_distance_matrix


def sinkhorn(
    cost_matrix: Tensor,
    epsilon: float = 0.01,
    num_iterations: int = 100,
) -> Tensor:
    """Log-domain Sinkhorn algorithm for entropic OT.

    Args:
        cost_matrix: (B, B) non-negative cost matrix.
        epsilon: entropic regularization parameter.
        num_iterations: number of Sinkhorn iterations.
    Returns:
        (B, B) doubly-stochastic transport plan.
    """
    B = cost_matrix.shape[0]
    log_K = -cost_matrix / epsilon  # (B, B)

    # Uniform marginals in log space
    log_a = -torch.log(torch.tensor(B, dtype=cost_matrix.dtype, device=cost_matrix.device)).expand(B)
    log_b = log_a.clone()

    log_u = torch.zeros(B, device=cost_matrix.device, dtype=cost_matrix.dtype)
    log_v = torch.zeros(B, device=cost_matrix.device, dtype=cost_matrix.dtype)

    for _ in range(num_iterations):
        log_v = log_b - torch.logsumexp(log_K + log_u.unsqueeze(1), dim=0)
        log_u = log_a - torch.logsumexp(log_K + log_v.unsqueeze(0), dim=1)

    # Recover coupling: pi = diag(u) K diag(v)
    log_pi = log_u.unsqueeze(1) + log_K + log_v.unsqueeze(0)
    return torch.exp(log_pi)


def compute_ot_coupling(
    x0_c: Tensor | None,
    x1_c: Tensor | None,
    x0_d: list[Tensor],
    x1_d: list[Tensor],
    x0_o: list[Tensor],
    x1_o: list[Tensor],
    epsilon: float = 0.01,
    num_iterations: int = 100,
) -> Tensor:
    """Compute OT coupling between source and target samples. Eqs 33-35.

    Returns:
        (B, B) transport plan.
    """
    cost = pairwise_product_distance_matrix(x0_c, x1_c, x0_d, x1_d, x0_o, x1_o)
    return sinkhorn(cost, epsilon, num_iterations)


def sample_ot_pairs(coupling: Tensor) -> Tensor:
    """Sample target indices from OT coupling for each source.

    Args:
        coupling: (B, B) transport plan.
    Returns:
        (B,) target index for each source.
    """
    B = coupling.shape[0]
    probs = coupling / coupling.sum(dim=1, keepdim=True).clamp(min=1e-12)
    return torch.multinomial(probs, num_samples=1).squeeze(-1)
