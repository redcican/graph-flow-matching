"""Geometric operations on the product manifold M = R^{d_c} x prod Delta^{n_j-1} x prod S^1.

Implements Fisher-Rao distances, geodesic interpolation, conditional velocities,
and all numerical safeguards described in Section 3 (Eqs 5-12).
"""

from __future__ import annotations

import torch
from torch import Tensor

EPS: float = 1e-8
SLERP_THRESHOLD: float = 1e-6


# ---------------------------------------------------------------------------
# Probability simplex utilities
# ---------------------------------------------------------------------------

def clamp_probabilities(p: Tensor, eps: float = EPS) -> Tensor:
    """Clamp probabilities to [eps, 1] and renormalize (Section 3, below Eq 9)."""
    p = p.clamp(min=eps)
    return p / p.sum(dim=-1, keepdim=True)


def sphere_map(p: Tensor) -> Tensor:
    """Map simplex to positive orthant of unit sphere: phi(p) = 2*sqrt(p). Eq 5."""
    return 2.0 * torch.sqrt(clamp_probabilities(p))


def sphere_map_inverse(s: Tensor) -> Tensor:
    """Inverse sphere map: p = (s/2)^2, renormalized."""
    p = (s / 2.0).pow(2)
    return p / p.sum(dim=-1, keepdim=True).clamp(min=EPS)


# ---------------------------------------------------------------------------
# Distance functions
# ---------------------------------------------------------------------------

def fisher_rao_distance(p: Tensor, q: Tensor) -> Tensor:
    """Fisher-Rao distance on the simplex. Eq 6.

    d_FR(p, q) = 2 * arccos(sum(sqrt(p_k * q_k))).
    """
    p = clamp_probabilities(p)
    q = clamp_probabilities(q)
    cos_val = (torch.sqrt(p) * torch.sqrt(q)).sum(dim=-1)
    cos_val = cos_val.clamp(-1.0 + EPS, 1.0 - EPS)
    return 2.0 * torch.acos(cos_val)


def circular_distance(theta1: Tensor, theta2: Tensor) -> Tensor:
    """Shortest arc distance on S^1. Eq 7."""
    diff = torch.abs(theta1 - theta2)
    return torch.min(diff, 2.0 * torch.pi - diff)


def product_manifold_distance_squared(
    x_c1: Tensor | None,
    x_c2: Tensor | None,
    x_d1: list[Tensor],
    x_d2: list[Tensor],
    x_o1: list[Tensor],
    x_o2: list[Tensor],
) -> Tensor:
    """Squared product manifold distance. Eq 8 / Eq 34 (for OT cost).

    Returns scalar distance per pair. Inputs are (B, ...) tensors.
    """
    dist_sq = torch.zeros(x_d1[0].shape[0], device=x_d1[0].device) if x_d1 else torch.tensor(0.0)

    if x_c1 is not None and x_c2 is not None:
        dist_sq = dist_sq + (x_c1 - x_c2).pow(2).sum(dim=-1)

    for p1, p2 in zip(x_d1, x_d2):
        dist_sq = dist_sq + fisher_rao_distance(p1, p2).pow(2)

    for t1, t2 in zip(x_o1, x_o2):
        dist_sq = dist_sq + circular_distance(t1, t2).pow(2)

    return dist_sq


def product_manifold_distance(
    x_c1: Tensor | None,
    x_c2: Tensor | None,
    x_d1: list[Tensor],
    x_d2: list[Tensor],
    x_o1: list[Tensor],
    x_o2: list[Tensor],
) -> Tensor:
    """Product manifold distance (Eq 8, Definition 1)."""
    return torch.sqrt(product_manifold_distance_squared(x_c1, x_c2, x_d1, x_d2, x_o1, x_o2) + EPS)


def pairwise_sample_distance_l1(
    x_c1: Tensor | None,
    x_c2: Tensor | None,
    x_d1: list[Tensor],
    x_d2: list[Tensor],
    x_o1: list[Tensor],
    x_o2: list[Tensor],
) -> Tensor:
    """Pairwise L1 distance matrix (B1, B2) for k-NN. Eq 25.

    Sums unsquared component distances:
    d(x,y) = ||x_c - y_c||_2 + Σ d_FR(x_d, y_d) + Σ d_circ(x_o, y_o).
    """
    B1 = x_d1[0].shape[0] if x_d1 else (x_c1.shape[0] if x_c1 is not None else x_o1[0].shape[0])
    B2 = x_d2[0].shape[0] if x_d2 else (x_c2.shape[0] if x_c2 is not None else x_o2[0].shape[0])
    device = x_d1[0].device if x_d1 else (x_c1.device if x_c1 is not None else x_o1[0].device)

    dist = torch.zeros(B1, B2, device=device)

    if x_c1 is not None and x_c2 is not None:
        # ||a - b||_2 (unsquared Euclidean)
        sq = x_c1.pow(2).sum(dim=-1, keepdim=True) + x_c2.pow(2).sum(dim=-1, keepdim=True).t() - 2.0 * x_c1 @ x_c2.t()
        dist = dist + torch.sqrt(sq.clamp(min=EPS))

    for p1, p2 in zip(x_d1, x_d2):
        p1c = clamp_probabilities(p1)
        p2c = clamp_probabilities(p2)
        cos_mat = torch.sqrt(p1c) @ torch.sqrt(p2c).t()
        cos_mat = cos_mat.clamp(-1.0 + EPS, 1.0 - EPS)
        dist = dist + 2.0 * torch.acos(cos_mat)  # unsquared Fisher-Rao

    for t1, t2 in zip(x_o1, x_o2):
        diff = torch.abs(t1.unsqueeze(1) - t2.unsqueeze(0))
        dist = dist + torch.min(diff, 2.0 * torch.pi - diff)  # unsquared circular

    return dist


def pairwise_product_distance_matrix(
    x_c1: Tensor | None,
    x_c2: Tensor | None,
    x_d1: list[Tensor],
    x_d2: list[Tensor],
    x_o1: list[Tensor],
    x_o2: list[Tensor],
) -> Tensor:
    """Pairwise squared distance matrix (B1, B2) for OT cost (Eq 34)."""
    B1 = x_d1[0].shape[0] if x_d1 else (x_c1.shape[0] if x_c1 is not None else x_o1[0].shape[0])
    B2 = x_d2[0].shape[0] if x_d2 else (x_c2.shape[0] if x_c2 is not None else x_o2[0].shape[0])
    device = x_d1[0].device if x_d1 else (x_c1.device if x_c1 is not None else x_o1[0].device)

    dist = torch.zeros(B1, B2, device=device)

    if x_c1 is not None and x_c2 is not None:
        # ||a - b||^2 = ||a||^2 + ||b||^2 - 2*a.b
        norm1 = x_c1.pow(2).sum(dim=-1, keepdim=True)  # (B1, 1)
        norm2 = x_c2.pow(2).sum(dim=-1, keepdim=True)  # (B2, 1)
        dist = dist + norm1 + norm2.t() - 2.0 * x_c1 @ x_c2.t()

    for p1, p2 in zip(x_d1, x_d2):
        p1c = clamp_probabilities(p1)
        p2c = clamp_probabilities(p2)
        cos_mat = torch.sqrt(p1c) @ torch.sqrt(p2c).t()
        cos_mat = cos_mat.clamp(-1.0 + EPS, 1.0 - EPS)
        dist = dist + (2.0 * torch.acos(cos_mat)).pow(2)

    for t1, t2 in zip(x_o1, x_o2):
        diff = torch.abs(t1.unsqueeze(1) - t2.unsqueeze(0))
        circ = torch.min(diff, 2.0 * torch.pi - diff)
        dist = dist + circ.pow(2)

    return dist


# ---------------------------------------------------------------------------
# Geodesic interpolation (Eq 9)
# ---------------------------------------------------------------------------

def geodesic_interpolation_continuous(
    x0: Tensor, x1: Tensor, t: Tensor
) -> Tensor:
    """Linear interpolation for continuous features."""
    return (1.0 - t) * x0 + t * x1


def geodesic_interpolation_categorical(
    x0: Tensor, x1: Tensor, t: Tensor
) -> Tensor:
    """Spherical linear interpolation on the simplex via sphere map. Eq 9 line 2.

    Args:
        x0, x1: (B, n_j) probability distributions on the simplex.
        t: (B, 1) or scalar interpolation parameter in [0, 1].
    """
    s0 = sphere_map(x0)  # (B, n_j)
    s1 = sphere_map(x1)

    # Compute angle between sphere-mapped points
    cos_theta = (s0 * s1).sum(dim=-1) / (
        s0.norm(dim=-1) * s1.norm(dim=-1) + EPS
    )
    cos_theta = cos_theta.clamp(-1.0 + EPS, 1.0 - EPS)
    theta = torch.acos(cos_theta)  # (B,)

    if t.dim() == 0:
        t_val = t
    else:
        t_val = t.squeeze(-1) if t.dim() > 1 else t  # (B,)

    sin_theta = torch.sin(theta).clamp(min=SLERP_THRESHOLD)
    coeff0 = torch.sin((1.0 - t_val) * theta) / sin_theta
    coeff1 = torch.sin(t_val * theta) / sin_theta

    # Fallback to linear interpolation when theta is small
    small = theta < SLERP_THRESHOLD
    coeff0 = torch.where(small, 1.0 - t_val, coeff0)
    coeff1 = torch.where(small, t_val, coeff1)

    s_t = coeff0.unsqueeze(-1) * s0 + coeff1.unsqueeze(-1) * s1
    return sphere_map_inverse(s_t)


def geodesic_interpolation_ordinal(
    theta0: Tensor, theta1: Tensor, t: Tensor
) -> Tensor:
    """Circular geodesic interpolation. Eq 9 line 3."""
    delta = torch.atan2(
        torch.sin(theta1 - theta0), torch.cos(theta1 - theta0)
    )
    t_val = t.squeeze(-1) if t.dim() > 1 else t
    return (theta0 + t_val * delta) % (2.0 * torch.pi)


# ---------------------------------------------------------------------------
# Conditional velocities (Eq 12)
# ---------------------------------------------------------------------------

def conditional_velocity_continuous(x0: Tensor, x1: Tensor) -> Tensor:
    """u_t^c = x1 - x0. Eq 12 line 1."""
    return x1 - x0


def conditional_velocity_categorical(
    x0: Tensor, x1: Tensor, t: Tensor
) -> Tensor:
    """Fisher-Rao conditional velocity. Eq 12 line 2.

    u_t = theta/sin(theta) * [-cos((1-t)*theta)*x0 + cos(t*theta)*x1]
    """
    x0 = clamp_probabilities(x0)
    x1 = clamp_probabilities(x1)
    theta = fisher_rao_distance(x0, x1)  # (B,)
    t_val = t.squeeze(-1) if t.dim() > 1 else t

    sin_theta = torch.sin(theta).clamp(min=SLERP_THRESHOLD)
    scale = theta / sin_theta

    c0 = -torch.cos((1.0 - t_val) * theta)
    c1 = torch.cos(t_val * theta)

    u = scale.unsqueeze(-1) * (c0.unsqueeze(-1) * x0 + c1.unsqueeze(-1) * x1)

    # Fallback for small theta: u_t ≈ x1 - x0
    small = (theta < SLERP_THRESHOLD).unsqueeze(-1)
    return torch.where(small, x1 - x0, u)


def conditional_velocity_ordinal(theta0: Tensor, theta1: Tensor) -> Tensor:
    """u_t^{o,l} = delta_l (signed angular displacement). Eq 12 line 3."""
    return torch.atan2(
        torch.sin(theta1 - theta0), torch.cos(theta1 - theta0)
    )


# ---------------------------------------------------------------------------
# Ordinal encoding (Section 3.1, line 203)
# ---------------------------------------------------------------------------

def ordinal_to_angle(level: Tensor, num_levels: int) -> Tensor:
    """Map ordinal level k to angle theta_k = pi * k / (K+1)."""
    return torch.pi * level.float() / (num_levels + 1)


def angle_to_ordinal(theta: Tensor, num_levels: int) -> Tensor:
    """Inverse: find closest ordinal level by minimizing circular distance."""
    levels = torch.arange(num_levels, device=theta.device, dtype=theta.dtype)
    angles = torch.pi * levels / (num_levels + 1)  # (K,)
    dists = circular_distance(theta.unsqueeze(-1), angles.unsqueeze(0))
    return dists.argmin(dim=-1)
