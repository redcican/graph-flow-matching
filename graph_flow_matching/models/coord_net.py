"""Coordinate-wise velocity networks (v_coord). Section 4.1, Eq 17.

Each manifold type has its own MLP. Outputs lie in the appropriate tangent space.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor


def _build_mlp(
    in_dim: int,
    hidden_dim: int,
    out_dim: int,
    num_layers: int,
) -> nn.Sequential:
    layers: list[nn.Module] = [nn.Linear(in_dim, hidden_dim), nn.GELU()]
    for _ in range(num_layers - 2):
        layers += [nn.Linear(hidden_dim, hidden_dim), nn.GELU()]
    layers.append(nn.Linear(hidden_dim, out_dim))
    return nn.Sequential(*layers)


class ContinuousVelocityNet(nn.Module):
    """5-layer MLP for continuous features. Output in R^{d_c}."""

    def __init__(
        self, d_c: int, hidden_dim: int = 256, num_layers: int = 5, time_embed_dim: int = 64
    ) -> None:
        super().__init__()
        self.net = _build_mlp(d_c + time_embed_dim, hidden_dim, d_c, num_layers)

    def forward(self, x_c: Tensor, t_emb: Tensor) -> Tensor:
        return self.net(torch.cat([x_c, t_emb], dim=-1))


class CategoricalVelocityNet(nn.Module):
    """5-layer MLP for a single categorical feature.

    Output is projected to simplex tangent space (zero-sum constraint).
    """

    def __init__(
        self, n_categories: int, hidden_dim: int = 256, num_layers: int = 5, time_embed_dim: int = 64
    ) -> None:
        super().__init__()
        self.net = _build_mlp(n_categories + time_embed_dim, hidden_dim, n_categories, num_layers)

    def forward(self, x_d: Tensor, t_emb: Tensor) -> Tensor:
        v = self.net(torch.cat([x_d, t_emb], dim=-1))
        return v - v.mean(dim=-1, keepdim=True)  # tangent space projection


class OrdinalVelocityNet(nn.Module):
    """5-layer MLP for a single ordinal feature. Input: (cos, sin); output: angular velocity."""

    def __init__(
        self, hidden_dim: int = 256, num_layers: int = 5, time_embed_dim: int = 64
    ) -> None:
        super().__init__()
        self.net = _build_mlp(2 + time_embed_dim, hidden_dim, 1, num_layers)

    def forward(self, theta: Tensor, t_emb: Tensor) -> Tensor:
        x = torch.stack([torch.cos(theta), torch.sin(theta)], dim=-1)
        return self.net(torch.cat([x, t_emb], dim=-1)).squeeze(-1)


class CoordinateWiseVelocity(nn.Module):
    """Combined coordinate-wise velocity for all feature types. Eq 17."""

    def __init__(
        self,
        d_c: int,
        categorical_dims: list[int],
        n_ordinal: int,
        hidden_dim: int = 256,
        num_layers: int = 5,
        time_embed_dim: int = 64,
    ) -> None:
        super().__init__()
        self.d_c = d_c

        self.cont_net = ContinuousVelocityNet(d_c, hidden_dim, num_layers, time_embed_dim) if d_c > 0 else None
        self.cat_nets = nn.ModuleList(
            [CategoricalVelocityNet(n, hidden_dim, num_layers, time_embed_dim) for n in categorical_dims]
        )
        self.ord_nets = nn.ModuleList(
            [OrdinalVelocityNet(hidden_dim, num_layers, time_embed_dim) for _ in range(n_ordinal)]
        )

    def forward(
        self,
        x_c: Tensor | None,
        x_d_list: list[Tensor],
        x_o_list: list[Tensor],
        t_emb: Tensor,
    ) -> tuple[Tensor | None, list[Tensor], list[Tensor]]:
        v_c = self.cont_net(x_c, t_emb) if self.cont_net is not None and x_c is not None else None
        v_d = [net(x_d, t_emb) for net, x_d in zip(self.cat_nets, x_d_list)]
        v_o = [net(x_o, t_emb) for net, x_o in zip(self.ord_nets, x_o_list)]
        return v_c, v_d, v_o
