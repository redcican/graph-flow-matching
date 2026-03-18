"""Sinusoidal time embedding for flow matching (Section 5.1: dimension 64)."""

from __future__ import annotations

import math

import torch
import torch.nn as nn
from torch import Tensor


class SinusoidalTimeEmbedding(nn.Module):
    """Transformer-style sinusoidal positional encoding for scalar time t."""

    def __init__(self, embed_dim: int = 64) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        half = embed_dim // 2
        freqs = torch.exp(-math.log(10000.0) * torch.arange(half, dtype=torch.float32) / half)
        self.register_buffer("freqs", freqs)

    def forward(self, t: Tensor) -> Tensor:
        """Map scalar times to sinusoidal embeddings.

        Args:
            t: (B,) time values in [0, 1].
        Returns:
            (B, embed_dim) embeddings.
        """
        if t.dim() == 0:
            t = t.unsqueeze(0)
        args = t.unsqueeze(-1) * self.freqs.unsqueeze(0)  # (B, half)
        return torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
