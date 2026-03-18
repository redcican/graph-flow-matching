"""ODE-based sampling from a trained GAFM model. Section 5.1.

Supports two solvers:
  - ``dopri5`` via torchdiffeq (adaptive, default when available)
  - Euler integration with manifold re-projection (fallback)
"""

from __future__ import annotations

import logging
import math

import pandas as pd
import torch
from torch import Tensor

from graph_flow_matching.models.manifold_ops import clamp_probabilities
from graph_flow_matching.models.velocity_field import GAFMConfig, GraphAugmentedFlowMatching
from graph_flow_matching.training.trainer import (
    DataPreprocessor,
    pack_flat,
    sample_prior,
    unpack_flat,
)

logger = logging.getLogger(__name__)


class GAFMSampler:
    """Generate synthetic samples by integrating the learned velocity field."""

    def __init__(
        self,
        model: GraphAugmentedFlowMatching,
        preprocessor: DataPreprocessor,
        config: GAFMConfig | None = None,
        device: torch.device | None = None,
    ) -> None:
        self.model = model
        self.preprocessor = preprocessor
        self.config = config or GAFMConfig()
        self.device = device or next(model.parameters()).device

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def sample(self, n: int, batch_size: int | None = None) -> pd.DataFrame:
        """Generate *n* synthetic rows as a DataFrame."""
        bs = batch_size or self.config.batch_size
        frames: list[pd.DataFrame] = []

        remaining = n
        while remaining > 0:
            B = min(bs, remaining)
            frames.append(self._sample_batch(B))
            remaining -= B

        return pd.concat(frames, ignore_index=True)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _sample_batch(self, B: int) -> pd.DataFrame:
        cfg = self.config
        pp = self.preprocessor

        # 1. Prior x_0
        x0_c, x0_d, x0_o = sample_prior(B, pp.d_c, pp.categorical_dims, pp.n_ordinal, self.device)

        # 2. Pack into flat state for ODE solver
        state = pack_flat(x0_c, x0_d, x0_o)

        # 3. Integrate from t=0 to t=1
        self.model.eval()
        with torch.no_grad():
            try:
                from torchdiffeq import odeint

                final_state = self._integrate_dopri5(odeint, state, B, cfg)
            except ImportError:
                logger.info("torchdiffeq not found; falling back to Euler integration")
                final_state = self._integrate_euler(state, B, cfg)

        # 4. Unpack and project to manifold
        x_c, x_d, x_o = unpack_flat(final_state, pp.d_c, pp.categorical_dims, pp.n_ordinal)
        x_d = [clamp_probabilities(p) for p in x_d]
        x_o = [theta % (2.0 * math.pi) for theta in x_o]

        # 5. Convert back to DataFrame
        return pp.inverse_transform(x_c, x_d, x_o)

    # ------------------------------------------------------------------
    # ODE solvers
    # ------------------------------------------------------------------

    def _integrate_dopri5(self, odeint, state: Tensor, B: int, cfg: GAFMConfig) -> Tensor:
        """Adaptive ODE solver (dopri5) via torchdiffeq."""
        pp = self.preprocessor

        def velocity_fn(t_scalar: Tensor, y: Tensor) -> Tensor:
            x_c, x_d, x_o = unpack_flat(y, pp.d_c, pp.categorical_dims, pp.n_ordinal)
            x_d = [clamp_probabilities(p) for p in x_d]
            x_o = [theta % (2.0 * math.pi) for theta in x_o]
            t_batch = t_scalar.expand(B)
            v_c, v_d, v_o = self.model(x_c, x_d, x_o, t_batch, stage=3)
            return pack_flat(v_c, v_d, v_o)

        t_span = torch.tensor([0.0, 1.0], device=self.device)
        result = odeint(
            velocity_fn, state, t_span,
            method="dopri5", atol=cfg.ode_atol, rtol=cfg.ode_rtol,
        )
        return result[-1]  # state at t=1

    def _integrate_euler(self, state: Tensor, B: int, cfg: GAFMConfig) -> Tensor:
        """Fixed-step Euler integration with manifold re-projection."""
        pp = self.preprocessor
        n_steps = cfg.n_ode_steps
        dt = 1.0 / n_steps
        y = state

        for i in range(n_steps):
            t_val = torch.tensor(i * dt, device=self.device)

            # Compute velocity
            x_c, x_d, x_o = unpack_flat(y, pp.d_c, pp.categorical_dims, pp.n_ordinal)
            x_d = [clamp_probabilities(p) for p in x_d]
            x_o = [theta % (2.0 * math.pi) for theta in x_o]
            t_batch = t_val.expand(B)
            v_c, v_d, v_o = self.model(x_c, x_d, x_o, t_batch, stage=3)

            # Euler step
            dy = pack_flat(v_c, v_d, v_o)
            y = y + dt * dy

            # Re-project to manifold
            x_c, x_d, x_o = unpack_flat(y, pp.d_c, pp.categorical_dims, pp.n_ordinal)
            x_d = [clamp_probabilities(p) for p in x_d]
            x_o = [theta % (2.0 * math.pi) for theta in x_o]
            y = pack_flat(x_c, x_d, x_o)

        return y
