"""Multi-stage training loop for Graph-Augmented Flow Matching. Algorithm 1.

Implements the full training pipeline:
  Phase 0 – Feature graph construction
  Phase 1 – Model initialization
  Phase 2 – Three-stage training with OT pairing and geodesic interpolation

Supports multi-GPU via ``nn.DataParallel`` with synchronized gradient
averaging and pinned-memory ``DataLoader`` (8 workers) for I/O overlap,
matching Section 5.1 of the manuscript.
"""

from __future__ import annotations

import logging
import math

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch import Tensor
from torch.optim import Adam
from torch.utils.data import DataLoader, TensorDataset

from graph_flow_matching.baselines.base import ColumnSpec
from graph_flow_matching.models.feature_gnn import build_feature_graph
from graph_flow_matching.models.manifold_ops import (
    clamp_probabilities,
    conditional_velocity_categorical,
    conditional_velocity_continuous,
    conditional_velocity_ordinal,
    geodesic_interpolation_categorical,
    geodesic_interpolation_continuous,
    geodesic_interpolation_ordinal,
    ordinal_to_angle,
    angle_to_ordinal,
)
from graph_flow_matching.models.ot_solver import compute_ot_coupling, sample_ot_pairs
from graph_flow_matching.models.velocity_field import GAFMConfig, GraphAugmentedFlowMatching

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Flat tensor packing / unpacking (shared with sampler)
# ---------------------------------------------------------------------------

def pack_flat(
    x_c: Tensor | None,
    x_d_list: list[Tensor],
    x_o_list: list[Tensor],
) -> Tensor:
    """Pack ``(x_c, x_d_list, x_o_list)`` into a single ``(B, D)`` tensor."""
    parts: list[Tensor] = []
    if x_c is not None:
        parts.append(x_c)
    parts.extend(x_d_list)
    for xo in x_o_list:
        parts.append(xo.unsqueeze(-1))
    return torch.cat(parts, dim=-1)


def unpack_flat(
    flat: Tensor,
    d_c: int,
    categorical_dims: list[int],
    n_ordinal: int,
) -> tuple[Tensor | None, list[Tensor], list[Tensor]]:
    """Inverse of :func:`pack_flat`."""
    offset = 0
    x_c: Tensor | None = None
    if d_c > 0:
        x_c = flat[:, offset : offset + d_c]
        offset += d_c

    x_d: list[Tensor] = []
    for n in categorical_dims:
        x_d.append(flat[:, offset : offset + n])
        offset += n

    x_o: list[Tensor] = []
    for _ in range(n_ordinal):
        x_o.append(flat[:, offset])
        offset += 1

    return x_c, x_d, x_o


# ---------------------------------------------------------------------------
# Data preprocessing
# ---------------------------------------------------------------------------

class DataPreprocessor:
    """Convert between DataFrame and product-manifold tensor representation.

    Continuous → standardized R^{d_c}.
    Categorical → one-hot on Δ^{n_j-1}.
    Ordinal → angles θ_k = πk/(K+1) on S^1.
    """

    def __init__(self, columns: list[ColumnSpec]) -> None:
        self.columns = columns
        self.cont_cols = [c for c in columns if c.dtype == "continuous"]
        self.cat_cols = [c for c in columns if c.dtype == "categorical"]
        self.ord_cols = [c for c in columns if c.dtype == "ordinal"]

        # Fitted parameters (set by fit_transform)
        self.cont_mean: Tensor | None = None
        self.cont_std: Tensor | None = None
        self.cat_encoders: dict[str, dict[str, int]] = {}
        self.ord_encoders: dict[str, dict[str, int]] = {}
        self.ord_num_levels: dict[str, int] = {}

    @property
    def d_c(self) -> int:
        return len(self.cont_cols)

    @property
    def categorical_dims(self) -> list[int]:
        return [len(c.categories) for c in self.cat_cols]

    @property
    def n_ordinal(self) -> int:
        return len(self.ord_cols)

    def fit_transform(
        self, df: pd.DataFrame, device: torch.device = torch.device("cpu"),
    ) -> tuple[Tensor | None, list[Tensor], list[Tensor]]:
        """Fit preprocessing parameters and transform the DataFrame."""
        # --- continuous ---
        x_c: Tensor | None = None
        if self.cont_cols:
            vals = df[[c.name for c in self.cont_cols]].values.astype(np.float32)
            self.cont_mean = torch.tensor(np.nanmean(vals, axis=0), device=device)
            self.cont_std = torch.tensor(np.nanstd(vals, axis=0) + 1e-8, device=device)
            x_c = (torch.tensor(vals, device=device) - self.cont_mean) / self.cont_std

        # --- categorical → one-hot ---
        x_d_list: list[Tensor] = []
        for col in self.cat_cols:
            encoder = {cat: i for i, cat in enumerate(col.categories)}
            self.cat_encoders[col.name] = encoder
            raw = df[col.name].astype(str).map(encoder)
            indices = raw.fillna(0).astype(int).values
            one_hot = torch.zeros(len(df), len(col.categories), device=device)
            one_hot.scatter_(1, torch.tensor(indices, device=device).unsqueeze(1), 1.0)
            x_d_list.append(one_hot)

        # --- ordinal → angle ---
        x_o_list: list[Tensor] = []
        for col in self.ord_cols:
            encoder = {cat: i for i, cat in enumerate(col.categories)}
            self.ord_encoders[col.name] = encoder
            self.ord_num_levels[col.name] = len(col.categories)
            raw = df[col.name].astype(str).map(encoder)
            levels = torch.tensor(raw.fillna(0).astype(int).values, device=device)
            x_o_list.append(ordinal_to_angle(levels, len(col.categories)))

        return x_c, x_d_list, x_o_list

    def inverse_transform(
        self,
        x_c: Tensor | None,
        x_d_list: list[Tensor],
        x_o_list: list[Tensor],
    ) -> pd.DataFrame:
        """Convert tensors back to a DataFrame."""
        data: dict[str, np.ndarray] = {}

        if x_c is not None and self.cont_cols:
            x_np = (x_c * self.cont_std + self.cont_mean).cpu().numpy()
            for i, col in enumerate(self.cont_cols):
                data[col.name] = x_np[:, i]

        for col, x_d in zip(self.cat_cols, x_d_list):
            indices = x_d.argmax(dim=-1).cpu().numpy()
            inv = {v: k for k, v in self.cat_encoders[col.name].items()}
            data[col.name] = np.array([inv.get(int(i), col.categories[0]) for i in indices])

        for col, x_o in zip(self.ord_cols, x_o_list):
            levels = angle_to_ordinal(x_o, self.ord_num_levels[col.name]).cpu().numpy()
            inv = {v: k for k, v in self.ord_encoders[col.name].items()}
            data[col.name] = np.array([inv.get(int(l), col.categories[0]) for l in levels])

        ordered = [c.name for c in self.columns if c.name in data]
        return pd.DataFrame({c: data[c] for c in ordered})


# ---------------------------------------------------------------------------
# Prior sampling on the product manifold
# ---------------------------------------------------------------------------

def sample_prior(
    batch_size: int,
    d_c: int,
    categorical_dims: list[int],
    n_ordinal: int,
    device: torch.device,
) -> tuple[Tensor | None, list[Tensor], list[Tensor]]:
    """Sample from π_0: N(0,I) × Dirichlet(1) × Uniform(0,2π)."""
    x0_c = torch.randn(batch_size, d_c, device=device) if d_c > 0 else None

    x0_d: list[Tensor] = []
    for n_cat in categorical_dims:
        exp = torch.distributions.Exponential(1.0).sample((batch_size, n_cat)).to(device)
        x0_d.append(exp / exp.sum(dim=-1, keepdim=True))

    x0_o = [torch.rand(batch_size, device=device) * 2.0 * torch.pi for _ in range(n_ordinal)]

    return x0_c, x0_d, x0_o


# ---------------------------------------------------------------------------
# Loss computation (Eqs 36-38)
# ---------------------------------------------------------------------------

def compute_loss(
    v_c: Tensor | None, v_d: list[Tensor], v_o: list[Tensor],
    u_c: Tensor | None, u_d: list[Tensor], u_o: list[Tensor],
    lambda_d: float = 1.0, lambda_o: float = 1.0,
    xt_d: list[Tensor] | None = None,
) -> Tensor:
    """L = L_cont + λ_d·L_cat + λ_o·L_ord (Eq 35).

    When *xt_d* is provided, categorical loss uses the Fisher-Rao metric
    (Eq 37): ||v−u||²_FR = Σ_k (v_k−u_k)² / (4·p_k) at interpolated
    point p = x_t^{d,j}.
    """
    parts: list[Tensor] = []

    if v_c is not None and u_c is not None:
        parts.append((v_c - u_c).pow(2).mean())

    for j, (vd, ud) in enumerate(zip(v_d, u_d)):
        if xt_d is not None:
            # Fisher-Rao weighted MSE (Eq 37)
            p = clamp_probabilities(xt_d[j])
            fr_weighted = (vd - ud).pow(2) / (4.0 * p)
            parts.append(lambda_d * fr_weighted.mean())
        else:
            parts.append(lambda_d * (vd - ud).pow(2).mean())

    for vo, uo in zip(v_o, u_o):
        parts.append(lambda_o * (vo - uo).pow(2).mean())

    if not parts:
        ref = v_d[0] if v_d else (v_c if v_c is not None else v_o[0])
        return torch.tensor(0.0, device=ref.device)
    return torch.stack(parts).sum()


# ---------------------------------------------------------------------------
# DataParallel-compatible training step
# ---------------------------------------------------------------------------

class _TrainStepModule(nn.Module):
    """Wraps GAFM for ``nn.DataParallel`` compatibility.

    Encapsulates geodesic interpolation, model forward pass, and loss
    computation inside a single ``forward`` call that operates only on
    flat ``(B, D)`` tensor inputs — safe for DataParallel scattering.

    OT pairing is performed *outside* this module on the primary GPU
    (full-batch operation) before the flat tensors are passed in.
    """

    def __init__(
        self,
        model: GraphAugmentedFlowMatching,
        d_c: int,
        categorical_dims: list[int],
        n_ordinal: int,
        lambda_d: float,
        lambda_o: float,
    ) -> None:
        super().__init__()
        self.model = model
        self.d_c = d_c
        self.cat_dims = list(categorical_dims)
        self.n_ord = n_ordinal
        self.lambda_d = lambda_d
        self.lambda_o = lambda_o
        # Set by the trainer before each stage's epoch loop
        self.stage: int = 3

    def forward(self, x0_flat: Tensor, x1_flat: Tensor, t: Tensor) -> Tensor:
        """Run one training step on a (possibly scattered) batch chunk.

        Args:
            x0_flat: ``(B, D)`` packed prior samples (already OT-paired).
            x1_flat: ``(B, D)`` packed data samples (already OT-paired).
            t: ``(B,)`` time values in [0, 1].

        Returns:
            ``(1,)`` scalar loss for this chunk (DataParallel gathers these).
        """
        x0_c, x0_d, x0_o = unpack_flat(x0_flat, self.d_c, self.cat_dims, self.n_ord)
        x1_c, x1_d, x1_o = unpack_flat(x1_flat, self.d_c, self.cat_dims, self.n_ord)
        t_col = t.unsqueeze(-1)

        # Geodesic interpolation → x_t (Eq 9)
        xt_c = (
            geodesic_interpolation_continuous(x0_c, x1_c, t_col)
            if x0_c is not None else None
        )
        xt_d = [geodesic_interpolation_categorical(p0, p1, t_col) for p0, p1 in zip(x0_d, x1_d)]
        xt_o = [geodesic_interpolation_ordinal(a0, a1, t) for a0, a1 in zip(x0_o, x1_o)]

        # Target velocities u_t (Eq 12)
        u_c = conditional_velocity_continuous(x0_c, x1_c) if x0_c is not None else None
        u_d = [conditional_velocity_categorical(p0, p1, t_col) for p0, p1 in zip(x0_d, x1_d)]
        u_o = [conditional_velocity_ordinal(a0, a1) for a0, a1 in zip(x0_o, x1_o)]

        # Model forward (Eq 16)
        v_c, v_d, v_o = self.model(xt_c, xt_d, xt_o, t, stage=self.stage)

        # Loss (Eq 35) — Fisher-Rao metric for categorical (Eq 37)
        # Returned as (1,) so DataParallel can gather
        loss = compute_loss(v_c, v_d, v_o, u_c, u_d, u_o,
                            self.lambda_d, self.lambda_o, xt_d=xt_d)
        return loss.unsqueeze(0)


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------

class GAFMTrainer:
    """Multi-stage training loop implementing Algorithm 1.

    Supports data-parallel training on multiple GPUs with synchronized
    gradient averaging and pinned-memory DataLoader for I/O overlap
    (Section 5.1).
    """

    def __init__(
        self,
        columns: list[ColumnSpec],
        config: GAFMConfig | None = None,
        device: torch.device | None = None,
    ) -> None:
        self.columns = columns
        self.config = config or GAFMConfig()
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.preprocessor = DataPreprocessor(columns)
        self.model: GraphAugmentedFlowMatching | None = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit(self, df: pd.DataFrame) -> GraphAugmentedFlowMatching:
        """Train the GAFM model on *df*. Returns the trained model."""
        cfg = self.config
        pp = self.preprocessor

        # 1. Preprocess to CPU (DataLoader will handle GPU transfer)
        x_c, x_d_list, x_o_list = pp.fit_transform(df, device=torch.device("cpu"))
        N = len(df)

        # 2. Feature graph (Phase 0)
        logger.info("Building feature graph (threshold=%.2f) ...", cfg.dependency_threshold)
        fg = build_feature_graph(df, self.columns, threshold=cfg.dependency_threshold)

        # 3. Initialize model on primary GPU (Phase 1)
        self.model = GraphAugmentedFlowMatching(
            d_c=pp.d_c,
            categorical_dims=pp.categorical_dims,
            n_ordinal=pp.n_ordinal,
            feature_graph=tuple(t.to(self.device) for t in fg),
            config=cfg,
        ).to(self.device)

        # 4. Train / val split
        val_size = max(1, int(N * cfg.val_fraction))
        perm = torch.randperm(N)
        val_idx, train_idx = perm[:val_size], perm[val_size:]
        N_train = len(train_idx)

        # Pack training data as flat CPU tensor for DataLoader
        train_flat = pack_flat(
            x_c[train_idx] if x_c is not None else None,
            [xd[train_idx] for xd in x_d_list],
            [xo[train_idx] for xo in x_o_list],
        )

        # Validation data → primary GPU
        val_data = (
            x_c[val_idx].to(self.device) if x_c is not None else None,
            [xd[val_idx].to(self.device) for xd in x_d_list],
            [xo[val_idx].to(self.device) for xo in x_o_list],
        )

        # 5. DataLoader with pinned memory and workers
        n_gpus = torch.cuda.device_count() if self.device.type == "cuda" else 0
        use_workers = cfg.num_workers if n_gpus > 0 else 0
        use_pin = cfg.pin_memory and n_gpus > 0
        eff_bs = min(cfg.batch_size, N_train)

        loader = DataLoader(
            TensorDataset(train_flat),
            batch_size=eff_bs,
            shuffle=True,
            num_workers=use_workers,
            pin_memory=use_pin,
            drop_last=(N_train > cfg.batch_size),
            persistent_workers=(use_workers > 0),
        )

        # 6. DataParallel training wrapper
        step_module = _TrainStepModule(
            self.model, pp.d_c, pp.categorical_dims, pp.n_ordinal,
            cfg.lambda_d, cfg.lambda_o,
        )
        if n_gpus > 1:
            step_module = nn.DataParallel(step_module)
            logger.info("DataParallel enabled on %d GPUs", n_gpus)

        # 7. Multi-stage training (Phase 2)
        stage_specs = [
            (1, cfg.stage1_fraction, 1.0),
            (2, cfg.stage2_fraction, cfg.stage2_lr_factor),
            (3, cfg.stage3_fraction, cfg.stage3_lr_factor),
        ]

        for stage, frac, lr_factor in stage_specs:
            n_epochs = max(1, int(cfg.num_epochs * frac))
            lr = cfg.lr * lr_factor
            optimizer = Adam(self.model.parameters(), lr=lr)

            # Set stage on the underlying module (DataParallel or raw)
            raw_step = step_module.module if isinstance(step_module, nn.DataParallel) else step_module
            raw_step.stage = stage

            logger.info("Stage %d: %d epochs, lr=%.2e", stage, n_epochs, lr)
            best_val, patience_ctr = float("inf"), 0

            for epoch in range(n_epochs):
                train_loss = self._train_epoch(loader, step_module, optimizer)
                val_loss = self._eval_loss(val_data, stage)

                if epoch % 10 == 0 or epoch == n_epochs - 1:
                    logger.info(
                        "  Stage %d  epoch %3d/%d  train=%.4f  val=%.4f",
                        stage, epoch + 1, n_epochs, train_loss, val_loss,
                    )

                if val_loss < best_val:
                    best_val = val_loss
                    patience_ctr = 0
                else:
                    patience_ctr += 1
                    if patience_ctr >= 20:
                        logger.info("  Early stopping at epoch %d", epoch + 1)
                        break

        self.model.eval()
        return self.model

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _train_epoch(
        self,
        loader: DataLoader,
        step_module: nn.Module,
        optimizer: Adam,
    ) -> float:
        cfg = self.config
        pp = self.preprocessor
        self.model.train()
        total_loss, n_batches = 0.0, 0

        for step, (x1_flat_cpu,) in enumerate(loader):
            if step >= cfg.max_steps_per_epoch:
                break

            x1_flat = x1_flat_cpu.to(self.device, non_blocking=True)
            B = x1_flat.shape[0]
            if B < 2:
                continue

            # Prior noise x_0 (on primary GPU)
            x0_c, x0_d, x0_o = sample_prior(
                B, pp.d_c, pp.categorical_dims, pp.n_ordinal, self.device,
            )

            # OT pairing — full-batch, primary GPU (Eqs 33-34)
            x1_c, x1_d, x1_o = unpack_flat(x1_flat, pp.d_c, pp.categorical_dims, pp.n_ordinal)
            coupling = compute_ot_coupling(
                x0_c, x1_c, x0_d, x1_d, x0_o, x1_o,
                epsilon=cfg.sinkhorn_epsilon, num_iterations=cfg.sinkhorn_iterations,
            )
            ot_idx = sample_ot_pairs(coupling)

            # Reorder x1 by OT, then repack both as flat tensors
            x1_c = x1_c[ot_idx] if x1_c is not None else None
            x1_d = [xd[ot_idx] for xd in x1_d]
            x1_o = [xo[ot_idx] for xo in x1_o]

            x0_flat = pack_flat(x0_c, x0_d, x0_o)
            x1_flat = pack_flat(x1_c, x1_d, x1_o)

            # Time t ~ U(0,1)
            t = torch.rand(B, device=self.device)

            # Forward pass — DataParallel scatters across GPUs
            per_gpu_losses = step_module(x0_flat, x1_flat, t)
            loss = per_gpu_losses.mean()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            n_batches += 1

        return total_loss / max(n_batches, 1)

    @torch.no_grad()
    def _eval_loss(
        self,
        val_data: tuple[Tensor | None, list[Tensor], list[Tensor]],
        stage: int,
    ) -> float:
        """Compute validation loss (single GPU, single batch)."""
        cfg = self.config
        pp = self.preprocessor
        self.model.eval()
        val_c, val_d, val_o = val_data

        N_val = val_d[0].shape[0] if val_d else (val_c.shape[0] if val_c is not None else val_o[0].shape[0])
        B = min(N_val, cfg.batch_size)

        x1_c = val_c[:B] if val_c is not None else None
        x1_d = [xd[:B] for xd in val_d]
        x1_o = [xo[:B] for xo in val_o]

        x0_c, x0_d, x0_o = sample_prior(B, pp.d_c, pp.categorical_dims, pp.n_ordinal, self.device)

        t = torch.rand(B, device=self.device)
        t_col = t.unsqueeze(-1)

        xt_c = geodesic_interpolation_continuous(x0_c, x1_c, t_col) if x0_c is not None else None
        xt_d = [geodesic_interpolation_categorical(p0, p1, t_col) for p0, p1 in zip(x0_d, x1_d)]
        xt_o = [geodesic_interpolation_ordinal(a0, a1, t) for a0, a1 in zip(x0_o, x1_o)]

        u_c = conditional_velocity_continuous(x0_c, x1_c) if x0_c is not None else None
        u_d = [conditional_velocity_categorical(p0, p1, t_col) for p0, p1 in zip(x0_d, x1_d)]
        u_o = [conditional_velocity_ordinal(a0, a1) for a0, a1 in zip(x0_o, x1_o)]

        v_c, v_d, v_o = self.model(xt_c, xt_d, xt_o, t, stage=stage)
        return compute_loss(v_c, v_d, v_o, u_c, u_d, u_o,
                            cfg.lambda_d, cfg.lambda_o, xt_d=xt_d).item()
