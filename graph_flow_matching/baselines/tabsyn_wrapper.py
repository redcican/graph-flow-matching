"""TabSyn baseline wrapper.

Upstream: https://github.com/coksvictoria/TTVAE/tree/main/baselines/tabsyn
(originally from https://github.com/amazon-science/tabsyn)
Setup:   git clone https://github.com/coksvictoria/TTVAE third_party/TTVAE
"""

from __future__ import annotations

import json
import logging
import sys
import tempfile
from argparse import Namespace
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

from graph_flow_matching.baselines.base import BaseGenerator, ColumnSpec
from graph_flow_matching.baselines.registry import register

logger = logging.getLogger(__name__)

_REPO_DIR = Path(__file__).resolve().parents[2] / "third_party" / "TTVAE"
_BASELINE_DIR = _REPO_DIR / "baselines" / "tabsyn"


def _ensure_on_path() -> None:
    for d in [str(_REPO_DIR), str(_BASELINE_DIR)]:
        if d not in sys.path:
            sys.path.insert(0, d)


def _prepare_data_dir(
    df: pd.DataFrame,
    columns: list[ColumnSpec],
    work_dir: Path,
) -> tuple[list[str], list[str], dict[str, LabelEncoder]]:
    """Write DataFrame to the numpy-file format expected by TabSyn."""
    num_cols = [s.name for s in columns if s.dtype == "continuous"]
    cat_cols = [s.name for s in columns if s.dtype in ("categorical", "ordinal")]
    label_encoders: dict[str, LabelEncoder] = {}

    if num_cols:
        X_num = df[num_cols].values.astype(np.float32)
        np.save(work_dir / "X_num_train.npy", X_num)

    if cat_cols:
        cat_arrays = []
        for col in cat_cols:
            le = LabelEncoder()
            le.fit(df[col].astype(str))
            label_encoders[col] = le
            cat_arrays.append(le.transform(df[col].astype(str)))
        X_cat = np.column_stack(cat_arrays).astype(np.int64)
        np.save(work_dir / "X_cat_train.npy", X_cat)

    y = np.zeros(len(df), dtype=np.int64)
    np.save(work_dir / "y_train.npy", y)

    num_classes = [
        int(df[s.name].nunique()) if s.dtype in ("categorical", "ordinal") else None
        for s in columns
    ]
    info = {
        "task_type": "binclass",
        "n_num_features": len(num_cols),
        "n_cat_features": len(cat_cols),
        "num_classes": num_classes,
        "train_size": len(df),
    }
    (work_dir / "info.json").write_text(json.dumps(info, indent=2))
    return num_cols, cat_cols, label_encoders


@register("tabsyn")
class TabSynGenerator(BaseGenerator):
    """Wrapper around TabSyn from the TTVAE baselines.

    TabSyn trains a VAE to learn a latent space, then applies
    score-based diffusion in that latent space for generation.
    Two-stage training: (1) VAE, (2) latent diffusion.
    """

    def __init__(
        self,
        vae_epochs: int = 100,
        diffusion_epochs: int = 100,
        batch_size: int = 256,
        lr: float = 1e-3,
        device: str = "cuda:0",
        **kwargs: Any,
    ) -> None:
        self._hparams = dict(
            vae_epochs=vae_epochs,
            diffusion_epochs=diffusion_epochs,
            batch_size=batch_size,
            lr=lr,
            device=device,
            **kwargs,
        )
        self._work_dir: Path | None = None
        self._columns: list[ColumnSpec] = []
        self._num_cols: list[str] = []
        self._cat_cols: list[str] = []
        self._label_encoders: dict[str, LabelEncoder] = {}
        self._tmpdir: tempfile.TemporaryDirectory | None = None

    def _make_args(self, **overrides: Any) -> Namespace:
        """Build an argparse.Namespace mimicking the upstream CLI."""
        args = Namespace(
            dataname="wrapped",
            datapath=str(self._work_dir),
            gpu=0,
            vae_epochs=self._hparams["vae_epochs"],
            tabsyn_epochs=self._hparams["diffusion_epochs"],
            batch_size=self._hparams["batch_size"],
            lr=self._hparams["lr"],
            save_path=str(self._work_dir / "synthetic.csv"),
        )
        for k, v in overrides.items():
            setattr(args, k, v)
        return args

    def fit(
        self,
        df: pd.DataFrame,
        columns: list[ColumnSpec],
        **kwargs: Any,
    ) -> None:
        _ensure_on_path()
        self._columns = columns

        self._tmpdir = tempfile.TemporaryDirectory(prefix="tabsyn_")
        self._work_dir = Path(self._tmpdir.name)

        self._num_cols, self._cat_cols, self._label_encoders = _prepare_data_dir(
            df, columns, self._work_dir
        )
        logger.info(
            "Fitting TabSyn stage 1: VAE (epochs=%d)",
            self._hparams["vae_epochs"],
        )

        try:
            from baselines.tabsyn.vae.main import main as train_vae
            from baselines.tabsyn.main import main as train_diffusion

            args = self._make_args()
            train_vae(args)

            logger.info(
                "Fitting TabSyn stage 2: diffusion (epochs=%d)",
                self._hparams["diffusion_epochs"],
            )
            train_diffusion(args)
        except ImportError:
            logger.warning(
                "TabSyn baseline not found at %s. "
                "Clone TTVAE repo first:\n"
                "  git clone https://github.com/coksvictoria/TTVAE %s",
                _BASELINE_DIR, _REPO_DIR,
            )
            raise

    def sample(self, n: int) -> pd.DataFrame:
        if self._work_dir is None:
            raise RuntimeError("Call fit() before sample()")

        from baselines.tabsyn.sample import main as sample_tabsyn

        args = self._make_args(num_samples=n)
        sample_tabsyn(args)

        sample_path = self._work_dir / "synthetic.csv"
        if sample_path.exists():
            return pd.read_csv(sample_path)

        # Fallback: reconstruct from numpy outputs
        parts: dict[str, np.ndarray] = {}
        X_num_path = self._work_dir / "X_num_syn.npy"
        X_cat_path = self._work_dir / "X_cat_syn.npy"

        if X_num_path.exists():
            X_num = np.load(X_num_path)
            for i, col in enumerate(self._num_cols):
                parts[col] = X_num[:, i]

        if X_cat_path.exists():
            X_cat = np.load(X_cat_path)
            for i, col in enumerate(self._cat_cols):
                le = self._label_encoders[col]
                encoded = np.clip(X_cat[:, i].astype(int), 0, len(le.classes_) - 1)
                parts[col] = le.inverse_transform(encoded)

        col_order = [s.name for s in self._columns]
        return pd.DataFrame(parts)[[c for c in col_order if c in parts]]

    @property
    def name(self) -> str:
        return "TabSyn"
