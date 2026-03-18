"""Evaluation metrics: statistical fidelity, utility, and privacy."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats
from scipy.spatial.distance import jensenshannon

from graph_flow_matching.baselines.base import ColumnSpec


# ---------------------------------------------------------------------------
# Statistical fidelity
# ---------------------------------------------------------------------------

def wasserstein1_per_column(
    real: pd.DataFrame,
    synthetic: pd.DataFrame,
    columns: list[ColumnSpec],
) -> dict[str, float]:
    """Wasserstein-1 distance for continuous columns."""
    results: dict[str, float] = {}
    for spec in columns:
        if spec.dtype == "continuous":
            r = real[spec.name].dropna().values
            s = synthetic[spec.name].dropna().values
            results[spec.name] = float(stats.wasserstein_distance(r, s))
    return results


def jsd_per_column(
    real: pd.DataFrame,
    synthetic: pd.DataFrame,
    columns: list[ColumnSpec],
) -> dict[str, float]:
    """Jensen-Shannon divergence for categorical/ordinal columns."""
    results: dict[str, float] = {}
    for spec in columns:
        if spec.dtype in ("categorical", "ordinal") and spec.categories is not None:
            r_counts = real[spec.name].value_counts(normalize=True)
            s_counts = synthetic[spec.name].value_counts(normalize=True)
            p = np.array([r_counts.get(c, 0.0) for c in spec.categories])
            q = np.array([s_counts.get(c, 0.0) for c in spec.categories])
            # Avoid zero divisions
            p = p + 1e-12
            q = q + 1e-12
            p /= p.sum()
            q /= q.sum()
            results[spec.name] = float(jensenshannon(p, q) ** 2)
    return results


def correlation_error(
    real: pd.DataFrame,
    synthetic: pd.DataFrame,
    columns: list[ColumnSpec],
) -> float:
    """Frobenius norm of the difference between correlation matrices
    (computed over continuous columns only)."""
    cont_cols = [s.name for s in columns if s.dtype == "continuous"]
    if len(cont_cols) < 2:
        return 0.0
    corr_real = real[cont_cols].corr().values
    corr_syn = synthetic[cont_cols].corr().values
    return float(np.linalg.norm(corr_real - corr_syn, "fro"))


# ---------------------------------------------------------------------------
# Downstream utility (MLE: machine learning efficiency)
# ---------------------------------------------------------------------------

def mle_score(
    real_train: pd.DataFrame,
    real_test: pd.DataFrame,
    synthetic: pd.DataFrame,
    target: str,
    columns: list[ColumnSpec],
) -> dict[str, float]:
    """Train on synthetic, test on real (TSTR) vs train on real, test on real.

    Returns dict with 'tstr_accuracy' and 'trtr_accuracy'.
    Requires scikit-learn.
    """
    from sklearn.ensemble import GradientBoostingClassifier
    from sklearn.metrics import accuracy_score
    from sklearn.preprocessing import LabelEncoder

    feature_cols = [s.name for s in columns if s.name != target]
    le = LabelEncoder()
    le.fit(pd.concat([real_train[target], real_test[target], synthetic[target]]))

    def _prepare(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        X = pd.get_dummies(df[feature_cols], drop_first=True).values.astype(float)
        y = le.transform(df[target])
        return X, y

    X_real, y_real = _prepare(real_train)
    X_test, y_test = _prepare(real_test)
    X_syn, y_syn = _prepare(synthetic)

    clf_real = GradientBoostingClassifier(n_estimators=100, random_state=42)
    clf_real.fit(X_real, y_real)
    trtr = accuracy_score(y_test, clf_real.predict(X_test))

    clf_syn = GradientBoostingClassifier(n_estimators=100, random_state=42)
    clf_syn.fit(X_syn, y_syn)
    tstr = accuracy_score(y_test, clf_syn.predict(X_test))

    return {"trtr_accuracy": float(trtr), "tstr_accuracy": float(tstr)}


# ---------------------------------------------------------------------------
# Privacy: distance to closest record
# ---------------------------------------------------------------------------

def dcr_scores(
    real: pd.DataFrame,
    synthetic: pd.DataFrame,
    columns: list[ColumnSpec],
) -> np.ndarray:
    """Minimum L2 distance from each synthetic row to the training set.

    Operates on one-hot encoded + standardized feature space.
    """
    feature_cols = [s.name for s in columns]
    real_enc = pd.get_dummies(real[feature_cols]).values.astype(float)
    syn_enc = pd.get_dummies(synthetic[feature_cols]).values.astype(float)

    # Align columns
    all_cols = sorted(
        set(pd.get_dummies(real[feature_cols]).columns)
        | set(pd.get_dummies(synthetic[feature_cols]).columns)
    )
    real_enc = pd.get_dummies(real[feature_cols]).reindex(columns=all_cols, fill_value=0).values.astype(float)
    syn_enc = pd.get_dummies(synthetic[feature_cols]).reindex(columns=all_cols, fill_value=0).values.astype(float)

    # Standardize
    mu = real_enc.mean(axis=0)
    sigma = real_enc.std(axis=0) + 1e-8
    real_enc = (real_enc - mu) / sigma
    syn_enc = (syn_enc - mu) / sigma

    # Compute pairwise distances in chunks to avoid OOM
    chunk_size = 1000
    dcr = np.full(syn_enc.shape[0], np.inf)
    for start in range(0, syn_enc.shape[0], chunk_size):
        end = min(start + chunk_size, syn_enc.shape[0])
        dists = np.linalg.norm(
            syn_enc[start:end, None, :] - real_enc[None, :, :], axis=2
        )
        dcr[start:end] = dists.min(axis=1)

    return dcr
