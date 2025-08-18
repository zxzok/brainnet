"""
brainnet.dynamic.hierarchical
=============================

This module implements dynamic functional connectivity analysis using
agglomerative hierarchical clustering.  The procedure mirrors the
K‑means based approach but replaces the clustering step with SciPy's
:mod:`scipy.cluster.hierarchy` utilities.  Sliding window correlation
matrices are generated, flattened into feature vectors and grouped into
a specified number of clusters.  The mean connectivity pattern of each
cluster is returned as the representative state together with the state
sequence and associated temporal metrics.

The optional dependency :mod:`scipy` is required for this analysis.  If
SciPy is unavailable, calling :func:`hierarchical_analysis` will raise a
``NotImplementedError``.
"""

from __future__ import annotations

from typing import List, Tuple, Optional

import numpy as np

from .config import DynamicConfig
from .model import DynamicStateModel
from .metrics import compute_state_metrics
from .window import sliding_window_connectivity

try:  # pragma: no cover - optional dependency
    from scipy.cluster.hierarchy import linkage, fcluster
except Exception:  # pragma: no cover - optional dependency
    linkage = fcluster = None  # type: ignore


def _flatten_upper_triangle(matrices: List[np.ndarray]) -> Tuple[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
    """Flatten the upper triangular part (excluding diagonal) of matrices.

    Parameters
    ----------
    matrices : list of np.ndarray
        List of symmetric matrices of identical shape.

    Returns
    -------
    features : np.ndarray
        2D array where each row corresponds to the flattened upper
        triangle of a matrix.
    iu : tuple of np.ndarray
        Indices of the upper triangular elements used for flattening.
    """
    if not matrices:
        return np.empty((0, 0)), (np.array([], dtype=int), np.array([], dtype=int))
    N = matrices[0].shape[0]
    iu = np.triu_indices(N, k=1)
    features = np.array([mat[iu] for mat in matrices])
    return features, iu


def hierarchical_analysis(
    roi_timeseries: np.ndarray, config: DynamicConfig, template: Optional[str] = None
) -> DynamicStateModel:
    """Perform dynamic connectivity analysis using hierarchical clustering.

    Parameters
    ----------
    roi_timeseries : np.ndarray
        Array of shape (T, N_ROI) with ROI time series.
    config : DynamicConfig
        Configuration specifying window parameters and number of states.

    Returns
    -------
    DynamicStateModel
        Model containing the mean connectivity pattern for each
        hierarchical cluster, the state sequence for each window and
        summary metrics.

    Raises
    ------
    NotImplementedError
        If SciPy is not installed.
    ValueError
        If no sliding windows can be generated from the input data.
    """
    # Generate sliding windows
    windows, window_metric = sliding_window_connectivity(
        roi_timeseries, config.window_length, config.step
    )
    if not windows:
        raise ValueError(
            "No windows could be generated; check window_length and data length"
        )

    if linkage is None or fcluster is None:
        raise NotImplementedError(
            "scipy is not installed; cannot perform hierarchical clustering analysis."
        )

    # Flatten windows for clustering
    features, _ = _flatten_upper_triangle(windows)
    # Perform hierarchical clustering and obtain labels
    Z = linkage(features, method="ward")
    labels = fcluster(Z, t=config.n_states, criterion="maxclust") - 1

    # Compute mean connectivity pattern for each state
    N = windows[0].shape[0]
    state_mats: List[np.ndarray] = []
    for s in range(config.n_states):
        if np.any(labels == s):
            state_mats.append(np.mean(np.array(windows)[labels == s], axis=0))
        else:
            state_mats.append(np.zeros((N, N), dtype=float))

    # Temporal metrics from state sequence
    metrics = compute_state_metrics(labels, config.n_states)
    extra = {
        "window_metric": window_metric,
    }

    return DynamicStateModel(
        method="hierarchical",
        n_states=config.n_states,
        states=state_mats,
        state_sequence=labels.astype(int),
        metrics=metrics,
        extra=extra,
        template=template,
    )


__all__ = ["hierarchical_analysis"]
