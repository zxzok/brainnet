"""
brainnet.dynamic.kmeans
======================

This module implements dynamic functional connectivity analysis
using K‑means clustering.  The algorithm follows these steps:

1. Compute sliding window connectivity matrices from ROI time series.
2. Flatten the upper triangular elements of each matrix to form a
   feature vector for clustering.
3. Apply K‑means to identify a set of discrete connectivity
   patterns (states).
4. Reconstruct centroid matrices for each state and assign each
   window to its nearest centroid.
5. Derive temporal metrics such as occupancy, dwell time and
   transition probabilities of the resulting state sequence.

The function ``kmeans_analysis`` is intended to be called by the
high level :class:`brainnet.dynamic.analyzer.DynamicAnalyzer` and
returns a :class:`brainnet.dynamic.model.DynamicStateModel` object.

Note
----
This implementation relies on :mod:`sklearn.cluster` for the K‑means
algorithm.  ``scikit‑learn`` must therefore be installed for this
function to work.
"""

from __future__ import annotations

from typing import List, Tuple, Optional

import numpy as np
from sklearn.cluster import KMeans

from .config import DynamicConfig
from .model import DynamicStateModel
from .metrics import compute_state_metrics
from .window import sliding_window_connectivity


def _flatten_upper_triangle(matrices: List[np.ndarray]) -> Tuple[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
    """Flatten the upper triangular part (excluding diagonal) of a list of matrices.

    Parameters
    ----------
    matrices : List[np.ndarray]
        List of symmetric matrices of identical shape (N×N).

    Returns
    -------
    features : np.ndarray
        2D array of shape (n_matrices, n_features) containing the
        flattened upper triangular parts of each matrix.
    iu : Tuple[np.ndarray, np.ndarray]
        Indices of the upper triangular elements used for flattening.
        This can be used to reconstruct matrices from flattened
        centroids.
    """
    if not matrices:
        return np.empty((0, 0)), (np.array([], dtype=int), np.array([], dtype=int))
    N = matrices[0].shape[0]
    iu = np.triu_indices(N, k=1)
    features = np.array([mat[iu] for mat in matrices])
    return features, iu


def kmeans_analysis(
    roi_timeseries: np.ndarray, config: DynamicConfig, template: Optional[str] = None
) -> DynamicStateModel:
    """Perform dynamic connectivity analysis using K‑means clustering.

    Parameters
    ----------
    roi_timeseries : np.ndarray
        Array of shape (T, N_ROI) with ROI time series.
    config : DynamicConfig
        Configuration specifying window length, step size, number of
        states and optional random seed.

    Returns
    -------
    DynamicStateModel
        Model containing the identified state patterns, the state
        sequence for each window and summary metrics.

    Raises
    ------
    ValueError
        If no sliding windows can be generated from the input data.
    """
    # Generate sliding windows and their mean absolute connectivity metric
    windows, window_metric = sliding_window_connectivity(
        roi_timeseries, config.window_length, config.step
    )
    if not windows:
        raise ValueError("No windows could be generated; check window_length and data length")
    # Flatten the upper triangle for clustering
    features, iu = _flatten_upper_triangle(windows)
    # Run K‑means clustering
    # Use an explicit integer for ``n_init`` instead of ``'auto'``.  Some
    # versions of scikit‑learn do not accept the string 'auto' and will
    # raise a TypeError.  Setting ``n_init`` to 10 is a reasonable
    # default that performs multiple random initialisations.
    kmeans = KMeans(
        n_clusters=config.n_states,
        random_state=config.random_state,
        n_init=10,
    )
    labels = kmeans.fit_predict(features)
    # Reconstruct centroid matrices
    N = windows[0].shape[0]
    centroids: List[np.ndarray] = []
    for cent in kmeans.cluster_centers_:
        mat = np.zeros((N, N), dtype=float)
        mat[iu] = cent
        # symmetrise
        mat = mat + mat.T
        np.fill_diagonal(mat, 0.0)
        centroids.append(mat)
    # Compute temporal metrics from state labels
    metrics = compute_state_metrics(labels, config.n_states)
    extra = {
        'window_metric': window_metric,
    }
    return DynamicStateModel(
        method='kmeans',
        n_states=config.n_states,
        states=centroids,
        state_sequence=np.array(labels, dtype=int),
        metrics=metrics,
        extra=extra,
        template=template,
    )


__all__ = [
    'kmeans_analysis',
]
