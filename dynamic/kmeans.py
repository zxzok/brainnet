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

The module also provides :func:`suggest_k` which evaluates silhouette
scores across candidate cluster counts to recommend an appropriate
number of states. When ``DynamicConfig.auto_n_states`` is set, this
recommendation overrides the configured ``n_states`` before running
K‑means.

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

from typing import List, Tuple, Optional, Iterable

import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

from .config import DynamicConfig
from .model import DynamicStateModel
from .metrics import compute_state_metrics
from .state_features import compute_state_features
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


# ---------------------------------------------------------------------------
def suggest_k(
    features: np.ndarray,
    k_values: Iterable[int] = range(2, 11),
    random_state: Optional[int] = None,
) -> int:
    """Recommend an optimal number of clusters using silhouette scores.

    Parameters
    ----------
    features : np.ndarray
        2D array of shape (n_samples, n_features) with flattened window
        connectivity features.
    k_values : Iterable[int], optional
        Candidate numbers of clusters to evaluate. Defaults to ``range(2, 11)``.
    random_state : int | None, optional
        Random seed for K‑means initialisation.

    Returns
    -------
    int
        The candidate K with the highest silhouette score.

    Raises
    ------
    ValueError
        If fewer than two samples are provided or no candidate K is valid.
    """
    n_samples = features.shape[0]
    if n_samples < 2:
        raise ValueError("At least two samples are required to compute silhouette scores")
    best_k: Optional[int] = None
    best_score = -1.0
    for k in k_values:
        if k >= n_samples:
            break
        labels = KMeans(n_clusters=k, n_init=10, random_state=random_state).fit_predict(features)
        score = silhouette_score(features, labels)
        if score > best_score:
            best_k, best_score = k, score
    if best_k is None:
        raise ValueError("Unable to determine a suitable number of states")
    return best_k


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
        states and optional random seed. If ``config.auto_n_states`` is
        True the optimal number of clusters is estimated via
        :func:`suggest_k` and ``n_states`` is overridden.

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
    if config.auto_n_states:
        config.n_states = suggest_k(features, random_state=config.random_state)
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
    # Derive graph features from centroid connectivity matrices
    state_features = compute_state_features(centroids)
    extra = {
        'window_metric': window_metric,
        'state_features': state_features,
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
    'suggest_k',
]
