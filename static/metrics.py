"""
brainnet.static.metrics
======================

This module defines data structures and functions for computing
graph‑theoretic metrics on functional connectivity matrices.  The
metrics implemented include node degree (weighted strength), weighted
clustering coefficient and global efficiency.  These utilities can be
combined independently or used via the high level
:class:`~brainnet.static.analyzer.StaticAnalyzer` class.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Sequence

import numpy as np


@dataclass
class GraphMetrics:
    """Store node‑wise and global graph metrics.

    Attributes
    ----------
    node_metrics : Dict[str, np.ndarray]
        Dictionary mapping metric names to arrays of length N_ROI for
        node‑wise metrics (e.g. ``degree`` or ``clustering``).
    global_metrics : Dict[str, float]
        Dictionary of global graph metrics (e.g. ``global_efficiency``).
    """

    node_metrics: Dict[str, np.ndarray] = field(default_factory=dict)
    global_metrics: Dict[str, float] = field(default_factory=dict)


def compute_degree(adj_mat: np.ndarray) -> np.ndarray:
    """Compute weighted degree (strength) of each node.

    Parameters
    ----------
    adj_mat : np.ndarray
        Weighted adjacency matrix (absolute values considered).

    Returns
    -------
    np.ndarray
        Array of length N giving the sum of absolute weights on each node.
    """
    abs_mat = np.abs(adj_mat)
    return np.sum(abs_mat, axis=1)


def compute_clustering(adj_mat: np.ndarray) -> np.ndarray:
    """Compute the weighted clustering coefficient for each node.

    The implementation follows Onnela et al. (2005) for weighted
    undirected graphs.  We normalise edge weights to the range
    [0, 1] prior to computing the cubic root term.

    Parameters
    ----------
    adj_mat : np.ndarray
        Weighted adjacency matrix (absolute values considered).

    Returns
    -------
    np.ndarray
        Array of length N with clustering coefficient per node.
    """
    abs_mat = np.abs(adj_mat)
    N = abs_mat.shape[0]
    max_val = np.max(abs_mat)
    if max_val > 0:
        W = abs_mat / max_val
    else:
        W = abs_mat.copy()
    clustering = np.zeros(N, dtype=float)
    for i in range(N):
        neighbors = np.where(W[i] > 0)[0]
        k_i = len(neighbors)
        if k_i < 2:
            clustering[i] = 0.0
            continue
        tri_sum = 0.0
        for idx_j in range(k_i):
            for idx_k in range(idx_j + 1, k_i):
                j = neighbors[idx_j]
                k = neighbors[idx_k]
                w_ij = W[i, j]
                w_ik = W[i, k]
                w_jk = W[j, k]
                if w_jk > 0:
                    tri_sum += (w_ij * w_ik * w_jk) ** (1.0 / 3.0)
        clustering[i] = (2.0 * tri_sum) / (k_i * (k_i - 1))
    return clustering


def compute_global_efficiency(adj_mat: np.ndarray) -> float:
    """Compute global efficiency of a weighted graph.

    Efficiency is defined as the average of the inverse shortest
    path lengths between all pairs of nodes.  Distances are
    computed as ``d_ij = 1 / w_ij`` for ``w_ij > 0``, and ``∞``
    for disconnected pairs.  Self distances are ignored.

    Parameters
    ----------
    adj_mat : np.ndarray
        Weighted adjacency matrix (absolute values considered).

    Returns
    -------
    float
        The global efficiency of the graph.
    """
    abs_mat = np.abs(adj_mat)
    N = abs_mat.shape[0]
    # build distance matrix: inverse of weights (inf where no edge)
    with np.errstate(divide='ignore'):
        dist = 1.0 / abs_mat
    dist[abs_mat == 0] = np.inf
    np.fill_diagonal(dist, 0.0)
    # Floyd‑Warshall algorithm for all pairs shortest paths
    D = dist.copy()
    for k in range(N):
        D = np.minimum(D, D[:, k][:, None] + D[k, :][None, :])
    # compute inverse distances excluding diagonal and inf
    mask = ~np.isinf(D) & (np.arange(N)[:, None] != np.arange(N)[None, :])
    if np.sum(mask) == 0:
        return 0.0
    inv_dist = 1.0 / D[mask]
    return float(np.mean(inv_dist))