"""
brainnet.dynamic.state_features
===============================

Utilities for computing graph-theoretic metrics from state connectivity matrices.

This module currently supports:

* ``global_efficiency`` – mean of the inverse shortest path lengths between all node pairs.
* ``modularity`` – Newman-Girvan modularity of the partition obtained via greedy optimisation.

These features rely on :mod:`networkx`. Install it via ``pip install networkx``
if graph metrics are required.
"""

from __future__ import annotations

from typing import Dict, List, Sequence, Iterable

import numpy as np

try:  # pragma: no cover - optional dependency check
    import networkx as nx
    from networkx.algorithms import community
except Exception:  # pragma: no cover
    nx = None  # type: ignore
    community = None  # type: ignore

SUPPORTED_METRICS = ("global_efficiency", "modularity")


def compute_state_features(
    matrices: Sequence[np.ndarray], metrics: Iterable[str] | None = None
) -> List[Dict[str, float]]:
    """Compute graph metrics for each connectivity matrix.

    Parameters
    ----------
    matrices : Sequence[np.ndarray]
        Iterable of square connectivity matrices (shape ``N×N``).
    metrics : Iterable[str], optional
        Names of metrics to compute.  Supported metrics are
        ``'global_efficiency'`` and ``'modularity'``.  If ``None``, all
        supported metrics are calculated.

    Returns
    -------
    List[Dict[str, float]]
        For each input matrix, a dictionary mapping metric names to
        their computed values.

    Raises
    ------
    NotImplementedError
        If :mod:`networkx` is not available.
    """
    if nx is None:  # pragma: no cover - simple dependency gate
        raise NotImplementedError(
            "networkx is required to compute state features"
        )
    if metrics is None:
        metrics = SUPPORTED_METRICS
    results: List[Dict[str, float]] = []
    for mat in matrices:
        G = nx.from_numpy_array(mat)
        feat: Dict[str, float] = {}
        if "global_efficiency" in metrics:
            feat["global_efficiency"] = nx.global_efficiency(G)
        if "modularity" in metrics:
            comms = community.greedy_modularity_communities(G, weight="weight")
            feat["modularity"] = community.modularity(G, comms, weight="weight")
        results.append(feat)
    return results


__all__ = ["compute_state_features", "SUPPORTED_METRICS"]
