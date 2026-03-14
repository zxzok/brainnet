"""
brainnet.dynamic.state_features
===============================

Utilities for computing graph-theoretic metrics from state connectivity matrices.

This module supports:

* ``global_efficiency`` – mean of the inverse shortest path lengths between all node pairs.
* ``modularity`` – Newman-Girvan modularity of the partition obtained via greedy optimisation.
* ``mean_connectivity`` – average absolute edge weight.
* ``network_density`` – fraction of edges with non-negligible weight.
* ``clustering_coefficient`` – weighted average clustering coefficient.
* ``assortativity`` – degree assortativity coefficient.
* ``characteristic_path_length`` – average shortest path length (on connected components).
* ``small_worldness`` – ratio of clustering coefficient to path length vs random graph.
* ``betweenness_centrality_mean`` – mean betweenness centrality across nodes.
* ``participation_coefficient`` – mean participation coefficient across nodes.

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

SUPPORTED_METRICS = (
    "global_efficiency",
    "modularity",
    "mean_connectivity",
    "network_density",
    "clustering_coefficient",
    "assortativity",
    "characteristic_path_length",
    "small_worldness",
    "betweenness_centrality_mean",
    "participation_coefficient",
)


def _participation_coefficient(G: "nx.Graph", partition: list) -> float:
    """Compute mean participation coefficient given a community partition."""
    # Build node-to-community mapping
    node_comm = {}
    for idx, comm_set in enumerate(partition):
        for node in comm_set:
            node_comm[node] = idx

    pc_values = []
    for node in G.nodes():
        ki = G.degree(node, weight="weight")
        if ki == 0:
            pc_values.append(0.0)
            continue
        # Strength to each community
        comm_strengths = {}
        for neighbor in G.neighbors(node):
            c = node_comm.get(neighbor, -1)
            w = G[node][neighbor].get("weight", 1.0)
            comm_strengths[c] = comm_strengths.get(c, 0.0) + abs(w)
        pc = 1.0 - sum((s / ki) ** 2 for s in comm_strengths.values())
        pc_values.append(pc)

    return float(np.mean(pc_values)) if pc_values else 0.0


def compute_state_features(
    matrices: Sequence[np.ndarray], metrics: Iterable[str] | None = None
) -> List[Dict[str, float]]:
    """Compute graph metrics for each connectivity matrix.

    Parameters
    ----------
    matrices : Sequence[np.ndarray]
        Iterable of square connectivity matrices (shape ``N×N``).
    metrics : Iterable[str], optional
        Names of metrics to compute.  If ``None``, all supported metrics
        are calculated.

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
    metrics_set = set(metrics)

    results: List[Dict[str, float]] = []
    for mat in matrices:
        # Use absolute values for graph construction
        abs_mat = np.abs(mat)
        np.fill_diagonal(abs_mat, 0.0)
        G = nx.from_numpy_array(abs_mat)
        feat: Dict[str, float] = {}

        if "global_efficiency" in metrics_set:
            feat["global_efficiency"] = nx.global_efficiency(G)

        if "modularity" in metrics_set:
            comms = community.greedy_modularity_communities(G, weight="weight")
            feat["modularity"] = community.modularity(G, comms, weight="weight")
        else:
            comms = None

        if "mean_connectivity" in metrics_set:
            mask = ~np.eye(mat.shape[0], dtype=bool)
            feat["mean_connectivity"] = float(np.mean(np.abs(mat[mask])))

        if "network_density" in metrics_set:
            n = mat.shape[0]
            max_edges = n * (n - 1) / 2
            if max_edges > 0:
                mask = ~np.eye(n, dtype=bool)
                n_edges = np.sum(np.abs(mat[mask]) > 0.05) / 2  # threshold at 0.05
                feat["network_density"] = float(n_edges / max_edges)
            else:
                feat["network_density"] = 0.0

        if "clustering_coefficient" in metrics_set:
            feat["clustering_coefficient"] = nx.average_clustering(G, weight="weight")

        if "assortativity" in metrics_set:
            try:
                feat["assortativity"] = nx.degree_assortativity_coefficient(G, weight="weight")
            except (nx.NetworkXError, ValueError):
                feat["assortativity"] = 0.0

        if "characteristic_path_length" in metrics_set:
            # Use largest connected component
            if nx.is_connected(G):
                feat["characteristic_path_length"] = nx.average_shortest_path_length(G, weight=None)
            else:
                largest_cc = max(nx.connected_components(G), key=len)
                subG = G.subgraph(largest_cc).copy()
                if len(subG) > 1:
                    feat["characteristic_path_length"] = nx.average_shortest_path_length(subG, weight=None)
                else:
                    feat["characteristic_path_length"] = 0.0

        if "small_worldness" in metrics_set:
            C = nx.average_clustering(G, weight="weight")
            if nx.is_connected(G) and len(G) > 3:
                L = nx.average_shortest_path_length(G, weight=None)
                # Compare against random graph expectations
                n = len(G)
                m = G.number_of_edges()
                k_mean = 2 * m / n if n > 0 else 0
                if k_mean > 0 and n > 0:
                    C_rand = k_mean / n
                    L_rand = np.log(n) / np.log(k_mean) if k_mean > 1 else n
                    gamma = C / C_rand if C_rand > 0 else 0
                    lam = L / L_rand if L_rand > 0 else 0
                    feat["small_worldness"] = float(gamma / lam) if lam > 0 else 0.0
                else:
                    feat["small_worldness"] = 0.0
            else:
                feat["small_worldness"] = 0.0

        if "betweenness_centrality_mean" in metrics_set:
            bc = nx.betweenness_centrality(G, weight="weight")
            feat["betweenness_centrality_mean"] = float(np.mean(list(bc.values())))

        if "participation_coefficient" in metrics_set:
            if comms is None:
                comms = community.greedy_modularity_communities(G, weight="weight")
            feat["participation_coefficient"] = _participation_coefficient(G, list(comms))

        results.append(feat)
    return results


__all__ = ["compute_state_features", "SUPPORTED_METRICS"]
