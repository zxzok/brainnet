"""
brainnet.static.analyzer
=======================

This module provides a high level interface for computing static
functional connectivity and graph metrics.  The :class:`StaticAnalyzer`
class ties together the lower level functions defined in
:mod:`brainnet.static.connectivity` and :mod:`brainnet.static.metrics`.
It also offers simple thresholding of connectivity matrices based on
absolute values or proportional sparsity.  Users may instantiate
``StaticAnalyzer`` with optional threshold parameters and call
``compute_connectivity`` and ``compute_graph_metrics`` on ROI time
series.
"""

from __future__ import annotations

from typing import Optional, Sequence

import numpy as np

from .connectivity import ConnectivityMatrix, compute_pearson_connectivity
from .metrics import (
    GraphMetrics,
    compute_degree,
    compute_clustering,
    compute_global_efficiency,
)


class StaticAnalyzer:
    """Compute static connectivity and graph metrics for ROI time series.

    Parameters
    ----------
    threshold : float | None, optional
        Absolute threshold on connection strength.  Edges with
        absolute value below this threshold are removed.  If both
        ``threshold`` and ``proportion`` are provided, ``threshold``
        takes precedence.
    proportion : float | None, optional
        Proportional sparsity threshold.  Only the top ``proportion``
        fraction of strongest connections (by absolute value) are
        retained.  Must be between 0 and 1.  Ignored if ``threshold``
        is provided.
    """

    def __init__(self, threshold: Optional[float] = None, proportion: Optional[float] = None) -> None:
        self.threshold = threshold
        self.proportion = proportion

    # -- connectivity computation -----------------------------------
    def compute_connectivity(
        self,
        roi_timeseries: np.ndarray,
        labels: Sequence[str],
        method: str = 'pearson',
        template: Optional[str] = None,
    ) -> ConnectivityMatrix:
        """Compute static functional connectivity matrix.

        Currently only Pearson correlation is supported.  If another
        method name is specified, a ``NotImplementedError`` is raised.

        Parameters
        ----------
        roi_timeseries : np.ndarray
            Array of shape (T, N_ROI) with ROI signals along columns.
        labels : Sequence[str]
            Names or identifiers for each ROI.
        method : str, optional
            Connectivity measure to use.  Only ``'pearson'`` is
            implemented.

        Returns
        -------
        ConnectivityMatrix
            Dataclass containing the connectivity matrix and labels.
        """
        if method != 'pearson':
            raise NotImplementedError(f"Connectivity method '{method}' is not implemented")
        return compute_pearson_connectivity(roi_timeseries, labels, template=template)

    # -- thresholding -----------------------------------------------
    def _apply_threshold(self, matrix: np.ndarray) -> np.ndarray:
        """Apply thresholding or sparsity to a connectivity matrix.

        If ``self.threshold`` is set, edges with absolute value
        below the threshold are removed.  Otherwise if ``self.proportion``
        is set, only the top ``proportion`` fraction of edges are
        retained.  The matrix is symmetrised after thresholding and
        diagonal elements are set to zero.

        Parameters
        ----------
        matrix : np.ndarray
            Weighted adjacency matrix (square).  The diagonal is
            assumed to be zero.

        Returns
        -------
        np.ndarray
            The thresholded matrix.
        """
        thr_mat = matrix.copy()
        if self.threshold is not None:
            mask = np.abs(thr_mat) < self.threshold
            thr_mat[mask] = 0.0
        elif self.proportion is not None and 0 < self.proportion < 1:
            N = thr_mat.shape[0]
            iu = np.triu_indices(N, k=1)
            vals = np.abs(thr_mat[iu])
            if vals.size > 0:
                cutoff_idx = int((1.0 - self.proportion) * vals.size)
                if cutoff_idx < 0:
                    cutoff_idx = 0
                sorted_vals = np.sort(vals)
                threshold_value = sorted_vals[cutoff_idx] if cutoff_idx < vals.size else sorted_vals[-1]
                mask = np.abs(thr_mat) < threshold_value
                thr_mat[mask] = 0.0
        # enforce symmetry and zero diagonal
        thr_mat = (thr_mat + thr_mat.T) / 2.0
        np.fill_diagonal(thr_mat, 0.0)
        return thr_mat

    # -- graph metrics ---------------------------------------------
    def compute_graph_metrics(self, conn_matrix: ConnectivityMatrix) -> GraphMetrics:
        """Derive graph metrics from a connectivity matrix.

        Parameters
        ----------
        conn_matrix : ConnectivityMatrix
            Functional connectivity matrix with labels.  The internal
            matrix is assumed symmetric and zero diagonal.

        Returns
        -------
        GraphMetrics
            Dataclass containing node‑wise and global graph metrics.
        """
        # threshold matrix if configured
        mat = conn_matrix.matrix.copy()
        mat = self._apply_threshold(mat)
        # compute metrics
        degrees = compute_degree(mat)
        clustering = compute_clustering(mat)
        efficiency = compute_global_efficiency(mat)
        node_metrics = {
            'degree': degrees,
            'clustering': clustering,
        }
        global_metrics = {
            'global_efficiency': efficiency,
        }
        return GraphMetrics(node_metrics=node_metrics, global_metrics=global_metrics)