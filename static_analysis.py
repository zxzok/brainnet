"""
static_analysis
================

This module contains functions and classes for computing static
functional connectivity matrices and deriving basic graph‑theoretic
metrics from them.  It is designed to operate on ROI time series
obtained from preprocessed fMRI data.  The focus is on clarity
rather than efficiency; for larger datasets users may wish to
parallelise or optimise certain operations.

The core entity is :class:`ConnectivityMatrix`, which stores the
connectivity values and associated ROI labels.  Graph metrics are
encapsulated in the :class:`GraphMetrics` dataclass.  The
:class:`StaticAnalyzer` class provides convenience methods to
compute the connectivity matrix using Pearson correlation and to
derive graph metrics such as weighted degree, clustering coefficient
and global efficiency.  Thresholding can be applied to emphasise
significant connections.
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, Optional, Sequence


# Re‑export the data classes from the new ``brainnet.static`` package.  The
# legacy ``static_analysis`` module used to define its own data structures,
# but these are now provided by the modular ``brainnet.static``
# subpackage.  Importing them here maintains backwards compatibility
# for code that still references ``brainnet.static_analysis.ConnectivityMatrix``
# or ``GraphMetrics``.
from .static import ConnectivityMatrix, GraphMetrics, compute_pearson_connectivity, compute_degree, compute_clustering, compute_global_efficiency


class StaticAnalyzer:
    """Compute static connectivity and graph metrics for ROI time series.

    This class mirrors the API of the legacy static analyzer but
    delegates the core computations to the modular functions in
    ``brainnet.static``.  It provides optional thresholding of
    connectivity matrices and computation of basic graph metrics.

    Parameters
    ----------
    threshold : float | None
        If provided, absolute connectivity values below this threshold
        will be set to zero before graph metrics are computed.  This
        helps focus analysis on the strongest connections.  If
        ``None``, no explicit thresholding is applied.
    proportion : float | None
        Alternatively to an absolute threshold, one can specify a
        proportion between 0 and 1.  In this case only the top
        ``proportion`` fraction of connections (by absolute value)
        will be retained.  If both ``threshold`` and ``proportion``
        are provided, ``threshold`` takes precedence.
    """

    def __init__(self, threshold: Optional[float] = None, proportion: Optional[float] = None) -> None:
        self.threshold = threshold
        self.proportion = proportion

    # -- connectivity computation ---------------------------------------
    def compute_connectivity(
        self,
        roi_timeseries: np.ndarray,
        labels: Sequence[str],
        method: str = 'pearson',
        template: Optional[str] = None,
    ) -> ConnectivityMatrix:
        """Compute a static functional connectivity matrix.

        This method currently supports only Pearson correlation.  It
        wraps the :func:`brainnet.static.compute_pearson_connectivity`
        function and returns the resulting :class:`~brainnet.static.ConnectivityMatrix`.

        Parameters
        ----------
        roi_timeseries : np.ndarray
            Array of shape (T, N_ROI) with ROI signals along columns.
        labels : Sequence[str]
            Names or identifiers for each ROI.
        method : str, optional
            Connectivity measure to use.  Only ``'pearson'`` is
            implemented.  Other values raise ``NotImplementedError``.

        Returns
        -------
        ConnectivityMatrix
            Dataclass containing the connectivity matrix and labels.
        """
        if method != 'pearson':
            raise NotImplementedError(f"Connectivity method '{method}' is not implemented")
        return compute_pearson_connectivity(roi_timeseries, labels, template=template)

    # -- thresholding ----------------------------------------------------
    def _apply_threshold(self, matrix: np.ndarray) -> np.ndarray:
        """Apply absolute threshold or proportional sparsity to matrix.

        This helper mirrors the behaviour of the legacy implementation.
        It does not depend on any external libraries.  Given a matrix,
        it zeros out small values based on the configured threshold
        or proportion and ensures the output is symmetric with a zero
        diagonal.

        Parameters
        ----------
        matrix : np.ndarray
            2D square matrix of connectivity values.

        Returns
        -------
        np.ndarray
            Thresholded matrix with same shape.  Values below the
            threshold/proportion are set to zero.  The matrix is
            symmetrised to ensure an undirected graph structure.
        """
        thr_mat = matrix.copy()
        # apply absolute threshold if specified
        if self.threshold is not None:
            mask = np.abs(thr_mat) < self.threshold
            thr_mat[mask] = 0.0
        # otherwise apply proportional threshold
        elif self.proportion is not None and 0 < self.proportion < 1:
            N = thr_mat.shape[0]
            iu = np.triu_indices(N, k=1)
            vals = np.abs(thr_mat[iu])
            if vals.size > 0:
                cutoff_idx = int((1.0 - self.proportion) * vals.size)
                if cutoff_idx < 0:
                    cutoff_idx = 0
                sorted_vals = np.sort(vals)
                threshold_value = (
                    sorted_vals[cutoff_idx]
                    if cutoff_idx < vals.size
                    else sorted_vals[-1]
                )
                mask = np.abs(thr_mat) < threshold_value
                thr_mat[mask] = 0.0
        # enforce symmetry and zero diagonal
        thr_mat = (thr_mat + thr_mat.T) / 2.0
        np.fill_diagonal(thr_mat, 0.0)
        return thr_mat

    # -- graph metrics ---------------------------------------------------
    def compute_graph_metrics(self, conn_matrix: ConnectivityMatrix) -> GraphMetrics:
        """Compute graph metrics from a connectivity matrix.

        This wraps the graph metric functions from ``brainnet.static``.
        The input connectivity matrix is first thresholded (if
        configured) and then degree, clustering coefficient and global
        efficiency are computed.  The results are returned in a
        :class:`GraphMetrics` dataclass.

        Parameters
        ----------
        conn_matrix : ConnectivityMatrix
            Functional connectivity matrix with labels.

        Returns
        -------
        GraphMetrics
            Node and global metrics derived from the thresholded matrix.
        """
        # copy the matrix and apply thresholding
        mat = conn_matrix.matrix.copy()
        mat = self._apply_threshold(mat)
        # compute metrics using functions from brainnet.static.metrics
        degrees = compute_degree(mat)
        clustering = compute_clustering(mat)
        efficiency = compute_global_efficiency(mat)
        node_metrics: Dict[str, np.ndarray] = {
            'degree': degrees,
            'clustering': clustering,
        }
        global_metrics: Dict[str, float] = {
            'global_efficiency': float(efficiency),
        }
        return GraphMetrics(node_metrics=node_metrics, global_metrics=global_metrics)


__all__ = [
    'ConnectivityMatrix',
    'GraphMetrics',
    'StaticAnalyzer',
]
