"""
brainnet.static.connectivity
===========================

This module defines data structures and functions for computing static
functional connectivity matrices from ROI time series.  Currently
only Pearson correlation is implemented.  The resulting
``ConnectivityMatrix`` stores the symmetric matrix along with the
associated ROI labels and metadata about the computation method.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np


@dataclass
class ConnectivityMatrix:
    """Encapsulate a functional connectivity matrix and its labels.

    Parameters
    ----------
    matrix : np.ndarray
        2D array of shape (N, N) representing the connectivity
        between N ROIs.  It is assumed to be symmetric with zeros
        on the diagonal.
    labels : Sequence[str]
        Sequence of ROI labels corresponding to the rows/columns of
        ``matrix``.  Length must match the matrix size.
    method : str, optional
        Name of the method used to compute the matrix (e.g. ``'pearson'``).
        Defaults to ``'pearson'``.
    """

    matrix: np.ndarray
    labels: Sequence[str]
    method: str = 'pearson'

    def copy(self) -> 'ConnectivityMatrix':
        """Return a deep copy of the connectivity matrix and labels."""
        return ConnectivityMatrix(self.matrix.copy(), list(self.labels), self.method)


def compute_pearson_connectivity(
    roi_timeseries: np.ndarray,
    labels: Sequence[str],
) -> ConnectivityMatrix:
    """Compute a Pearson correlation connectivity matrix for ROI signals.

    Given a time×ROI matrix ``roi_timeseries``, this function computes
    the Pearson correlation coefficient between each pair of ROIs.
    The resulting matrix is symmetric with zeros on the diagonal.

    Parameters
    ----------
    roi_timeseries : np.ndarray
        Array of shape (T, N_ROI) where each column contains the
        signal for one region of interest.
    labels : Sequence[str]
        Names or identifiers for each ROI.  Length must equal ``N_ROI``.

    Returns
    -------
    ConnectivityMatrix
        Dataclass containing the correlation matrix and ROI labels.
    """
    if roi_timeseries.ndim != 2:
        raise ValueError("roi_timeseries must be a 2D array of shape (T, N_ROI)")
    T, N = roi_timeseries.shape
    if len(labels) != N:
        raise ValueError("Number of labels must match number of columns in roi_timeseries")
    # compute correlation; transpose so variables (ROIs) are rows for np.corrcoef
    corr = np.corrcoef(roi_timeseries.T)
    # replace nan due to zero variance with zero
    corr = np.nan_to_num(corr, nan=0.0)
    # zero the diagonal
    np.fill_diagonal(corr, 0.0)
    return ConnectivityMatrix(matrix=corr, labels=list(labels), method='pearson')