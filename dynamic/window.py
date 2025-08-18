"""
brainnet.dynamic.window
======================

This module provides helper functions to compute time‑resolved
connectivity matrices by sliding a window across ROI time series.
For each window the Pearson correlation between all pairs of ROIs
is computed and the diagonal is zeroed.  A simple summary metric,
the mean absolute connectivity strength, is also returned for each
window.  These outputs are used by dynamic analysis routines such
as K‑means and HMM clustering.

Functions
---------

``sliding_window_connectivity(roi_timeseries, window_length, step)``
    Compute correlation matrices for sliding windows and a summary
    metric for each window.
"""

from __future__ import annotations

from typing import List, Tuple

import numpy as np


def sliding_window_connectivity(
    roi_timeseries: np.ndarray,
    window_length: int,
    step: int,
) -> Tuple[List[np.ndarray], np.ndarray]:
    """Compute connectivity matrices for sliding windows.

    Parameters
    ----------
    roi_timeseries : np.ndarray
        Array of shape (T, N_ROI) containing the time series for each
        region of interest.  ``T`` is the number of time points and
        ``N_ROI`` is the number of ROIs.
    window_length : int
        Length of the sliding window in time points.  Must be
        positive and no larger than ``T``.
    step : int
        Step size (in time points) between successive windows.  Must
        be positive.

    Returns
    -------
    windows : list of np.ndarray
        List of connectivity matrices, each of shape
        (N_ROI, N_ROI).  The matrices are symmetrised with zero
        diagonal.
    window_metric : np.ndarray
        One‑dimensional array of length equal to the number of
        windows.  Each entry is the mean absolute connectivity
        strength (excluding the diagonal) in the corresponding
        window.

    Raises
    ------
    ValueError
        If the specified window length or step is invalid.
    """
    T, N = roi_timeseries.shape
    if window_length <= 0 or step <= 0:
        raise ValueError("window_length and step must be positive integers")
    if window_length > T:
        raise ValueError("window_length exceeds number of time points in data")
    windows: List[np.ndarray] = []
    metrics = []
    # slide the window across the time series
    for start in range(0, T - window_length + 1, step):
        end = start + window_length
        segment = roi_timeseries[start:end]
        # compute correlation matrix; transpose so variables are rows
        corr = np.corrcoef(segment.T)
        corr = np.nan_to_num(corr, nan=0.0)
        # remove self connections
        np.fill_diagonal(corr, 0.0)
        # enforce symmetry exactly (rounding errors)
        corr = (corr + corr.T) / 2.0
        windows.append(corr)
        # compute mean absolute connectivity excluding diagonal
        mask = ~np.eye(N, dtype=bool)
        metrics.append(float(np.mean(np.abs(corr[mask]))))
    return windows, np.array(metrics, dtype=float)


__all__ = [
    'sliding_window_connectivity',
]
