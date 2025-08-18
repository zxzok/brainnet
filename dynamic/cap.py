"""
brainnet.dynamic.cap
===================

This module implements co‑activation pattern (CAP) analysis for
dynamic functional connectivity.  CAP analysis identifies short
epochs of high global activation in the ROI time series and
clusters these events into distinct spatial patterns.  Unlike
sliding window methods, CAP focuses on instantaneous events
exceeding a specified threshold, making it sensitive to transient
coordinated activations.  The output includes the spatial pattern
for each state, a state sequence over time (with unassigned
intervals marked as -1) and temporal metrics computed from this
sequence.

Note
----
CAP analysis relies on the same K‑means clustering used in the
sliding window approach.  It requires ``scikit‑learn`` to be
installed.  If no events exceed the threshold, an exception will
be raised.
"""

from __future__ import annotations

from typing import List, Optional

import numpy as np
from sklearn.cluster import KMeans

from .config import DynamicConfig
from .model import DynamicStateModel
from .metrics import compute_state_metrics


def cap_analysis(
    roi_timeseries: np.ndarray, config: DynamicConfig, template: Optional[str] = None
) -> DynamicStateModel:
    """Perform co‑activation pattern (CAP) analysis.

    Parameters
    ----------
    roi_timeseries : np.ndarray
        Array of shape (T, N_ROI) containing ROI signals.
    config : DynamicConfig
        Configuration specifying the number of states and CAP
        detection threshold.

    Returns
    -------
    DynamicStateModel
        Model containing the CAP patterns, the state sequence and
        summary metrics.  Unassigned time points are labelled as -1
        in the state sequence.

    Raises
    ------
    ValueError
        If no high‑activation events are detected given the
        threshold.
    """
    T, N = roi_timeseries.shape
    # z‑score each ROI time series (subtract mean, divide by std)
    stds = np.std(roi_timeseries, axis=0, ddof=0) + 1e-8
    zscored = (roi_timeseries - roi_timeseries.mean(axis=0)) / stds
    # Compute global activation as mean absolute z‑scores across ROIs
    global_amp = np.mean(np.abs(zscored), axis=1)
    # Determine threshold based on mean and std of global amplitude
    mu = float(np.mean(global_amp))
    sigma = float(np.std(global_amp) + 1e-8)
    threshold = mu + config.cap_threshold * sigma
    # Identify event indices where global activation exceeds threshold
    event_idx = np.where(global_amp > threshold)[0]
    if event_idx.size == 0:
        raise ValueError(
            "No CAP events detected; consider lowering cap_threshold or checking data quality"
        )
    # Extract event frames and cluster them with K‑means
    samples = zscored[event_idx]  # shape (#events, N_ROI)
    # Use an explicit integer for ``n_init`` rather than the string
    # 'auto'.  This avoids TypeError in some scikit‑learn versions
    # where 'auto' is not recognised.  A value of 10 performs the
    # clustering from ten different initial centroids and selects
    # the best outcome.
    kmeans = KMeans(
        n_clusters=config.n_states,
        random_state=config.random_state,
        n_init=10,
    )
    labels = kmeans.fit_predict(samples)
    # Compute average activation pattern for each state
    states: List[np.ndarray] = []
    for i in range(config.n_states):
        if np.any(labels == i):
            pattern = samples[labels == i].mean(axis=0)
        else:
            # if no events assigned to state i, use zeros
            pattern = np.zeros(N, dtype=float)
        states.append(pattern)
    # Build state sequence: -1 indicates non‑event, event indices get labels
    seq = -np.ones(T, dtype=int)
    seq[event_idx] = labels
    # Compute temporal metrics (will ignore negative labels automatically)
    metrics = compute_state_metrics(seq, config.n_states)
    extra = {
        'event_idx': event_idx,
        'global_amp': global_amp,
    }
    return DynamicStateModel(
        method='cap',
        n_states=config.n_states,
        states=states,
        state_sequence=seq,
        metrics=metrics,
        extra=extra,
        template=template,
    )


__all__ = [
    'cap_analysis',
]
