"""
brainnet.dynamic.metrics
=======================

This module implements functions for computing temporal statistics
from sequences of discrete brain states.  Given a state assignment
over time, we can quantify how often the system resides in each
state, how long consecutive periods of activity in each state last,
the likelihood of transitioning between states and the total number
of transitions observed.  These metrics are used to summarise the
output of dynamic connectivity analyses performed by other
components of the :mod:`brainnet.dynamic` package.

Functions
---------

``compute_state_metrics(sequence: np.ndarray, n_states: int)``
    Compute occupancy, mean dwell time, transition probabilities and
    transition count from a state sequence.

See Also
--------
brainnet.dynamic.model.DynamicMetrics
    Dataclass encapsulating the metrics computed here.
"""

from __future__ import annotations

import numpy as np

from .model import DynamicMetrics


def compute_state_metrics(sequence: np.ndarray, n_states: int) -> DynamicMetrics:
    """Compute temporal metrics from a sequence of state labels.

    Parameters
    ----------
    sequence : np.ndarray
        One‑dimensional array of integer state labels.  Negative
        values indicate undefined or unassigned time points (e.g. in
        CAP analysis).  The valid labels should range from 0 to
        ``n_states-1``.
    n_states : int
        Total number of possible states.  This determines the
        dimensions of the returned metrics even if some states are not
        present in the input sequence.

    Returns
    -------
    DynamicMetrics
        Dataclass containing the occupancy, mean dwell time,
        transition probability matrix and the total number of
        transitions.
    """
    # Filter out invalid entries
    valid_mask = sequence >= 0
    valid_seq = sequence[valid_mask]
    # If no valid states, return zeros
    if valid_seq.size == 0:
        occupancy = np.zeros(n_states, dtype=float)
        mean_dwell = np.zeros(n_states, dtype=float)
        trans_probs = np.zeros((n_states, n_states), dtype=float)
        n_trans = 0
        return DynamicMetrics(occupancy, mean_dwell, trans_probs, n_trans)
    # Compute occupancy: fraction of valid time points/windows in each state
    occupancy = np.zeros(n_states, dtype=float)
    for s in range(n_states):
        occupancy[s] = np.mean(valid_seq == s)
    # Compute mean dwell time: average length of consecutive runs for each state
    mean_dwell = np.zeros(n_states, dtype=float)
    for s in range(n_states):
        lengths = []
        current = 0
        for val in valid_seq:
            if val == s:
                current += 1
            elif current > 0:
                lengths.append(current)
                current = 0
        if current > 0:
            lengths.append(current)
        mean_dwell[s] = float(np.mean(lengths)) if lengths else 0.0
    # Compute transition counts
    trans_mat = np.zeros((n_states, n_states), dtype=float)
    n_trans = 0
    for i in range(len(valid_seq) - 1):
        a = valid_seq[i]
        b = valid_seq[i + 1]
        if a != b:
            trans_mat[a, b] += 1.0
            n_trans += 1
    # Convert counts to probabilities (row normalised)
    row_sums = trans_mat.sum(axis=1, keepdims=True)
    with np.errstate(divide='ignore', invalid='ignore'):
        trans_probs = trans_mat / row_sums
    trans_probs[np.isnan(trans_probs)] = 0.0
    return DynamicMetrics(occupancy, mean_dwell, trans_probs, n_trans)


__all__ = [
    'compute_state_metrics',
]
