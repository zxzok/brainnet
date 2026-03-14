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


def _dwell_lengths(sequence: np.ndarray, state: int) -> list:
    """Return list of consecutive run lengths for a given state."""
    lengths = []
    current = 0
    for val in sequence:
        if val == state:
            current += 1
        elif current > 0:
            lengths.append(current)
            current = 0
    if current > 0:
        lengths.append(current)
    return lengths


def _lempel_ziv_complexity(sequence: np.ndarray) -> float:
    """Compute normalized Lempel-Ziv complexity of a discrete sequence.

    Returns a value in [0, 1] where higher means more complex/random.
    """
    n = len(sequence)
    if n <= 1:
        return 0.0
    s = sequence.tolist()
    # Count distinct subsequences (LZ76 algorithm)
    i = 0
    c = 1  # complexity counter
    l = 1  # current prefix length
    k = 1
    k_max = 1
    while True:
        if s[i + k - 1] == s[l + k - 1]:
            k += 1
            if l + k > n:
                c += 1
                break
        else:
            if k > k_max:
                k_max = k
            i += 1
            if i == l:
                c += 1
                l += k_max
                if l + 1 > n:
                    break
                i = 0
                k = 1
                k_max = 1
            else:
                k = 1
    # Normalize by theoretical upper bound for random sequence
    n_states = len(np.unique(sequence))
    if n_states <= 1:
        return 0.0
    b = max(n_states, 2)
    upper_bound = n / np.log2(n) * np.log2(b) if n > 0 else 1.0
    return min(c / upper_bound, 1.0) if upper_bound > 0 else 0.0


def compute_state_metrics(sequence: np.ndarray, n_states: int) -> DynamicMetrics:
    """Compute temporal metrics from a sequence of state labels.

    Parameters
    ----------
    sequence : np.ndarray
        One-dimensional array of integer state labels.  Negative
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
        Dataclass containing all temporal metrics.
    """
    # Filter out invalid entries
    valid_mask = sequence >= 0
    valid_seq = sequence[valid_mask]

    # If no valid states, return zeros
    if valid_seq.size == 0:
        z = np.zeros(n_states, dtype=float)
        return DynamicMetrics(
            occupancy=z.copy(), mean_dwell_time=z.copy(),
            transition_matrix=np.zeros((n_states, n_states), dtype=float),
            n_transitions=0,
            dwell_time_std=z.copy(), max_dwell_time=z.copy(),
            occupancy_entropy=0.0, transition_entropy=0.0,
            mean_recurrence_interval=z.copy(),
            switching_rate=0.0, state_complexity=0.0,
            temporal_autocorrelation=0.0,
        )

    # --- Occupancy ---
    occupancy = np.zeros(n_states, dtype=float)
    for s in range(n_states):
        occupancy[s] = np.mean(valid_seq == s)

    # --- Dwell time statistics ---
    mean_dwell = np.zeros(n_states, dtype=float)
    dwell_std = np.zeros(n_states, dtype=float)
    max_dwell = np.zeros(n_states, dtype=float)
    for s in range(n_states):
        lengths = _dwell_lengths(valid_seq, s)
        if lengths:
            mean_dwell[s] = float(np.mean(lengths))
            dwell_std[s] = float(np.std(lengths))
            max_dwell[s] = float(np.max(lengths))

    # --- Transition counts & probabilities ---
    trans_mat = np.zeros((n_states, n_states), dtype=float)
    n_trans = 0
    for i in range(len(valid_seq) - 1):
        a = valid_seq[i]
        b = valid_seq[i + 1]
        if a != b:
            trans_mat[a, b] += 1.0
            n_trans += 1

    row_sums = trans_mat.sum(axis=1, keepdims=True)
    with np.errstate(divide='ignore', invalid='ignore'):
        trans_probs = trans_mat / row_sums
    trans_probs[np.isnan(trans_probs)] = 0.0

    # --- Occupancy entropy (Shannon) ---
    occ_nonzero = occupancy[occupancy > 0]
    occupancy_entropy = float(-np.sum(occ_nonzero * np.log2(occ_nonzero))) if len(occ_nonzero) > 0 else 0.0

    # --- Transition entropy ---
    tp_flat = trans_probs[trans_probs > 0]
    transition_entropy = float(-np.sum(tp_flat * np.log2(tp_flat))) if len(tp_flat) > 0 else 0.0

    # --- Mean recurrence interval ---
    mean_recurrence = np.zeros(n_states, dtype=float)
    for s in range(n_states):
        indices = np.where(valid_seq == s)[0]
        if len(indices) > 1:
            intervals = np.diff(indices)
            mean_recurrence[s] = float(np.mean(intervals))

    # --- Switching rate ---
    switching_rate = float(n_trans / (len(valid_seq) - 1)) if len(valid_seq) > 1 else 0.0

    # --- State complexity (Lempel-Ziv) ---
    state_complexity = _lempel_ziv_complexity(valid_seq)

    # --- Temporal autocorrelation (lag-1) ---
    if len(valid_seq) > 1:
        seq_float = valid_seq.astype(float)
        mean_s = np.mean(seq_float)
        var_s = np.var(seq_float)
        if var_s > 0:
            temporal_autocorrelation = float(
                np.mean((seq_float[:-1] - mean_s) * (seq_float[1:] - mean_s)) / var_s
            )
        else:
            temporal_autocorrelation = 0.0
    else:
        temporal_autocorrelation = 0.0

    return DynamicMetrics(
        occupancy=occupancy,
        mean_dwell_time=mean_dwell,
        transition_matrix=trans_probs,
        n_transitions=n_trans,
        dwell_time_std=dwell_std,
        max_dwell_time=max_dwell,
        occupancy_entropy=occupancy_entropy,
        transition_entropy=transition_entropy,
        mean_recurrence_interval=mean_recurrence,
        switching_rate=switching_rate,
        state_complexity=state_complexity,
        temporal_autocorrelation=temporal_autocorrelation,
    )


__all__ = [
    'compute_state_metrics',
]
