"""
brainnet.dynamic.hmm
===================

This module implements dynamic connectivity analysis using Gaussian
hidden Markov models (HMMs).  A multivariate Gaussian HMM is fit to
the ROI time series, assuming that each hidden state corresponds to
a distinct distribution of brain activity.  The number of states
is specified by the user.  After fitting the HMM, the most likely
sequence of hidden states is obtained (via Viterbi decoding) and
temporal metrics are computed from this sequence.  The state
patterns (means and covariances) are returned as part of the
resulting :class:`brainnet.dynamic.model.DynamicStateModel`.

This implementation requires the optional dependency ``hmmlearn``.
If ``hmmlearn`` is not installed, attempting to call
``hmm_analysis`` will raise ``NotImplementedError``.
"""

from __future__ import annotations

from typing import List, Optional

import numpy as np

from .config import DynamicConfig
from .model import DynamicStateModel
from .metrics import compute_state_metrics

try:
    from hmmlearn.hmm import GaussianHMM  # type: ignore
except Exception:
    GaussianHMM = None  # type: ignore


def hmm_analysis(
    roi_timeseries: np.ndarray, config: DynamicConfig, template: Optional[str] = None
) -> DynamicStateModel:
    """Perform dynamic connectivity analysis using a Gaussian HMM.

    Parameters
    ----------
    roi_timeseries : np.ndarray
        Array of shape (T, N_ROI) with ROI time series.
    config : DynamicConfig
        Configuration specifying the number of states and random seed.

    Returns
    -------
    DynamicStateModel
        Model containing the HMM state parameters, the inferred state
        sequence and summary metrics.

    Raises
    ------
    NotImplementedError
        If the optional dependency ``hmmlearn`` is not available.
    ValueError
        If the configuration is invalid relative to the data.
    """
    T, N = roi_timeseries.shape
    # validate configuration
    config.validate(T)
    if GaussianHMM is None:
        raise NotImplementedError(
            "hmmlearn is not installed; cannot perform HMM analysis."
        )
    # Fit Gaussian HMM to ROI time series.  Each observation is a vector
    # of ROI signals.  The number of states is given by config.n_states.
    hmm = GaussianHMM(
        n_components=config.n_states,
        covariance_type='full',
        n_iter=100,
        random_state=config.random_state,
    )
    hmm.fit(roi_timeseries)
    # Decode the most likely sequence of hidden states
    hidden_states = hmm.predict(roi_timeseries)
    # Collect state patterns: store mean and covariance for each state
    states: List[dict] = []
    for i in range(config.n_states):
        means = hmm.means_[i]
        cov = hmm.covars_[i]
        states.append({'mean': means, 'cov': cov})
    # Compute temporal metrics from the hidden state sequence
    metrics = compute_state_metrics(hidden_states, config.n_states)
    extra = {
        'hmm': hmm,
    }
    return DynamicStateModel(
        method='hmm',
        n_states=config.n_states,
        states=states,
        state_sequence=hidden_states.astype(int),
        metrics=metrics,
        extra=extra,
        template=template,
    )


__all__ = [
    'hmm_analysis',
]
