"""
brainnet.dynamic.hmm
===================

This module implements dynamic connectivity analysis using Gaussian
hidden Markov models (HMMs).  A multivariate Gaussian HMM is fit to
the ROI time series, assuming that each hidden state corresponds to
a distinct distribution of brain activity.  The number of states can
either be specified by the user or, when
``DynamicConfig.auto_n_states`` is set, chosen automatically by
minimising the Bayesian or Akaike information criteria.  After
fitting the HMM, the most likely sequence of hidden states is
obtained (via Viterbi decoding) and temporal metrics are computed
from this sequence.  The state patterns (means and covariances) are
returned as part of the resulting
:class:`brainnet.dynamic.model.DynamicStateModel`.

This implementation requires the optional dependency ``hmmlearn``.
If ``hmmlearn`` is not installed, attempting to call
``hmm_analysis`` will raise ``NotImplementedError``.
"""

from __future__ import annotations

from typing import Iterable, List, Optional

import numpy as np

from .config import DynamicConfig
from .model import DynamicStateModel
from .metrics import compute_state_metrics
from .state_features import compute_state_features

try:
    from hmmlearn.hmm import GaussianHMM  # type: ignore
except Exception:
    GaussianHMM = None  # type: ignore


def _num_hmm_parameters(n_states: int, n_features: int) -> int:
    """Return the number of free parameters in a Gaussian HMM."""

    cov_params = n_features * (n_features + 1) / 2
    mean_params = n_features
    emission = n_states * (cov_params + mean_params)
    transition = n_states * (n_states - 1)
    startprob = n_states - 1
    return int(emission + transition + startprob)


def suggest_n_states(
    roi_timeseries: np.ndarray,
    candidates: Iterable[int] = range(2, 11),
    criterion: str = 'bic',
    random_state: Optional[int] = None,
) -> int:
    """Select the optimal number of HMM states using information criteria.

    Parameters
    ----------
    roi_timeseries : np.ndarray
        Array of shape (T, N) with ROI signals.
    candidates : Iterable[int], optional
        Candidate numbers of states to evaluate. Defaults to ``range(2, 11)``.
    criterion : str, optional
        Either ``'bic'`` (default) or ``'aic'`` specifying which
        information criterion to minimise.
    random_state : int | None, optional
        Random seed for model initialisation.
    """

    T, N = roi_timeseries.shape
    best_score = np.inf
    best_k: Optional[int] = None
    for k in candidates:
        hmm = GaussianHMM(
            n_components=k,
            covariance_type='full',
            n_iter=100,
            random_state=random_state,
        )
        hmm.fit(roi_timeseries)
        log_lik = hmm.score(roi_timeseries)
        n_params = _num_hmm_parameters(k, N)
        if criterion == 'bic':
            score = -2 * log_lik + n_params * np.log(T)
        elif criterion == 'aic':
            score = -2 * log_lik + 2 * n_params
        else:
            raise ValueError("criterion must be 'bic' or 'aic'")
        if score < best_score:
            best_score = score
            best_k = k
    if best_k is None:
        raise ValueError("Unable to determine a suitable number of states")
    return best_k


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
    if config.auto_n_states:
        config.n_states = suggest_n_states(
            roi_timeseries,
            criterion=config.n_states_criterion,
            random_state=config.random_state,
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
    cov_mats: List[np.ndarray] = []
    for i in range(config.n_states):
        means = hmm.means_[i]
        cov = hmm.covars_[i]
        states.append({'mean': means, 'cov': cov})
        cov_mats.append(cov)
    # Compute temporal metrics from the hidden state sequence
    metrics = compute_state_metrics(hidden_states, config.n_states)
    # Derive graph features from the covariance matrices of each state
    state_features = compute_state_features(cov_mats)
    extra = {
        'hmm': hmm,
        'state_features': state_features,
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
    'suggest_n_states',
]
