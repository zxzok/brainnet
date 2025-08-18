"""
dynamic_analysis
================

This module implements simple algorithms for estimating time‑varying
functional connectivity and identifying recurring network states.  The
approach is based on dividing ROI time series into overlapping
windows, computing a connectivity matrix for each window, and then
grouping these matrices using clustering (e.g. K‑means).  Basic
metrics describing the temporal dynamics of the network states are
also computed.

The implementation here is intentionally straightforward and does
not include more sophisticated models such as autoregressive
hidden Markov models.  Nevertheless, it provides a starting point
for exploratory analyses of dynamic functional connectivity (dFC).

Example
-------
>>> from brainnet.dynamic_analysis import DynamicAnalyzer, DynamicConfig
>>> cfg = DynamicConfig(window_length=30, step=5, n_states=4)
>>> dyn = DynamicAnalyzer(cfg)
>>> result = dyn.analyse(roi_timeseries)
>>> print(result.metrics.occupancy)
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

from sklearn.cluster import KMeans
try:
    # hmmlearn is optional; used for HMM modelling if available
    from hmmlearn.hmm import GaussianHMM
except Exception:
    GaussianHMM = None


@dataclass
class DynamicConfig:
    """Configuration for dynamic connectivity analysis.

    Parameters
    ----------
    window_length : int
        Length of sliding window in time points.
    step : int
        Step size (in time points) for sliding window.  Windows start
        at indices 0, ``step``, ``2*step``, … until no full window
        remains.
    n_states : int
        Number of discrete states/clusters to identify when using
        clustering or HMM.
    method : str
        Method to identify states.  ``'kmeans'`` (default) applies
        K‑means clustering to the flattened connectivity matrices.
        ``'hmm'`` fits a Gaussian hidden Markov model to the ROI
        time series.  ``'cap'`` performs co‑activation pattern
        analysis on the ROI signals using a threshold and K‑means.
    cap_threshold : float
        Threshold (in standard deviations) for selecting high‑amplitude
        events in CAP analysis.
    random_state : int | None
        Random seed for clustering and HMM initialisation.
    """

    window_length: int
    step: int
    n_states: int = 4
    method: str = 'kmeans'
    cap_threshold: float = 1.5
    random_state: Optional[int] = None

    def validate(self, n_timepoints: int) -> None:
        if self.window_length <= 0 or self.step <= 0:
            raise ValueError("window_length and step must be positive integers")
        if self.window_length > n_timepoints:
            raise ValueError("window_length exceeds number of time points in data")
        if self.method not in {'kmeans', 'hmm', 'cap'}:
            raise ValueError(f"Unknown dynamic method '{self.method}'")


@dataclass
class DynamicMetrics:
    """Quantify temporal properties of discrete brain states.

    Attributes
    ----------
    occupancy : np.ndarray
        Fraction of time spent in each state.  Shape (n_states,).
    mean_dwell_time : np.ndarray
        Average consecutive duration (in windows or time points) of
        each state.  Shape (n_states,).
    transition_matrix : np.ndarray
        Square matrix of size (n_states, n_states) where entry (i,j)
        gives the probability of transitioning from state i to state j.
    n_transitions : int
        Total number of state changes across the sequence.
    """

    occupancy: np.ndarray
    mean_dwell_time: np.ndarray
    transition_matrix: np.ndarray
    n_transitions: int


@dataclass
class DynamicStateModel:
    """Representation of a dynamic state model.

    Attributes
    ----------
    method : str
        Method used to identify states ('kmeans', 'hmm' or 'cap').
    n_states : int
        Number of states in the model.
    states : List[np.ndarray]
        List of state patterns.  For K‑means this contains the state
        centroids as connectivity matrices (shape N_ROI×N_ROI).  For
        CAP it contains average activation vectors (length N_ROI).  For
        HMM it could contain the state means or covariances.
    state_sequence : np.ndarray
        Array of length equal to the number of windows (for kmeans/HMM)
        or time points (for CAP) giving the index of the state at each
        time.
    metrics : DynamicMetrics
        Quantitative measures of the temporal dynamics.
    extra : Dict
        Dictionary for storing method‑specific additional outputs (e.g.
        raw window matrices or responsibilities).  Not guaranteed to
        be present.
    """

    method: str
    n_states: int
    states: List[np.ndarray]
    state_sequence: np.ndarray
    metrics: DynamicMetrics
    extra: Dict = field(default_factory=dict)


class DynamicAnalyzer:
    """Compute time‑varying connectivity and derive state dynamics.

    An instance of this class uses a :class:`DynamicConfig` to define
    how dynamic analysis should be performed.  See the
    :class:`DynamicConfig` documentation for details on the
    configuration options.
    """

    def __init__(self, config: DynamicConfig) -> None:
        self.config = config

    # --------------------------------------------------------------
    def _compute_sliding_windows(self, roi_timeseries: np.ndarray) -> Tuple[List[np.ndarray], np.ndarray]:
        """Compute connectivity matrices for sliding windows.

        Parameters
        ----------
        roi_timeseries : np.ndarray
            Array of shape (T, N_ROI) where T is the number of
            time points and N_ROI is the number of regions.

        Returns
        -------
        windows : list of np.ndarray
            Each element is a 2D connectivity matrix (N_ROI×N_ROI) for
            the corresponding window.
        window_metric : np.ndarray
            1D array of length equal to number of windows containing
            the mean absolute connectivity strength in each window.
            This can be used for visualising the dynamic spectrum.
        """
        T, N = roi_timeseries.shape
        self.config.validate(T)
        windows: List[np.ndarray] = []
        window_metric = []
        # iterate over windows
        for start in range(0, T - self.config.window_length + 1, self.config.step):
            end = start + self.config.window_length
            segment = roi_timeseries[start:end]
            # correlation matrix
            corr = np.corrcoef(segment.T)
            corr = np.nan_to_num(corr, nan=0.0)
            np.fill_diagonal(corr, 0.0)
            windows.append(corr)
            # compute mean absolute connectivity (excluding diagonal)
            mask = ~np.eye(N, dtype=bool)
            mean_abs = np.mean(np.abs(corr[mask]))
            window_metric.append(mean_abs)
        return windows, np.array(window_metric)

    # --------------------------------------------------------------
    def analyse(self, roi_timeseries: np.ndarray, template: Optional[str] = None) -> DynamicStateModel:
        """Run dynamic connectivity analysis and state identification.

        Parameters
        ----------
        roi_timeseries : np.ndarray
            Time×ROI matrix.

        Returns
        -------
        DynamicStateModel
            Encapsulating the discovered states, the state sequence and
            associated metrics.
        """
        method = self.config.method
        if method == 'kmeans':
            return self._analyse_kmeans(roi_timeseries, template)
        elif method == 'hmm':
            return self._analyse_hmm(roi_timeseries, template)
        elif method == 'cap':
            return self._analyse_cap(roi_timeseries, template)
        else:
            raise ValueError(f"Unknown dynamic analysis method '{method}'")

    # --------------------------------------------------------------
    def _analyse_kmeans(self, roi_timeseries: np.ndarray, template: Optional[str]) -> DynamicStateModel:
        windows, window_metric = self._compute_sliding_windows(roi_timeseries)
        if not windows:
            raise ValueError("No windows could be generated; check window_length and data length")
        # flatten upper triangle of each connectivity matrix to vector
        N = windows[0].shape[0]
        iu = np.triu_indices(N, k=1)
        features = np.array([w[iu] for w in windows])  # shape (n_windows, n_features)
        # Use an explicit integer for ``n_init`` instead of ``'auto'``.  Some
        # versions of scikit‑learn do not support the string 'auto' and
        # will raise a TypeError.  A value of 10 is a sensible default
        # that runs the algorithm multiple times and selects the best
        # clustering.
        kmeans = KMeans(
            n_clusters=self.config.n_states,
            random_state=self.config.random_state,
            n_init=10
        )
        labels = kmeans.fit_predict(features)
        # reconstruct centroid matrices
        centroids: List[np.ndarray] = []
        for cent in kmeans.cluster_centers_:
            mat = np.zeros((N, N))
            mat[iu] = cent
            mat = mat + mat.T
            np.fill_diagonal(mat, 0.0)
            centroids.append(mat)
        # compute metrics
        metrics = self._compute_state_metrics(labels)
        extra = {
            'window_metric': window_metric,
        }
        return DynamicStateModel(
            method='kmeans',
            n_states=self.config.n_states,
            states=centroids,
            state_sequence=labels,
            metrics=metrics,
            extra=extra,
            template=template,
        )

    # --------------------------------------------------------------
    def _analyse_hmm(self, roi_timeseries: np.ndarray, template: Optional[str]) -> DynamicStateModel:
        T, N = roi_timeseries.shape
        self.config.validate(T)
        if GaussianHMM is None:
            raise NotImplementedError("hmmlearn is not available; cannot run HMM analysis")
        # Fit a Gaussian HMM to ROI time series.  We assume each time point
        # is an observation vector of length N.
        hmm = GaussianHMM(
            n_components=self.config.n_states,
            covariance_type='full',
            n_iter=100,
            random_state=self.config.random_state,
        )
        hmm.fit(roi_timeseries)
        hidden_states = hmm.predict(roi_timeseries)
        # state means or covariances could serve as patterns
        states = []
        for i in range(self.config.n_states):
            means = hmm.means_[i]  # shape (N,)
            cov = hmm.covars_[i]   # shape (N,N)
            # store both mean and covariance in a dict
            states.append({'mean': means, 'cov': cov})
        metrics = self._compute_state_metrics(hidden_states)
        extra = {
            'hmm': hmm,
        }
        return DynamicStateModel(
            method='hmm',
            n_states=self.config.n_states,
            states=states,
            state_sequence=hidden_states,
            metrics=metrics,
            extra=extra,
            template=template,
        )

    # --------------------------------------------------------------
    def _analyse_cap(self, roi_timeseries: np.ndarray, template: Optional[str]) -> DynamicStateModel:
        # Co‑activation pattern analysis: identify high‑amplitude frames and cluster them
        T, N = roi_timeseries.shape
        # z‑score each ROI time series
        zscored = (roi_timeseries - roi_timeseries.mean(axis=0)) / (roi_timeseries.std(axis=0) + 1e-8)
        # compute global activation measure: mean across ROI absolute values
        global_amp = np.mean(np.abs(zscored), axis=1)
        # select frames where global activation exceeds threshold (in std dev)
        thresh = self.config.cap_threshold
        # threshold based on mean and std of global_amp
        mu = np.mean(global_amp)
        sigma = np.std(global_amp) + 1e-8
        event_idx = np.where(global_amp > mu + thresh * sigma)[0]
        if event_idx.size == 0:
            raise ValueError("No CAP events detected; consider lowering cap_threshold")
        # cluster event frames
        samples = zscored[event_idx]  # shape (#events, N)
        # Use an explicit integer for ``n_init`` rather than 'auto'.  This
        # avoids compatibility issues with scikit‑learn versions prior to
        # 1.2 where 'auto' is not accepted.  A value of 10 runs the
        # clustering from ten random initialisations.
        kmeans = KMeans(
            n_clusters=self.config.n_states,
            random_state=self.config.random_state,
            n_init=10
        )
        labels = kmeans.fit_predict(samples)
        # compute state patterns as mean activation across events assigned to each state
        states: List[np.ndarray] = []
        for i in range(self.config.n_states):
            if np.any(labels == i):
                pattern = samples[labels == i].mean(axis=0)
            else:
                pattern = np.zeros(N)
            states.append(pattern)
        # assign state label to each time point: default -1, events have labels
        seq = -np.ones(T, dtype=int)
        seq[event_idx] = labels
        metrics = self._compute_state_metrics(seq)
        extra = {
            'event_idx': event_idx,
            'global_amp': global_amp,
        }
        return DynamicStateModel(
            method='cap',
            n_states=self.config.n_states,
            states=states,
            state_sequence=seq,
            metrics=metrics,
            extra=extra,
            template=template,
        )

    # --------------------------------------------------------------
    def _compute_state_metrics(self, sequence: np.ndarray) -> DynamicMetrics:
        """Compute occupancy, dwell time and transition statistics.

        Parameters
        ----------
        sequence : np.ndarray
            1D array of state labels.  Negative values are ignored
            when counting occupancy and transitions (used for CAP
            non‑events).  The length of the array corresponds to the
            number of windows (for kmeans/HMM) or time points (for CAP).

        Returns
        -------
        DynamicMetrics
            Temporal metrics summarising the state sequence.
        """
        # consider only non‑negative labels for metrics
        valid_mask = sequence >= 0
        valid_seq = sequence[valid_mask]
        if valid_seq.size == 0:
            # no valid states
            n_states = self.config.n_states
            occupancy = np.zeros(n_states)
            mean_dwell = np.zeros(n_states)
            trans_mat = np.zeros((n_states, n_states))
            n_trans = 0
            return DynamicMetrics(occupancy, mean_dwell, trans_mat, n_trans)
        n_states = self.config.n_states
        # occupancy: fraction of valid time spent in each state
        occupancy = np.zeros(n_states, dtype=float)
        for s in range(n_states):
            occupancy[s] = np.mean(valid_seq == s)
        # dwell time: compute runs of same state
        mean_dwell = np.zeros(n_states, dtype=float)
        for s in range(n_states):
            # find lengths of consecutive runs of state s
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
            if lengths:
                mean_dwell[s] = float(np.mean(lengths))
            else:
                mean_dwell[s] = 0.0
        # transition matrix
        trans_mat = np.zeros((n_states, n_states), dtype=float)
        n_trans = 0
        for i in range(len(valid_seq) - 1):
            a = valid_seq[i]
            b = valid_seq[i + 1]
            if a != b:
                trans_mat[a, b] += 1.0
                n_trans += 1
        # normalise rows to probabilities
        row_sums = trans_mat.sum(axis=1, keepdims=True)
        with np.errstate(divide='ignore', invalid='ignore'):
            trans_probs = trans_mat / row_sums
        trans_probs[np.isnan(trans_probs)] = 0.0
        return DynamicMetrics(occupancy, mean_dwell, trans_probs, n_trans)


__all__ = [
    'DynamicConfig',
    'DynamicMetrics',
    'DynamicStateModel',
    'DynamicAnalyzer',
]