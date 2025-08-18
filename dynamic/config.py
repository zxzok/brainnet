"""
brainnet.dynamic.config
======================

This module defines configuration data classes for dynamic
functional connectivity analysis.  The primary class
:class:`DynamicConfig` specifies parameters for sliding window
construction and state identification via clustering, hidden Markov
models (HMM) or co‑activation pattern (CAP) analysis.  It includes
basic validation to ensure configuration options are consistent with
the input data.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass
class DynamicConfig:
    """Configuration options for dynamic connectivity analysis.

    Attributes
    ----------
    window_length : int
        Length of the sliding window in time points.
    step : int
        Step size (in time points) between successive windows.
    n_states : int, optional
        Number of discrete states or clusters to identify.  The
        interpretation depends on the chosen method.  Defaults to 4.
    auto_n_states : bool, optional
        If True, the optimal number of states is estimated automatically
        using silhouette scores.  When enabled ``n_states`` is treated as
        an initial guess and replaced by the recommended value.
    method : str, optional
        Method used for state identification.  Supported values are
        ``'kmeans'`` (default), ``'hmm'`` and ``'cap'``.
    cap_threshold : float, optional
        Threshold in standard deviations for selecting high amplitude
        events in CAP analysis.  Defaults to 1.5.
    output_dir : str | None, optional
        If provided, results from :class:`brainnet.dynamic.analyzer.DynamicAnalyzer`
        will be written to this directory using helpers in
        :mod:`brainnet.dynamic.io`.
    random_state : int | None, optional
        Random seed for initialising clustering algorithms.  Defaults
        to None, meaning no specific seed is set.
    """

    window_length: int
    step: int
    n_states: int = 4
    auto_n_states: bool = False
    method: str = 'kmeans'
    cap_threshold: float = 1.5
    output_dir: Optional[str] = None
    random_state: Optional[int] = None

    def validate(self, n_timepoints: int) -> None:
        """Validate configuration parameters relative to input data.

        Parameters
        ----------
        n_timepoints : int
            Number of time points in the ROI time series.

        Raises
        ------
        ValueError
            If window_length or step are invalid or exceed the data
            length, or if an unsupported method is specified.
        """
        if self.window_length <= 0 or self.step <= 0:
            raise ValueError("window_length and step must be positive integers")
        if self.window_length > n_timepoints:
            raise ValueError("window_length exceeds number of time points in data")
        if self.method not in {'kmeans', 'hmm', 'cap'}:
            raise ValueError(f"Unknown dynamic method '{self.method}'")
