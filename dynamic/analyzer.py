"""
brainnet.dynamic.analyzer
========================

This module defines the :class:`DynamicAnalyzer` class, a high level
interface for performing dynamic functional connectivity analysis on
ROI time series.  It uses the configuration supplied via
:class:`brainnet.dynamic.config.DynamicConfig` to select the
appropriate analysis method (K‑means, HMM or CAP), compute sliding
window connectivity where necessary, identify discrete brain states
and derive summary metrics of the temporal dynamics.

The class delegates to specialised functions in the :mod:`kmeans`,
:mod:`hmm` and :mod:`cap` modules depending on the chosen method.
Users can instantiate ``DynamicAnalyzer`` with a configuration and
then call :meth:`analyse` on ROI time series to obtain a
:class:`brainnet.dynamic.model.DynamicStateModel` object.
"""

from __future__ import annotations

import numpy as np

from .config import DynamicConfig
from .model import DynamicStateModel
from .kmeans import kmeans_analysis
from .hmm import hmm_analysis
from .cap import cap_analysis


class DynamicAnalyzer:
    """High level wrapper for dynamic connectivity analysis.

    Parameters
    ----------
    config : DynamicConfig
        Configuration specifying window length, step size, number of
        states, analysis method and other options.

    Examples
    --------
    >>> from brainnet.dynamic import DynamicConfig, DynamicAnalyzer
    >>> cfg = DynamicConfig(window_length=30, step=5, n_states=4, method='kmeans')
    >>> analyzer = DynamicAnalyzer(cfg)
    >>> result = analyzer.analyse(roi_timeseries)
    >>> print(result.metrics.occupancy)
    """

    def __init__(self, config: DynamicConfig) -> None:
        self.config = config

    # --------------------------------------------------------------
    def analyse(self, roi_timeseries: np.ndarray) -> DynamicStateModel:
        """Run dynamic analysis on ROI time series.

        This method inspects the ``method`` field of the configuration
        to determine which analysis function to call.  Supported
        methods are:

        * ``'kmeans'`` – sliding window K‑means clustering.
        * ``'hmm'`` – Gaussian hidden Markov model fitting.
        * ``'cap'`` – co‑activation pattern analysis.

        Parameters
        ----------
        roi_timeseries : np.ndarray
            Array of shape (T, N_ROI) representing ROI signals.

        Returns
        -------
        DynamicStateModel
            The result of the selected dynamic analysis method.

        Raises
        ------
        ValueError
            If the configured method is unknown.
        """
        method = self.config.method
        if method == 'kmeans':
            return kmeans_analysis(roi_timeseries, self.config)
        elif method == 'hmm':
            return hmm_analysis(roi_timeseries, self.config)
        elif method == 'cap':
            return cap_analysis(roi_timeseries, self.config)
        else:
            raise ValueError(f"Unknown dynamic analysis method '{method}'")


__all__ = [
    'DynamicAnalyzer',
]
