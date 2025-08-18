"""
brainnet.dynamic
================

This subpackage provides tools for analysing time‑varying functional
connectivity.  The code is broken into modules that handle
configuration, sliding window construction, state identification via
different methods and computation of temporal metrics.  The high
level :class:`DynamicAnalyzer` class orchestrates these components
based on user‑provided configuration.

Modules
-------

config
    Defines the :class:`DynamicConfig` dataclass used to specify
    parameters for dynamic analysis.

model
    Defines :class:`DynamicMetrics` and :class:`DynamicStateModel` to
    encapsulate the results of dynamic analysis.

window
    Provides functions to compute connectivity matrices for sliding
    windows and associated summary metrics.

metrics
    Provides functions to compute temporal statistics (occupancy,
    dwell time, transition probabilities) from a state sequence.

kmeans, hmm, cap
    Implement specific methods for state identification via K‑means
    clustering, Gaussian HMMs or co‑activation pattern (CAP) analysis.

analyzer
    Contains :class:`DynamicAnalyzer`, a high level interface that
    coordinates sliding window computation, state identification
    according to the chosen method and computation of temporal metrics.
"""

from .config import DynamicConfig
from .model import DynamicMetrics, DynamicStateModel
from .analyzer import DynamicAnalyzer

__all__ = [
    'DynamicConfig',
    'DynamicMetrics',
    'DynamicStateModel',
    'DynamicAnalyzer',
]