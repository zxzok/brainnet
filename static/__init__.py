"""
brainnet.static
================

This subpackage contains components for computing static functional
connectivity and deriving graph‑theoretic metrics on brain networks.
The code is organised into modules that separate the core data
structures (connectivity matrices and graph metrics) from the
computations themselves.  Users can import individual functions or
classes from these modules or use the high level ``StaticAnalyzer``
class to tie the pieces together.

Modules
-------

connectivity
    Defines the :class:`ConnectivityMatrix` dataclass and functions
    for computing Pearson correlation matrices between ROI time
    series.

metrics
    Defines the :class:`GraphMetrics` dataclass and functions for
    calculating node and global graph measures (degree, clustering,
    global efficiency).

analyzer
    Provides :class:`StaticAnalyzer`, a convenience class that
    computes a connectivity matrix, applies thresholding and
    derives graph metrics in one call.
"""

from .connectivity import ConnectivityMatrix, compute_pearson_connectivity
from .metrics import GraphMetrics, compute_degree, compute_clustering, compute_global_efficiency
from .analyzer import StaticAnalyzer

__all__ = [
    'ConnectivityMatrix',
    'compute_pearson_connectivity',
    'GraphMetrics',
    'compute_degree',
    'compute_clustering',
    'compute_global_efficiency',
    'StaticAnalyzer',
]