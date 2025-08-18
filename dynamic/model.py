"""
brainnet.dynamic.model
=======================

This module defines dataclasses to encapsulate the results of dynamic
functional connectivity analyses.  Two primary objects are provided:

``DynamicMetrics``
    Stores quantitative summaries of a discrete state sequence such as
    how long the system spends in each state, how long it dwells
    consecutively in a given state, the transition probabilities
    between states and the number of observed transitions.

``DynamicStateModel``
    Represents the outcome of a dynamic analysis method (e.g. K‑means
    clustering, hidden Markov model or co‑activation patterns).  It
    contains the identified state patterns, the state assignment for
    each time point or window, the associated metrics and a freeform
    dictionary for method‑specific additional outputs.

The definitions here mirror the structures used in the legacy
``dynamic_analysis`` module but are separated to allow reuse by
multiple analysis routines implemented in other submodules.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np


@dataclass
class DynamicMetrics:
    """Quantify temporal properties of a discrete state sequence.

    Attributes
    ----------
    occupancy : np.ndarray
        Fraction of time (or number of windows) spent in each state.
        Shape ``(n_states,)``.
    mean_dwell_time : np.ndarray
        Average consecutive duration of each state.  Shape
        ``(n_states,)``.
    transition_matrix : np.ndarray
        Matrix of size ``(n_states, n_states)`` where entry ``(i, j)``
        gives the probability of transitioning from state ``i`` to
        state ``j``.  Rows sum to one for states that are visited.
    n_transitions : int
        Total number of state changes observed in the sequence.
    """

    occupancy: np.ndarray
    mean_dwell_time: np.ndarray
    transition_matrix: np.ndarray
    n_transitions: int


@dataclass
class DynamicStateModel:
    """Encapsulate the result of a dynamic connectivity analysis.

    Parameters
    ----------
    method : str
        Name of the method used to identify states (e.g. ``'kmeans'``,
        ``'hmm'`` or ``'cap'``).
    n_states : int
        Number of discrete states identified.
    states : List[np.ndarray | Any]
        List of state patterns.  For clustering methods this will
        contain connectivity matrices (shape ``N_ROI×N_ROI``) or
        activation vectors (length ``N_ROI``).  For HMMs this may
        include dictionaries with parameters such as means and
        covariances.
    state_sequence : np.ndarray
        One‑dimensional array indicating which state is active at
        each time point or window.  Length equals the number of
        windows (for clustering/HMM) or the number of time points
        (for CAP).  Entries should be integers in the range
        ``0..n_states-1``; negative values denote undefined or
        unassigned time points.
    metrics : DynamicMetrics
        Summary statistics describing the temporal dynamics of the
        state sequence.
    extra : Dict[str, Any]
        Dictionary for optional method‑specific outputs.  This is
        deliberately untyped to allow storing arbitrary additional
        information (e.g. raw window matrices, HMM objects, event
        indices).  Consumers should consult the documentation of
        individual analysis functions for details on possible keys.
    template : str | None
        Name of the template/parcellation used for ROI extraction, if
        applicable.
    """

    method: str
    n_states: int
    states: List[Any]
    state_sequence: np.ndarray
    metrics: DynamicMetrics
    extra: Dict[str, Any] = field(default_factory=dict)
    template: Optional[str] = None
