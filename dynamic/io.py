"""Utility helpers for saving dynamic analysis outputs.

This module provides functions to persist results from dynamic
connectivity analyses to disk. Each helper writes data in both CSV and
NumPy formats for easy inspection and efficient reloading.
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np

from .model import DynamicMetrics


def _ensure_dir(path: Path) -> None:
    """Create directory if it does not already exist."""
    path.mkdir(parents=True, exist_ok=True)


def save_state_sequence(sequence: np.ndarray, output_dir: str | Path) -> None:
    """Save a state sequence to ``output_dir``.

    Parameters
    ----------
    sequence : np.ndarray
        One-dimensional array of state labels.
    output_dir : str or Path
        Destination directory. It will be created if necessary.
    """
    out = Path(output_dir)
    _ensure_dir(out)
    np.savetxt(out / "state_sequence.csv", sequence, fmt="%d", delimiter=",")
    np.save(out / "state_sequence.npy", sequence)


def save_state_matrices(matrices: Sequence[np.ndarray], output_dir: str | Path) -> None:
    """Persist state connectivity/activation matrices.

    The matrices are stacked into a single array before saving. A CSV
    file containing a row per state with the flattened values is written
    alongside a binary ``.npy`` file retaining the original shape.

    Parameters
    ----------
    matrices : sequence of np.ndarray
        List or array of state patterns.
    output_dir : str or Path
        Destination directory. It will be created if necessary.
    """
    out = Path(output_dir)
    _ensure_dir(out)
    arr = np.asarray(matrices)
    np.save(out / "state_matrices.npy", arr)
    flat = arr.reshape(arr.shape[0], -1)
    np.savetxt(out / "state_matrices.csv", flat, delimiter=",")


def save_metrics(metrics: DynamicMetrics, output_dir: str | Path) -> None:
    """Save summary metrics describing the state sequence.

    Parameters
    ----------
    metrics : DynamicMetrics
        Dataclass containing occupancy, mean dwell time, transition
        matrix and number of transitions.
    output_dir : str or Path
        Destination directory. It will be created if necessary.
    """
    out = Path(output_dir)
    _ensure_dir(out)
    np.savetxt(out / "occupancy.csv", metrics.occupancy, delimiter=",")
    np.savetxt(out / "mean_dwell_time.csv", metrics.mean_dwell_time, delimiter=",")
    np.savetxt(out / "transition_matrix.csv", metrics.transition_matrix, delimiter=",")
    np.savetxt(out / "n_transitions.csv", np.array([metrics.n_transitions]), fmt="%d", delimiter=",")
    np.savez(
        out / "metrics.npz",
        occupancy=metrics.occupancy,
        mean_dwell_time=metrics.mean_dwell_time,
        transition_matrix=metrics.transition_matrix,
        n_transitions=metrics.n_transitions,
    )


__all__ = [
    "save_state_sequence",
    "save_state_matrices",
    "save_metrics",
]
