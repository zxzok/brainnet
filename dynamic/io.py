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


def load_state_sequence(input_dir: str | Path) -> np.ndarray:
    """Load a saved state sequence from ``input_dir``.

    The function will look for a binary ``.npy`` file first for efficient
    loading and fall back to a CSV file if necessary.

    Parameters
    ----------
    input_dir : str or Path
        Directory containing ``state_sequence`` files.

    Returns
    -------
    np.ndarray
        One-dimensional array of state labels.
    """

    inp = Path(input_dir)
    npy = inp / "state_sequence.npy"
    csv = inp / "state_sequence.csv"
    if npy.exists():
        return np.load(npy)
    if csv.exists():
        data = np.loadtxt(csv, delimiter=",")
        return data.astype(int)
    raise FileNotFoundError("No state sequence file found in" f" {input_dir}.")


def load_state_matrices(input_dir: str | Path) -> np.ndarray:
    """Load saved state matrices from ``input_dir``.

    Parameters
    ----------
    input_dir : str or Path
        Directory containing ``state_matrices`` files.

    Returns
    -------
    np.ndarray
        Array of state patterns with the same shape as originally saved.
    """

    inp = Path(input_dir)
    npy = inp / "state_matrices.npy"
    csv = inp / "state_matrices.csv"
    if npy.exists():
        return np.load(npy)
    if csv.exists():
        data = np.loadtxt(csv, delimiter=",")
        n_states, n_features = data.shape
        root = int(np.sqrt(n_features))
        if root * root == n_features:
            return data.reshape(n_states, root, root)
        return data
    raise FileNotFoundError("No state matrices file found in" f" {input_dir}.")


def load_metrics(input_dir: str | Path) -> DynamicMetrics:
    """Load :class:`DynamicMetrics` from ``input_dir``.

    Parameters
    ----------
    input_dir : str or Path
        Directory containing the saved metrics files.

    Returns
    -------
    DynamicMetrics
        Metrics object with the same fields as originally saved.
    """

    inp = Path(input_dir)
    npz = inp / "metrics.npz"
    if npz.exists():
        data = np.load(npz)
        return DynamicMetrics(
            occupancy=data["occupancy"],
            mean_dwell_time=data["mean_dwell_time"],
            transition_matrix=data["transition_matrix"],
            n_transitions=int(data["n_transitions"]),
        )

    occ = np.loadtxt(inp / "occupancy.csv", delimiter=",")
    dwell = np.loadtxt(inp / "mean_dwell_time.csv", delimiter=",")
    trans = np.loadtxt(inp / "transition_matrix.csv", delimiter=",")
    n_trans = np.loadtxt(inp / "n_transitions.csv", delimiter=",")
    if np.ndim(n_trans) > 0:
        n_trans = int(np.squeeze(n_trans))
    else:
        n_trans = int(n_trans)
    return DynamicMetrics(
        occupancy=occ,
        mean_dwell_time=dwell,
        transition_matrix=trans,
        n_transitions=n_trans,
    )


__all__ = [
    "save_state_sequence",
    "load_state_sequence",
    "save_state_matrices",
    "load_state_matrices",
    "save_metrics",
    "load_metrics",
]
