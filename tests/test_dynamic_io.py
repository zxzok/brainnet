"""Tests for dynamic analysis IO helpers."""

from __future__ import annotations

from pathlib import Path
import importlib.util
import types
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]

# Create a lightweight package so relative imports work
package = types.ModuleType("dynamic")
package.__path__ = [str(ROOT / "dynamic")]
sys.modules.setdefault("dynamic", package)

model_spec = importlib.util.spec_from_file_location("dynamic.model", ROOT / "dynamic" / "model.py")
model = importlib.util.module_from_spec(model_spec)
sys.modules["dynamic.model"] = model
model_spec.loader.exec_module(model)

io_spec = importlib.util.spec_from_file_location("dynamic.io", ROOT / "dynamic" / "io.py")
io = importlib.util.module_from_spec(io_spec)
sys.modules["dynamic.io"] = io
io_spec.loader.exec_module(io)

save_state_sequence = io.save_state_sequence
load_state_sequence = io.load_state_sequence
save_state_matrices = io.save_state_matrices
load_state_matrices = io.load_state_matrices
save_metrics = io.save_metrics
load_metrics = io.load_metrics
DynamicMetrics = model.DynamicMetrics


def test_state_sequence_roundtrip(tmp_path: Path) -> None:
    seq = np.array([0, 1, 2, 1, 0])
    save_state_sequence(seq, tmp_path)

    # Load from npy
    loaded = load_state_sequence(tmp_path)
    assert np.array_equal(seq, loaded)

    # Load from csv fallback
    (tmp_path / "state_sequence.npy").unlink()
    loaded_csv = load_state_sequence(tmp_path)
    assert np.array_equal(seq, loaded_csv)


def test_state_matrices_roundtrip(tmp_path: Path) -> None:
    mats = [np.array([[1, 2], [3, 4]]), np.array([[5, 6], [7, 8]])]
    save_state_matrices(mats, tmp_path)

    loaded = load_state_matrices(tmp_path)
    assert np.array_equal(np.asarray(mats), loaded)

    # CSV fallback should reconstruct square matrices
    (tmp_path / "state_matrices.npy").unlink()
    loaded_csv = load_state_matrices(tmp_path)
    assert np.array_equal(np.asarray(mats), loaded_csv)


def test_metrics_roundtrip(tmp_path: Path) -> None:
    metrics = DynamicMetrics(
        occupancy=np.array([0.5, 0.5]),
        mean_dwell_time=np.array([2.0, 2.0]),
        transition_matrix=np.array([[0.0, 1.0], [1.0, 0.0]]),
        n_transitions=4,
    )
    save_metrics(metrics, tmp_path)

    loaded = load_metrics(tmp_path)
    assert np.array_equal(metrics.occupancy, loaded.occupancy)
    assert np.array_equal(metrics.mean_dwell_time, loaded.mean_dwell_time)
    assert np.array_equal(metrics.transition_matrix, loaded.transition_matrix)
    assert metrics.n_transitions == loaded.n_transitions

    # CSV/NPY fallback
    (tmp_path / "metrics.npz").unlink()
    loaded_csv = load_metrics(tmp_path)
    assert np.array_equal(metrics.occupancy, loaded_csv.occupancy)
    assert np.array_equal(metrics.mean_dwell_time, loaded_csv.mean_dwell_time)
    assert np.array_equal(metrics.transition_matrix, loaded_csv.transition_matrix)
    assert metrics.n_transitions == loaded_csv.n_transitions

