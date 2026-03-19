"""Tool definitions and dispatcher for the BrainNet orchestrator."""

from __future__ import annotations

from typing import Any

import numpy as np

from brainnet.session_store import SessionStore


# ---------------------------------------------------------------------------
# Tool JSON schema definitions (Anthropic tool-use format)
# ---------------------------------------------------------------------------

TOOL_DEFINITIONS: list[dict[str, Any]] = [
    {
        "name": "load_dataset",
        "description": "Load an fMRI dataset from OpenNeuro or a local BIDS path.",
        "input_schema": {
            "type": "object",
            "properties": {
                "openneuro_id": {
                    "type": "string",
                    "description": "OpenNeuro dataset accession ID (e.g. ds000228).",
                },
                "local_path": {
                    "type": "string",
                    "description": "Absolute path to a local BIDS dataset directory.",
                },
            },
            "required": [],
        },
    },
    {
        "name": "list_subjects",
        "description": "List available subjects in a loaded BIDS dataset.",
        "input_schema": {
            "type": "object",
            "properties": {
                "dataset_path": {
                    "type": "string",
                    "description": "Path to the BIDS dataset directory.",
                },
            },
            "required": ["dataset_path"],
        },
    },
    {
        "name": "preprocess",
        "description": "Run preprocessing on a functional MRI file: smoothing, bandpass filtering, and ROI extraction.",
        "input_schema": {
            "type": "object",
            "properties": {
                "input_path": {
                    "type": "string",
                    "description": "Path to the input NIfTI file.",
                },
                "smoothing_fwhm": {
                    "type": "number",
                    "description": "Smoothing kernel FWHM in mm.",
                },
                "bandpass_low": {
                    "type": "number",
                    "description": "Low-frequency cutoff for bandpass filter in Hz.",
                },
                "bandpass_high": {
                    "type": "number",
                    "description": "High-frequency cutoff for bandpass filter in Hz.",
                },
                "atlas": {
                    "type": "string",
                    "description": "Atlas name for ROI extraction (e.g. aal, schaefer100).",
                },
            },
            "required": ["input_path", "smoothing_fwhm", "bandpass_low", "bandpass_high", "atlas"],
        },
    },
    {
        "name": "inspect_qc",
        "description": "Inspect quality-control metrics for preprocessed ROI time series, including SNR and outlier detection.",
        "input_schema": {
            "type": "object",
            "properties": {
                "roi_timeseries_id": {
                    "type": "string",
                    "description": "Session ID of the stored ROI time-series data.",
                },
            },
            "required": ["roi_timeseries_id"],
        },
    },
    {
        "name": "static_connectivity",
        "description": "Compute a static functional connectivity matrix from ROI time series.",
        "input_schema": {
            "type": "object",
            "properties": {
                "roi_timeseries_id": {
                    "type": "string",
                    "description": "Session ID of the stored ROI time-series data.",
                },
                "method": {
                    "type": "string",
                    "description": "Connectivity estimation method (e.g. pearson, partial, tangent).",
                },
            },
            "required": ["roi_timeseries_id"],
        },
    },
    {
        "name": "dynamic_connectivity",
        "description": "Compute dynamic functional connectivity using sliding windows and state clustering.",
        "input_schema": {
            "type": "object",
            "properties": {
                "roi_timeseries_id": {
                    "type": "string",
                    "description": "Session ID of the stored ROI time-series data.",
                },
                "method": {
                    "type": "string",
                    "description": "Dynamic connectivity method (e.g. sliding_window, hmm).",
                },
                "n_states": {
                    "type": "integer",
                    "description": "Number of connectivity states to identify.",
                },
                "window_length": {
                    "type": "integer",
                    "description": "Sliding window length in TRs.",
                },
                "step": {
                    "type": "integer",
                    "description": "Sliding window step size in TRs.",
                },
            },
            "required": ["roi_timeseries_id", "method", "n_states", "window_length", "step"],
        },
    },
    {
        "name": "extract_features",
        "description": "Extract graph-theoretic and summary features from a completed analysis strategy.",
        "input_schema": {
            "type": "object",
            "properties": {
                "strategy_id": {
                    "type": "string",
                    "description": "Session ID of the stored strategy result.",
                },
            },
            "required": ["strategy_id"],
        },
    },
    {
        "name": "compare_strategies",
        "description": "Compare multiple analysis strategies and produce a summary table.",
        "input_schema": {
            "type": "object",
            "properties": {
                "strategy_ids": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "List of strategy IDs to compare.",
                },
            },
            "required": ["strategy_ids"],
        },
    },
    {
        "name": "generate_report",
        "description": "Generate a summary report for one or more analysis strategies.",
        "input_schema": {
            "type": "object",
            "properties": {
                "strategy_ids": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "List of strategy IDs to include in the report.",
                },
            },
            "required": [],
        },
    },
]


# ---------------------------------------------------------------------------
# Tool implementations
# ---------------------------------------------------------------------------

def _tool_inspect_qc(session: SessionStore, params: dict) -> dict:
    roi_data = session.get_roi_data(params["roi_timeseries_id"])
    if roi_data is None:
        return {"error": f"ROI data not found: {params['roi_timeseries_id']}"}
    ts = roi_data.timeseries
    snr = (np.mean(ts, axis=0) / np.maximum(np.std(ts, axis=0), 1e-10)).tolist()
    global_signal = np.mean(ts, axis=1)
    z = np.abs((global_signal - np.mean(global_signal)) / max(np.std(global_signal), 1e-10))
    outliers = [int(i) for i in np.where(z > 3.0)[0]]
    return {
        "motion_summary": roi_data.qc,
        "snr_per_roi": [round(s, 2) for s in snr],
        "outlier_timepoints": outliers,
    }


def _stub(session: SessionStore, params: dict) -> dict:
    return {"error": "Not yet implemented"}


# ---------------------------------------------------------------------------
# Dispatcher
# ---------------------------------------------------------------------------

_TOOL_DISPATCH: dict[str, Any] = {
    "load_dataset": _stub,
    "list_subjects": _stub,
    "preprocess": _stub,
    "inspect_qc": _tool_inspect_qc,
    "static_connectivity": _stub,
    "dynamic_connectivity": _stub,
    "extract_features": _stub,
    "compare_strategies": _stub,
    "generate_report": _stub,
}


def execute_tool(session: SessionStore, tool_name: str, tool_input: dict) -> dict:
    """Dispatch a tool call to the appropriate handler."""
    fn = _TOOL_DISPATCH.get(tool_name)
    if fn is None:
        return {"error": f"Unknown tool: {tool_name}"}
    try:
        return fn(session, tool_input)
    except Exception as exc:
        return {"error": str(exc)}
