"""Tool definitions and dispatcher for the BrainNet orchestrator."""

from __future__ import annotations

from typing import Any

import numpy as np

from brainnet.session_store import SessionStore, RoiData, StrategyResult


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


def _tool_static_connectivity(session: SessionStore, params: dict) -> dict:
    method = params.get("method", "pearson")
    if method != "pearson":
        return {"error": f"Method '{method}' not yet supported. Only 'pearson' is available."}

    roi_data = session.get_roi_data(params["roi_timeseries_id"])
    if roi_data is None:
        return {"error": f"ROI data not found: {params['roi_timeseries_id']}"}

    try:
        from brainnet.static_analysis import StaticAnalyzer
    except ImportError:
        return {"error": "Static analysis dependencies not installed. Run: pip install -e .[analysis]"}

    analyzer = StaticAnalyzer()
    conn_matrix = analyzer.compute_connectivity(roi_data.timeseries, roi_data.labels, method=method)
    graph_metrics = analyzer.compute_graph_metrics(conn_matrix)
    metrics = graph_metrics.global_metrics

    strategy = StrategyResult(
        strategy_type="static",
        method=method,
        params={"method": method},
        artifacts=conn_matrix,
        metrics_summary=metrics,
    )
    sid = session.store_strategy(strategy)
    return {"strategy_id": sid, "global_metrics": metrics}


def _tool_dynamic_connectivity(session: SessionStore, params: dict) -> dict:
    roi_data = session.get_roi_data(params["roi_timeseries_id"])
    if roi_data is None:
        return {"error": f"ROI data not found: {params['roi_timeseries_id']}"}

    try:
        from brainnet.dynamic_analysis import DynamicConfig, DynamicAnalyzer
    except ImportError:
        return {"error": "Dynamic analysis dependencies not installed. Run: pip install -e .[analysis]"}

    config = DynamicConfig(
        method=params.get("method", "kmeans"),
        n_states=params.get("n_states", 4),
        window_length=params.get("window_length", 30),
        step=params.get("step", 10),
    )
    analyzer = DynamicAnalyzer(config)
    artifacts = analyzer.analyse(roi_data.timeseries)

    dm = artifacts.metrics
    seq_len = max(len(artifacts.state_sequence), 1)
    switching_rate = float(dm.n_transitions) / seq_len
    occ = dm.occupancy
    occupancy_entropy = float(-np.sum(occ * np.log(occ + 1e-10)))
    metrics = {
        "n_states": config.n_states,
        "switching_rate": round(switching_rate, 4),
        "occupancy_entropy": round(occupancy_entropy, 4),
        "mean_dwell_time": [round(float(d), 2) for d in dm.mean_dwell_time],
        "n_transitions": dm.n_transitions,
    }

    strategy = StrategyResult(
        strategy_type="dynamic",
        method=config.method,
        params={"method": config.method, "n_states": config.n_states,
                "window_length": config.window_length, "step": config.step},
        artifacts=artifacts,
        metrics_summary=metrics,
    )
    sid = session.store_strategy(strategy)
    return {"strategy_id": sid, "n_states": config.n_states,
            "switching_rate": metrics["switching_rate"],
            "occupancy_entropy": metrics["occupancy_entropy"]}


def _tool_extract_features(session: SessionStore, params: dict) -> dict:
    strategy = session.get_strategy(params["strategy_id"])
    if strategy is None:
        return {"error": f"Strategy not found: {params['strategy_id']}"}
    return {
        "strategy_id": params["strategy_id"],
        "strategy_type": strategy.strategy_type,
        "method": strategy.method,
        "features": strategy.metrics_summary,
    }


def _tool_compare_strategies(session: SessionStore, params: dict) -> dict:
    from brainnet.strategy_compare import compare_strategies as _compare

    strategy_ids = params.get("strategy_ids", [])
    strategies = {}
    for sid in strategy_ids:
        s = session.get_strategy(sid)
        if s is None:
            return {"error": f"Strategy not found: {sid}"}
        strategies[sid] = s

    try:
        result = _compare(strategies)
    except ValueError as exc:
        return {"error": str(exc)}

    return {
        "comparison_table": result.comparison_table,
        "metric_differences": result.metric_differences,
        "ranking_axes": result.ranking_axes,
        "recommended_strategy": result.recommended_strategy,
        "confidence": result.confidence,
        "reasoning_hints": result.reasoning_hints,
        "limitations": result.limitations,
    }


def _tool_load_dataset(session: SessionStore, params: dict) -> dict:
    try:
        from brainnet.data_management import DatasetIndex
    except ImportError:
        return {"error": "Data management dependencies not installed."}

    local_path = params.get("local_path")
    openneuro_id = params.get("openneuro_id")

    try:
        if local_path:
            idx = DatasetIndex(local_path)
            return {"dataset_path": local_path, "n_subjects": len(idx.list_subjects()), "status": "loaded"}
        elif openneuro_id:
            idx = DatasetIndex(openneuro_id, source="openneuro")
            return {"dataset_path": openneuro_id, "n_subjects": len(idx.list_subjects()), "status": "loaded"}
        else:
            return {"error": "Provide either 'local_path' or 'openneuro_id'."}
    except Exception as exc:
        return {"error": str(exc), "suggestion": "Check dataset ID or local path."}


def _tool_list_subjects(session: SessionStore, params: dict) -> dict:
    try:
        from brainnet.data_management import DatasetIndex
    except ImportError:
        return {"error": "Data management dependencies not installed."}

    try:
        idx = DatasetIndex(params["dataset_path"])
        subjects = idx.list_subjects()
        return {"subjects": subjects, "count": len(subjects)}
    except Exception as exc:
        return {"error": str(exc)}


def _tool_preprocess(session: SessionStore, params: dict) -> dict:
    try:
        from brainnet.preprocessing_full import (
            PreprocessPipelineConfig,
            PreprocessPipeline,
            SmoothingConfig,
            TemporalFilterConfig,
            RoiExtractionConfig,
        )
    except ImportError:
        return {"error": "Preprocessing dependencies not installed. Run: pip install -e .[analysis]"}

    config = PreprocessPipelineConfig(
        smoothing=SmoothingConfig(
            enabled=True,
            fwhm=params.get("smoothing_fwhm", 6.0),
        ),
        temporal_filter=TemporalFilterConfig(
            enabled=True,
            low_cut=params.get("bandpass_low", 0.01),
            high_cut=params.get("bandpass_high", 0.1),
        ),
        roi_extraction=RoiExtractionConfig(
            enabled=True,
            atlas_path=params.get("atlas", "default"),
        ),
    )

    try:
        pipeline = PreprocessPipeline(config)
        result = pipeline.run(params["input_path"])
        roi_ts = result.get("roi_timeseries")
        labels = result.get("roi_labels", [])
        qc = result.get("qc_metrics", {})
        if roi_ts is None:
            return {"error": "Preprocessing completed but no ROI timeseries extracted. Check atlas configuration."}
        roi = RoiData(
            timeseries=roi_ts,
            labels=labels,
            qc=qc,
            source_path=params["input_path"],
        )
        rid = session.store_roi_data(roi)
        return {
            "roi_timeseries_id": rid,
            "n_rois": len(labels),
            "n_timepoints": roi_ts.shape[0],
            "qc_summary": qc,
        }
    except Exception as exc:
        return {"error": str(exc), "suggestion": "Check input path and preprocessing parameters."}


def _tool_generate_report(session: SessionStore, params: dict) -> dict:
    strategy_ids = params.get("strategy_ids", list(session.strategies.keys()))
    report_lines = ["# Analysis Report", ""]

    for sid in strategy_ids:
        strategy = session.get_strategy(sid)
        if strategy is None:
            continue
        report_lines.append(f"## Strategy: {sid}")
        report_lines.append(f"- Type: {strategy.strategy_type}")
        report_lines.append(f"- Method: {strategy.method}")
        report_lines.append(f"- Params: {strategy.params}")
        report_lines.append(f"- Metrics: {strategy.metrics_summary}")
        report_lines.append("")

    report_lines.append("---")
    report_lines.append("*This report is auto-generated. All findings are candidate observations.*")

    report_text = "\n".join(report_lines)
    return {"report": report_text, "n_strategies": len(strategy_ids)}


# ---------------------------------------------------------------------------
# Dispatcher
# ---------------------------------------------------------------------------

_TOOL_DISPATCH: dict[str, Any] = {
    "inspect_qc": _tool_inspect_qc,
    "static_connectivity": _tool_static_connectivity,
    "dynamic_connectivity": _tool_dynamic_connectivity,
    "extract_features": _tool_extract_features,
    "compare_strategies": _tool_compare_strategies,
    "load_dataset": _tool_load_dataset,
    "list_subjects": _tool_list_subjects,
    "preprocess": _tool_preprocess,
    "generate_report": _tool_generate_report,
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
