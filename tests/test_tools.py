from __future__ import annotations

import json

import numpy as np
import pytest

from brainnet.session_store import SessionManager, SessionStore, RoiData
from brainnet.tools import (
    TOOL_DEFINITIONS,
    execute_tool,
)


def test_tool_definitions_are_valid_json_schemas():
    """Every tool definition has name, description, and input_schema."""
    assert len(TOOL_DEFINITIONS) == 9
    for tool in TOOL_DEFINITIONS:
        assert "name" in tool
        assert "description" in tool
        assert "input_schema" in tool
        schema = tool["input_schema"]
        assert schema.get("type") == "object"
        assert "properties" in schema


def test_tool_names():
    names = {t["name"] for t in TOOL_DEFINITIONS}
    expected = {
        "load_dataset",
        "list_subjects",
        "preprocess",
        "inspect_qc",
        "static_connectivity",
        "dynamic_connectivity",
        "extract_features",
        "compare_strategies",
        "generate_report",
    }
    assert names == expected


def _make_session_with_roi() -> tuple[SessionStore, str]:
    """Helper: create a session with synthetic ROI data."""
    session = SessionStore(session_id="test")
    roi = RoiData(
        timeseries=np.random.randn(50, 10),
        labels=[f"ROI{i}" for i in range(10)],
        qc={"tsnr": 45.0},
        source_path="/tmp/test.nii.gz",
    )
    rid = session.store_roi_data(roi)
    return session, rid


def test_inspect_qc_returns_metrics():
    session, rid = _make_session_with_roi()
    result = execute_tool(session, "inspect_qc", {"roi_timeseries_id": rid})
    assert "snr_per_roi" in result
    assert "outlier_timepoints" in result
    assert len(result["snr_per_roi"]) == 10


def test_inspect_qc_missing_roi():
    session = SessionStore(session_id="test")
    result = execute_tool(session, "inspect_qc", {"roi_timeseries_id": "nonexistent"})
    assert "error" in result


def test_unknown_tool_returns_error():
    session = SessionStore(session_id="test")
    result = execute_tool(session, "nonexistent_tool", {})
    assert "error" in result
