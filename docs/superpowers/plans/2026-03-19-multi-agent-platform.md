# Multi-Agent Platform Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a Claude-powered analysis orchestrator to BrainNet that lets users describe goals in natural language, automatically runs 2-3 analysis strategies, compares results, and recommends the best approach — all via a web chat interface.

**Architecture:** Single Claude agent (Anthropic SDK) receives user goals, calls tool-wrapped existing analysis functions, stores intermediate results in an in-memory SessionStore, and streams responses to a Flask chat blueprint via SSE-formatted streaming fetch responses.

**Tech Stack:** Python 3.10+, Flask, Anthropic SDK, existing BrainNet modules (preprocessing_full, static, dynamic, visualization)

**Spec:** `docs/superpowers/specs/2026-03-19-multi-agent-platform-design.md`

---

## File Map

| File | Responsibility | Task |
|------|---------------|------|
| `session_store.py` | In-memory session state: ROI data, strategy results, conversation history, eviction | 1 |
| `strategy_compare.py` | Cross-strategy metric comparison, 3-axis ranking, confidence levels | 2 |
| `tools.py` | 9 tool JSON schemas + Python implementations + adapter logic | 3, 4 |
| `orchestrator.py` | Claude API conversation loop, tool dispatch, token budget, SSE formatting | 5 |
| `web_chat.py` | Flask blueprint: chat routes, SSE streaming, session management | 6 |
| `templates/chat.html` | Chat UI: messages, tool indicators, streaming, dark theme | 7 |
| `web_app.py` | (modify) Register chat blueprint, add nav link | 6 |
| `pyproject.toml` | (modify) Add `anthropic` to `[llm]` extra | 3 |

---

### Task 1: SessionStore

**Files:**
- Create: `session_store.py`
- Test: `tests/test_session_store.py`

- [ ] **Step 1: Write failing tests for SessionStore**

```python
# tests/test_session_store.py
from __future__ import annotations

import time

import numpy as np
import pytest

from brainnet.session_store import RoiData, SessionManager, SessionStore, StrategyResult


def test_create_session():
    mgr = SessionManager(max_sessions=10, timeout_seconds=1800)
    session = mgr.create_session()
    assert session.session_id
    assert session.roi_data == {}
    assert session.strategies == {}
    assert session.messages == []


def test_get_session():
    mgr = SessionManager(max_sessions=10, timeout_seconds=1800)
    session = mgr.create_session()
    retrieved = mgr.get_session(session.session_id)
    assert retrieved is session


def test_get_nonexistent_session():
    mgr = SessionManager(max_sessions=10, timeout_seconds=1800)
    assert mgr.get_session("nonexistent") is None


def test_max_sessions_enforced():
    mgr = SessionManager(max_sessions=2, timeout_seconds=1800)
    mgr.create_session()
    mgr.create_session()
    with pytest.raises(RuntimeError, match="Maximum.*sessions"):
        mgr.create_session()


def test_expired_sessions_evicted():
    mgr = SessionManager(max_sessions=10, timeout_seconds=0)
    session = mgr.create_session()
    sid = session.session_id
    time.sleep(0.05)
    mgr.evict_expired()
    assert mgr.get_session(sid) is None


def test_store_and_retrieve_roi_data():
    session = SessionStore(session_id="test")
    roi = RoiData(
        timeseries=np.ones((10, 3)),
        labels=["A", "B", "C"],
        qc={"tsnr": 50.0},
        source_path="/tmp/test.nii.gz",
    )
    rid = session.store_roi_data(roi)
    assert rid.startswith("roi_")
    assert session.get_roi_data(rid) is roi


def test_store_and_retrieve_strategy():
    session = SessionStore(session_id="test")
    strategy = StrategyResult(
        strategy_type="static",
        method="pearson",
        params={"method": "pearson"},
        artifacts=None,
        metrics_summary={"modularity": 0.5},
    )
    sid = session.store_strategy(strategy)
    assert sid.startswith("strategy_")
    assert session.get_strategy(sid) is strategy


def test_estimate_tokens():
    session = SessionStore(session_id="test")
    session.messages = [{"role": "user", "content": "hello world"}]
    tokens = session.estimate_tokens()
    assert tokens > 0
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python3 -m pytest tests/test_session_store.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'brainnet.session_store'`

- [ ] **Step 3: Implement SessionStore**

```python
# session_store.py
"""In-memory session state for the BrainNet orchestrator."""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass
class RoiData:
    """Preprocessed ROI time series and QC metrics."""

    timeseries: np.ndarray
    labels: list[str]
    qc: dict[str, Any]
    source_path: str


@dataclass
class StrategyResult:
    """Output of one analysis strategy."""

    strategy_type: str
    method: str
    params: dict[str, Any]
    artifacts: Any  # AnalysisArtifacts or None
    metrics_summary: dict[str, Any]


@dataclass
class SessionStore:
    """State for a single orchestrator chat session."""

    session_id: str
    roi_data: dict[str, RoiData] = field(default_factory=dict)
    strategies: dict[str, StrategyResult] = field(default_factory=dict)
    messages: list[dict[str, Any]] = field(default_factory=list)
    last_active: float = field(default_factory=time.time)

    def store_roi_data(self, data: RoiData) -> str:
        rid = f"roi_{uuid.uuid4().hex[:8]}"
        self.roi_data[rid] = data
        return rid

    def get_roi_data(self, roi_id: str) -> RoiData | None:
        return self.roi_data.get(roi_id)

    def store_strategy(self, result: StrategyResult) -> str:
        sid = f"strategy_{uuid.uuid4().hex[:8]}"
        self.strategies[sid] = result
        return sid

    def get_strategy(self, strategy_id: str) -> StrategyResult | None:
        return self.strategies.get(strategy_id)

    def estimate_tokens(self) -> int:
        text = str(self.messages)
        return len(text) // 4

    def touch(self) -> None:
        self.last_active = time.time()


class SessionManager:
    """Manage multiple concurrent sessions with limits and eviction."""

    def __init__(self, *, max_sessions: int = 10, timeout_seconds: int = 1800) -> None:
        self._sessions: dict[str, SessionStore] = {}
        self._max_sessions = max_sessions
        self._timeout_seconds = timeout_seconds

    def create_session(self) -> SessionStore:
        self.evict_expired()
        if len(self._sessions) >= self._max_sessions:
            raise RuntimeError(
                f"Maximum {self._max_sessions} sessions reached. "
                "Please wait for an existing session to expire."
            )
        sid = uuid.uuid4().hex
        session = SessionStore(session_id=sid)
        self._sessions[sid] = session
        return session

    def get_session(self, session_id: str) -> SessionStore | None:
        session = self._sessions.get(session_id)
        if session is not None:
            session.touch()
        return session

    def evict_expired(self) -> None:
        now = time.time()
        expired = [
            sid
            for sid, s in self._sessions.items()
            if now - s.last_active > self._timeout_seconds
        ]
        for sid in expired:
            del self._sessions[sid]
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python3 -m pytest tests/test_session_store.py -v`
Expected: All 8 tests PASS

- [ ] **Step 5: Commit**

```bash
git add session_store.py tests/test_session_store.py
git commit -m "feat: add SessionStore for orchestrator state management"
```

---

### Task 2: Strategy Comparison Module

**Files:**
- Create: `strategy_compare.py`
- Test: `tests/test_strategy_compare.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_strategy_compare.py
from __future__ import annotations

import pytest

from brainnet.strategy_compare import compare_strategies, ComparisonResult
from brainnet.session_store import StrategyResult


def _make_static_strategy(modularity: float = 0.5, clustering: float = 0.3) -> StrategyResult:
    return StrategyResult(
        strategy_type="static",
        method="pearson",
        params={"method": "pearson"},
        artifacts=None,
        metrics_summary={
            "modularity": modularity,
            "clustering_coefficient": clustering,
            "small_world": 1.2,
            "n_connections": 50,
        },
    )


def _make_dynamic_strategy(
    method: str = "kmeans",
    n_states: int = 4,
    switching_rate: float = 0.15,
    occupancy_entropy: float = 1.3,
) -> StrategyResult:
    return StrategyResult(
        strategy_type="dynamic",
        method=method,
        params={"method": method, "n_states": n_states, "window_length": 30, "step": 10},
        artifacts=None,
        metrics_summary={
            "n_states": n_states,
            "switching_rate": switching_rate,
            "occupancy_entropy": occupancy_entropy,
            "temporal_complexity": 0.7,
        },
    )


def test_compare_two_dynamic_strategies():
    strategies = {
        "s1": _make_dynamic_strategy(n_states=4, switching_rate=0.15),
        "s2": _make_dynamic_strategy(n_states=3, switching_rate=0.10),
    }
    result = compare_strategies(strategies)
    assert isinstance(result, ComparisonResult)
    assert result.confidence in ("high", "moderate", "low")
    assert len(result.limitations) >= 1
    assert result.comparison_table  # not empty


def test_compare_static_vs_dynamic():
    strategies = {
        "s1": _make_static_strategy(),
        "s2": _make_dynamic_strategy(),
    }
    result = compare_strategies(strategies)
    assert result.confidence in ("high", "moderate", "low")
    assert len(result.ranking_axes) == 3
    assert "information_richness" in result.ranking_axes
    assert "stability" in result.ranking_axes
    assert "interpretability" in result.ranking_axes


def test_compare_requires_at_least_two():
    with pytest.raises(ValueError, match="at least 2"):
        compare_strategies({"s1": _make_static_strategy()})


def test_limitations_always_populated():
    strategies = {
        "s1": _make_static_strategy(),
        "s2": _make_dynamic_strategy(),
    }
    result = compare_strategies(strategies)
    assert any("single subject" in lim.lower() for lim in result.limitations)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python3 -m pytest tests/test_strategy_compare.py -v`
Expected: FAIL — `ModuleNotFoundError`

- [ ] **Step 3: Implement strategy_compare.py**

```python
# strategy_compare.py
"""Cross-strategy metric comparison and ranking for BrainNet orchestrator."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from brainnet.session_store import StrategyResult


@dataclass
class ComparisonResult:
    """Output of a multi-strategy comparison."""

    comparison_table: dict[str, dict[str, Any]]
    metric_differences: dict[str, Any]
    ranking_axes: dict[str, dict[str, float]]  # axis -> {strategy_id: score}
    recommended_strategy: str
    confidence: str  # "high", "moderate", "low"
    reasoning_hints: list[str]
    limitations: list[str]


def _score_information_richness(strategy: StrategyResult) -> float:
    """More distinct non-zero metrics = higher score."""
    return float(len([v for v in strategy.metrics_summary.values() if v]))


def _score_stability(strategy: StrategyResult) -> float:
    """Fewer parameters = more stable (heuristic for prototype)."""
    return 1.0 / max(len(strategy.params), 1)


def _score_interpretability(strategy: StrategyResult) -> float:
    """Static methods are simpler to interpret than dynamic."""
    base = 1.0 if strategy.strategy_type == "static" else 0.5
    return base / max(len(strategy.params), 1)


def _build_comparison_table(
    strategies: dict[str, StrategyResult],
) -> dict[str, dict[str, Any]]:
    """Build a per-strategy table of all available metrics."""
    return {sid: dict(s.metrics_summary) for sid, s in strategies.items()}


def _compute_metric_differences(
    strategies: dict[str, StrategyResult],
) -> dict[str, Any]:
    """Compute pairwise differences for shared numeric metrics."""
    all_keys: set[str] = set()
    for s in strategies.values():
        all_keys.update(s.metrics_summary.keys())

    diffs: dict[str, Any] = {}
    ids = list(strategies.keys())
    for key in sorted(all_keys):
        vals = {}
        for sid in ids:
            v = strategies[sid].metrics_summary.get(key)
            if isinstance(v, (int, float)):
                vals[sid] = v
        if len(vals) >= 2:
            values = list(vals.values())
            diffs[key] = {
                "values": vals,
                "range": max(values) - min(values),
            }
    return diffs


def _determine_confidence(ranking_axes: dict[str, dict[str, float]]) -> tuple[str, str]:
    """Return (confidence, recommended_strategy)."""
    # Count how many axes each strategy wins
    wins: dict[str, int] = {}
    for axis_scores in ranking_axes.values():
        if not axis_scores:
            continue
        best = max(axis_scores, key=lambda k: axis_scores[k])
        wins[best] = wins.get(best, 0) + 1

    if not wins:
        return "low", ""

    best_strategy = max(wins, key=lambda k: wins[k])
    best_count = wins[best_strategy]

    if best_count >= 3:
        confidence = "high"
    elif best_count >= 2:
        confidence = "moderate"
    else:
        confidence = "low"

    return confidence, best_strategy


def compare_strategies(strategies: dict[str, StrategyResult]) -> ComparisonResult:
    """Compare multiple analysis strategies and produce a ranking."""
    if len(strategies) < 2:
        raise ValueError("compare_strategies requires at least 2 strategies")

    comparison_table = _build_comparison_table(strategies)
    metric_differences = _compute_metric_differences(strategies)

    ranking_axes = {
        "information_richness": {
            sid: _score_information_richness(s) for sid, s in strategies.items()
        },
        "stability": {sid: _score_stability(s) for sid, s in strategies.items()},
        "interpretability": {sid: _score_interpretability(s) for sid, s in strategies.items()},
    }

    confidence, recommended = _determine_confidence(ranking_axes)

    reasoning_hints = []
    for axis, scores in ranking_axes.items():
        best = max(scores, key=lambda k: scores[k])
        reasoning_hints.append(f"{axis}: {best} scores highest")

    limitations = [
        "Comparison based on single subject — results may not generalize.",
        "Ranking heuristics are simplified for prototype; not a formal statistical test.",
        "Parameter sensitivity not assessed — different parameters may change ranking.",
    ]

    types_present = {s.strategy_type for s in strategies.values()}
    if len(types_present) > 1:
        limitations.append(
            "Cross-type comparison (static vs dynamic) uses proxy metrics "
            "that may not capture all differences."
        )

    return ComparisonResult(
        comparison_table=comparison_table,
        metric_differences=metric_differences,
        ranking_axes=ranking_axes,
        recommended_strategy=recommended,
        confidence=confidence,
        reasoning_hints=reasoning_hints,
        limitations=limitations,
    )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python3 -m pytest tests/test_strategy_compare.py -v`
Expected: All 4 tests PASS

- [ ] **Step 5: Commit**

```bash
git add strategy_compare.py tests/test_strategy_compare.py
git commit -m "feat: add strategy comparison module with 3-axis ranking"
```

---

### Task 3: Tool Definitions and Data Access / Preprocessing Tools

**Files:**
- Create: `tools.py`
- Test: `tests/test_tools.py`
- Modify: `pyproject.toml` (add `anthropic` to `[llm]`)

- [ ] **Step 1: Add anthropic dependency to pyproject.toml**

In `pyproject.toml`, change the `[llm]` extra from:
```toml
llm = [
  "openai>=1.0",
]
```
to:
```toml
llm = [
  "openai>=1.0",
  "anthropic>=0.40",
]
```

- [ ] **Step 2: Write failing tests for tool schemas and data access tools**

```python
# tests/test_tools.py
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
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `python3 -m pytest tests/test_tools.py -v`
Expected: FAIL — `ModuleNotFoundError`

- [ ] **Step 4: Implement tools.py with schemas and data access / preprocessing tools**

Create `tools.py` with:
1. All 9 tool JSON schema definitions (`TOOL_DEFINITIONS` list)
2. `execute_tool(session, tool_name, tool_input)` dispatcher function
3. Implementations for: `inspect_qc` (computes SNR per ROI and outlier detection)
4. Stub implementations for heavy tools (`load_dataset`, `list_subjects`, `preprocess`, `static_connectivity`, `dynamic_connectivity`, `extract_features`, `compare_strategies`, `generate_report`) that delegate to existing modules — each returns `{"error": "..."}` if dependencies are missing

The tool implementations follow this pattern:
```python
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
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `python3 -m pytest tests/test_tools.py -v`
Expected: All tests PASS (5 tests: schemas valid, tool names, inspect_qc, inspect_qc missing, unknown tool)

- [ ] **Step 6: Commit**

```bash
git add tools.py tests/test_tools.py pyproject.toml
git commit -m "feat: add tool registry with schemas and data access tools"
```

---

### Task 4: Analysis and Comparison Tool Implementations

**Files:**
- Modify: `tools.py`
- Modify: `tests/test_tools.py`

- [ ] **Step 1: Write failing tests for analysis tools**

Add to `tests/test_tools.py`:

```python
def test_static_connectivity_with_synthetic_data():
    session, rid = _make_session_with_roi()
    result = execute_tool(session, "static_connectivity", {
        "roi_timeseries_id": rid,
        "method": "pearson",
    })
    if "error" in result:
        # Only skip for the specific known dependency error
        assert "not installed" in result["error"].lower() or "dependencies" in result["error"].lower(), \
            f"Unexpected tool error: {result['error']}"
        pytest.skip("Analysis dependencies not installed")
    assert "strategy_id" in result
    assert "global_metrics" in result
    strategy = session.get_strategy(result["strategy_id"])
    assert strategy is not None
    assert strategy.strategy_type == "static"


def test_static_connectivity_partial_not_supported():
    session, rid = _make_session_with_roi()
    result = execute_tool(session, "static_connectivity", {
        "roi_timeseries_id": rid,
        "method": "partial",
    })
    assert "error" in result
    assert "not yet supported" in result["error"].lower()


def test_dynamic_connectivity_with_synthetic_data():
    session, rid = _make_session_with_roi()
    result = execute_tool(session, "dynamic_connectivity", {
        "roi_timeseries_id": rid,
        "method": "kmeans",
        "n_states": 2,
        "window_length": 10,
        "step": 5,
    })
    if "error" in result:
        assert "not installed" in result["error"].lower() or "dependencies" in result["error"].lower(), \
            f"Unexpected tool error: {result['error']}"
        pytest.skip("Analysis dependencies not installed")
    assert "strategy_id" in result
    assert "switching_rate" in result


def test_compare_strategies_tool():
    session = SessionStore(session_id="test")
    from brainnet.session_store import StrategyResult

    s1 = StrategyResult("static", "pearson", {}, None, {"modularity": 0.5, "small_world": 1.2})
    s2 = StrategyResult("dynamic", "kmeans", {"n_states": 4}, None, {"switching_rate": 0.15, "occupancy_entropy": 1.3})
    sid1 = session.store_strategy(s1)
    sid2 = session.store_strategy(s2)

    result = execute_tool(session, "compare_strategies", {"strategy_ids": [sid1, sid2]})
    assert "comparison_table" in result
    assert "confidence" in result
    assert "limitations" in result


def test_generate_report_tool():
    session = SessionStore(session_id="test")
    from brainnet.session_store import StrategyResult

    s1 = StrategyResult("static", "pearson", {"method": "pearson"}, None, {"modularity": 0.5})
    sid1 = session.store_strategy(s1)

    result = execute_tool(session, "generate_report", {"strategy_ids": [sid1]})
    assert "report" in result
    assert "pearson" in result["report"]
    assert "candidate observations" in result["report"]


def test_tool_result_is_json_serializable():
    """Tool results must be JSON-serializable for the Anthropic API tool_result messages."""
    import json

    session, rid = _make_session_with_roi()
    result = execute_tool(session, "inspect_qc", {"roi_timeseries_id": rid})
    # Must not raise
    serialized = json.dumps(result, default=str)
    assert isinstance(serialized, str)
```

- [ ] **Step 2: Run tests to verify new tests fail**

Run: `python3 -m pytest tests/test_tools.py -v`
Expected: New tests FAIL

- [ ] **Step 3: Implement remaining tool functions in tools.py**

Add these implementations to `tools.py`. Each wraps existing BrainNet modules and stores results in the session.

```python
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

    # StaticAnalyzer.__init__ takes optional threshold/proportion, not data
    analyzer = StaticAnalyzer()
    # Data is passed to compute_connectivity, which returns a ConnectivityMatrix
    conn_matrix = analyzer.compute_connectivity(roi_data.timeseries, roi_data.labels, method=method)
    # compute_graph_metrics takes the ConnectivityMatrix, returns GraphMetrics
    graph_metrics = analyzer.compute_graph_metrics(conn_matrix)
    metrics = graph_metrics.global_metrics  # dict[str, float]

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

    # DynamicConfig fields: window_length, step, n_states, method (no 'enabled' field)
    config = DynamicConfig(
        method=params.get("method", "kmeans"),
        n_states=params.get("n_states", 4),
        window_length=params.get("window_length", 30),
        step=params.get("step", 10),
    )
    # DynamicAnalyzer takes only config; data is passed to analyse()
    analyzer = DynamicAnalyzer(config)
    artifacts = analyzer.analyse(roi_data.timeseries)  # returns DynamicStateModel

    # Extract metrics from artifacts.metrics (DynamicMetrics dataclass)
    dm = artifacts.metrics  # has: occupancy, mean_dwell_time, transition_matrix, n_transitions
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
            # Local BIDS dataset — just validate it exists
            idx = DatasetIndex(local_path)
            return {"dataset_path": local_path, "n_subjects": len(idx.list_subjects()), "status": "loaded"}
        elif openneuro_id:
            # OpenNeuro — DatasetIndex supports source='openneuro' with dataset ID as root
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
        # DatasetIndex.list_subjects() returns sorted subject labels
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

    # Construct config with adapter logic: flat params → nested config objects
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
        # PreprocessPipeline.run(func_path) returns a dict with pipeline results
        pipeline = PreprocessPipeline(config)
        result = pipeline.run(params["input_path"])
        # Extract ROI timeseries and QC from result dict
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
```

Register all functions in the `_TOOL_DISPATCH` dict:
```python
_TOOL_DISPATCH = {
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python3 -m pytest tests/test_tools.py -v`
Expected: All tests PASS (analysis tests may skip if deps not installed)

- [ ] **Step 5: Commit**

```bash
git add tools.py tests/test_tools.py
git commit -m "feat: add analysis and comparison tool implementations"
```

---

### Task 5: Orchestrator (Claude API Loop)

**Files:**
- Create: `orchestrator.py`
- Test: `tests/test_orchestrator.py`

- [ ] **Step 1: Write failing tests with mocked Claude API**

```python
# tests/test_orchestrator.py
from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest

from brainnet.orchestrator import (
    SYSTEM_PROMPT,
    run_orchestrator,
    sse_format,
)
from brainnet.session_store import SessionStore


def test_sse_format():
    line = sse_format("text_delta", {"text": "hello"})
    assert line.startswith("event: text_delta\n")
    assert '"text": "hello"' in line
    assert line.endswith("\n\n")


def test_system_prompt_contains_safety_rules():
    assert "candidate observations" in SYSTEM_PROMPT
    assert "Maximum 3 strategies" in SYSTEM_PROMPT


def test_run_orchestrator_streams_text():
    session = SessionStore(session_id="test")

    # Mock a simple Claude response that just returns text
    mock_stream = MagicMock()
    mock_stream.__enter__ = MagicMock(return_value=mock_stream)
    mock_stream.__exit__ = MagicMock(return_value=False)

    # Simulate one text event
    text_event = MagicMock()
    text_event.type = "content_block_delta"
    text_event.delta = MagicMock()
    text_event.delta.type = "text_delta"
    text_event.delta.text = "Hello!"

    mock_stream.__iter__ = MagicMock(return_value=iter([text_event]))
    mock_stream.get_final_message.return_value = MagicMock(stop_reason="end_turn")

    with patch("brainnet.orchestrator._create_stream", return_value=mock_stream):
        events = list(run_orchestrator(session, "test message"))

    assert len(events) >= 1
    assert any("text_delta" in e for e in events)
    assert any("done" in e for e in events)
    assert session.messages[0]["role"] == "user"
    assert session.messages[0]["content"] == "test message"


def test_run_orchestrator_handles_tool_use():
    """Test the tool-calling path where Claude requests a tool and gets results."""
    session = SessionStore(session_id="test")

    # First response: Claude wants to call a tool
    tool_block = MagicMock()
    tool_block.type = "tool_use"
    tool_block.name = "inspect_qc"
    tool_block.input = {"roi_timeseries_id": "nonexistent"}
    tool_block.id = "tool_123"

    first_msg = MagicMock()
    first_msg.content = [tool_block]
    first_msg.stop_reason = "tool_use"

    first_stream = MagicMock()
    first_stream.__enter__ = MagicMock(return_value=first_stream)
    first_stream.__exit__ = MagicMock(return_value=False)
    first_stream.__iter__ = MagicMock(return_value=iter([]))
    first_stream.get_final_message.return_value = first_msg

    # Second response: Claude sends text after getting tool result
    text_event = MagicMock()
    text_event.type = "content_block_delta"
    text_event.delta = MagicMock()
    text_event.delta.type = "text_delta"
    text_event.delta.text = "The QC data was not found."

    text_block = MagicMock()
    text_block.type = "text"

    second_msg = MagicMock()
    second_msg.content = [text_block]
    second_msg.stop_reason = "end_turn"

    second_stream = MagicMock()
    second_stream.__enter__ = MagicMock(return_value=second_stream)
    second_stream.__exit__ = MagicMock(return_value=False)
    second_stream.__iter__ = MagicMock(return_value=iter([text_event]))
    second_stream.get_final_message.return_value = second_msg

    call_count = 0

    def mock_create_stream(*args, **kwargs):
        nonlocal call_count
        call_count += 1
        return first_stream if call_count == 1 else second_stream

    with patch("brainnet.orchestrator._create_stream", side_effect=mock_create_stream):
        events = list(run_orchestrator(session, "check QC"))

    # Should have tool_start, tool_result, text_delta, and done events
    event_types = [e.split("\n")[0] for e in events]
    assert any("tool_start" in e for e in event_types)
    assert any("tool_result" in e for e in event_types)
    assert any("done" in e for e in event_types)

    # Session should have user message, assistant (tool_use), user (tool_result), assistant (text)
    assert len(session.messages) >= 3
    assert session.messages[0]["role"] == "user"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python3 -m pytest tests/test_orchestrator.py -v`
Expected: FAIL — `ModuleNotFoundError`

- [ ] **Step 3: Implement orchestrator.py**

```python
# orchestrator.py
"""Claude-powered analysis orchestrator for BrainNet."""

from __future__ import annotations

import json
import os
from typing import Any, Generator

from brainnet.session_store import SessionStore
from brainnet.tools import TOOL_DEFINITIONS, execute_tool

SYSTEM_PROMPT = """You are BrainNet's analysis orchestrator, a scientific fMRI analysis assistant.

Given a user's analysis goal:
1. Load and preprocess the requested data
2. Propose 2-3 distinct analysis strategies that address the goal
3. Execute each strategy using your tools
4. Compare results using the compare_strategies tool
5. Summarize findings and recommend the best approach

Rules:
- Always preprocess data before running analysis
- Maximum 3 strategies per comparison round (user can request more)
- Report data quality issues honestly — do not proceed if QC fails
- Frame all findings as candidate observations, not confirmed discoveries
- Acknowledge limitations in the comparison
- If a tool fails, explain why and try alternative parameters
- If a required dependency is missing, tell the user what to install
"""

TOKEN_BUDGET = 50_000
TOKEN_WARNING_THRESHOLD = 0.8


def sse_format(event: str, data: dict[str, Any]) -> str:
    """Format an SSE event string."""
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


def _get_client():
    """Lazily import and create the Anthropic client."""
    try:
        import anthropic
    except ImportError:
        raise RuntimeError(
            "The anthropic package is required for the chat feature. "
            "Install it with: pip install -e .[llm]"
        )
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise RuntimeError(
            "ANTHROPIC_API_KEY environment variable is required for the chat feature."
        )
    return anthropic.Anthropic(api_key=api_key)


def _create_stream(client, messages, system, tools, max_tokens):
    """Create a streaming message — extracted for test mocking."""
    return client.messages.stream(
        model="claude-sonnet-4-20250514",
        system=system,
        messages=messages,
        tools=tools,
        max_tokens=max_tokens,
    )


def _summarize_old_tool_results(session: SessionStore) -> None:
    """Trim older tool_result content when approaching token budget."""
    if session.estimate_tokens() < int(TOKEN_BUDGET * TOKEN_WARNING_THRESHOLD):
        return
    for msg in session.messages[:-6]:  # keep last 6 messages intact
        if msg.get("role") == "user" and isinstance(msg.get("content"), list):
            for block in msg["content"]:
                if isinstance(block, dict) and block.get("type") == "tool_result":
                    content = block.get("content", "")
                    if isinstance(content, str) and len(content) > 200:
                        block["content"] = content[:200] + "... [truncated]"


def run_orchestrator(
    session: SessionStore, user_message: str
) -> Generator[str, None, None]:
    """Synchronous generator that streams SSE events for one user turn."""
    session.messages.append({"role": "user", "content": user_message})
    _summarize_old_tool_results(session)

    token_estimate = session.estimate_tokens()
    if token_estimate > int(TOKEN_BUDGET * TOKEN_WARNING_THRESHOLD):
        yield sse_format("text_delta", {
            "text": f"\n\n⚠️ Session using ~{token_estimate} tokens "
            f"(budget: {TOKEN_BUDGET}). Consider starting a new session.\n\n"
        })

    client = _get_client()

    while True:
        with _create_stream(
            client, session.messages, SYSTEM_PROMPT, TOOL_DEFINITIONS, 4096
        ) as stream:
            assistant_content = []
            for event in stream:
                if hasattr(event, "delta"):
                    if getattr(event.delta, "type", None) == "text_delta":
                        yield sse_format("text_delta", {"text": event.delta.text})

            final = stream.get_final_message()

        # Record assistant message
        session.messages.append({
            "role": "assistant",
            "content": final.content,
        })

        # Process any tool calls
        tool_uses = [b for b in final.content if getattr(b, "type", None) == "tool_use"]
        if not tool_uses:
            yield sse_format("done", {})
            break

        tool_results = []
        for tool_block in tool_uses:
            yield sse_format("tool_start", {
                "tool": tool_block.name,
                "params": tool_block.input,
            })
            result = execute_tool(session, tool_block.name, tool_block.input)
            yield sse_format("tool_result", {
                "tool": tool_block.name,
                "summary": result,
            })
            tool_results.append({
                "type": "tool_result",
                "tool_use_id": tool_block.id,
                "content": json.dumps(result, ensure_ascii=False, default=str),
            })

        session.messages.append({"role": "user", "content": tool_results})

        if final.stop_reason == "end_turn":
            yield sse_format("done", {})
            break
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python3 -m pytest tests/test_orchestrator.py -v`
Expected: All 4 tests PASS

- [ ] **Step 5: Commit**

```bash
git add orchestrator.py tests/test_orchestrator.py
git commit -m "feat: add Claude orchestrator with streaming tool dispatch"
```

---

### Task 6: Web Chat Blueprint

**Files:**
- Create: `web_chat.py`
- Modify: `web_app.py`
- Modify: `tests/test_entrypoints.py`

- [ ] **Step 1: Write failing tests for chat routes**

Add to `tests/test_entrypoints.py`:

```python
def test_chat_route_exists(monkeypatch, tmp_path) -> None:
    instance_dir = tmp_path / "instance"
    monkeypatch.setenv("BRAINNET_INSTANCE_DIR", str(instance_dir))
    sys.modules.pop("brainnet.web_app", None)
    sys.modules.pop("brainnet.web_chat", None)

    module = importlib.import_module("brainnet.web_app")
    client = module.app.test_client()

    routes = {rule.rule for rule in module.app.url_map.iter_rules()}
    assert "/chat" in routes or "/chat/" in routes

    response = client.get("/chat")
    assert response.status_code in (200, 302)


def test_chat_new_session(monkeypatch, tmp_path) -> None:
    instance_dir = tmp_path / "instance"
    monkeypatch.setenv("BRAINNET_INSTANCE_DIR", str(instance_dir))
    sys.modules.pop("brainnet.web_app", None)
    sys.modules.pop("brainnet.web_chat", None)

    module = importlib.import_module("brainnet.web_app")
    client = module.app.test_client()

    response = client.post("/chat/new")
    assert response.status_code == 200
    data = response.get_json()
    assert "session_id" in data
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python3 -m pytest tests/test_entrypoints.py::test_chat_route_exists -v`
Expected: FAIL

- [ ] **Step 3: Implement web_chat.py**

```python
# web_chat.py
"""Flask blueprint for BrainNet chat interface."""

from __future__ import annotations

import json

from flask import Blueprint, Response, jsonify, render_template, request

from brainnet.session_store import SessionManager

chat_bp = Blueprint("chat", __name__)
_session_manager = SessionManager(max_sessions=10, timeout_seconds=1800)


def _check_api_key() -> str | None:
    """Return an error message if ANTHROPIC_API_KEY is not set."""
    import os

    if not os.environ.get("ANTHROPIC_API_KEY"):
        return (
            "ANTHROPIC_API_KEY is not set. "
            "Set this environment variable to enable the chat feature."
        )
    return None


@chat_bp.route("/chat")
def chat_page():
    api_error = _check_api_key()
    return render_template("chat.html", api_error=api_error)


@chat_bp.route("/chat/new", methods=["POST"])
def new_session():
    try:
        session = _session_manager.create_session()
        return jsonify({"session_id": session.session_id})
    except RuntimeError as exc:
        return jsonify({"error": str(exc)}), 429


@chat_bp.route("/chat/send", methods=["POST"])
def send_message():
    api_error = _check_api_key()
    if api_error:
        return jsonify({"error": api_error}), 503

    data = request.get_json()
    if not data:
        return jsonify({"error": "JSON body required"}), 400

    session_id = data.get("session_id")
    message = data.get("message", "").strip()
    if not session_id or not message:
        return jsonify({"error": "session_id and message required"}), 400

    session = _session_manager.get_session(session_id)
    if session is None:
        return jsonify({"error": "Session not found or expired"}), 404

    from brainnet.orchestrator import run_orchestrator

    def generate():
        try:
            for event in run_orchestrator(session, message):
                yield event
        except Exception as exc:
            yield f"event: error\ndata: {json.dumps({'message': str(exc)})}\n\n"

    return Response(generate(), mimetype="text/event-stream")


@chat_bp.route("/chat/history")
def chat_history():
    session_id = request.args.get("session_id")
    if not session_id:
        return jsonify({"error": "session_id required"}), 400
    session = _session_manager.get_session(session_id)
    if session is None:
        return jsonify({"error": "Session not found"}), 404
    return jsonify({"messages": session.messages})


@chat_bp.route("/chat/status")
def chat_status():
    session_id = request.args.get("session_id")
    if not session_id:
        return jsonify({"error": "session_id required"}), 400
    session = _session_manager.get_session(session_id)
    if session is None:
        return jsonify({"error": "Session not found"}), 404
    return jsonify({
        "session_id": session.session_id,
        "n_strategies": len(session.strategies),
        "n_roi_datasets": len(session.roi_data),
        "estimated_tokens": session.estimate_tokens(),
    })
```

- [ ] **Step 4: Register blueprint in web_app.py**

Add to `web_app.py` after existing blueprint/route registrations. Wrap in a broad try/except so chat import errors don't break the entire web app:

```python
try:
    try:
        from brainnet.web_chat import chat_bp
    except ImportError:
        from web_chat import chat_bp
    app.register_blueprint(chat_bp)
except Exception:
    pass  # Chat feature unavailable — web app still works
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `python3 -m pytest tests/test_entrypoints.py -v`
Expected: All tests PASS (including new chat tests)

- [ ] **Step 6: Run full test suite**

Run: `python3 -m pytest -v`
Expected: All existing tests still pass

- [ ] **Step 7: Commit**

```bash
git add web_chat.py web_app.py tests/test_entrypoints.py
git commit -m "feat: add Flask chat blueprint with SSE streaming"
```

---

### Task 7: Chat UI Template

**Files:**
- Create: `templates/chat.html`

- [ ] **Step 1: Create chat.html template**

Create `templates/chat.html` with:
- Dark theme (background: `#0a0a14`, cards: `#1a1a2e`)
- Nav bar matching existing BrainNet templates with links to Patients, OpenNeuro, and Analysis Chat
- Chat message area with auto-scroll
- User messages (right-aligned, `#2a2a4a` background)
- Assistant messages (left-aligned, `#1a1a2e` background, `#7c7cff` left border)
- Tool call indicators inside assistant messages (monospace, green/orange/red icons)
- Text input + Send button at the bottom
- "New Session" button
- JavaScript for:
  - `newSession()`: POST to `/chat/new`, store session_id
  - `sendMessage()`: POST to `/chat/send`, consume ReadableStream, parse SSE lines
  - `appendText(text)`: Append to current assistant message bubble
  - `showToolStart(name, params)`: Add orange spinner tool indicator
  - `showToolResult(name, summary)`: Update indicator to green checkmark
  - `scrollToBottom()`: Auto-scroll chat container
- `{% if api_error %}` block showing the API key error message instead of the chat input
- Graceful fallback when JavaScript is disabled

- [ ] **Step 2: Verify the template renders**

Run: `python3 -c "from brainnet.web_app import app; c = app.test_client(); r = c.get('/chat'); print(r.status_code, len(r.data))"`
Expected: `200` with HTML content

- [ ] **Step 3: Commit**

```bash
git add templates/chat.html
git commit -m "feat: add chat UI template with streaming support"
```

---

### Task 8: Integration Smoke Test and Lint

**Files:**
- Modify: `tests/test_entrypoints.py`

- [ ] **Step 1: Run full lint**

Run: `python3 -m ruff check .`
Expected: No errors. Fix any that appear.

- [ ] **Step 2: Run full test suite**

Run: `python3 -m pytest -v`
Expected: All tests pass

- [ ] **Step 3: Verify chat endpoint end-to-end with mock**

Add to `tests/test_entrypoints.py`:

```python
def test_chat_send_without_api_key(monkeypatch, tmp_path) -> None:
    """Send endpoint returns 503 when ANTHROPIC_API_KEY is not set."""
    instance_dir = tmp_path / "instance"
    monkeypatch.setenv("BRAINNET_INSTANCE_DIR", str(instance_dir))
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    sys.modules.pop("brainnet.web_app", None)
    sys.modules.pop("brainnet.web_chat", None)

    module = importlib.import_module("brainnet.web_app")
    client = module.app.test_client()

    # Create session first
    resp = client.post("/chat/new")
    session_id = resp.get_json()["session_id"]

    # Try to send a message
    resp = client.post(
        "/chat/send",
        json={"session_id": session_id, "message": "hello"},
    )
    assert resp.status_code == 503
```

- [ ] **Step 4: Run tests**

Run: `python3 -m pytest -v`
Expected: All tests pass

- [ ] **Step 5: Final commit**

```bash
git add tests/test_entrypoints.py
git commit -m "test: add integration smoke tests for chat endpoint"
```

---

## Execution Order

Tasks must be executed in order (1→8). Each task depends on the previous:

```
Task 1 (SessionStore) → Task 2 (Strategy Compare) → Task 3 (Tool Schemas)
  → Task 4 (Tool Implementations) → Task 5 (Orchestrator) → Task 6 (Web Blueprint)
  → Task 7 (Chat UI) → Task 8 (Integration)
```
