# BrainNet Multi-Agent Platform Design

**Date**: 2026-03-19
**Status**: Draft
**Scope**: Working prototype of LLM-powered multi-strategy fMRI analysis

## 1. Overview

Build a Claude-powered orchestrator that enables users to describe analysis goals in natural language, automatically generates and executes multiple analysis strategies using existing BrainNet modules, compares results, and recommends the best approach — all through a web chat interface.

### Design Decisions

- **Single orchestrator agent** (not multi-agent): One Claude conversation manages the full session. Simplest path to a working prototype; tools can be promoted to independent agents later.
- **Claude (Anthropic SDK)** as the LLM provider.
- **Tool-calling wrapper**: Existing analysis functions exposed as Claude tools. Claude decides which tools to invoke and in what order.
- **Web chat interface**: Chat panel integrated into the existing Flask app at `/chat`.

### End-to-End Flow

```
User types goal → Flask SSE endpoint → Claude API (streaming)
  → Claude plans 2-3 strategies
  → Claude calls tools (preprocess, analyze, compare)
  → Tool results stored in SessionStore
  → Claude synthesizes findings → streamed to chat UI
  → User can refine → cycle repeats
```

## 2. Architecture

### Layers

1. **Chat Interface** (`web_chat.py` + `templates/chat.html`) — Flask blueprint with SSE streaming
2. **Orchestrator** (`orchestrator.py`) — Claude API conversation loop with tool dispatch
3. **Tool Layer** (`tools.py`) — Tool JSON schemas + Python implementations wrapping existing modules
4. **Session Store** (`session_store.py`) — In-memory dict holding ROI data, strategy results, conversation history
5. **Strategy Comparison** (`strategy_compare.py`) — New module for cross-strategy metric comparison
6. **Execution Layer** — Existing `analysis_service.py`, `preprocessing.py`, `static/`, `dynamic/` (unchanged)

### Data Flow

```
Chat Interface
  ↕ SSE (text_delta, tool_start, tool_result, done)
Orchestrator (Claude API)
  ↕ tool_use / tool_result messages
Tool Layer
  ↕ function calls
SessionStore ←→ Existing Analysis Modules
```

## 3. Tool Registry

### Data Access

#### `load_dataset`
- **Purpose**: Load a local BIDS dataset or fetch from OpenNeuro
- **Input**: `dataset_path: str | None`, `openneuro_id: str | None`, `subject: str`, `task: str`
- **Output**: `dataset_root: str`, `available_runs: list[str]`, `n_timepoints: int`
- **Wraps**: `DatasetIndex`, `DatasetManager.fetch_from_openneuro`

#### `list_subjects`
- **Purpose**: List available subjects and tasks in a dataset
- **Input**: `dataset_path: str`
- **Output**: `subjects: list[{id, tasks, runs}]`
- **Wraps**: `DatasetIndex.get_functional_runs`

### Preprocessing

#### `preprocess`
- **Purpose**: Run preprocessing pipeline, extract ROI time series
- **Input**: `run_path: str`, `smoothing_fwhm: float = 3.0`, `bandpass_low: float = 0.01`, `bandpass_high: float = 0.1`, `atlas: str | None = None`
- **Output**: `roi_timeseries_id: str`, `n_rois: int`, `n_timepoints: int`, `roi_labels: list[str]`, `qc_summary: dict`
- **Wraps**: `PreprocessPipeline.run` via `PreprocessPipelineConfig`
- **Adapter logic**: The tool constructs a `PreprocessPipelineConfig` from the flat parameters — specifically `SmoothingConfig(fwhm=smoothing_fwhm)`, `TemporalFilterConfig(low_cut=bandpass_low, high_cut=bandpass_high)`, and `RoiExtractionConfig(atlas_path=atlas)`. Other config steps (slice timing, motion correction, nuisance regression) use defaults from `build_cli_preprocess_config()`. This translation layer lives in `tools.py`.
- **Side effect**: Stores ROI time series in SessionStore under `roi_timeseries_id`

#### `inspect_qc`
- **Purpose**: Get detailed QC metrics for a preprocessed run
- **Input**: `roi_timeseries_id: str`
- **Output**: `motion_summary: dict`, `snr_per_roi: list[float]`, `outlier_timepoints: list[int]`
- **Wraps**: QC metrics from preprocess output
- **Note**: `snr_per_roi` and `outlier_timepoints` are new computations not in the current pipeline. `snr_per_roi` is computed as mean/std per ROI column. `outlier_timepoints` uses a simple z-score threshold (|z| > 3) on framewise displacement or global signal. These are added in `tools.py`, not in the preprocessing module itself.

### Analysis

#### `static_connectivity`
- **Purpose**: Compute static connectivity matrix and graph metrics
- **Input**: `roi_timeseries_id: str`, `method: "pearson"` (only Pearson is currently supported; partial correlation is deferred)
- **Output**: `strategy_id: str`, `global_metrics: dict`, `top_connections: list`, `modularity: float`, `small_world: float`
- **Wraps**: `StaticAnalyzer.compute_connectivity`, `StaticAnalyzer.compute_graph_metrics`
- **Note**: The existing `StaticAnalyzer` always uses Pearson correlation. The `method` parameter is accepted for forward-compatibility but only `"pearson"` is implemented in the prototype. Requesting `"partial"` returns an error with a message that it is not yet supported.
- **Side effect**: Stores full `AnalysisArtifacts` in SessionStore under `strategy_id`

#### `dynamic_connectivity`
- **Purpose**: Run dynamic state analysis
- **Input**: `roi_timeseries_id: str`, `method: "kmeans" | "hmm" | "cap"`, `n_states: int = 4`, `window_length: int = 30`, `step: int = 10`
- **Output**: `strategy_id: str`, `state_occupancies: list`, `transition_matrix: list[list]`, `switching_rate: float`, `temporal_complexity: float`
- **Wraps**: `DynamicAnalyzer(DynamicConfig(...)).analyse(roi_timeseries)`
- **Adapter logic**: The tool looks up the ndarray from SessionStore via `roi_timeseries_id`, constructs a `DynamicConfig(window_length=..., step=..., n_states=..., method=...)`, creates a `DynamicAnalyzer(config)`, and calls `analyse(roi_ts)`. This translation layer lives in `tools.py`.
- **Side effect**: Stores full `AnalysisArtifacts` in SessionStore under `strategy_id`

#### `extract_features`
- **Purpose**: Extract comprehensive features from analysis results and return them to Claude (does NOT write to SQLite)
- **Input**: `strategy_id: str`
- **Output**: `feature_names: list[str]`, `feature_values: list[float]`, `feature_types: list[str]`, `summary_stats: dict`
- **Reuses logic from**: `persist_analysis_features` (same feature extraction, but returns data instead of writing to DB)

### Comparison & Reporting

#### `compare_strategies`
- **Purpose**: Compare results from multiple analysis strategies
- **Input**: `strategy_ids: list[str]`
- **Output**: `comparison_table: dict`, `metric_differences: dict`, `recommended_strategy: str`, `confidence: str`, `reasoning_hints: list[str]`, `limitations: list[str]`
- **Implementation**: New module (`strategy_compare.py`)
- **Comparison logic**:
  - **Within-type** (e.g. two dynamic strategies): Compare directly on shared metrics — occupancy entropy, switching rate, temporal complexity, state count stability.
  - **Cross-type** (static vs dynamic): Compare on the common metric categories both produce — global graph metrics (modularity, clustering coefficient) are available from static analysis; for dynamic strategies, compute equivalent graph metrics from the mean connectivity across states. Also compare information content (entropy measures) and complexity.
  - **Recommendation**: The tool does NOT pick a winner. It ranks strategies on 3 axes: (1) information richness (more distinct metrics captured), (2) stability (lower variance across parameters), (3) interpretability (fewer parameters = simpler to explain). Claude uses these rankings plus the user's goal to make the final recommendation.
  - **Confidence levels**: `"high"` (clear winner on all axes), `"moderate"` (winner on 2/3 axes), `"low"` (no clear winner or insufficient data). Claude must state the confidence level when presenting results.
  - **Limitations field**: Always populated with at least: sample size (single subject), analysis parameter sensitivity, and any missing comparisons (e.g. if HMM was unavailable).

#### `generate_report`
- **Purpose**: Generate HTML report for one or more strategies
- **Input**: `strategy_ids: list[str]`, `subject_id: str`, `include_comparison: bool = True`
- **Output**: `report_path: str`, `report_url: str`
- **Wraps**: `ReportGenerator.generate`
- **Adapter logic**: The existing `ReportGenerator.generate` takes concrete artifacts (`conn_matrix`, `graph_metrics`, `dyn_model`, etc.), not strategy IDs. The tool resolves `strategy_ids` from SessionStore into the required artifact parameters. For multi-strategy reports (`include_comparison=True`), the tool calls `generate` once per strategy and produces a comparison summary page linking them.

## 4. SessionStore Design

```python
@dataclass
class SessionStore:
    session_id: str
    roi_data: dict[str, RoiData]        # keyed by roi_timeseries_id
    strategies: dict[str, StrategyResult] # keyed by strategy_id
    messages: list[dict]                  # Claude conversation history

@dataclass
class RoiData:
    timeseries: ndarray
    labels: list[str]
    qc: dict
    source_path: str

@dataclass
class StrategyResult:
    strategy_type: str          # "static" or "dynamic"
    method: str                 # "pearson", "kmeans", "hmm", etc.
    params: dict                # full parameter set
    artifacts: AnalysisArtifacts
    metrics_summary: dict       # condensed metrics for Claude
```

Storage is in-memory (dict of session_id → SessionStore). For the prototype, sessions are lost on server restart. Persistence can be added later.

### Session Limits

- **Maximum concurrent sessions**: 10. New session creation fails with a message when the limit is reached.
- **Session timeout**: Sessions expire after 30 minutes of inactivity (no new messages). Expired sessions are evicted on the next request.
- **Memory estimate**: A typical session with 3 strategies stores ~3 MB of ndarray data (200 timepoints × 100 ROIs × 3 strategies × connectivity matrices). 10 sessions ≈ 30 MB, acceptable for a prototype.

## 5. Orchestrator Design

### System Prompt

```
You are BrainNet's analysis orchestrator, a scientific fMRI analysis assistant.

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
```

### Conversation Loop

The orchestrator uses **synchronous** Python generators (not `async`), matching the existing synchronous Flask app. The Anthropic SDK provides a synchronous streaming interface via `client.messages.stream()`. Flask serves the generator as a streaming response via `Response(generator(), mimetype='text/event-stream')`.

```python
def run_orchestrator(session: SessionStore, user_message: str):
    """Synchronous generator yielding SSE-formatted strings."""
    session.messages.append({"role": "user", "content": user_message})

    while True:
        with client.messages.stream(
            model="claude-sonnet-4-20250514",
            system=SYSTEM_PROMPT,
            messages=session.messages,
            tools=TOOL_DEFINITIONS,
            max_tokens=4096,
        ) as stream:
            for event in stream:
                if event is text_delta:
                    yield sse_format("text_delta", {"text": event.text})
                elif event is tool_use:
                    yield sse_format("tool_start", {"tool": name, "params": input})
                    result = execute_tool(session, name, input)
                    yield sse_format("tool_result", {"tool": name, "summary": result})

        if stream.stop_reason == "end_turn":
            yield sse_format("done", {})
            break
```

### Token Budget

- Session history capped at ~50K tokens (estimated via `anthropic`'s token counting or character-based heuristic of ~4 chars/token)
- **Summarization strategy**: When history exceeds 40K tokens, a deterministic Python function (not Claude) replaces older `tool_result` content blocks with condensed summaries. For each tool type, there is a fixed summarizer: e.g. `static_connectivity` results are reduced to `{modularity, small_world, n_connections}`, `dynamic_connectivity` to `{n_states, switching_rate, occupancy_entropy}`. The original full results remain in `SessionStore` — only the conversation history is trimmed.
- User is warned when token usage crosses 80% of the budget

## 6. Web Chat Interface

### Flask Blueprint (`web_chat.py`)

| Route | Method | Purpose |
|-------|--------|---------|
| `/chat` | GET | Render chat page |
| `/chat/send` | POST | Send message, return SSE stream |
| `/chat/new` | POST | Create new session |
| `/chat/history` | GET | Get past messages for session |
| `/chat/status` | GET | Session info (strategies run, token usage) |

### SSE Event Types

| Event | Data | Purpose |
|-------|------|---------|
| `text_delta` | `{text: "..."}` | Streaming text chunks from Claude |
| `tool_start` | `{tool: "...", params: {...}}` | Tool execution started |
| `tool_result` | `{tool: "...", summary: "..."}` | Tool execution completed |
| `error` | `{message: "..."}` | Error occurred |
| `done` | `{}` | Response complete |

### Chat UI (`templates/chat.html`)

- Integrated into existing Flask app navigation (new "Analysis Chat" tab)
- Dark theme consistent with existing BrainNet templates
- Message bubbles: user (right-aligned), assistant (left-aligned, with tool call indicators)
- Tool call indicators: green checkmark (complete), orange spinner (running), red X (failed)
- Auto-scroll on new messages
- Text input with Send button
- "New Session" button

### Client-Side Streaming

The client uses `fetch()` with `ReadableStream` to consume the streaming response from the POST endpoint. This is **not** standard `EventSource` (which only supports GET). The client manually parses SSE-formatted lines (`data: {...}\n\n`) from the stream.

```javascript
// POST message, consume streaming response with manual SSE parsing
const response = await fetch('/chat/send', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({session_id, message}),
});
const reader = response.body.getReader();
const decoder = new TextDecoder();
let buffer = '';
while (true) {
    const {done, value} = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, {stream: true});
    // Split on double-newline, parse each SSE event as JSON
    // Update DOM: append text, show tool indicators, etc.
}
```

## 7. Stopping Conditions

The orchestrator stops exploring when any of these conditions is met:

1. **Strategy budget**: Max 3 strategies per comparison round. User can explicitly request more.
2. **Data quality failure**: Preprocessing QC flags >30% outlier timepoints. Claude reports the issue and stops.
3. **Convergent results**: All strategies produce metrics within 10% of each other. Claude notes convergence and recommends based on simplicity.
4. **User goal satisfied**: Claude judges the goal is answered based on conversation context.
5. **Dependency missing**: Required optional dependency not installed. Claude tells user what to install.
6. **Token budget**: Session approaching 50K tokens. Claude summarizes findings and suggests starting a new session.

## 8. Error Handling

- **Tool execution errors**: Reported back to Claude as structured `{"error": "message", "suggestion": "..."}`. Claude can reason about them and try alternative parameters.
- **All strategies fail**: Claude explains common causes (data quality, missing dependencies, incompatible parameters) and suggests alternatives.
- **Network errors** (OpenNeuro): Retried once with exponential backoff, then reported to user.
- **Claude API errors**: Displayed to user with retry option.
- **Invalid tool parameters**: Validated before execution; error returned to Claude with valid ranges.

## 9. Scientific Safety

Aligned with AGENTS.md and WORKFLOW.md principles:

- System prompt instructs Claude to frame findings as **candidate observations**, not confirmed discoveries
- `compare_strategies` tool always returns a `limitations` field that Claude must acknowledge
- No strategy is labeled "best" without Claude explaining the comparison methodology and caveats
- If results are inconclusive, Claude says so explicitly
- The system never claims statistical significance without actual statistical testing

## 10. New Files

| File | Purpose | Lines (est.) |
|------|---------|-------------|
| `orchestrator.py` | Claude API integration, conversation loop, tool dispatch | ~200 |
| `tools.py` | Tool JSON schemas + Python implementations + adapter logic | ~550 |
| `session_store.py` | In-memory session state management | ~80 |
| `strategy_compare.py` | Cross-strategy metric comparison and ranking | ~150 |
| `web_chat.py` | Flask blueprint for chat routes + SSE streaming | ~120 |
| `templates/chat.html` | Chat UI with streaming message rendering | ~200 |

### Modified Files

| File | Change |
|------|--------|
| `web_app.py` | Register chat blueprint, add nav link |
| `pyproject.toml` | Add `anthropic` to optional dependencies (`[llm]` extra) |
| `requirements.txt` | Add `anthropic` |

### Dependencies

- `anthropic` (Anthropic Python SDK) — added to the existing `[llm]` extra alongside `openai` (which is already there for LLM interpretation). Both coexist; `openai` is used by `llm_interpretation.py`, `anthropic` is used by the orchestrator.
- No other new dependencies

## 11. Environment Variables

| Variable | Purpose | Required |
|----------|---------|----------|
| `ANTHROPIC_API_KEY` | Claude API authentication | Yes, for chat feature |

The chat feature gracefully degrades when `ANTHROPIC_API_KEY` is not set — the `/chat` route shows a message explaining the API key is needed.

## 12. Testing Strategy

- **Unit tests for tools**: Each tool function tested independently with mock data
- **Unit tests for SessionStore**: State management, ID generation, cleanup
- **Unit tests for strategy_compare**: Comparison logic with known inputs
- **Integration test for orchestrator**: Mock the Claude API, verify tool dispatch loop
- **Smoke test for web chat**: Flask test client, verify routes return correct status codes
- **No real Claude API calls in tests**: All LLM interactions mocked

## 13. What This Design Does NOT Include

These are explicitly deferred for future iterations:

- **Multi-agent coordination**: No separate specialist agents — single orchestrator only
- **Persistent sessions**: Sessions are in-memory, lost on restart
- **Authentication**: No user auth on the chat endpoint
- **Parallel strategy execution**: Strategies run sequentially
- **Custom atlas selection**: Uses default atlas only
- **CLI chat mode**: Web only for the prototype
- **Cost tracking**: No token usage accounting beyond the budget cap
