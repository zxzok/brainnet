# CLAUDE.md — BrainNet

## Project Overview

BrainNet is a modular Python pipeline for functional MRI (fMRI) preprocessing and analysis. It provides tools for BIDS dataset indexing, fMRI preprocessing, static/dynamic functional connectivity analysis, graph-theoretic metrics, interactive HTML report generation, and a Flask web interface for patient management.

## Repository Structure

```
brainnet/
├── main.py                    # CLI entry point, pipeline orchestration
├── web_app.py                 # Flask web application (patient CRUD, analysis, reports)
├── __init__.py                # Package exports with graceful optional imports
├── data_management.py         # BIDS dataset indexing (DatasetIndex, BIDSFile)
├── preprocessing.py           # Simple preprocessing pipeline
├── preprocessing_full.py      # Modular preprocessing with ABC-based steps
├── static_analysis.py         # Legacy static connectivity wrapper
├── dynamic_analysis.py        # Legacy dynamic connectivity wrapper
├── visualization.py           # Plotly-based HTML report generation
├── llm_interpretation.py      # LLM-based result summarization (OpenAI / transformers)
├── openneuro_client.py        # OpenNeuro dataset downloading
├── multimodal.py              # Multi-modal data handling
├── templates.py               # Atlas template utilities
├── requirements.txt           # Python dependencies
│
├── static/                    # Modular static connectivity package
│   ├── analyzer.py            #   StaticAnalyzer orchestrator
│   ├── connectivity.py        #   ConnectivityMatrix, Pearson correlation
│   └── metrics.py             #   GraphMetrics (degree, clustering, efficiency, modularity)
│
├── dynamic/                   # Modular dynamic connectivity package
│   ├── analyzer.py            #   DynamicAnalyzer orchestrator
│   ├── config.py              #   DynamicConfig dataclass
│   ├── model.py               #   DynamicStateModel, DynamicMetrics
│   ├── window.py              #   Sliding window computation
│   ├── kmeans.py              #   K-means state identification
│   ├── hmm.py                 #   Gaussian HMM analysis
│   ├── cap.py                 #   Co-activation pattern analysis
│   ├── metrics.py             #   Temporal statistics (occupancy, dwell time)
│   ├── state_features.py      #   Graph metrics per state
│   └── io.py                  #   Save/load results to disk
│
├── templates/                 # Flask HTML templates (Jinja2)
└── tests/                     # pytest test suite
    ├── test_dataset_index.py
    ├── test_openneuro_client.py
    ├── test_roi_labels.py
    ├── test_static_metrics.py
    ├── test_dynamic_io.py
    └── test_hmm_auto_n_states.py
```

## Development Commands

```bash
# Install dependencies
pip install -r requirements.txt

# Run all tests
pytest tests/

# Run a single test file
pytest tests/test_static_metrics.py -v

# Run CLI pipeline
python -m brainnet.main /path/to/bids_dataset --subject 01 --task rest

# Start web application
python -m brainnet.web_app
```

## Architecture

**Pipeline flow:**
```
BIDS Dataset → DatasetIndex → Preprocessing → ROI Extraction
    ├── StaticAnalyzer → ConnectivityMatrix → GraphMetrics
    └── DynamicAnalyzer → DynamicStateModel → Temporal Dynamics
        → ReportGenerator (Plotly HTML + optional LLM summary)
```

**Key abstractions:**
- **Dataclass configs** (`PreprocConfig`, `DynamicConfig`, `ReportConfig`) — all have `.validate()` methods
- **Analyzer classes** (`StaticAnalyzer`, `DynamicAnalyzer`) — orchestrate analysis pipelines
- **Data containers** (`ConnectivityMatrix`, `GraphMetrics`, `DynamicStateModel`, `PreprocessedData`) — immutable results
- **ABC-based steps** (`ProcessingStep` in `preprocessing_full.py`) — extensible preprocessing

**Two-tier module design:** Legacy wrapper modules (`static_analysis.py`, `dynamic_analysis.py`) re-export from the newer modular packages (`static/`, `dynamic/`).

## Code Conventions

### Naming
- **Modules/functions/variables:** `snake_case`
- **Classes:** `PascalCase`
- **Private members:** `_prefixed`
- **Module-level loggers:** `logger = logging.getLogger(__name__)`

### Type Hints
- Use `from __future__ import annotations` at the top of every module
- Full type annotations on function signatures and dataclass fields
- Use `X | None` or `Optional[X]` for optional types

### Docstrings
- **NumPy-style** with `Parameters`, `Returns`, `Raises`, `Examples` sections
- Include `>>>` interactive examples where helpful

### Optional Dependencies
Always wrap optional imports in try/except, assigning `None` on failure:
```python
try:
    import nibabel as nib
except ImportError:
    nib = None
```
Check before use and provide fallback behavior or raise `RuntimeError`.

### Section Comments
Use dash-delimited section headers within modules:
```python
# -- discovery ---------------------------------------------------------
```

## Key Patterns to Follow

1. **Config validation** — Every config dataclass should have a `.validate()` method that raises `ValueError` with descriptive messages
2. **Graceful degradation** — Optional features (LLM summaries, HMM analysis, NIfTI loading) fall back gracefully when dependencies are missing
3. **External tool checks** — Use `shutil.which()` before calling external executables (FSL, ANTs, SPM)
4. **Subprocess calls** — Use `subprocess.run(..., capture_output=True, check=True)` with try/except for `FileNotFoundError` and `CalledProcessError`
5. **Package exports** — Update `__init__.py` `__all__` when adding new public classes

## Testing

- **Framework:** pytest
- **Patterns used:**
  - `monkeypatch` for mocking external calls
  - `tmp_path` fixture for temporary file operations
  - `@pytest.mark.parametrize` for parameterized tests
  - `pytest.importorskip("hmmlearn")` for optional dependency tests
- **Test naming:** `test_<module_name>.py` files with `test_<behavior>()` functions
- **No CI/CD pipeline** — tests run locally

## External Dependencies

Some features require external neuroimaging software (not bundled):
- **FSL** (`FSLDIR` env var) — slice timing, motion correction, spatial normalization
- **SPM** (MATLAB/Octave) — alternative preprocessing steps
- **ANTs** — registration and normalization

## Environment Variables

| Variable | Purpose |
|---|---|
| `FSLDIR` | FSL installation directory |
| `OPENNEURO_CACHE_DIR` | Override OpenNeuro cache location (default: `~/.cache/openneuro`) |
| `OPENAI_API_KEY` | LLM interpretation features (or `~/.openai_api_key` file) |

## Database

The web app uses SQLite (`brainnet.db`) with tables for patients, MRI images, features, and OpenNeuro dataset tracking. Schema is auto-created by `web_app.py` on startup.
