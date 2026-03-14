# CLAUDE.md — BrainNet

## Project Overview

BrainNet is a modular Python platform for functional MRI (fMRI) preprocessing and connectivity analysis. It provides BIDS dataset indexing, OpenNeuro integration, fMRI preprocessing, static/dynamic functional connectivity analysis, graph-theoretic metrics, interactive visualizations, and a Flask web interface for patient/data management.

## Repository Structure

```
brainnet/
├── main.py                     # CLI entry point, pipeline orchestration
├── web_app.py                  # Flask web app (~1000 lines): routes, DB, analysis
├── __init__.py                 # Package exports with graceful optional imports
├── requirements.txt            # Python dependencies
│
├── data_management.py          # BIDS indexing (DatasetIndex, BIDSFile, DatasetManager)
├── openneuro_client.py         # OpenNeuro GraphQL API client
├── preprocessing.py            # Simple preprocessing pipeline (Preprocessor)
├── preprocessing_full.py       # Modular preprocessing with ABC steps (PreprocessPipeline)
├── static_analysis.py          # StaticAnalyzer wrapper (re-exports from static/)
├── dynamic_analysis.py         # Dynamic analysis wrapper (re-exports from dynamic/)
├── visualization.py            # Plotly HTML report generation (ReportGenerator)
├── llm_interpretation.py       # LLM result summarization (OpenAI / transformers)
├── templates.py                # Atlas template management (AAL, Harvard-Oxford, Schaefer)
├── multimodal.py               # Multi-modal data handling
│
├── static/                     # Modular static connectivity package
│   ├── connectivity.py         #   ConnectivityMatrix, Pearson correlation
│   ├── metrics.py              #   GraphMetrics (degree, clustering, efficiency)
│   └── analyzer.py             #   StaticAnalyzer orchestrator
│
├── dynamic/                    # Modular dynamic connectivity package
│   ├── config.py               #   DynamicConfig dataclass
│   ├── analyzer.py             #   DynamicAnalyzer orchestrator
│   ├── model.py                #   DynamicStateModel, DynamicMetrics
│   ├── window.py               #   Sliding window computation
│   ├── kmeans.py               #   K-means state identification
│   ├── hmm.py                  #   Gaussian HMM analysis
│   ├── cap.py                  #   Co-activation pattern analysis
│   ├── metrics.py              #   Temporal stats (occupancy, dwell time, transitions)
│   ├── state_features.py       #   Graph metrics per state
│   └── io.py                   #   Save/load results to disk
│
├── templates/                  # 19 Jinja2 HTML templates (Bootstrap 5)
│   ├── base.html               #   Layout: navbar (Home, Patients, Data Hub, System)
│   ├── data_hub.html           #   Downloaded datasets + active downloads
│   ├── openneuro.html          #   OpenNeuro dataset search/list
│   ├── openneuro_detail.html   #   Dataset metadata + download actions
│   ├── dataset_browse.html     #   Subject/session/run explorer with checkboxes
│   ├── features_detail.html    #   Feature visualization (Plotly heatmap, state plots)
│   ├── system_status.html      #   Dependency status check
│   └── ...                     #   patient/image CRUD pages
│
└── tests/                      # 6 pytest test modules
```

## Development Commands

```bash
# Install all dependencies
pip install -r requirements.txt
pip install nibabel nilearn scikit-learn plotly hmmlearn

# Run tests
pytest tests/

# Start web app
python web_app.py              # Runs on http://localhost:6525

# CLI pipeline
python main.py /path/to/bids --subject 01 --task rest
python main.py --openneuro-id ds000114 --subject 01 --task rest
```

## Architecture

### Pipeline flow
```
BIDS/NIfTI → DatasetIndex → PreprocessPipeline → ROI time series
    ├── StaticAnalyzer → ConnectivityMatrix → GraphMetrics
    └── DynamicAnalyzer → DynamicStateModel (states, occupancy, transitions)
        → features stored in SQLite
        → Plotly visualizations (heatmap, state timeline, bar charts)
        → ReportGenerator (HTML report + optional LLM summary)
```

### Key abstractions
- **Dataclass configs** — `PreprocConfig`, `PreprocessPipelineConfig`, `DynamicConfig`, `ReportConfig` with `.validate()` methods
- **Analyzer classes** — `StaticAnalyzer`, `DynamicAnalyzer` orchestrate pipelines
- **Data containers** — `ConnectivityMatrix`, `GraphMetrics`, `DynamicStateModel`, `PreprocessedData`
- **ABC processing steps** — `ProcessingStep` base in `preprocessing_full.py`
- **Two-tier modules** — Legacy wrappers (`static_analysis.py`) re-export from modular packages (`static/`)

### Web app design
- **Lazy imports** — Heavy deps (numpy/scipy/nibabel) imported inside functions; app starts without them
- **Background tasks** — `ThreadPoolExecutor(max_workers=2)` for downloads and analysis
- **Status tracking** — `download_tasks` table with polling via `GET /api/download_status/<id>`
- **Adaptive analysis** — Dynamic params auto-adjust to data length: `window_length = min(30, max(5, T/5))`

### Database tables
| Table | Purpose |
|---|---|
| `patients` | Patient records (id, name, age, sex, diagnosis) |
| `mri_images` | MRI image records (path, type, patient FK) |
| `features` | Computed features (name, value, type: static/dynamic/static_node) |
| `openneuro_datasets` | Downloaded datasets (metadata, path, status: downloading/ready/failed) |
| `download_tasks` | Download attempt log (status, error_message) |

## Web Routes Summary

### Pages
| Route | Purpose |
|---|---|
| `GET /` | Home — patient list |
| `GET /patients` | Patient list |
| `GET /patient/<id>` | Patient detail (images, OpenNeuro datasets) |
| `GET /data` | Data Hub — downloaded datasets + active downloads |
| `GET /openneuro` | OpenNeuro search + browse |
| `GET /openneuro/<id>` | Dataset detail + download actions |
| `GET /openneuro/<id>/browse` | Browse dataset subjects/runs, select for analysis |
| `GET /features/<image_id>` | Feature visualization (heatmap, state plots, metrics) |
| `GET /system/status` | Dependency check |

### Actions
| Route | Purpose |
|---|---|
| `POST /openneuro/download` | Start dataset download |
| `POST /openneuro/<id>/delete` | Delete downloaded dataset |
| `POST /openneuro/<id>/analyze` | Run analysis on selected runs |
| `POST /features/<id>/recompute` | Re-run analysis |
| `POST /features/<id>/delete` | Delete computed features |
| `GET /api/features/<id>/export` | Export features as JSON |

## Code Conventions

### Naming
- **Modules/functions/variables:** `snake_case`
- **Classes:** `PascalCase`
- **Private members:** `_prefixed`
- **Module-level loggers:** `logger = logging.getLogger(__name__)`

### Type Hints
- `from __future__ import annotations` at top of every module
- Full annotations on function signatures and dataclass fields

### Docstrings
- **NumPy-style** with `Parameters`, `Returns`, `Raises`, `Examples` sections

### Optional Dependencies
```python
try:
    import nibabel as nib
except ImportError:
    nib = None
```
Check before use, provide fallback or raise `RuntimeError`.

### Key Patterns
1. Config dataclasses with `.validate()` raising `ValueError`
2. Graceful degradation when deps missing (try/except → None)
3. `shutil.which()` before calling external executables
4. `subprocess.run(..., capture_output=True, check=True)` with error handling
5. Update `__init__.py` `__all__` when adding public classes
6. Section comments: `# -- discovery ---------------------------------------------------------`

## Testing

- **Framework:** pytest
- **Fixtures:** `monkeypatch`, `tmp_path`, `@pytest.mark.parametrize`, `pytest.importorskip`
- **Naming:** `test_<module>.py` → `test_<behavior>()`
- **No CI/CD** — tests run locally

## Environment Variables

| Variable | Purpose |
|---|---|
| `FSLDIR` | FSL installation directory |
| `OPENNEURO_CACHE_DIR` | OpenNeuro cache (default: `~/.cache/openneuro`) |
| `OPENAI_API_KEY` | LLM features (or `~/.openai_api_key` file) |
