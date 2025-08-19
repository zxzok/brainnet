# BrainNet Preprocessing Pipeline

This project provides a modular preprocessing pipeline for functional MRI data. Several steps can optionally rely on external neuroimaging software packages. These tools are **not** bundled with the library and must be installed separately when the corresponding method is enabled.

## Repository structure

The library is split into small, composable modules.  Key components are:

- `data_management.py` & `openneuro_client.py` – indexing BIDS datasets and
  downloading example data from OpenNeuro.
- `preprocessing.py` / `preprocessing_full.py` – wrapper classes for common
  fMRI preprocessing steps such as slice‑timing, motion correction and spatial
  normalisation.
- `static` and `static_analysis.py` – utilities to compute ROI‑to‑ROI
  connectivity matrices and graph‑theoretic metrics.
- `dynamic` and `dynamic_analysis.py` – sliding‑window, HMM and
  co‑activation‑pattern analyses for studying time‑varying connectivity.
- `visualization.py` & `templates/` – HTML report generation.
- `web_app.py` – a small Flask application with an SQLite backend for uploading
  images, running analyses and viewing reports.

## Running the full pipeline

The ``brainnet.main`` module ties together dataset indexing, preprocessing,
static and dynamic connectivity analyses, and report generation. If you have
an OpenNeuro dataset ID, the data can be fetched automatically and processed
in a single command:

```bash
python -m brainnet.main --openneuro-id ds000114 --subject 01 --task rest
```

To generate a report for a patient already stored in the web application's
database you can call the CLI with ``--patient-id``. The report will include a
short interpretation produced by an optional Large Language Model (LLM):

```bash
python -m brainnet.main --patient-id 1
```

LLM access requires an API key for either OpenAI or a locally available
transformer model. Set the ``OPENAI_API_KEY`` environment variable or place the
key in ``~/.openai_api_key``. If no key or model is available the report will
fall back to a simple rule-based summary.

Within the Flask web interface a **Generate Report** button is available on the
patient detail page and an equivalent endpoint is exposed at
``/patient/<patient_id>/report``.

## Dataset indexing

The :class:`brainnet.data_management.DatasetIndex` helper can enumerate
BIDS‑style datasets across multiple datatypes. For example:

```python
from brainnet.data_management import DatasetIndex

index = DatasetIndex('/path/to/bids', datatypes=['func', 'anat', 'dwi'])
subjects = index.list_subjects()
anat_imgs = index.get_files('anat', subjects[0])
for img in anat_imgs:
    print(img.path, img.metadata.get('acq'))
```

Metadata from file names (e.g. ``acq-highres``) and sidecar JSON files
are accessible via the :pyattr:`~brainnet.data_management.BIDSFile.metadata`
attribute.

## OpenNeuro downloads and caching

The optional [`openneuro-py`](https://github.com/brainlife/openneuro-py)
dependency allows the pipeline to fetch example datasets from
[OpenNeuro](https://openneuro.org). Using this feature requires outbound
HTTPS access to `openneuro.org` so the environment must permit network
connections. Downloaded datasets are cached under
`~/.cache/openneuro` by default; set the `OPENNEURO_CACHE_DIR`
environment variable to change the cache location. For more robust and
resumable large-file transfers you may also install the optional
`datalad` package.

## External Dependencies

| Step | Method | Required command |
|------|--------|-----------------|
| Slice timing correction | `fsl` | `slicetimer` |
| Motion correction | `fsl` | `mcflirt` |
| Spatial normalisation | `fsl` | `flirt` |
| Spatial normalisation | `ants` | `antsRegistration`, `antsApplyTransforms` |
| Any SPM based step | `spm` | MATLAB/Octave with SPM on the MATLAB path |
| Dynamic state features | `networkx` | `networkx` Python package |

### FSL

1. Download and install [FSL](https://fsl.fmrib.ox.ac.uk/fsl/fslwiki/FslInstallation).
2. Set the `FSLDIR` environment variable and source the `fsl.sh` script, e.g.:
   ```bash
   export FSLDIR=/usr/local/fsl
   source $FSLDIR/etc/fslconf/fsl.sh
   export PATH="$FSLDIR/bin:$PATH"
   ```

### SPM

1. Install MATLAB or Octave.
2. Download [SPM](https://www.fil.ion.ucl.ac.uk/spm/software/).
3. Add the SPM directory to the MATLAB path and ensure the `matlab` (or `octave`) command is available in your `PATH`.

### ANTs

1. Install [ANTs](https://antsx.github.io/).
2. Make sure `antsRegistration` and `antsApplyTransforms` are on your `PATH` (e.g., `conda install -c conda-forge ants`).

### Dynamic state features

The dynamic analysis utilities can compute graph metrics such as
global efficiency and modularity for each connectivity state. These
features require the optional dependency [`networkx`](https://networkx.org/):

```bash
pip install networkx
```

## Example

```python
from preprocessing_full import (
    PreprocessPipeline, PreprocessPipelineConfig,
    SliceTimingConfig, MotionCorrectionConfig, SpatialNormalizationConfig
)

config = PreprocessPipelineConfig(
    slice_timing=SliceTimingConfig(method='fsl'),
    motion=MotionCorrectionConfig(method='fsl'),
    spatial_norm=SpatialNormalizationConfig(method='ants', template_path='tpl.nii.gz')
)
pipe = PreprocessPipeline(config)
outputs = pipe.run('func.nii.gz')
```

Enable the appropriate methods only after the corresponding software has been installed and configured.

## Static connectivity

Static network analysis computes a single connectivity matrix from the full
ROI time series and extracts graph‑theoretic descriptors.  The
``brainnet.static_analysis`` module wraps these utilities in a convenient
``StaticAnalyzer`` class:

```python
from brainnet.static_analysis import StaticAnalyzer

analyzer = StaticAnalyzer(threshold=0.2)
conn = analyzer.compute_connectivity(roi_ts, labels)
metrics = analyzer.compute_graph_metrics(conn)
print(metrics.global_metrics["global_efficiency"])
```

The low‑level implementations live in the modular ``brainnet.static``
subpackage and can be used independently for custom pipelines.

## Dynamic connectivity

The `brainnet.dynamic` module performs sliding-window functional connectivity analyses. For K‑means clustering you can allow
the library to recommend the number of states via silhouette scores, while for Gaussian HMMs the optimal state count can be chosen automatically using information criteria (BIC or AIC):

```python
from brainnet.dynamic import DynamicConfig, DynamicAnalyzer

# K-means with automatic K selection
cfg = DynamicConfig(window_length=30, step=5, auto_n_states=True)
model = DynamicAnalyzer(cfg).analyse(roi_timeseries)

# HMM using BIC to choose the number of states
cfg_hmm = DynamicConfig(window_length=30, step=5, method='hmm', auto_n_states=True, n_states_criterion='bic')
model_hmm = DynamicAnalyzer(cfg_hmm).analyse(roi_timeseries)
print(model_hmm.n_states)
```

Setting `auto_n_states=True` triggers an internal evaluation of candidate `K` values and overrides `n_states` with the best recommendation.

## Web application

Running ``python -m brainnet.web_app`` launches a Flask server that exposes the
pipeline through a browser interface.  The application stores patient records
and computed features in ``brainnet.db`` and allows uploading scans, triggering
the preprocessing/analysis workflow in the background and downloading the
resulting HTML report.  The same report can be retrieved programmatically via
``/patient/<id>/report``.
