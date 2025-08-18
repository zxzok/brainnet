# BrainNet Preprocessing Pipeline

This project provides a modular preprocessing pipeline for functional MRI data. Several steps can optionally rely on external neuroimaging software packages. These tools are **not** bundled with the library and must be installed separately when the corresponding method is enabled.

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
