# BrainNet Preprocessing Pipeline

This project provides a modular preprocessing pipeline for functional MRI data. Several steps can optionally rely on external neuroimaging software packages. These tools are **not** bundled with the library and must be installed separately when the corresponding method is enabled.

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
