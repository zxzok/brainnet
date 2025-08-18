"""
preprocessing_full
==================

This module defines a more extensible and modular preprocessing
pipeline for functional MRI data, built upon modern software
architecture principles.  Rather than a monolithic function that
applies several operations in sequence, this design decomposes the
preprocessing into discrete steps with well defined interfaces.  Each
step encapsulates a specific transformation or estimation and can be
individually configured, enabled or replaced.  The overall pipeline
coordinates loading of raw data, execution of the configured steps
and assembly of the outputs.

Overview of the pipeline components
-----------------------------------

1. **Image loading** – The raw NIfTI image is loaded from disk using
   :func:`nibabel.load` into a 4D array along with its affine
   transformation.  Sequence parameters such as the repetition time
   (TR) and slice timing information are extracted from the header or
   provided metadata.
2. **Slice timing correction** – Corrects for differences in slice
   acquisition times within a volume.  Implementations may shift
   voxel time courses by fractional TRs based on a slice timing table.
3. **Motion correction** – Realigns volumes in the time series to
   remove motion artefacts.  Rigid body registration algorithms
   available in external software (e.g. FSL's MCFLIRT, SPM's
   Realign) are typically used.  If such tools are unavailable the
   step may be skipped or approximate methods applied.
4. **Spatial normalisation** – Optionally transforms the data to a
   standard anatomical space (e.g. MNI) using an anatomical template
   and a registration algorithm.  This step facilitates group
   comparisons but may be omitted when working in native space.
5. **Spatial smoothing** – Applies a Gaussian filter to each volume
   with a configurable full width at half maximum (FWHM), trading off
   spatial resolution for improved signal to noise ratio.
6. **Temporal filtering** – Removes slow drifts and/or high frequency
   noise from the voxel or ROI time series using a Butterworth
   high‑pass, low‑pass or band‑pass filter depending on the specified
   cutoff frequencies.
7. **Nuisance regression** – Regresses out confounding signals
   (e.g. motion parameters, white matter/CSF averages) using
   ordinary least squares.
8. **ROI extraction** – Condenses the voxel time series into mean
   time series for each region of interest defined by a labelled
   atlas.
9. **Quality control metrics** – Computes simple QC metrics such as
   temporal signal to noise ratio (tSNR) for later inspection.

Steps 2–4 are complex operations for which reliable implementations
generally depend on external neuroimaging software.  This module
defines their interfaces and placeholders; users should plug in
their preferred tools or wrappers.  For example, one could
implement :class:`MotionCorrectionStep` using Nipype interfaces to
FSL or SPM.  A minimal fallback motion correction implementation is
provided for demonstration.

Note
----
The full pipeline requires ``nibabel`` for loading NIfTI images.
External dependencies like FSL/SPM/Nipype are not included here
but can be integrated by extending the relevant step classes.  The
default implementations provided for slice timing, motion
correction and normalisation are no-ops; they simply return the
input data.  Users should implement or substitute proper methods
where appropriate.
"""

from __future__ import annotations

import os
import subprocess
import logging
import shutil
import tempfile
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Sequence

import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter
from scipy.signal import butter, filtfilt

try:
    import nibabel as nib
except Exception:
    nib = None  # nibabel may not be available


logger = logging.getLogger(__name__)


class ProcessingStep(ABC):
    """Abstract base class for a preprocessing step.

    Each step operates on a 4D data array of shape (X, Y, Z, T) and
    may also access or update a metadata dictionary.  Steps may
    produce auxiliary outputs such as motion parameters which can be
    stored in the metadata under agreed keys.
    """

    @abstractmethod
    def run(self, data: np.ndarray, meta: Dict[str, Any]) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Execute the processing step.

        Parameters
        ----------
        data : np.ndarray
            4D array (X,Y,Z,T) representing the fMRI time series.
        meta : dict
            Mutable dictionary of metadata which may include TR,
            slice timing table, motion parameters, etc.

        Returns
        -------
        Tuple[np.ndarray, dict]
            The transformed data array and updated metadata.  The
            metadata should include any new information produced by
            the step.  The input metadata may be modified in place.
        """
        raise NotImplementedError


# -----------------------------------------------------------------------------
# Config dataclasses

@dataclass
class SliceTimingConfig:
    enabled: bool = True
    method: str = 'none'  # 'none', 'fsl', 'spm'
    slice_times: Optional[Sequence[float]] = None  # length equals number of slices
    extra_args: Dict[str, Any] = field(default_factory=dict)


@dataclass
class MotionCorrectionConfig:
    enabled: bool = True
    method: str = 'none'  # 'none', 'fsl', 'spm', 'simple'
    reference_volume: int = 0  # index of reference volume for alignment
    extra_args: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SpatialNormalizationConfig:
    enabled: bool = False
    method: str = 'none'  # 'none', 'fsl', 'spm', 'ants'
    template_path: Optional[str] = None
    extra_args: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SmoothingConfig:
    enabled: bool = False
    fwhm: float = 6.0  # voxels


@dataclass
class TemporalFilterConfig:
    enabled: bool = False
    low_cut: Optional[float] = None
    high_cut: Optional[float] = None
    order: int = 2


@dataclass
class NuisanceRegressionConfig:
    enabled: bool = False
    confound_columns: Optional[List[str]] = None
    detrend: bool = True


@dataclass
class RoiExtractionConfig:
    enabled: bool = False
    atlas_path: Optional[str] = None


@dataclass
class PreprocessPipelineConfig:
    """Aggregate configuration for the preprocessing pipeline."""
    slice_timing: SliceTimingConfig = field(default_factory=SliceTimingConfig)
    motion: MotionCorrectionConfig = field(default_factory=MotionCorrectionConfig)
    spatial_norm: SpatialNormalizationConfig = field(default_factory=SpatialNormalizationConfig)
    smoothing: SmoothingConfig = field(default_factory=SmoothingConfig)
    temporal_filter: TemporalFilterConfig = field(default_factory=TemporalFilterConfig)
    nuisance: NuisanceRegressionConfig = field(default_factory=NuisanceRegressionConfig)
    roi_extraction: RoiExtractionConfig = field(default_factory=RoiExtractionConfig)
    retain_4d: bool = False


# -----------------------------------------------------------------------------
# Step implementations

class SliceTimingStep(ProcessingStep):
    """Apply slice timing correction based on acquisition times.

    This implementation is a placeholder.  A proper implementation
    should resample the time series along the temporal dimension to
    align slice acquisition times.  If ``slice_times`` is not
    provided in the configuration, the step will simply return the
    input data unchanged.
    """

    def __init__(self, config: SliceTimingConfig) -> None:
        self.config = config

    def run(self, data: np.ndarray, meta: Dict[str, Any]) -> Tuple[np.ndarray, Dict[str, Any]]:
        if not self.config.enabled or self.config.method == 'none':
            return data, meta
        method = self.config.method.lower()
        if method == 'fsl':
            if nib is None:
                raise RuntimeError("nibabel is required for slice timing but not available")
            cmd_path = shutil.which('slicetimer')
            if cmd_path is None:
                logger.error('FSL slicetimer not found in PATH')
                raise RuntimeError('slicetimer command not found')
            tr = meta.get('TR')
            if tr is None:
                logger.error('TR is required for FSL slicetimer')
                raise RuntimeError('TR missing for slice timing')
            with tempfile.TemporaryDirectory() as tmpdir:
                in_path = os.path.join(tmpdir, 'in.nii.gz')
                out_path = os.path.join(tmpdir, 'out.nii.gz')
                img = nib.Nifti1Image(data, meta.get('affine', np.eye(4)))
                nib.save(img, in_path)
                cmd = [cmd_path, '-i', in_path, '-o', out_path, '--repeat', str(tr)]
                if self.config.slice_times is not None:
                    times_file = os.path.join(tmpdir, 'times.txt')
                    np.savetxt(times_file, np.array(self.config.slice_times), fmt='%.6f')
                    cmd.extend(['--tcustom', times_file])
                for k, v in self.config.extra_args.items():
                    cmd.append(str(k))
                    if v is not None:
                        cmd.append(str(v))
                try:
                    subprocess.run(cmd, check=True, capture_output=True)
                    out_img = nib.load(out_path)
                    data = np.asanyarray(out_img.get_fdata())
                except FileNotFoundError:
                    logger.error('slicetimer command not found')
                    raise RuntimeError('slicetimer command not found')
                except subprocess.CalledProcessError as e:
                    logger.error('slicetimer failed: %s', e.stderr.decode('utf-8', 'ignore'))
                    raise RuntimeError('slice timing correction failed') from e
            return data, meta
        elif method == 'spm':
            matlab = shutil.which('matlab') or shutil.which('octave')
            if matlab is None:
                logger.error('MATLAB/Octave not found for SPM slice timing')
                raise RuntimeError('MATLAB/Octave command not found')
            if nib is None:
                raise RuntimeError('nibabel is required for slice timing but not available')
            with tempfile.TemporaryDirectory() as tmpdir:
                in_path = os.path.join(tmpdir, 'in.nii')
                out_path = os.path.join(tmpdir, 'out.nii')
                img = nib.Nifti1Image(data, meta.get('affine', np.eye(4)))
                nib.save(img, in_path)
                script = (
                    "try, spm('defaults','fmri'); "
                    f"P='{in_path}'; "
                    "spm_slice_timing(P); "
                    f"movefile('a{os.path.basename(in_path)}','{out_path}'); "
                    "catch e, disp(getReport(e)); exit(1); end; exit;"
                )
                cmd = [matlab, '-nodisplay', '-nosplash', '-r', script]
                try:
                    subprocess.run(cmd, check=True, capture_output=True)
                    out_img = nib.load(out_path)
                    data = np.asanyarray(out_img.get_fdata())
                except FileNotFoundError:
                    logger.error('MATLAB/Octave not found for SPM slice timing')
                    raise RuntimeError('MATLAB/Octave command not found')
                except subprocess.CalledProcessError as e:
                    logger.error('SPM slice timing failed: %s', e.stderr.decode('utf-8', 'ignore'))
                    raise RuntimeError('SPM slice timing failed') from e
            return data, meta
        else:
            raise ValueError(f"Unknown slice timing method '{self.config.method}'")


class MotionCorrectionStep(ProcessingStep):
    """Realign volumes to remove head motion.

    By default this step does nothing, but users can specify
    ``method='simple'`` to align each volume to a reference using a
    rigid‑body transformation estimated via centre of mass alignment.
    Other options like ``'fsl'`` or ``'spm'`` would invoke external
    tools (if installed) via subprocess or Nipype.  The motion
    parameters are stored in the metadata under the key ``'motion_params'``.
    """

    def __init__(self, config: MotionCorrectionConfig) -> None:
        self.config = config

    def run(self, data: np.ndarray, meta: Dict[str, Any]) -> Tuple[np.ndarray, Dict[str, Any]]:
        if not self.config.enabled or self.config.method == 'none':
            return data, meta
        method = self.config.method.lower()
        if method == 'simple':
            # Simple alignment: compute centre of mass shifts and apply translation
            # This is a toy example and not a valid motion correction algorithm
            ref_idx = self.config.reference_volume
            ref_vol = data[..., ref_idx]
            # compute centre of mass of reference
            coords = np.array(np.meshgrid(
                np.arange(ref_vol.shape[0]),
                np.arange(ref_vol.shape[1]),
                np.arange(ref_vol.shape[2]),
                indexing='ij'
            ))
            # weights by signal intensity
            w = ref_vol
            ref_com = np.array([(coords[i] * w).sum() / w.sum() for i in range(3)])
            motion_params = []
            corrected = np.empty_like(data)
            for t in range(data.shape[-1]):
                vol = data[..., t]
                wv = vol
                vol_com = np.array([(coords[i] * wv).sum() / (wv.sum() + 1e-8) for i in range(3)])
                shift = ref_com - vol_com
                # apply translation by shift (nearest neighbour for speed)
                corrected[..., t] = self._translate_volume(vol, shift)
                motion_params.append(np.hstack([shift, [0, 0, 0]]))  # rotation set to zero
            meta['motion_params'] = np.array(motion_params)
            return corrected, meta
        elif method == 'fsl':
            if nib is None:
                raise RuntimeError('nibabel is required for motion correction but not available')
            cmd_path = shutil.which('mcflirt')
            if cmd_path is None:
                logger.error('FSL mcflirt not found in PATH')
                raise RuntimeError('mcflirt command not found')
            with tempfile.TemporaryDirectory() as tmpdir:
                in_path = os.path.join(tmpdir, 'in.nii.gz')
                out_path = os.path.join(tmpdir, 'out.nii.gz')
                img = nib.Nifti1Image(data, meta.get('affine', np.eye(4)))
                nib.save(img, in_path)
                cmd = [cmd_path, '-in', in_path, '-out', out_path, '-refvol', str(self.config.reference_volume)]
                for k, v in self.config.extra_args.items():
                    cmd.append(str(k))
                    if v is not None:
                        cmd.append(str(v))
                try:
                    subprocess.run(cmd, check=True, capture_output=True)
                    out_img = nib.load(out_path)
                    data = np.asanyarray(out_img.get_fdata())
                    par_path = out_path + '.par'
                    if os.path.exists(par_path):
                        meta['motion_params'] = np.loadtxt(par_path)
                except FileNotFoundError:
                    logger.error('mcflirt command not found')
                    raise RuntimeError('mcflirt command not found')
                except subprocess.CalledProcessError as e:
                    logger.error('mcflirt failed: %s', e.stderr.decode('utf-8', 'ignore'))
                    raise RuntimeError('FSL mcflirt failed') from e
            return data, meta
        elif method == 'spm':
            matlab = shutil.which('matlab') or shutil.which('octave')
            if matlab is None:
                logger.error('MATLAB/Octave not found for SPM motion correction')
                raise RuntimeError('MATLAB/Octave command not found')
            if nib is None:
                raise RuntimeError('nibabel is required for motion correction but not available')
            with tempfile.TemporaryDirectory() as tmpdir:
                in_path = os.path.join(tmpdir, 'in.nii')
                out_path = os.path.join(tmpdir, 'out.nii')
                img = nib.Nifti1Image(data, meta.get('affine', np.eye(4)))
                nib.save(img, in_path)
                script = (
                    "try, spm('defaults','fmri'); "
                    f"P='{in_path}'; "
                    "spm_realign(P); spm_reslice(P); "
                    f"movefile('r{os.path.basename(in_path)}','{out_path}'); "
                    f"save('rp_{os.path.basename(in_path).split('.')[0]}.txt',rp_{os.path.basename(in_path).split('.')[0]}); "
                    "catch e, disp(getReport(e)); exit(1); end; exit;"
                )
                cmd = [matlab, '-nodisplay', '-nosplash', '-r', script]
                try:
                    subprocess.run(cmd, check=True, capture_output=True)
                    out_img = nib.load(out_path)
                    data = np.asanyarray(out_img.get_fdata())
                    par_path = os.path.join(tmpdir, f"rp_{os.path.basename(in_path).split('.')[0]}.txt")
                    if os.path.exists(par_path):
                        meta['motion_params'] = np.loadtxt(par_path)
                except FileNotFoundError:
                    logger.error('MATLAB/Octave not found for SPM motion correction')
                    raise RuntimeError('MATLAB/Octave command not found')
                except subprocess.CalledProcessError as e:
                    logger.error('SPM motion correction failed: %s', e.stderr.decode('utf-8', 'ignore'))
                    raise RuntimeError('SPM motion correction failed') from e
            return data, meta
        else:
            raise ValueError(f"Unknown motion correction method '{self.config.method}'")

    def _translate_volume(self, vol: np.ndarray, shift: Sequence[float]) -> np.ndarray:
        """Translate a 3D volume by fractional voxels using nearest neighbour.

        This is a coarse and inefficient implementation used solely for
        demonstration.  For real applications consider using
        scipy.ndimage.shift or nibabel processing.
        """
        from scipy.ndimage import shift as nd_shift
        return nd_shift(vol, shift, order=0, mode='constant', cval=0.0)


class SpatialNormalizationStep(ProcessingStep):
    """Transform data to a standard space.

    No default implementation is provided; this is a placeholder for
    integration with external registration tools such as FSL FLIRT/FNIRT,
    SPM or ANTs.  The template path should be specified in the config.
    """

    def __init__(self, config: SpatialNormalizationConfig) -> None:
        self.config = config

    def run(self, data: np.ndarray, meta: Dict[str, Any]) -> Tuple[np.ndarray, Dict[str, Any]]:
        if not self.config.enabled or self.config.method == 'none':
            return data, meta
        method = self.config.method.lower()
        if method == 'fsl':
            if nib is None:
                raise RuntimeError('nibabel is required for spatial normalization but not available')
            cmd_path = shutil.which('flirt')
            if cmd_path is None:
                logger.error('FSL flirt not found in PATH')
                raise RuntimeError('flirt command not found')
            if not self.config.template_path:
                raise RuntimeError('template_path must be provided for FSL normalization')
            with tempfile.TemporaryDirectory() as tmpdir:
                in_path = os.path.join(tmpdir, 'in.nii.gz')
                out_path = os.path.join(tmpdir, 'out.nii.gz')
                img = nib.Nifti1Image(data, meta.get('affine', np.eye(4)))
                nib.save(img, in_path)
                cmd = [cmd_path, '-in', in_path, '-ref', self.config.template_path, '-out', out_path]
                for k, v in self.config.extra_args.items():
                    cmd.append(str(k))
                    if v is not None:
                        cmd.append(str(v))
                try:
                    subprocess.run(cmd, check=True, capture_output=True)
                    out_img = nib.load(out_path)
                    data = np.asanyarray(out_img.get_fdata())
                except FileNotFoundError:
                    logger.error('flirt command not found')
                    raise RuntimeError('flirt command not found')
                except subprocess.CalledProcessError as e:
                    logger.error('flirt failed: %s', e.stderr.decode('utf-8', 'ignore'))
                    raise RuntimeError('FSL flirt failed') from e
            return data, meta
        elif method == 'ants':
            if nib is None:
                raise RuntimeError('nibabel is required for spatial normalization but not available')
            reg_cmd = shutil.which('antsRegistration')
            apply_cmd = shutil.which('antsApplyTransforms')
            if reg_cmd is None or apply_cmd is None:
                logger.error('ANTs commands not found in PATH')
                raise RuntimeError('ANTs commands not found')
            if not self.config.template_path:
                raise RuntimeError('template_path must be provided for ANTs normalization')
            with tempfile.TemporaryDirectory() as tmpdir:
                in_path = os.path.join(tmpdir, 'in.nii.gz')
                out_prefix = os.path.join(tmpdir, 'ants_')
                warped = out_prefix + 'Warped.nii.gz'
                img = nib.Nifti1Image(data, meta.get('affine', np.eye(4)))
                nib.save(img, in_path)
                cmd = [
                    reg_cmd,
                    '--dimensionality', '3',
                    '--output', out_prefix,
                    '--transform', 'rigid[0.1]',
                    '--metric', f"MI[{self.config.template_path},{in_path},1,32,Regular,0.25]",
                    '--convergence', '[1000x500x250x0,1e-6,10]',
                    '--shrink-factors', '8x4x2x1',
                    '--smoothing-sigmas', '4x2x1x0vox',
                    '--use-histogram-matching', '0',
                    '--interpolation', 'Linear'
                ]
                for k, v in self.config.extra_args.items():
                    cmd.append(str(k))
                    if v is not None:
                        cmd.append(str(v))
                try:
                    subprocess.run(cmd, check=True, capture_output=True)
                    apply = [
                        apply_cmd,
                        '-d', '3',
                        '-i', in_path,
                        '-r', self.config.template_path,
                        '-o', warped,
                        '-t', out_prefix + '0GenericAffine.mat'
                    ]
                    subprocess.run(apply, check=True, capture_output=True)
                    out_img = nib.load(warped)
                    data = np.asanyarray(out_img.get_fdata())
                except FileNotFoundError:
                    logger.error('ANTs commands not found')
                    raise RuntimeError('ANTs commands not found')
                except subprocess.CalledProcessError as e:
                    logger.error('ANTs registration failed: %s', e.stderr.decode('utf-8', 'ignore'))
                    raise RuntimeError('ANTs normalization failed') from e
            return data, meta
        elif method == 'spm':
            matlab = shutil.which('matlab') or shutil.which('octave')
            if matlab is None:
                logger.error('MATLAB/Octave not found for SPM normalization')
                raise RuntimeError('MATLAB/Octave command not found')
            if nib is None:
                raise RuntimeError('nibabel is required for spatial normalization but not available')
            if not self.config.template_path:
                raise RuntimeError('template_path must be provided for SPM normalization')
            with tempfile.TemporaryDirectory() as tmpdir:
                in_path = os.path.join(tmpdir, 'in.nii')
                out_path = os.path.join(tmpdir, 'out.nii')
                img = nib.Nifti1Image(data, meta.get('affine', np.eye(4)))
                nib.save(img, in_path)
                script = (
                    "try, spm('defaults','fmri'); "
                    f"P='{in_path}'; T='{self.config.template_path}'; "
                    "est=spm_vol(T); res=spm_vol(P); "
                    "spm_normalise(est, res, 'param.mat', '', res, ''); "
                    "spm_write_sn(P,'param.mat'); "
                    f"movefile('w{os.path.basename(in_path)}','{out_path}'); "
                    "catch e, disp(getReport(e)); exit(1); end; exit;"
                )
                cmd = [matlab, '-nodisplay', '-nosplash', '-r', script]
                try:
                    subprocess.run(cmd, check=True, capture_output=True)
                    out_img = nib.load(out_path)
                    data = np.asanyarray(out_img.get_fdata())
                except FileNotFoundError:
                    logger.error('MATLAB/Octave not found for SPM normalization')
                    raise RuntimeError('MATLAB/Octave command not found')
                except subprocess.CalledProcessError as e:
                    logger.error('SPM normalization failed: %s', e.stderr.decode('utf-8', 'ignore'))
                    raise RuntimeError('SPM normalization failed') from e
            return data, meta
        else:
            raise ValueError(f"Unknown spatial normalization method '{self.config.method}'")


class SmoothingStep(ProcessingStep):
    """Apply spatial Gaussian smoothing."""

    def __init__(self, config: SmoothingConfig) -> None:
        self.config = config

    def run(self, data: np.ndarray, meta: Dict[str, Any]) -> Tuple[np.ndarray, Dict[str, Any]]:
        if not self.config.enabled or self.config.fwhm <= 0:
            return data, meta
        sigma = self.config.fwhm / (2.35482)
        smoothed = np.empty_like(data)
        for t in range(data.shape[-1]):
            smoothed[..., t] = gaussian_filter(
                data[..., t], sigma=sigma, mode='constant', cval=0.0
            )
        return smoothed, meta


class TemporalFilterStep(ProcessingStep):
    """Apply Butterworth temporal filter along the time axis.

    Depending on which cutoff frequencies are provided this step will
    perform high‑pass, low‑pass or band‑pass filtering.
    """

    def __init__(self, config: TemporalFilterConfig) -> None:
        self.config = config

    def run(self, data: np.ndarray, meta: Dict[str, Any]) -> Tuple[np.ndarray, Dict[str, Any]]:
        if not self.config.enabled:
            return data, meta
        tr = meta.get('TR')
        if tr is None:
            # cannot filter without TR
            return data, meta
        fs = 1.0 / tr
        nyq = 0.5 * fs
        low = self.config.low_cut / nyq if self.config.low_cut else None
        high = self.config.high_cut / nyq if self.config.high_cut else None
        if low is not None and not 0 < low < 1:
            raise ValueError("low_cut must be between 0 and the Nyquist frequency")
        if high is not None and not 0 < high < 1:
            raise ValueError("high_cut must be between 0 and the Nyquist frequency")
        if low is None and high is None:
            return data, meta
        if low is not None and high is not None:
            if low >= high:
                raise ValueError("low_cut must be less than high_cut when both are specified")
            btype = "bandpass"
            wn = [low, high]
        elif high is not None:
            btype = "lowpass"
            wn = high
        else:
            btype = "highpass"
            wn = low
        b, a = butter(self.config.order, wn, btype=btype, analog=False)
        # reshape to (V,T)
        orig_shape = data.shape
        flat = data.reshape(-1, orig_shape[-1])
        # apply filter along axis 1
        filtered = filtfilt(b, a, flat, axis=1)
        data = filtered.reshape(orig_shape)
        return data, meta


class NuisanceRegressionStep(ProcessingStep):
    """Regress out confound signals from voxel time series."""

    def __init__(self, config: NuisanceRegressionConfig) -> None:
        self.config = config

    def run(self, data: np.ndarray, meta: Dict[str, Any]) -> Tuple[np.ndarray, Dict[str, Any]]:
        if not self.config.enabled or not self.config.confound_columns:
            return data, meta
        confounds = meta.get('confounds')
        if confounds is None:
            return data, meta
        regressors = confounds[self.config.confound_columns].values
        # demean and optionally detrend regressors
        regressors = regressors - regressors.mean(axis=0, keepdims=True)
        if self.config.detrend:
            t = np.arange(regressors.shape[0], dtype=float)
            t = (t - t.mean()) / (t.std() + 1e-8)
            regressors = np.column_stack([t, regressors])
        # add intercept
        intercept = np.ones((regressors.shape[0], 1))
        design = np.hstack([intercept, regressors])
        # apply regression to each voxel
        orig_shape = data.shape
        flat = data.reshape(-1, orig_shape[-1])
        pinv = np.linalg.pinv(design)
        betas = pinv @ flat.T  # (p, V)
        fitted = (design @ betas).T
        clean = flat - fitted
        data = clean.reshape(orig_shape)
        return data, meta


class RoiExtractionStep(ProcessingStep):
    """Extract mean ROI time series from an atlas."""

    def __init__(self, config: RoiExtractionConfig) -> None:
        self.config = config
        self._atlas_data: Optional[np.ndarray] = None
        self._labels: Optional[List[int]] = None
        if self.config.enabled and self.config.atlas_path:
            if nib is None:
                raise RuntimeError("nibabel is required for ROI extraction but not available")
            atlas_img = nib.load(self.config.atlas_path)
            self._atlas_data = np.asanyarray(atlas_img.get_fdata())
            labels = np.unique(self._atlas_data)
            labels = labels[labels != 0]
            self._labels = labels.astype(int).tolist()

    def run(self, data: np.ndarray, meta: Dict[str, Any]) -> Tuple[np.ndarray, Dict[str, Any]]:
        if not self.config.enabled:
            return data, meta
        if self._atlas_data is None or self._labels is None:
            raise ValueError("ROI extraction requested but atlas not loaded")
        T = data.shape[-1]
        atlas = self._atlas_data
        labels = self._labels
        roi_ts = np.zeros((T, len(labels)), dtype=float)
        for idx, lbl in enumerate(labels):
            mask = atlas == lbl
            if not np.any(mask):
                continue
            voxels = data[mask, :]
            roi_ts[:, idx] = voxels.mean(axis=0)
        meta['roi_timeseries'] = roi_ts
        meta['roi_labels'] = labels
        return data, meta


class QCMetricsStep(ProcessingStep):
    """Compute basic quality control metrics such as tSNR."""

    def run(self, data: np.ndarray, meta: Dict[str, Any]) -> Tuple[np.ndarray, Dict[str, Any]]:
        # temporal SNR: mean/SD per voxel averaged
        flat = data.reshape(-1, data.shape[-1])
        mean_signal = flat.mean(axis=1)
        std_signal = flat.std(axis=1) + 1e-8
        tSNR = float(np.mean(mean_signal / std_signal))
        qc = meta.get('qc_metrics', {})
        qc['tSNR'] = tSNR
        meta['qc_metrics'] = qc
        return data, meta


# -----------------------------------------------------------------------------
# Pipeline class

class PreprocessPipeline:
    """Coordinate the execution of configured preprocessing steps."""

    def __init__(self, config: PreprocessPipelineConfig) -> None:
        self.config = config
        # instantiate steps based on configuration
        self.steps: List[ProcessingStep] = []
        # Order matters: loading is handled externally in run()
        if config.slice_timing.enabled:
            self.steps.append(SliceTimingStep(config.slice_timing))
        if config.motion.enabled:
            self.steps.append(MotionCorrectionStep(config.motion))
        if config.spatial_norm.enabled:
            self.steps.append(SpatialNormalizationStep(config.spatial_norm))
        if config.smoothing.enabled:
            self.steps.append(SmoothingStep(config.smoothing))
        if config.temporal_filter.enabled:
            self.steps.append(TemporalFilterStep(config.temporal_filter))
        if config.nuisance.enabled:
            self.steps.append(NuisanceRegressionStep(config.nuisance))
        # QC metrics before ROI extraction ensures 4D data available
        self.steps.append(QCMetricsStep())
        if config.roi_extraction.enabled:
            self.steps.append(RoiExtractionStep(config.roi_extraction))

    def run(self, func_path: str, confounds_path: Optional[str] = None) -> Dict[str, Any]:
        """Execute the pipeline on a single functional run.

        Parameters
        ----------
        func_path : str
            Path to the functional NIfTI file.
        confounds_path : str, optional
            Path to a TSV/CSV file containing confound regressors.

        Returns
        -------
        dict
            A dictionary containing outputs including (but not limited to):
            - 'data': cleaned 4D array or None depending on retain_4d
            - 'roi_timeseries': ROI×time matrix if ROI extraction enabled
            - 'roi_labels': list of integer labels for ROI atlas
            - 'qc_metrics': quality metrics
            - 'affine': affine transformation matrix
        """
        if nib is None:
            raise RuntimeError("nibabel is required to load NIfTI images but is not available")
        # load functional image
        img = nib.load(func_path)
        data = np.asanyarray(img.get_fdata())
        affine = img.affine.copy()
        header = img.header
        meta: Dict[str, Any] = {'affine': affine}
        # TR from header zooms
        try:
            meta['TR'] = float(header.get_zooms()[-1])
        except Exception:
            meta['TR'] = None
        # load confounds if provided
        if confounds_path:
            try:
                df = pd.read_csv(confounds_path, sep=None, engine='python')
            except Exception:
                df = pd.read_csv(confounds_path, sep=',')
            meta['confounds'] = df
        # run each processing step
        current_data = data
        for step in self.steps:
            current_data, meta = step.run(current_data, meta)
        # prepare outputs
        outputs: Dict[str, Any] = {}
        if self.config.retain_4d:
            outputs['data'] = current_data
        outputs['affine'] = affine
        # propagate ROI outputs if present
        if 'roi_timeseries' in meta:
            outputs['roi_timeseries'] = meta['roi_timeseries']
            # generate simple ROI names if none provided
            outputs['roi_labels'] = [str(lbl) for lbl in meta.get('roi_labels', [])]
        if 'qc_metrics' in meta:
            outputs['qc_metrics'] = meta['qc_metrics']
        if 'motion_params' in meta:
            outputs['motion_params'] = meta['motion_params']
        return outputs


__all__ = [
    'SliceTimingConfig',
    'MotionCorrectionConfig',
    'SpatialNormalizationConfig',
    'SmoothingConfig',
    'TemporalFilterConfig',
    'NuisanceRegressionConfig',
    'RoiExtractionConfig',
    'PreprocessPipelineConfig',
    'PreprocessPipeline',
]