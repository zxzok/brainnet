"""
preprocessing
=============

This module implements a minimal fMRI preprocessing pipeline.  The
intent is not to replace established pipelines such as fMRIPrep but
rather to provide a simple yet functional example that prepares data
for connectivity analyses.  The pipeline supports spatial smoothing,
temporal filtering and nuisance regression.  It can optionally
extract region‑of‑interest (ROI) time series from a labelled atlas.

Because this module is designed to be self contained and portable, it
relies solely on a handful of standard Python libraries: ``numpy`` for
numerical computations, ``scipy`` for signal processing and
``pandas`` for confound handling.  External neuroimaging packages
like ``nibabel`` are only used if available; if they are missing the
pipeline will raise a clear error when attempting to load NIfTI files.

Preprocessing proceeds in the following stages:

1. **Loading** – A 4D fMRI dataset is loaded from disk using
   :func:`nibabel.load`.  If ``nibabel`` is not available, a
   :class:`RuntimeError` is raised.
2. **Spatial smoothing** – Optionally apply a Gaussian filter to the
   spatial dimensions (x, y, z) of each volume.  The full width at
   half maximum (FWHM) parameter is converted to a standard deviation
   and used with :func:`scipy.ndimage.gaussian_filter`.
3. **Temporal filtering** – Optional Butterworth bandpass filtering of
   the time series along the final dimension.  This removes slow
   drifts and high frequency noise.
4. **Nuisance regression** – Optional removal of confounding
   regressors (e.g. motion parameters) using ordinary least squares.
5. **ROI extraction** – If configured, compute mean time courses for
   each region defined in a labelled atlas NIfTI file.

The results of preprocessing are returned as a :class:`PreprocessedData`
object which stores the cleaned 4D data array (if retained) and/or
the ROI time series matrix.  A small set of quality metrics such as
the temporal signal‑to‑noise ratio (tSNR) is also provided for
inspection.

Note
----
For any real study we strongly recommend using a community vetted
preprocessing toolchain.  The implementation here is intentionally
minimal and may not account for all idiosyncrasies of fMRI data.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Sequence

import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter
from scipy.signal import butter, filtfilt

try:
    import nibabel as nib
except Exception:
    nib = None  # nibabel may not be available in all environments


@dataclass
class PreprocConfig:
    """Configuration options for the preprocessing pipeline.

    Parameters
    ----------
    smoothing_fwhm : float | None
        Full width at half maximum (in voxels) for spatial Gaussian
        smoothing.  If ``None`` no smoothing is applied.
    low_cut : float | None
        Low frequency cutoff (Hz) for bandpass filtering.  Frequencies
        below this value will be attenuated.  If ``None``, no high‑pass
        filter is applied.
    high_cut : float | None
        High frequency cutoff (Hz) for bandpass filtering.  Frequencies
        above this value will be attenuated.  If ``None``, no low‑pass
        filter is applied.
    filter_order : int
        Order of the Butterworth bandpass filter.  A second‑order filter
        provides a reasonable compromise between roll‑off and ripple.
    confound_columns : List[str] | None
        List of column names to select from a confounds TSV/CSV file.
        These regressors will be regressed out of the data.  If
        ``None`` or empty, no nuisance regression is performed.
    detrend : bool
        If ``True``, remove linear trends from each voxel/ROI time
        course prior to filtering and nuisance regression.
    extract_roi : bool
        Whether to compute ROI mean time courses from an atlas.
    roi_atlas_path : str | None
        Path to a labelled atlas NIfTI file.  Required if
        ``extract_roi`` is ``True``.
    retain_4d : bool
        If ``True``, keep the processed 4D data in memory in the
        :class:`PreprocessedData` object.  Otherwise the 4D data
        attribute will be set to ``None`` and only ROI time series will
        be stored.
    """

    smoothing_fwhm: Optional[float] = None
    low_cut: Optional[float] = None
    high_cut: Optional[float] = None
    filter_order: int = 2
    confound_columns: Optional[List[str]] = None
    detrend: bool = True
    extract_roi: bool = False
    roi_atlas_path: Optional[str] = None
    retain_4d: bool = False

    def validate(self) -> None:
        if self.extract_roi and not self.roi_atlas_path:
            raise ValueError("ROI extraction requested but no atlas path provided")
        if self.low_cut is not None and self.high_cut is not None:
            if self.low_cut >= self.high_cut:
                raise ValueError("low_cut must be less than high_cut for bandpass filtering")


@dataclass
class PreprocessedData:
    """Container for outputs of the preprocessing pipeline.

    Attributes
    ----------
    data : np.ndarray | None
        The preprocessed 4D data array with shape (X, Y, Z, T).  This
        may be ``None`` if ``retain_4d`` was set to ``False``.
    roi_timeseries : np.ndarray | None
        Array of shape (T, N_ROI) containing the mean time course for
        each ROI.  ``None`` if ROI extraction was not performed.
    tr : float
        Repetition time (TR) in seconds, extracted from the metadata.
    qc_metrics : Dict[str, float]
        Dictionary of basic quality control metrics (e.g. tSNR).
    affine : np.ndarray | None
        4×4 affine transform of the original image.  ``None`` if no
        image was loaded.
    mask : np.ndarray | None
        Brain mask array (bool) used during ROI extraction, or ``None``
        if not available.
    """
    data: Optional[np.ndarray] = None
    roi_timeseries: Optional[np.ndarray] = None
    tr: Optional[float] = None
    qc_metrics: Dict[str, float] = field(default_factory=dict)
    affine: Optional[np.ndarray] = None
    mask: Optional[np.ndarray] = None

    def get_roi_dataframe(self, roi_labels: Sequence[str]) -> pd.DataFrame:
        """Return ROI time series as a :class:`pandas.DataFrame`.

        Parameters
        ----------
        roi_labels : list of str
            Names of the ROIs in the same order as the columns of
            ``roi_timeseries``.

        Returns
        -------
        pandas.DataFrame
            A DataFrame with shape (T, N_ROI) where each column is a
            labelled time series.
        """
        if self.roi_timeseries is None:
            raise ValueError("No ROI time series available")
        return pd.DataFrame(self.roi_timeseries, columns=list(roi_labels))


class Preprocessor:
    """Perform simple preprocessing on fMRI datasets.

    This class encapsulates a set of operations that can be applied to a
    single functional run.  An instance can be reused across runs
    provided the configuration remains constant.
    """

    def __init__(self, config: PreprocConfig) -> None:
        self.config = config
        self.config.validate()
        # load atlas once if ROI extraction requested
        self._atlas_data: Optional[np.ndarray] = None
        self._atlas_labels: Optional[List[int]] = None
        if self.config.extract_roi:
            if nib is None:
                raise RuntimeError("nibabel is required for ROI extraction but not available")
            atlas_img = nib.load(self.config.roi_atlas_path)
            self._atlas_data = np.asanyarray(atlas_img.get_fdata())
            # unique non‑zero labels sorted
            labels = np.unique(self._atlas_data)
            labels = labels[labels != 0]
            self._atlas_labels = labels.astype(int).tolist()

    # ------------------------------------------------------------------
    def preprocess(self, func_path: str, confounds_path: Optional[str] = None) -> PreprocessedData:
        """Run the preprocessing pipeline on a single functional run.

        Parameters
        ----------
        func_path : str
            Path to a 4D functional MRI dataset in NIfTI format.
        confounds_path : str, optional
            Path to a TSV or CSV file containing confound regressors.
            These will be selected using ``confound_columns`` from
            :class:`PreprocConfig` and regressed out.

        Returns
        -------
        PreprocessedData
            An object containing the processed data and auxiliary
            information.
        """
        if nib is None:
            raise RuntimeError(
                "Cannot load NIfTI file because nibabel is not installed."
            )
        img = nib.load(func_path)
        data = np.asanyarray(img.get_fdata())  # shape (X, Y, Z, T)
        affine = img.affine.copy()
        header = img.header
        # extract TR from header or metadata
        tr = None
        try:
            tr = float(header.get_zooms()[-1])
        except Exception:
            tr = None

        # apply detrending if requested
        if self.config.detrend:
            # remove linear trend from each voxel's time series
            # shape: X,Y,Z,T -> reshape to (V,T)
            orig_shape = data.shape
            n_voxels = np.prod(orig_shape[:-1])
            flat = data.reshape((n_voxels, orig_shape[-1]))
            t = np.arange(flat.shape[1], dtype=float)
            t = (t - t.mean()) / (t.std() + 1e-8)
            # regress out t and intercept
            intercept = np.ones_like(t)
            design = np.vstack([intercept, t]).T  # shape (T, 2)
            # compute pseudoinverse
            pinv = np.linalg.pinv(design)
            betas = pinv @ flat.T  # (2, V)
            trend = (design @ betas).T  # (V, T)
            flat_detrended = flat - trend
            data = flat_detrended.reshape(orig_shape)

        # spatial smoothing
        if self.config.smoothing_fwhm is not None and self.config.smoothing_fwhm > 0:
            # convert FWHM to standard deviation: sigma = fwhm / (2*sqrt(2*ln2))
            sigma = self.config.smoothing_fwhm / (2.35482)
            # apply Gaussian filter along spatial axes for each time point
            smoothed = np.empty_like(data)
            for t in range(data.shape[-1]):
                smoothed[..., t] = gaussian_filter(
                    data[..., t], sigma=sigma, mode='constant', cval=0.0
                )
            data = smoothed

        # temporal filtering
        if self.config.low_cut is not None or self.config.high_cut is not None:
            if tr is None:
                raise ValueError(
                    "Cannot perform temporal filtering because repetition time (TR) is unknown"
                )
            fs = 1.0 / tr
            nyq = 0.5 * fs
            low = self.config.low_cut / nyq if self.config.low_cut else 0.0
            high = self.config.high_cut / nyq if self.config.high_cut else 1.0
            if low <= 0 and high >= 1:
                # no filtering
                pass
            else:
                b, a = butter(self.config.filter_order, [low, high], btype='bandpass', analog=False)
                # apply filter along time axis for each voxel
                orig_shape = data.shape
                n_voxels = np.prod(orig_shape[:-1])
                flat = data.reshape((n_voxels, orig_shape[-1]))
                # zero‑phase filtering
                try:
                    filtered = filtfilt(b, a, flat, axis=1)
                except Exception:
                    # fall back to forward filter if filtfilt fails
                    filtered = np.apply_along_axis(lambda x: np.convolve(x, b, mode='same'), 1, flat)
                data = filtered.reshape(orig_shape)

        # load confounds and perform nuisance regression
        confounds: Optional[np.ndarray] = None
        if confounds_path and self.config.confound_columns:
            # read using pandas; support TSV or CSV
            try:
                df = pd.read_csv(confounds_path, sep=None, engine='python')
            except Exception:
                df = pd.read_csv(confounds_path, sep=',')
            # select specified confound columns, drop rows with NaN
            columns = [col for col in self.config.confound_columns if col in df.columns]
            if columns:
                regressors = df[columns].values
                # demean regressors
                regressors = regressors - regressors.mean(axis=0, keepdims=True)
                # add intercept
                design = np.hstack([np.ones((regressors.shape[0], 1)), regressors])
                # apply regression to each voxel
                orig_shape = data.shape
                n_voxels = np.prod(orig_shape[:-1])
                flat = data.reshape((n_voxels, orig_shape[-1]))
                pinv = np.linalg.pinv(design)
                betas = pinv @ flat.T  # shape (n_reg+1, V)
                fitted = (design @ betas).T  # (V, T)
                flat_clean = flat - fitted
                data = flat_clean.reshape(orig_shape)
                confounds = regressors

        # compute QC metric: temporal SNR (mean / std per voxel averaged)
        # flatten to V x T
        flat = data.reshape(-1, data.shape[-1])
        mean_signal = flat.mean(axis=1)  # (V,)
        std_signal = flat.std(axis=1) + 1e-8
        tSNR = float(np.mean(mean_signal / std_signal))
        qc = {'tSNR': tSNR}

        # ROI extraction
        roi_ts: Optional[np.ndarray] = None
        if self.config.extract_roi and self._atlas_data is not None:
            roi_ts = self._extract_roi_time_series(data)

        # optionally discard 4D data to save memory
        data_out: Optional[np.ndarray] = data if self.config.retain_4d else None

        return PreprocessedData(
            data=data_out,
            roi_timeseries=roi_ts,
            tr=tr,
            qc_metrics=qc,
            affine=affine,
            mask=None,
        )

    # ------------------------------------------------------------------
    def _extract_roi_time_series(self, data: np.ndarray) -> np.ndarray:
        """Compute mean time courses for each ROI defined in the atlas.

        Parameters
        ----------
        data : np.ndarray
            Preprocessed 4D data array (X, Y, Z, T).

        Returns
        -------
        np.ndarray
            2D array of shape (T, N_ROI) containing mean signals per ROI.
        """
        if self._atlas_data is None or self._atlas_labels is None:
            raise ValueError("ROI extraction requested but atlas not loaded")
        atlas = self._atlas_data
        labels = self._atlas_labels
        # flatten spatial dims to index voxels by ROI
        T = data.shape[-1]
        roi_ts = np.zeros((T, len(labels)), dtype=float)
        for idx, lbl in enumerate(labels):
            mask = atlas == lbl
            if not np.any(mask):
                continue
            # mean across voxels in mask
            voxels = data[mask, :]
            roi_ts[:, idx] = voxels.mean(axis=0)
        return roi_ts


__all__ = [
    'PreprocConfig',
    'PreprocessedData',
    'Preprocessor',
]