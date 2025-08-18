"""Utilities for loading and managing atlas templates.

The :func:`load_template` function fetches atlas images and labels for
common parcellations such as AAL, Harvard‑Oxford and Schaefer.  The
:func:`construct_network` helper builds a connectivity matrix from
preprocessed ROI time series for a specific template.

These utilities rely on :mod:`nilearn` to provide atlas data.  If
``nilearn`` is not installed the functions will raise informative errors.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Sequence, Optional, Union

import numpy as np

try:  # nilearn and nibabel are optional dependencies
    from nilearn import datasets
    import nibabel as nib
except Exception:  # pragma: no cover - handled at runtime
    datasets = None
    nib = None

from .static import compute_pearson_connectivity, ConnectivityMatrix


@dataclass
class Template:
    """Representation of an atlas template."""

    name: str
    atlas_path: str
    labels: Sequence[str]


def load_template(name: str) -> Template:
    """Load a common atlas template by name.

    Parameters
    ----------
    name : str
        Name of the template.  Supported values include ``'aal'``,
        ``'harvardoxford'`` and ``'schaefer'``.

    Returns
    -------
    Template
        Dataclass containing the atlas image path and label list.
    """
    if datasets is None:
        raise RuntimeError("nilearn is required to load atlas templates")
    lname = name.lower()
    if lname == "aal":
        atlas = datasets.fetch_atlas_aal()
        labels = list(atlas.labels)
        return Template("aal", atlas.maps, labels)
    if lname == "harvardoxford":
        atlas = datasets.fetch_atlas_harvard_oxford('cort-prob-2mm')
        labels = list(atlas.labels)
        return Template("harvardoxford", atlas.maps, labels)
    if lname.startswith("schaefer"):
        # name may specify number of ROIs and networks e.g. "schaefer-200-7"
        parts = lname.split('-')
        n_rois = 100
        networks = 7
        if len(parts) >= 2 and parts[1].isdigit():
            n_rois = int(parts[1])
        if len(parts) >= 3 and parts[2].isdigit():
            networks = int(parts[2])
        atlas = datasets.fetch_atlas_schaefer_2018(
            n_rois=n_rois, yeo_networks=networks, resolution_mm=2
        )
        labels = list(atlas.labels)
        return Template(f"schaefer-{n_rois}-{networks}", atlas.maps, labels)
    raise ValueError(f"Unknown template '{name}'")


def construct_network(
    data: Union['PreprocessedData', np.ndarray],
    template: Union[str, Template],
) -> ConnectivityMatrix:
    """Construct a connectivity matrix for a given template.

    Parameters
    ----------
    data : :class:`~brainnet.preprocessing.PreprocessedData` or ndarray
        If a :class:`PreprocessedData` instance is provided the ROI time
        series and labels for the requested template are extracted from it.
        Alternatively a 2D ``ndarray`` of ROI time series can be supplied
        directly.
    template : str or Template
        Template identifier.  When ``data`` is an array, ``template`` must
        be a :class:`Template` instance providing the ROI labels.

    Returns
    -------
    ConnectivityMatrix
        Pearson correlation connectivity matrix with the template name
        recorded in the ``template`` attribute.
    """
    if isinstance(template, Template):
        tmpl_name = template.name
        labels = template.labels
    else:
        tmpl_name = template
        labels = None

    if hasattr(data, 'roi_timeseries'):
        roi_ts = data.roi_timeseries[tmpl_name]
        labels = data.roi_labels[tmpl_name]
    else:
        roi_ts = np.asarray(data)
        if labels is None:
            raise ValueError("labels required when data is raw array and template is not Template")
    conn = compute_pearson_connectivity(roi_ts, labels, template=tmpl_name)
    return conn


__all__ = [
    'Template',
    'load_template',
    'construct_network',
]
