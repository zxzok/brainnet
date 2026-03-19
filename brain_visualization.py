"""
brain_visualization
====================

Generate brain network visualisation images using nilearn and matplotlib.
Images are produced as base64-encoded PNG strings suitable for direct
embedding in HTML ``<img>`` tags.

Supported visualisations:

* **Glass brain connectome** — 3D projection of ROI nodes and edges on a
  glass brain surface, using nilearn's ``plot_connectome``.
* **Connectivity matrix** — coloured matrix heatmap via ``plot_matrix``.
* **Connectivity circle** — chord diagram via nilearn's
  ``plot_connectivity_circle`` (if available).
* **ROI map overlay** — atlas ROIs rendered on an anatomical background.

All public functions accept numpy arrays and return ``str`` (base64 data
URI) so they can be injected into Jinja2 templates.
"""

from __future__ import annotations

import base64
import io
from typing import Optional, Sequence

import numpy as np

# ── optional dependency guards ──────────────────────────────────────

try:
    import matplotlib
    matplotlib.use("Agg")  # non-interactive backend for server use
    # Configure CJK font support for Chinese labels
    matplotlib.rcParams['font.sans-serif'] = [
        'Noto Sans SC', 'PingFang SC', 'Microsoft YaHei',
        'SimHei', 'Arial Unicode MS', 'DejaVu Sans',
    ]
    matplotlib.rcParams['axes.unicode_minus'] = False
    import matplotlib.pyplot as plt
except ImportError:
    plt = None  # type: ignore[assignment]

try:
    from nilearn import plotting, datasets, image
    import nibabel as nib
except ImportError:
    plotting = None  # type: ignore[assignment]
    datasets = None  # type: ignore[assignment]
    image = None  # type: ignore[assignment]
    nib = None  # type: ignore[assignment]


# ── MNI coordinate utilities ────────────────────────────────────────

# Default MNI coordinates for common atlas ROIs (subset of AAL 90).
# When the caller does not provide coordinates we fall back to these.
_AAL_90_COORDS: np.ndarray | None = None


def _get_default_coords(n_rois: int) -> np.ndarray:
    """Return MNI coordinates for *n_rois* regions.

    If nilearn is available the coordinates are extracted from the AAL
    atlas.  Otherwise a simple spherical arrangement is used.
    """
    global _AAL_90_COORDS
    if datasets is not None and _AAL_90_COORDS is None:
        try:
            atlas = datasets.fetch_atlas_aal()
            _AAL_90_COORDS = plotting.find_parcellation_cut_coords(atlas.maps)
        except Exception:
            _AAL_90_COORDS = np.empty((0, 3))

    if _AAL_90_COORDS is not None and len(_AAL_90_COORDS) >= n_rois:
        return _AAL_90_COORDS[:n_rois]

    # Fallback: distribute points on a sphere
    indices = np.arange(n_rois)
    phi = np.arccos(1 - 2 * (indices + 0.5) / n_rois)
    theta = np.pi * (1 + 5 ** 0.5) * indices
    r = 60  # approximate brain radius in mm
    x = r * np.sin(phi) * np.cos(theta)
    y = r * np.sin(phi) * np.sin(theta)
    z = r * np.cos(phi)
    return np.column_stack([x, y, z])


def _fig_to_base64(fig) -> str:
    """Render a matplotlib Figure to a data-URI string."""
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=120, bbox_inches="tight",
                facecolor="#0e1019", edgecolor="none")
    plt.close(fig)
    buf.seek(0)
    b64 = base64.b64encode(buf.read()).decode("ascii")
    return f"data:image/png;base64,{b64}"


# ── public API ──────────────────────────────────────────────────────

def plot_glass_brain_connectome(
    connectivity_matrix: np.ndarray,
    labels: Sequence[str] | None = None,
    coords: np.ndarray | None = None,
    threshold: float | str = "80%",
    title: str = "Brain Connectivity",
) -> str:
    """Render a glass-brain connectome projection.

    Parameters
    ----------
    connectivity_matrix : ndarray, shape (n, n)
        Symmetric connectivity matrix.
    labels : sequence of str, optional
        ROI labels (currently used for metadata only).
    coords : ndarray, shape (n, 3), optional
        MNI coordinates for each ROI.  If ``None`` default AAL
        coordinates are used.
    threshold : float or str
        Absolute threshold below which edges are hidden.  A string like
        ``"80%"`` keeps only the top 20 % of edges.
    title : str
        Title displayed above the figure.

    Returns
    -------
    str
        Base64 data-URI of the rendered PNG image.
    """
    if plt is None or plotting is None:
        return ""
    n = connectivity_matrix.shape[0]
    if coords is None:
        coords = _get_default_coords(n)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5),
                             subplot_kw={"projection": "3d"} if False else {})
    views = ["x", "y", "z"]
    titles = ["Sagittal", "Coronal", "Axial"]

    for ax, view, vtitle in zip(axes, views, titles):
        display = plotting.plot_connectome(
            connectivity_matrix,
            coords,
            edge_threshold=threshold,
            display_mode=view,
            node_size=30,
            edge_kwargs={"linewidth": 0.8, "alpha": 0.6},
            node_kwargs={"alpha": 0.85},
            colorbar=False,
            axes=ax,
            title=vtitle,
        )
    fig.suptitle(title, color="white", fontsize=14, y=1.02)
    fig.patch.set_facecolor("#0e1019")
    return _fig_to_base64(fig)


def plot_connectivity_matrix_brain(
    connectivity_matrix: np.ndarray,
    labels: Sequence[str] | None = None,
    title: str = "Connectivity Matrix",
) -> str:
    """Render the connectivity matrix as a styled heatmap.

    Uses nilearn's ``plot_matrix`` when available, otherwise falls back
    to a plain matplotlib ``imshow``.
    """
    if plt is None:
        return ""

    n = connectivity_matrix.shape[0]
    tick_labels = list(labels) if labels else [f"R{i}" for i in range(n)]

    fig, ax = plt.subplots(figsize=(8, 7))
    if plotting is not None:
        plotting.plot_matrix(
            connectivity_matrix,
            labels=tick_labels,
            colorbar=True,
            vmin=-1, vmax=1,
            axes=ax,
            title=title,
        )
    else:
        im = ax.imshow(connectivity_matrix, cmap="RdBu_r", vmin=-1, vmax=1)
        ax.set_title(title, color="white")
        fig.colorbar(im, ax=ax, shrink=0.8)

    fig.patch.set_facecolor("#0e1019")
    ax.set_facecolor("#0e1019")
    return _fig_to_base64(fig)


def plot_brain_regions(
    atlas_name: str = "aal",
    title: str = "Brain Atlas",
) -> str:
    """Overlay atlas ROI boundaries on a glass brain.

    This gives the user a visual reference for which brain regions are
    included in the analysis.
    """
    if plt is None or plotting is None or datasets is None:
        return ""

    try:
        if atlas_name.lower() == "aal":
            atlas = datasets.fetch_atlas_aal()
        elif atlas_name.lower() == "harvardoxford":
            atlas = datasets.fetch_atlas_harvard_oxford("cort-prob-2mm")
        elif atlas_name.lower().startswith("schaefer"):
            parts = atlas_name.lower().split("-")
            n_rois = int(parts[1]) if len(parts) >= 2 and parts[1].isdigit() else 100
            atlas = datasets.fetch_atlas_schaefer_2018(n_rois=n_rois, resolution_mm=2)
        else:
            return ""

        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        for ax, view in zip(axes, ["x", "y", "z"]):
            plotting.plot_roi(
                atlas.maps, display_mode=view, axes=ax,
                title="", cut_coords=3, alpha=0.5,
            )
        fig.suptitle(title, color="white", fontsize=13, y=1.02)
        fig.patch.set_facecolor("#0e1019")
        return _fig_to_base64(fig)
    except Exception:
        return ""


def plot_connectome_circle(
    connectivity_matrix: np.ndarray,
    labels: Sequence[str] | None = None,
    title: str = "Connectivity Circle",
    n_lines: int | None = None,
) -> str:
    """Render a chord-diagram style connectivity circle.

    Uses nilearn's ``plot_connectivity_circle`` (available from
    nilearn ≥ 0.10).
    """
    if plt is None:
        return ""

    n = connectivity_matrix.shape[0]
    node_names = list(labels) if labels else [f"R{i}" for i in range(n)]
    if n_lines is None:
        n_lines = max(n, 20)

    try:
        from nilearn.plotting import plot_connectivity_circle as _pcc
        fig, _ = _pcc(
            connectivity_matrix,
            node_names,
            n_lines=n_lines,
            title=title,
            colorbar=True,
            colorbar_size=0.3,
            facecolor="#0e1019",
            textcolor="white",
            node_colors=None,
        )
        fig.patch.set_facecolor("#0e1019")
        return _fig_to_base64(fig)
    except ImportError:
        pass

    # Fallback: simple circular layout with matplotlib
    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw={"polar": True})
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
    ax.set_xticks(angles)
    ax.set_xticklabels(node_names, fontsize=6, color="white")
    ax.set_yticklabels([])
    ax.set_title(title, color="white", pad=20)

    # Draw top edges
    flat = np.abs(connectivity_matrix[np.triu_indices(n, k=1)])
    thresh = np.percentile(flat, 80) if len(flat) > 0 else 0
    for i in range(n):
        for j in range(i + 1, n):
            w = connectivity_matrix[i, j]
            if abs(w) > thresh:
                color = "#38d9c5" if w > 0 else "#e15759"
                ax.plot([angles[i], angles[j]], [1, 1],
                        color=color, alpha=min(abs(w), 0.7), linewidth=0.5)
    fig.patch.set_facecolor("#0e1019")
    ax.set_facecolor("#0e1019")
    return _fig_to_base64(fig)


def plot_node_strength_brain(
    connectivity_matrix: np.ndarray,
    coords: np.ndarray | None = None,
    title: str = "Node Strength",
) -> str:
    """Render node strength as sized spheres on a glass brain."""
    if plt is None or plotting is None:
        return ""

    n = connectivity_matrix.shape[0]
    if coords is None:
        coords = _get_default_coords(n)
    strengths = np.abs(connectivity_matrix).sum(axis=1)

    try:
        fig, ax = plt.subplots(figsize=(12, 4))
        display = plotting.plot_markers(
            strengths,
            coords,
            display_mode="z",
            axes=ax,
            title=title,
            node_cmap="coolwarm",
            alpha=0.8,
        )
        fig.patch.set_facecolor("#0e1019")
        return _fig_to_base64(fig)
    except Exception:
        return ""


# ── Docker-generated image loading ─────────────────────────────────

def load_nifti_snapshot(
    nifti_path: str,
    display_mode: str = "ortho",
    title: str = "",
) -> str:
    """Generate a brain slice visualisation from a NIfTI file.

    This is useful for displaying results produced by external tools
    such as fmriprep, SPM, or FSL.
    """
    if plt is None or plotting is None or nib is None:
        return ""
    try:
        img = nib.load(nifti_path)
        fig, ax = plt.subplots(figsize=(10, 4))
        plotting.plot_stat_map(
            img, display_mode=display_mode, axes=ax,
            title=title, bg_img="MNI152",
        )
        fig.patch.set_facecolor("#0e1019")
        return _fig_to_base64(fig)
    except Exception:
        return ""


__all__ = [
    "plot_glass_brain_connectome",
    "plot_connectivity_matrix_brain",
    "plot_brain_regions",
    "plot_connectome_circle",
    "plot_node_strength_brain",
    "load_nifti_snapshot",
]
