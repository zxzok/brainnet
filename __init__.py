"""
brainnet
=======

This package provides a modular set of components for managing, preprocessing,
analyzing and visualising brain imaging data, with a particular focus on
functional MRI (fMRI) connectivity analyses.  The design follows the
architecture described in the accompanying system design document.  While
only a subset of the full specification is implemented here, the code
establishes a clear structure and provides extensible interfaces for future
development.

The key modules include:

* ``data_management`` – indexing and parsing BIDS-like datasets.
* ``preprocessing`` – simple fMRI preprocessing pipeline with smoothing,
  filtering and nuisance regression.
* ``static_analysis`` – computation of static functional connectivity
  matrices and basic graph theoretic metrics.
* ``dynamic_analysis`` – sliding‑window based estimation of time‑varying
  connectivity and state modelling via clustering.
* ``visualization`` – utilities for building interactive reports using
  Plotly.

End users can import individual classes or functions from these modules
or use them in combination to build a complete analysis pipeline.  See
``brainnet.main`` for a rudimentary example of how the components can be
tied together.

Note
----
This code relies on external libraries such as ``numpy``, ``pandas``,
``scipy`` and ``plotly`` which are included in the base environment.
Additional dependencies such as ``nibabel`` or ``sklearn`` are imported
where needed but may not be available in every runtime environment.  If
these packages are missing the corresponding functionality will not work.
"""

from .data_management import BIDSFile, DatasetIndex, DatasetManager

# The remaining imports are optional and may require additional third‑party
# dependencies.  They are wrapped in ``try`` blocks so that importing the
# top‑level package does not fail if those dependencies are absent.  Modules
# that cannot be imported will simply expose ``None`` placeholders.
try:  # pragma: no cover - optional dependency chain
    from .preprocessing import PreprocConfig, Preprocessor, PreprocessedData
except Exception:  # pragma: no cover
    PreprocConfig = Preprocessor = PreprocessedData = None

try:  # pragma: no cover - optional dependency chain
    from .preprocessing_full import (
        PreprocessPipelineConfig,
        PreprocessPipeline,
        SliceTimingConfig,
        MotionCorrectionConfig,
        SpatialNormalizationConfig,
        SmoothingConfig,
        TemporalFilterConfig,
        NuisanceRegressionConfig,
        RoiExtractionConfig,
    )
except Exception:  # pragma: no cover
    (
        PreprocessPipelineConfig,
        PreprocessPipeline,
        SliceTimingConfig,
        MotionCorrectionConfig,
        SpatialNormalizationConfig,
        SmoothingConfig,
        TemporalFilterConfig,
        NuisanceRegressionConfig,
        RoiExtractionConfig,
    ) = (None,) * 9

try:  # pragma: no cover - optional dependency chain
    from .static import (
        ConnectivityMatrix,
        GraphMetrics,
        StaticAnalyzer,
        compute_pearson_connectivity,
        compute_degree,
        compute_clustering,
        compute_global_efficiency,
    )
except Exception:  # pragma: no cover
    (
        ConnectivityMatrix,
        GraphMetrics,
        StaticAnalyzer,
        compute_pearson_connectivity,
        compute_degree,
        compute_clustering,
        compute_global_efficiency,
    ) = (None,) * 7

try:  # pragma: no cover - optional dependency chain
    from .dynamic import (
        DynamicAnalyzer,
        DynamicConfig,
        DynamicStateModel,
        DynamicMetrics,
    )
except Exception:  # pragma: no cover
    DynamicAnalyzer = DynamicConfig = DynamicStateModel = DynamicMetrics = None

try:  # pragma: no cover - optional dependency chain
    from .visualization import ReportConfig, ReportGenerator
except Exception:  # pragma: no cover
    ReportConfig = ReportGenerator = None

__all__ = [
    'BIDSFile',
    'DatasetIndex',
    'DatasetManager',
    'PreprocConfig',
    'Preprocessor',
    'PreprocessedData',
    'PreprocessPipelineConfig',
    'PreprocessPipeline',
    'SliceTimingConfig',
    'MotionCorrectionConfig',
    'SpatialNormalizationConfig',
    'SmoothingConfig',
    'TemporalFilterConfig',
    'NuisanceRegressionConfig',
    'RoiExtractionConfig',
    'StaticAnalyzer',
    'ConnectivityMatrix',
    'GraphMetrics',
    'DynamicAnalyzer',
    'DynamicConfig',
    'DynamicStateModel',
    'DynamicMetrics',
    'ReportConfig',
    'ReportGenerator',
]