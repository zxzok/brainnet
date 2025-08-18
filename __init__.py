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

from .data_management import BIDSFile, DatasetIndex, Patient, DatasetManager
# Expose the simple and full preprocessing pipelines.  ``PreprocConfig`` and
# ``Preprocessor`` implement a lightweight pipeline suitable for quick
# connectivity analyses, whereas ``PreprocessPipelineConfig`` and
# ``PreprocessPipeline`` (defined in ``preprocessing_full``) provide a
# more modular architecture with pluggable steps.  Users can choose the
# pipeline appropriate to their needs.  See the documentation in
# ``preprocessing.py`` and ``preprocessing_full.py`` for details.
from .preprocessing import PreprocConfig, Preprocessor, PreprocessedData
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
# -----------------------------------------------------------------------------
# Import static connectivity and graph analysis utilities from the new modular
# ``brainnet.static`` subpackage.  This subpackage defines the
# ``ConnectivityMatrix`` and ``GraphMetrics`` dataclasses along with the
# ``StaticAnalyzer`` class, which replaces the legacy ``static_analysis``
# module.  The old module remains in the repository for backwards
# compatibility, but new code should import from ``brainnet.static`` instead.
from .static import (
    ConnectivityMatrix,
    GraphMetrics,
    StaticAnalyzer,
    compute_pearson_connectivity,
    compute_degree,
    compute_clustering,
    compute_global_efficiency,
)
# Import dynamic analysis classes from the new modular package.  The
# legacy ``dynamic_analysis`` module remains for backwards
# compatibility but users are encouraged to import from
# ``brainnet.dynamic`` instead.  See the documentation in
# ``brainnet.dynamic.__init__`` for details.
from .dynamic import (
    DynamicAnalyzer,
    DynamicConfig,
    DynamicStateModel,
    DynamicMetrics,
)
from .visualization import ReportConfig, ReportGenerator

__all__ = [
    'BIDSFile',
    'DatasetIndex',
    'Patient',
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