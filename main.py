"""
main
====

This script demonstrates how to assemble the components of the
``brainnet`` package into a complete analysis workflow.  It is not
intended to be a polished command‑line interface but rather serves
as a blueprint for your own scripts or notebooks.  The workflow is
as follows:

1. **Index the dataset** using :class:`brainnet.data_management.DatasetIndex`.
2. **Preprocess** one or more functional runs with
   :class:`brainnet.preprocessing.Preprocessor` to obtain ROI time
   series.  A simple preprocessing configuration is provided.
3. **Compute static connectivity** using
   :class:`brainnet.static_analysis.StaticAnalyzer` and derive graph
   metrics.
4. **Run dynamic analysis** using
   :class:`brainnet.dynamic_analysis.DynamicAnalyzer` to identify
   temporal states and compute their dynamics.
5. **Generate an interactive report** with
   :class:`brainnet.visualization.ReportGenerator`.

Run this module as a script with the path to your BIDS dataset to
process a single subject and produce a report.  You may need to
adjust the configuration parameters to suit your data and analysis
requirements.  Note that this script assumes the necessary
dependencies (nibabel, plotly, sklearn, etc.) are installed.

Example
-------
python -m brainnet.main /path/to/bids_dataset --subject 01 --task rest

"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np

from .data_management import DatasetIndex
# Import both preprocessing pipelines.  The simple Preprocessor is kept for
# compatibility, but here we demonstrate usage of the more modular
# PreprocessPipeline.
from .preprocessing import PreprocConfig, Preprocessor
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
# Import the StaticAnalyzer from the modular static package instead of the
# legacy ``static_analysis`` module.  The legacy module remains for
# compatibility but new code should use ``brainnet.static``.
from .static import StaticAnalyzer
# Import dynamic analysis classes from the modular dynamic package.  The
# old ``dynamic_analysis`` module is still available but the new
# subpackage provides the same interfaces in a more structured way.
from .dynamic import DynamicConfig, DynamicAnalyzer
from .visualization import ReportConfig, ReportGenerator


def run_pipeline(dataset_path: str, subject: str, task: str, output_dir: str) -> None:
    # index dataset
    index = DatasetIndex(dataset_path)
    # select functional run
    runs = [rf for rf in index.get_functional_runs(subject) if rf.task == task]
    if not runs:
        raise ValueError(f"No runs found for subject {subject}, task {task}")
    run = runs[0]
    # ---------------------------------------------------------------------
    # Configure and execute the preprocessing pipeline.  This example uses
    # the extensible ``PreprocessPipeline`` with individual step configs.  If
    # you prefer the simpler pipeline, replace this block with construction
    # of ``PreprocConfig`` and ``Preprocessor`` as before.
    # Slice timing and spatial normalisation are disabled by default here.
    pre_cfg = PreprocessPipelineConfig(
        slice_timing=SliceTimingConfig(enabled=False),
        motion=MotionCorrectionConfig(enabled=False),
        spatial_norm=SpatialNormalizationConfig(enabled=False),
        smoothing=SmoothingConfig(enabled=True, fwhm=3.0),
        temporal_filter=TemporalFilterConfig(enabled=True, low_cut=0.01, high_cut=0.1, order=2),
        nuisance=NuisanceRegressionConfig(enabled=False),
        roi_extraction=RoiExtractionConfig(enabled=True, atlas_path=None),  # set atlas_path
        retain_4d=False,
    )
    preproc = PreprocessPipeline(pre_cfg)
    outputs = preproc.run(run.path)
    roi_ts = outputs.get('roi_timeseries')
    if roi_ts is None:
        raise RuntimeError("ROI extraction failed; check atlas path")
    # define ROI labels: use atlas labels if provided, otherwise simple indices
    roi_labels = outputs.get('roi_labels') or [f'ROI{i}' for i in range(roi_ts.shape[1])]
    # static connectivity and graph metrics
    static_analyzer = StaticAnalyzer(threshold=0.2)
    conn = static_analyzer.compute_connectivity(roi_ts, roi_labels)
    graph_metrics = static_analyzer.compute_graph_metrics(conn)
    # dynamic analysis
    dyn_cfg = DynamicConfig(window_length=30, step=10, n_states=4, method='kmeans')
    dyn_analyzer = DynamicAnalyzer(dyn_cfg)
    dyn_model = dyn_analyzer.analyse(roi_ts)
    # generate report
    rep_cfg = ReportConfig(output_dir=output_dir)
    rep_gen = ReportGenerator(rep_cfg)
    qc_metrics = outputs.get('qc_metrics', {})
    report_path = rep_gen.generate(
        subject_id=subject,
        conn_matrix=conn,
        graph_metrics=graph_metrics,
        dyn_model=dyn_model,
        roi_labels=roi_labels,
        qc_metrics=qc_metrics,
    )
    print(f"Report written to {report_path}")


def parse_args(args: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run brainnet pipeline on a BIDS dataset")
    parser.add_argument('dataset', type=str, help='Path to BIDS dataset root')
    parser.add_argument('--subject', type=str, required=True, help='Subject label (without sub-)')
    parser.add_argument('--task', type=str, required=True, help='Task name (e.g. rest)')
    parser.add_argument('--output', type=str, default='reports', help='Output directory for reports')
    return parser.parse_args(args)


def main(argv: Optional[list[str]] = None) -> None:
    args = parse_args(argv or [])
    run_pipeline(args.dataset, args.subject, args.task, args.output)


if __name__ == '__main__':
    main()