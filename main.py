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
import sqlite3

import numpy as np

from typing import Optional

from .data_management import DatasetIndex, DatasetManager
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
from .web_app import process_image  # reuse processing helper


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
    dyn_out = Path(output_dir) / 'dynamic'
    dyn_cfg = DynamicConfig(window_length=30, step=10, n_states=4,
                            method='kmeans', output_dir=dyn_out)
    dyn_analyzer = DynamicAnalyzer(dyn_cfg)
    dyn_model = dyn_analyzer.analyse(roi_ts)
    _ = np.load(dyn_out / 'state_sequence.npy')
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


def generate_patient_report(patient_id: int, output_dir: str) -> None:
    """Generate report for a patient stored in the SQLite database."""
    conn = sqlite3.connect('brainnet.db')
    cur = conn.cursor()
    cur.execute('SELECT id, patient_id, name, age, sex FROM patients WHERE id = ?', (patient_id,))
    patient = cur.fetchone()
    if not patient:
        raise SystemExit("Patient not found")
    cur.execute('SELECT id, image_path FROM mri_images WHERE patient_id = ?', (patient_id,))
    images = cur.fetchall()
    if not images:
        raise SystemExit("No images for patient")
    for img_id, path in images:
        cur.execute('SELECT COUNT(*) FROM features WHERE image_id = ?', (img_id,))
        if cur.fetchone()[0] == 0:
            process_image(img_id, path)
    conn.close()

    first_id, first_path = images[0]
    pipeline = PreprocessPipeline(PreprocessPipelineConfig())
    preproc = pipeline.run(first_path)
    roi_ts = preproc.get('roi_timeseries')
    labels = preproc.get('roi_labels') or []
    static_analyzer = StaticAnalyzer()
    conn_matrix = static_analyzer.compute_connectivity(roi_ts, labels)
    graph_metrics = static_analyzer.compute_graph_metrics(conn_matrix)
    dyn_cfg = DynamicConfig(window_length=10, step=5, n_states=2)
    dyn_analyzer = DynamicAnalyzer(dyn_cfg)
    dyn_model = dyn_analyzer.analyse(roi_ts)

    rep_cfg = ReportConfig(output_dir=output_dir)
    rep_gen = ReportGenerator(rep_cfg)
    patient_info = {"Name": patient[2], "Sex": patient[4] or '', "Age": patient[3] or ''}
    report_path = rep_gen.generate(
        subject_id=patient[1],
        conn_matrix=conn_matrix,
        graph_metrics=graph_metrics,
        dyn_model=dyn_model,
        roi_labels=labels,
        qc_metrics=preproc.get('qc_metrics', {}),
        patient_info=patient_info,
    )
    print(f"Report written to {report_path}")


def parse_args(args: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run brainnet pipeline on a BIDS dataset")
    parser.add_argument(
        'dataset', nargs='?', type=str, default=None,
        help='Path to BIDS dataset root')
    parser.add_argument(
        '--openneuro-id', type=str, default=None,
        help='OpenNeuro dataset identifier (e.g. ds000114)')
    parser.add_argument('--subject', type=str, help='Subject label (without sub-)')
    parser.add_argument('--task', type=str, help='Task name (e.g. rest)')
    parser.add_argument('--output', type=str, default='reports', help='Output directory for reports')
    parser.add_argument('--patient-id', type=int, default=None, help='Generate report for patient in DB')
    return parser.parse_args(args)


def main(argv: Optional[list[str]] = None) -> None:
    args = parse_args(argv or [])
    if args.patient_id is not None:
        generate_patient_report(args.patient_id, args.output)
        return
    if not args.subject or not args.task:
        raise SystemExit('--subject and --task are required when not using --patient-id')
    dataset_path = args.dataset
    if args.openneuro_id:
        dataset_path = DatasetManager.fetch_from_openneuro(args.openneuro_id)
    if dataset_path is None:
        raise SystemExit('Either a dataset path or --openneuro-id must be provided')
    run_pipeline(dataset_path, args.subject, args.task, args.output)


if __name__ == '__main__':
    main()
