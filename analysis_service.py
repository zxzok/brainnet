"""Shared analysis services used by BrainNet entrypoints.

The goal of this module is to keep CLI and Web execution paths aligned while
avoiding duplicated preprocessing, analysis, feature persistence, and report
generation logic.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

try:
    from brainnet.data_management import DatasetIndex, DatasetManager
    from brainnet.runtime import connect_db
    from brainnet.workflow import resolve_output_dir
except ImportError:
    from data_management import DatasetIndex, DatasetManager
    from runtime import connect_db
    from workflow import resolve_output_dir


_ANALYSIS_DEPS_ERROR = None
try:
    from brainnet.dynamic import DynamicAnalyzer, DynamicConfig
    from brainnet.preprocessing_full import (
        MotionCorrectionConfig,
        NuisanceRegressionConfig,
        PreprocessPipeline,
        PreprocessPipelineConfig,
        RoiExtractionConfig,
        SliceTimingConfig,
        SmoothingConfig,
        SpatialNormalizationConfig,
        TemporalFilterConfig,
    )
    from brainnet.static import StaticAnalyzer
    from brainnet.visualization import ReportConfig, ReportGenerator
except Exception as exc:  # pragma: no cover - dependency edge case
    try:
        from dynamic import DynamicAnalyzer, DynamicConfig
        from preprocessing_full import (
            MotionCorrectionConfig,
            NuisanceRegressionConfig,
            PreprocessPipeline,
            PreprocessPipelineConfig,
            RoiExtractionConfig,
            SliceTimingConfig,
            SmoothingConfig,
            SpatialNormalizationConfig,
            TemporalFilterConfig,
        )
        from static import StaticAnalyzer
        from visualization import ReportConfig, ReportGenerator
    except Exception:  # pragma: no cover - dependency edge case
        _ANALYSIS_DEPS_ERROR = str(exc)
        (
            DynamicAnalyzer,
            DynamicConfig,
            MotionCorrectionConfig,
            NuisanceRegressionConfig,
            PreprocessPipeline,
            PreprocessPipelineConfig,
            RoiExtractionConfig,
            SliceTimingConfig,
            SmoothingConfig,
            SpatialNormalizationConfig,
            TemporalFilterConfig,
            StaticAnalyzer,
            ReportConfig,
            ReportGenerator,
        ) = (None,) * 14


@dataclass
class AnalysisArtifacts:
    """Materialized outputs of one BrainNet analysis execution."""

    source_path: str
    roi_timeseries: Any
    roi_labels: list[str]
    qc_metrics: dict[str, Any]
    conn_matrix: Any
    graph_metrics: Any
    dyn_model: Any


def get_analysis_deps_error() -> str | None:
    """Return the cached dependency error, if any."""

    return _ANALYSIS_DEPS_ERROR


def ensure_analysis_dependencies() -> None:
    """Raise a clear error when the scientific stack is unavailable."""

    if _ANALYSIS_DEPS_ERROR:
        raise RuntimeError(
            f"Analysis dependencies are not installed: {_ANALYSIS_DEPS_ERROR}. "
            "Run: python3 -m pip install -e .[dev]"
        )


def build_web_preprocess_config() -> PreprocessPipelineConfig:
    """Return the lightweight preprocessing config used by the Web app."""

    ensure_analysis_dependencies()
    return PreprocessPipelineConfig(
        roi_extraction=RoiExtractionConfig(enabled=True),
    )


def build_cli_preprocess_config() -> PreprocessPipelineConfig:
    """Return the CLI preprocessing config used for subject-level reports."""

    ensure_analysis_dependencies()
    return PreprocessPipelineConfig(
        slice_timing=SliceTimingConfig(enabled=False),
        motion=MotionCorrectionConfig(enabled=False),
        spatial_norm=SpatialNormalizationConfig(enabled=False),
        smoothing=SmoothingConfig(enabled=True, fwhm=3.0),
        temporal_filter=TemporalFilterConfig(enabled=True, low_cut=0.01, high_cut=0.1, order=2),
        nuisance=NuisanceRegressionConfig(enabled=False),
        roi_extraction=RoiExtractionConfig(enabled=True, atlas_path=None),
        retain_4d=False,
    )


def recommend_dynamic_config(
    n_timepoints: int,
    *,
    output_dir: str | Path | None = None,
    method: str = "kmeans",
) -> DynamicConfig:
    """Choose conservative dynamic-analysis parameters from ROI length."""

    ensure_analysis_dependencies()
    window_length = min(30, max(5, n_timepoints // 5))
    step = max(1, window_length // 3)
    n_states = min(4, max(2, n_timepoints // max(window_length * 2, 1)))
    return DynamicConfig(
        window_length=window_length,
        step=step,
        n_states=n_states,
        method=method,
        output_dir=str(output_dir) if output_dir else None,
    )


def run_image_analysis(
    filepath: str,
    *,
    preprocess_config: PreprocessPipelineConfig | None = None,
    dynamic_config: DynamicConfig | None = None,
) -> AnalysisArtifacts:
    """Run preprocessing, static analysis, and dynamic analysis for one image."""

    ensure_analysis_dependencies()

    pipeline = PreprocessPipeline(preprocess_config or build_web_preprocess_config())
    preproc = pipeline.run(filepath)
    roi_ts = preproc.get("roi_timeseries")
    if roi_ts is None:
        raise ValueError("Preprocessing produced no ROI time series")

    labels = list(preproc.get("roi_labels") or [])
    if not labels:
        labels = [f"ROI{i}" for i in range(roi_ts.shape[1])]

    static_analyzer = StaticAnalyzer()
    conn_matrix = static_analyzer.compute_connectivity(roi_ts, labels)
    graph_metrics = static_analyzer.compute_graph_metrics(conn_matrix)

    dyn_cfg = dynamic_config or recommend_dynamic_config(roi_ts.shape[0])
    dyn_analyzer = DynamicAnalyzer(dyn_cfg)
    dyn_model = dyn_analyzer.analyse(roi_ts)

    return AnalysisArtifacts(
        source_path=filepath,
        roi_timeseries=roi_ts,
        roi_labels=labels,
        qc_metrics=preproc.get("qc_metrics", {}),
        conn_matrix=conn_matrix,
        graph_metrics=graph_metrics,
        dyn_model=dyn_model,
    )


def write_analysis_report(
    *,
    subject_id: str,
    artifacts: AnalysisArtifacts,
    output_dir: str | None = None,
    patient_info: dict[str, str] | None = None,
) -> str:
    """Generate and persist a BrainNet HTML report."""

    report_dir = resolve_output_dir(output_dir)
    rep_cfg = ReportConfig(output_dir=str(report_dir))
    rep_gen = ReportGenerator(rep_cfg)
    return rep_gen.generate(
        subject_id=subject_id,
        conn_matrix=artifacts.conn_matrix,
        graph_metrics=artifacts.graph_metrics,
        dyn_model=artifacts.dyn_model,
        roi_labels=artifacts.roi_labels,
        qc_metrics=artifacts.qc_metrics,
        patient_info=patient_info,
    )


def _insert_feature(cursor: Any, image_id: int, name: str, value: float, feature_type: str) -> None:
    cursor.execute(
        "INSERT INTO features (image_id, feature_name, feature_value, feature_type) VALUES (?, ?, ?, ?)",
        (image_id, name, value, feature_type),
    )


def persist_analysis_features(image_id: int, artifacts: AnalysisArtifacts) -> None:
    """Replace stored features for one image with fresh analysis outputs."""

    db = connect_db()
    cur = db.cursor()
    cur.execute("DELETE FROM features WHERE image_id = ?", (image_id,))

    for metric_name, values in artifacts.graph_metrics.node_metrics.items():
        for idx, value in enumerate(values):
            label = artifacts.roi_labels[idx] if idx < len(artifacts.roi_labels) else str(idx)
            _insert_feature(cur, image_id, f"{metric_name}_{label}", float(value), "static_node")

    for name, value in artifacts.graph_metrics.global_metrics.items():
        _insert_feature(cur, image_id, name, float(value), "static")

    metrics = artifacts.dyn_model.metrics
    for idx, value in enumerate(metrics.occupancy):
        _insert_feature(cur, image_id, f"state_{idx}_occupancy", float(value), "dynamic")
    for idx, value in enumerate(metrics.mean_dwell_time):
        _insert_feature(cur, image_id, f"state_{idx}_dwell_time", float(value), "dynamic")
    for idx, value in enumerate(metrics.dwell_time_std):
        _insert_feature(cur, image_id, f"state_{idx}_dwell_time_std", float(value), "dynamic")
    for idx, value in enumerate(metrics.max_dwell_time):
        _insert_feature(cur, image_id, f"state_{idx}_max_dwell_time", float(value), "dynamic")
    for idx, value in enumerate(metrics.mean_recurrence_interval):
        _insert_feature(cur, image_id, f"state_{idx}_recurrence_interval", float(value), "dynamic")

    transition_matrix = metrics.transition_matrix
    for src_idx in range(transition_matrix.shape[0]):
        for dst_idx in range(transition_matrix.shape[1]):
            _insert_feature(
                cur,
                image_id,
                f"transition_{src_idx}_to_{dst_idx}",
                float(transition_matrix[src_idx, dst_idx]),
                "dynamic",
            )

    for name, value in [
        ("occupancy_entropy", metrics.occupancy_entropy),
        ("transition_entropy", metrics.transition_entropy),
        ("switching_rate", metrics.switching_rate),
        ("state_complexity", metrics.state_complexity),
        ("temporal_autocorrelation", metrics.temporal_autocorrelation),
        ("n_transitions", float(metrics.n_transitions)),
    ]:
        _insert_feature(cur, image_id, name, float(value), "dynamic")

    try:
        try:
            from brainnet.dynamic.state_features import compute_state_features
        except ImportError:
            from dynamic.state_features import compute_state_features

        for state_idx, feature_map in enumerate(compute_state_features(artifacts.dyn_model.states)):
            for feat_name, feat_value in feature_map.items():
                _insert_feature(
                    cur,
                    image_id,
                    f"state_{state_idx}_{feat_name}",
                    float(feat_value),
                    "dynamic",
                )
    except (ImportError, NotImplementedError):
        pass

    _insert_feature(
        cur,
        image_id,
        "_connectivity_matrix",
        0.0,
        json.dumps(artifacts.conn_matrix.matrix.tolist()),
    )
    _insert_feature(
        cur,
        image_id,
        "_connectivity_labels",
        0.0,
        json.dumps(list(artifacts.roi_labels)),
    )
    _insert_feature(
        cur,
        image_id,
        "_state_sequence",
        0.0,
        json.dumps(artifacts.dyn_model.state_sequence.tolist()),
    )

    db.commit()
    db.close()


def record_analysis_error(image_id: int, message: str) -> None:
    """Replace stored features with a single analysis error marker."""

    db = connect_db()
    cur = db.cursor()
    cur.execute("DELETE FROM features WHERE image_id = ?", (image_id,))
    _insert_feature(cur, image_id, "error", 0.0, message)
    db.commit()
    db.close()


def analyze_and_store_image(
    image_id: int,
    filepath: str,
    *,
    preprocess_config: PreprocessPipelineConfig | None = None,
    dynamic_config: DynamicConfig | None = None,
) -> None:
    """Run image analysis and persist the derived features."""

    try:
        artifacts = run_image_analysis(
            filepath,
            preprocess_config=preprocess_config,
            dynamic_config=dynamic_config,
        )
        persist_analysis_features(image_id, artifacts)
    except Exception as exc:  # pragma: no cover - best effort logging
        record_analysis_error(image_id, str(exc))


def _image_has_non_error_features(image_id: int) -> bool:
    db = connect_db()
    cur = db.cursor()
    cur.execute(
        "SELECT COUNT(*) FROM features WHERE image_id = ? AND feature_name != 'error'",
        (image_id,),
    )
    count = cur.fetchone()[0]
    db.close()
    return bool(count)


def generate_patient_report(patient_id: int, output_dir: str | None = None) -> str:
    """Generate a patient report from stored MRI images and analysis outputs."""

    conn = connect_db()
    cur = conn.cursor()
    cur.execute("SELECT id, patient_id, name, age, sex FROM patients WHERE id = ?", (patient_id,))
    patient = cur.fetchone()
    if not patient:
        conn.close()
        raise ValueError("Patient not found")

    cur.execute("SELECT id, image_path FROM mri_images WHERE patient_id = ?", (patient_id,))
    images = cur.fetchall()
    conn.close()
    if not images:
        raise ValueError("No images for patient")

    for image_id, image_path in images:
        if not _image_has_non_error_features(image_id):
            analyze_and_store_image(image_id, image_path)

    artifacts = None
    last_error = None
    for _, image_path in images:
        try:
            artifacts = run_image_analysis(image_path, preprocess_config=build_web_preprocess_config())
            break
        except Exception as exc:
            last_error = exc

    if artifacts is None:
        raise ValueError(f"No analyzable images for patient: {last_error}")

    patient_info = {"Name": patient[2], "Sex": patient[4] or "", "Age": patient[3] or ""}
    return write_analysis_report(
        subject_id=patient[1],
        artifacts=artifacts,
        output_dir=output_dir,
        patient_info=patient_info,
    )


def run_subject_analysis(
    dataset_path: str,
    subject: str,
    task: str,
    output_dir: str | None = None,
) -> str:
    """Run the documented CLI subject analysis workflow and return the report path."""

    index = DatasetIndex(dataset_path)
    runs = [run_file for run_file in index.get_functional_runs(subject) if run_file.task == task]
    if not runs:
        raise ValueError(f"No runs found for subject {subject}, task {task}")

    report_dir = resolve_output_dir(output_dir)
    dynamic_dir = report_dir / f"dynamic_sub-{subject}_task-{task}"
    dynamic_dir.mkdir(parents=True, exist_ok=True)
    artifacts = run_image_analysis(
        runs[0].path,
        preprocess_config=build_cli_preprocess_config(),
        dynamic_config=DynamicConfig(
            window_length=30,
            step=10,
            n_states=4,
            method="kmeans",
            output_dir=str(dynamic_dir),
        ),
    )
    return write_analysis_report(
        subject_id=subject,
        artifacts=artifacts,
        output_dir=str(report_dir),
    )


def resolve_dataset_path(dataset_path: str | None, openneuro_id: str | None) -> str:
    """Resolve either a local dataset path or an OpenNeuro dataset into a local path."""

    if openneuro_id:
        return str(DatasetManager.fetch_from_openneuro(openneuro_id).root)
    if dataset_path is None:
        raise ValueError("Either a dataset path or --openneuro-id must be provided")
    return dataset_path


__all__ = [
    "AnalysisArtifacts",
    "analyze_and_store_image",
    "build_cli_preprocess_config",
    "build_web_preprocess_config",
    "ensure_analysis_dependencies",
    "generate_patient_report",
    "get_analysis_deps_error",
    "persist_analysis_features",
    "record_analysis_error",
    "recommend_dynamic_config",
    "resolve_dataset_path",
    "run_image_analysis",
    "run_subject_analysis",
    "write_analysis_report",
]
