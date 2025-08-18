"""
multimodal_preprocessing
========================

This module defines :class:`MultimodalPreprocessor`, a thin orchestration
layer that coordinates preprocessing of different MRI modalities.  The
preprocessor is configured via small dataclasses that allow users to
enable or disable individual modalities and to tweak basic parameters.

Only the functional MRI (fMRI) branch is fully implemented here and
leverages the existing :class:`~brainnet.preprocessing_full.PreprocessPipeline`.
The anatomical (T1/T2) and diffusion (DWI) branches contain minimal
placeholders illustrating where calls to external tools such as FSL or
MRtrix could be inserted.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict

from .preprocessing_full import PreprocessPipeline, PreprocessPipelineConfig


# ---------------------------------------------------------------------------
# Configuration dataclasses


@dataclass
class AnatPreprocConfig:
    """Configuration for anatomical (T1/T2) preprocessing."""

    enabled: bool = False
    skull_strip: bool = True
    bias_correct: bool = True


@dataclass
class DWIPreprocConfig:
    """Configuration for diffusion MRI preprocessing."""

    enabled: bool = False
    method: str = "mrtrix"  # or ``fsl``


@dataclass
class FMRIPreprocConfig:
    """Configuration for functional MRI preprocessing."""

    enabled: bool = False
    pipeline: PreprocessPipelineConfig = field(default_factory=PreprocessPipelineConfig)


@dataclass
class MultimodalPreprocConfig:
    """Aggregate configuration for all modalities."""

    t1: AnatPreprocConfig = field(default_factory=AnatPreprocConfig)
    t2: AnatPreprocConfig = field(default_factory=AnatPreprocConfig)
    dwi: DWIPreprocConfig = field(default_factory=DWIPreprocConfig)
    fmri: FMRIPreprocConfig = field(default_factory=FMRIPreprocConfig)


# ---------------------------------------------------------------------------
# Multimodal preprocessor implementation


class MultimodalPreprocessor:
    """Run preprocessing for multiple imaging modalities.

    Parameters
    ----------
    config:
        Instance of :class:`MultimodalPreprocConfig` describing which
        modalities are enabled and their respective parameters.
    """

    def __init__(self, config: MultimodalPreprocConfig) -> None:
        self.config = config

    # -- Anatomical -----------------------------------------------------
    def _run_anat_preproc(self, image_path: str, cfg: AnatPreprocConfig, label: str) -> str:
        if not cfg.enabled:
            return image_path
        # Placeholder for calls to packages such as FSL's FAST or ANTs
        print(f"Running {label} preprocessing (skull_strip={cfg.skull_strip}, bias_correct={cfg.bias_correct})")
        return image_path

    def run_t1_preproc(self, t1_path: str) -> str:
        """Preprocess a T1‑weighted image."""

        return self._run_anat_preproc(t1_path, self.config.t1, "T1")

    def run_t2_preproc(self, t2_path: str) -> str:
        """Preprocess a T2‑weighted image."""

        return self._run_anat_preproc(t2_path, self.config.t2, "T2")

    # -- Diffusion ------------------------------------------------------
    def run_dwi_preproc(self, dwi_path: str) -> str:
        """Preprocess diffusion data using the configured backend.

        The function currently serves as a placeholder.  Depending on the
        ``method`` field, it would normally invoke MRtrix or FSL commands
        via subprocess or dedicated Python bindings.
        """

        cfg = self.config.dwi
        if not cfg.enabled:
            return dwi_path
        method = cfg.method.lower()
        if method == "mrtrix":
            print("DWI preprocessing via MRtrix would be executed here")
        elif method == "fsl":
            print("DWI preprocessing via FSL would be executed here")
        else:
            raise ValueError(f"Unknown DWI preprocessing method: {cfg.method}")
        return dwi_path

    # -- Functional -----------------------------------------------------
    def run_fmri_preproc(self, fmri_path: str) -> Dict[str, Any]:
        """Run fMRI preprocessing using :class:`PreprocessPipeline`."""

        cfg = self.config.fmri
        if not cfg.enabled:
            return {}
        pipeline = PreprocessPipeline(cfg.pipeline)
        return pipeline.run(fmri_path)

    # -- Orchestration --------------------------------------------------
    def run_all(self, data_paths: Dict[str, str]) -> Dict[str, Any]:
        """Execute preprocessing for all enabled modalities.

        Parameters
        ----------
        data_paths:
            Mapping from modality labels (``'t1'``, ``'t2'``, ``'dwi'``,
            ``'fmri'``) to file paths.

        Returns
        -------
        dict
            Mapping of modality label to preprocessing outputs.  For
            anatomical and diffusion modalities the returned value is the
            (potentially modified) input path, whereas for fMRI the full
            pipeline outputs are returned.
        """

        results: Dict[str, Any] = {}
        if "t1" in data_paths:
            results["t1"] = self.run_t1_preproc(data_paths["t1"])
        if "t2" in data_paths:
            results["t2"] = self.run_t2_preproc(data_paths["t2"])
        if "dwi" in data_paths:
            results["dwi"] = self.run_dwi_preproc(data_paths["dwi"])
        if "fmri" in data_paths:
            results["fmri"] = self.run_fmri_preproc(data_paths["fmri"])
        return results
