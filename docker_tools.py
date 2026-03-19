"""
docker_tools
=============

Manage and execute neuroimaging preprocessing tools (fmriprep, SPM, FSL)
through Docker containers.  Each tool is represented by a configuration
dataclass that knows its Docker image, typical mount points and how to
construct the container command line.

The module purposefully does **not** import Docker libraries at module
level – all interaction is via ``subprocess`` so no additional Python
packages are required.  If Docker is not installed on the host the
helpers will raise :class:`DockerNotAvailableError`.

Usage
-----
>>> from brainnet.docker_tools import FmriprepRunner
>>> runner = FmriprepRunner(bids_dir="/data/bids", output_dir="/data/out")
>>> runner.run(subject="01")
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from dataclasses import dataclass, field
from typing import Dict, List, Optional


class DockerNotAvailableError(RuntimeError):
    """Raised when Docker is not installed or the daemon is not running."""


def docker_available() -> bool:
    """Return ``True`` if the ``docker`` CLI is reachable."""
    try:
        result = subprocess.run(
            ["docker", "info"],
            capture_output=True, timeout=10,
        )
        return result.returncode == 0
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False


def _ensure_docker() -> None:
    if not docker_available():
        raise DockerNotAvailableError(
            "Docker 未安装或 Docker 守护进程未运行。"
            "请安装 Docker Desktop 并确保其正在运行。"
        )


# ── fmriprep ────────────────────────────────────────────────────────

@dataclass
class FmriprepRunner:
    """Run fmriprep via its official Docker image.

    Parameters
    ----------
    bids_dir : str
        Path to the BIDS-formatted input dataset.
    output_dir : str
        Path where fmriprep outputs will be written.
    fs_license_file : str, optional
        Path to the FreeSurfer license file.
    image : str
        Docker image name (default: ``nipreps/fmriprep:latest``).
    extra_args : list of str
        Additional arguments forwarded to fmriprep.
    """

    bids_dir: str
    output_dir: str
    fs_license_file: str = ""
    image: str = "nipreps/fmriprep:latest"
    extra_args: List[str] = field(default_factory=list)

    def build_command(self, subject: str, task: str = "") -> List[str]:
        """Construct the ``docker run`` command list."""
        cmd = [
            "docker", "run", "--rm",
            "-v", f"{os.path.abspath(self.bids_dir)}:/data:ro",
            "-v", f"{os.path.abspath(self.output_dir)}:/out",
        ]
        if self.fs_license_file and os.path.isfile(self.fs_license_file):
            cmd += ["-v", f"{os.path.abspath(self.fs_license_file)}:/opt/freesurfer/license.txt:ro"]
        cmd += [
            self.image,
            "/data", "/out", "participant",
            "--participant-label", subject,
            "--nthreads", "4",
            "--omp-nthreads", "4",
            "--output-spaces", "MNI152NLin2009cAsym",
        ]
        if task:
            cmd += ["--task-id", task]
        cmd += self.extra_args
        return cmd

    def run(self, subject: str, task: str = "", timeout: int = 7200) -> subprocess.CompletedProcess:
        """Execute fmriprep for a single subject."""
        _ensure_docker()
        cmd = self.build_command(subject, task)
        return subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)

    def status_dict(self) -> Dict:
        """Return a JSON-serialisable status summary."""
        return {
            "tool": "fmriprep",
            "image": self.image,
            "docker_available": docker_available(),
            "bids_dir": self.bids_dir,
            "output_dir": self.output_dir,
        }


# ── FSL ─────────────────────────────────────────────────────────────

@dataclass
class FSLRunner:
    """Run FSL tools via Docker.

    Uses the official ``brainlife/fsl`` image which contains a full FSL
    installation.
    """

    work_dir: str
    image: str = "brainlife/fsl:6.0.7"

    def run_bet(self, input_path: str, output_path: str,
                frac: float = 0.5) -> subprocess.CompletedProcess:
        """Brain extraction using ``bet``."""
        _ensure_docker()
        abs_work = os.path.abspath(self.work_dir)
        cmd = [
            "docker", "run", "--rm",
            "-v", f"{abs_work}:/work",
            self.image,
            "bet", f"/work/{os.path.relpath(input_path, abs_work)}",
            f"/work/{os.path.relpath(output_path, abs_work)}",
            "-f", str(frac), "-R",
        ]
        return subprocess.run(cmd, capture_output=True, text=True, timeout=600)

    def run_flirt(self, input_path: str, ref_path: str,
                  output_path: str) -> subprocess.CompletedProcess:
        """Linear registration using ``flirt``."""
        _ensure_docker()
        abs_work = os.path.abspath(self.work_dir)
        cmd = [
            "docker", "run", "--rm",
            "-v", f"{abs_work}:/work",
            self.image,
            "flirt",
            "-in", f"/work/{os.path.relpath(input_path, abs_work)}",
            "-ref", f"/work/{os.path.relpath(ref_path, abs_work)}",
            "-out", f"/work/{os.path.relpath(output_path, abs_work)}",
        ]
        return subprocess.run(cmd, capture_output=True, text=True, timeout=600)

    def run_feat(self, design_file: str) -> subprocess.CompletedProcess:
        """Run a full FEAT analysis given a ``.fsf`` design file."""
        _ensure_docker()
        abs_work = os.path.abspath(self.work_dir)
        cmd = [
            "docker", "run", "--rm",
            "-v", f"{abs_work}:/work",
            self.image,
            "feat", f"/work/{os.path.relpath(design_file, abs_work)}",
        ]
        return subprocess.run(cmd, capture_output=True, text=True, timeout=3600)

    def status_dict(self) -> Dict:
        return {
            "tool": "fsl",
            "image": self.image,
            "docker_available": docker_available(),
            "work_dir": self.work_dir,
        }


# ── SPM via MATLAB Runtime Container ───────────────────────────────

@dataclass
class SPMRunner:
    """Run SPM batch scripts via the SPM standalone Docker image.

    The ``spm-standalone`` image ships the MATLAB Compiler Runtime so no
    MATLAB licence is required.
    """

    work_dir: str
    image: str = "spmcentral/spm:standalone-latest"

    def run_batch(self, batch_script: str) -> subprocess.CompletedProcess:
        """Execute an SPM batch MATLAB script."""
        _ensure_docker()
        abs_work = os.path.abspath(self.work_dir)
        cmd = [
            "docker", "run", "--rm",
            "-v", f"{abs_work}:/work",
            self.image,
            "script", f"/work/{os.path.relpath(batch_script, abs_work)}",
        ]
        return subprocess.run(cmd, capture_output=True, text=True, timeout=3600)

    def run_segment(self, input_path: str) -> subprocess.CompletedProcess:
        """Unified segmentation of a structural image."""
        _ensure_docker()
        abs_work = os.path.abspath(self.work_dir)
        cmd = [
            "docker", "run", "--rm",
            "-v", f"{abs_work}:/work",
            self.image,
            "run_spm12.sh", "/opt/mcr/v97/",
            "segment", f"/work/{os.path.relpath(input_path, abs_work)}",
        ]
        return subprocess.run(cmd, capture_output=True, text=True, timeout=3600)

    def status_dict(self) -> Dict:
        return {
            "tool": "spm",
            "image": self.image,
            "docker_available": docker_available(),
            "work_dir": self.work_dir,
        }


# ── helper: pull images ─────────────────────────────────────────────

def pull_image(image: str) -> subprocess.CompletedProcess:
    """Pull a Docker image.  Returns the ``CompletedProcess``."""
    _ensure_docker()
    return subprocess.run(
        ["docker", "pull", image],
        capture_output=True, text=True, timeout=1800,
    )


def list_available_tools() -> List[Dict]:
    """Return status information for all supported Docker tools."""
    has_docker = docker_available()
    return [
        {
            "name": "fmriprep",
            "description": "fMRIPrep — 标准化 fMRI 预处理流水线",
            "image": "nipreps/fmriprep:latest",
            "docker_available": has_docker,
        },
        {
            "name": "fsl",
            "description": "FSL — FMRIB 软件库 (BET, FLIRT, FEAT 等)",
            "image": "brainlife/fsl:6.0.7",
            "docker_available": has_docker,
        },
        {
            "name": "spm",
            "description": "SPM — 统计参数映射 (独立版，无需 MATLAB)",
            "image": "spmcentral/spm:standalone-latest",
            "docker_available": has_docker,
        },
    ]


__all__ = [
    "DockerNotAvailableError",
    "docker_available",
    "FmriprepRunner",
    "FSLRunner",
    "SPMRunner",
    "pull_image",
    "list_available_tools",
]
