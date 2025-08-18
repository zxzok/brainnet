"""Multimodal preprocessing orchestration utilities.

This module defines :class:`MultimodalPreprocessor` which coordinates
execution of modality specific preprocessing routines.  Each modality is
associated with a :class:`ModalityConfig` describing how it should be
run.  Results for each modality are stored within a dedicated directory
under a user supplied working directory using a standard layout::

    work_dir/
        <modality>/
            intermediate/
            output/

The preprocessor maintains a simple cache: if a modality has already
produced results (indicated by a ``.complete`` marker in its output
folder) it will be skipped on subsequent runs.  A logging facility writes
parameters, executed commands and wall clock duration for each modality
into ``multimodal.log`` in the working directory.
"""

from __future__ import annotations

import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, Optional
import logging


@dataclass
class ModalityConfig:
    """Configuration for a single modality.

    Parameters
    ----------
    enabled:
        Whether the modality should be executed.
    runner:
        Optional Python callable implementing the modality.  The callable
        will receive three positional arguments ``intermediate_dir``,
        ``output_dir`` and ``params`` (a dict of any additional
        configuration parameters).
    command:
        Shell command to execute for the modality.  Either ``runner`` or
        ``command`` may be provided.  If both are supplied ``command``
        takes precedence.
    params:
        Additional parameters passed to ``runner`` or used to format the
        shell command via ``str.format``.
    """

    enabled: bool = True
    runner: Optional[Callable[[Path, Path, Dict[str, Any]], Any]] = None
    command: Optional[str] = None
    params: Dict[str, Any] = field(default_factory=dict)


class MultimodalPreprocessor:
    """Coordinate preprocessing across multiple modalities.

    The class manages a working directory, executes enabled modalities
    and records detailed logs for reproducibility.
    """

    def __init__(self, work_dir: Path | str, modalities: Dict[str, ModalityConfig]) -> None:
        self.work_dir = Path(work_dir).expanduser().resolve()
        self.modalities = modalities
        self.work_dir.mkdir(parents=True, exist_ok=True)

        # set up logger writing to work_dir/multimodal.log
        self.logger = logging.getLogger(__name__ + ".MultimodalPreprocessor")
        self.logger.setLevel(logging.INFO)
        log_path = self.work_dir / "multimodal.log"
        handler = logging.FileHandler(log_path, encoding="utf-8")
        formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
        handler.setFormatter(formatter)
        # avoid duplicate handlers if multiple instances are created
        if not self.logger.handlers:
            self.logger.addHandler(handler)
        else:
            # replace handlers to ensure logging to the correct directory
            self.logger.handlers = [handler]

    # ------------------------------------------------------------------
    def _modality_dirs(self, name: str) -> tuple[Path, Path, Path]:
        """Return (root, intermediate, output) directories for a modality."""

        root = self.work_dir / name
        intermediate = root / "intermediate"
        output = root / "output"
        for d in (root, intermediate, output):
            d.mkdir(parents=True, exist_ok=True)
        return root, intermediate, output

    # ------------------------------------------------------------------
    def run(self) -> Dict[str, Any]:
        """Execute all enabled modalities and return their results.

        If a modality has previously completed (as indicated by a
        ``.complete`` marker file in its output directory) it will be
        skipped on subsequent runs.
        """

        results: Dict[str, Any] = {}
        for name, cfg in self.modalities.items():
            if not cfg.enabled:
                self.logger.info("Modality '%s' disabled; skipping", name)
                continue

            root, inter_dir, out_dir = self._modality_dirs(name)
            marker = out_dir / ".complete"
            if marker.exists():
                self.logger.info("Using cached results for '%s'", name)
                continue

            self.logger.info("Starting modality '%s'", name)
            self.logger.info("Parameters: %s", cfg.params)
            start = time.time()
            try:
                if cfg.command:
                    cmd = cfg.command.format(**cfg.params)
                    self.logger.info("Command: %s", cmd)
                    subprocess.run(cmd, shell=True, check=True, cwd=root)
                    result = None
                elif cfg.runner:
                    self.logger.info("Runner: %s", cfg.runner)
                    result = cfg.runner(inter_dir, out_dir, cfg.params)
                else:
                    self.logger.warning("No runner or command for modality '%s'", name)
                    continue
            except Exception as exc:  # pragma: no cover - best effort logging
                self.logger.exception("Modality '%s' failed: %s", name, exc)
                raise
            finally:
                elapsed = time.time() - start
                self.logger.info("Finished '%s' in %.2f s", name, elapsed)

            marker.touch()
            results[name] = result
        return results
