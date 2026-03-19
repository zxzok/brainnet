"""Runtime paths and helpers for local application state.

This module centralizes filesystem locations for mutable artifacts such as
the SQLite database and uploaded files so they can be redirected in tests,
CI, and local development environments.
"""

from __future__ import annotations

import os
import sqlite3
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent
DEFAULT_INSTANCE_DIR = REPO_ROOT / "instance"


def _ensure_parent_dir(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def get_instance_dir() -> Path:
    """Return the directory for mutable local runtime state."""

    instance_dir = Path(os.environ.get("BRAINNET_INSTANCE_DIR", DEFAULT_INSTANCE_DIR))
    instance_dir.mkdir(parents=True, exist_ok=True)
    return instance_dir


def get_database_path() -> Path:
    """Return the SQLite database path for the current runtime."""

    db_override = os.environ.get("BRAINNET_DB_PATH")
    if db_override:
        return _ensure_parent_dir(Path(db_override))
    return _ensure_parent_dir(get_instance_dir() / "brainnet.db")


def get_upload_dir() -> Path:
    """Return the directory where uploaded images should be stored."""

    upload_override = os.environ.get("BRAINNET_UPLOAD_DIR")
    upload_dir = Path(upload_override) if upload_override else get_instance_dir() / "uploads"
    upload_dir.mkdir(parents=True, exist_ok=True)
    return upload_dir


def get_reports_dir() -> Path:
    """Return the directory where generated reports should be stored."""

    reports_override = os.environ.get("BRAINNET_REPORT_DIR")
    reports_dir = Path(reports_override) if reports_override else get_instance_dir() / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    return reports_dir


def get_openneuro_cache_dir() -> Path:
    """Return the directory used for downloaded OpenNeuro datasets."""

    cache_override = os.environ.get("OPENNEURO_CACHE_DIR")
    cache_dir = Path(cache_override) if cache_override else get_instance_dir() / "openneuro_datasets"
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir


def connect_db() -> sqlite3.Connection:
    """Open a SQLite connection to the configured BrainNet database."""

    return sqlite3.connect(get_database_path())
