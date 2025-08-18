"""
data_management
================

This module implements a simple indexer for BIDS‑like datasets. The aim
is to provide a lightweight way to enumerate subjects, sessions and
imaging files without forcing users to load large NIfTI images up front.
Each file is represented by a :class:`BIDSFile` object which stores the
path to the imaging file along with parsed metadata. The dataset indexer
can retrieve lists of files and associated metadata for subsequent
processing.

In addition to the low‑level :class:`DatasetIndex`, a high‑level
convenience wrapper :class:`DatasetManager` is provided. It augments the
index with demographic information loaded from a ``participants.tsv``
file, exposing helper methods to query patient metadata alongside run
information.

The implementation supports common BIDS datatypes such as ``func``,
``anat`` and ``dwi``. It handles optional session hierarchies
(``ses-*``) and will attempt to locate corresponding JSON sidecar files
containing sequence parameters. Metadata embedded in file names (e.g.
``task-rest`` or ``dir-AP``) are also parsed and exposed on each
returned :class:`BIDSFile` instance.

Example
-------
>>> from brainnet.data_management import DatasetIndex
>>> index = DatasetIndex('/path/to/bids_dataset', datatypes=['func', 'anat', 'dwi'])
>>> subjects = index.list_subjects()
>>> runs = index.get_functional_runs(subjects[0])
>>> t1w = index.get_files('anat', subjects[0])
>>> dwi = index.get_files('dwi', subjects[0])
>>> for run in runs:
...     print(run.path, run.metadata.get('RepetitionTime'))

>>> for img in t1w:
...     print(img.path, img.suffix)

>>> for img in dwi:
...     print(img.path, img.metadata.get('dir'))

Note
----
This module does not attempt to fully validate BIDS compliance. If a
directory structure does not follow BIDS conventions the resulting index
may be incomplete or incorrect. Users may want to employ the official
BIDS validator during dataset preparation.
"""

from __future__ import annotations

import csv
import json
import logging
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

logger = logging.getLogger(__name__)


@dataclass
class Patient:
    """Representation of a single participant and optional metadata."""

    participant_id: str
    age: Optional[float] = None
    sex: Optional[str] = None
    diagnosis: Optional[str] = None
    data: Dict[str, str] = field(default_factory=dict)


@dataclass
class BIDSFile:
    """Represent a single imaging file and its metadata."""

    subject: str
    session: Optional[str]
    task: Optional[str]
    run: Optional[str]
    suffix: str
    datatype: str
    path: str
    metadata: Dict = field(default_factory=dict)

    def __repr__(self) -> str:  # pragma: no cover - simple representation
        return (
            f"BIDSFile(subject={self.subject!r}, session={self.session!r}, "
            f"task={self.task!r}, run={self.run!r}, suffix={self.suffix!r}, "
            f"datatype={self.datatype!r}, path={self.path!r})"
        )


def _parse_float(value: Optional[str]) -> Optional[float]:
    """Safely parse a string into a float."""

    if value in (None, ""):
        return None
    try:
        return float(value)
    except ValueError:  # pragma: no cover - logging side effect
        logger.warning("Invalid float value %s in participants.tsv", value)
        return None


def parse_participants_tsv(path: str) -> Dict[str, Patient]:
    """Parse a BIDS ``participants.tsv`` file into :class:`Patient` objects."""

    participants: Dict[str, Patient] = {}
    with open(path, newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            pid = row.get("participant_id")
            if not pid:
                logger.error("Row missing participant_id: %s", row)
                raise ValueError("participant_id is required in participants.tsv")
            age = _parse_float(row.get("age"))
            sex = row.get("sex") or None
            diagnosis = row.get("diagnosis") or None
            data = {
                k: v
                for k, v in row.items()
                if k not in {"participant_id", "age", "sex", "diagnosis"} and v != ""
            }
            participants[pid] = Patient(
                participant_id=pid,
                age=age,
                sex=sex,
                diagnosis=diagnosis,
                data=data,
            )
    return participants


class DatasetIndex:
    """Index a BIDS‑like dataset.

    Parameters
    ----------
    root:
        Path to the dataset root.
    datatypes:
        BIDS datatypes to index. May be a string or a sequence of
        strings, e.g. ``"anat"`` or ``["func", "dwi"]``. If omitted, only
        functional data are indexed.
    """

    def __init__(
        self, root: str, datatypes: Optional[Sequence[str] | str] | None = None
    ) -> None:
        self.root = os.path.abspath(root)
        if not os.path.exists(self.root):
            raise FileNotFoundError(f"Dataset path does not exist: {self.root}")

        if datatypes is None:
            self.datatypes = ["func"]
        elif isinstance(datatypes, str):
            self.datatypes = [datatypes]
        else:
            self.datatypes = list(datatypes)

        self._subjects: List[str] = []
        self._index: Dict[str, Dict[Optional[str], List[BIDSFile]]] = {}
        self._parse_dataset()

    # -- discovery ---------------------------------------------------------
    def _parse_dataset(self) -> None:
        """Discover subjects, sessions and run files for configured datatypes."""

        for entry in sorted(os.listdir(self.root)):
            if entry.startswith("sub-") and os.path.isdir(os.path.join(self.root, entry)):
                subject_label = entry[len("sub-") :]
                self._subjects.append(subject_label)
                self._index[subject_label] = {}
                subj_path = os.path.join(self.root, entry)
                has_session = False
                for ses_entry in sorted(os.listdir(subj_path)):
                    ses_path = os.path.join(subj_path, ses_entry)
                    if ses_entry.startswith("ses-") and os.path.isdir(ses_path):
                        has_session = True
                        session_label = ses_entry[len("ses-") :]
                        self._index[subject_label][session_label] = []
                        self._discover_files(
                            subject_label, session_label, ses_path, self.datatypes
                        )
                if not has_session:
                    self._index[subject_label][None] = []
                    self._discover_files(subject_label, None, subj_path, self.datatypes)

    def _discover_files(
        self,
        subject: str,
        session: Optional[str],
        base_path: str,
        datatypes: Optional[Sequence[str]] | None = None,
    ) -> None:
        """Populate index with imaging files located under ``base_path``."""

        datatypes = datatypes or ["func"]
        for dtype in datatypes:
            dtype_dir = os.path.join(base_path, dtype)
            if not os.path.isdir(dtype_dir):
                continue
            for fname in sorted(os.listdir(dtype_dir)):
                if not (fname.endswith(".nii") or fname.endswith(".nii.gz")):
                    continue
                name, _ = os.path.splitext(fname)
                if name.endswith(".nii"):
                    name, _ = os.path.splitext(name)
                tokens = name.split("_")
                suffix = None
                entities: Dict[str, str] = {}
                for tok in tokens:
                    if tok.startswith("sub-") or tok.startswith("ses-"):
                        continue
                    if "-" in tok:
                        key, value = tok.split("-", 1)
                        entities[key] = value
                    else:
                        suffix = tok
                if suffix is None:
                    suffix = "bold" if dtype == "func" else dtype
                task = entities.get("task")
                run = entities.get("run")
                fpath = os.path.join(dtype_dir, fname)
                json_path = fpath.replace(".nii.gz", ".json").replace(".nii", ".json")
                metadata: Dict[str, str] = dict(entities)
                if os.path.exists(json_path):
                    try:
                        with open(json_path, "r") as jf:
                            metadata.update(json.load(jf))
                    except Exception:  # pragma: no cover - best effort
                        metadata = dict(entities)
                bids_file = BIDSFile(
                    subject=subject,
                    session=session,
                    task=task,
                    run=run,
                    suffix=suffix,
                    datatype=dtype,
                    path=fpath,
                    metadata=metadata,
                )
                self._index[subject][session].append(bids_file)

    # -- query --------------------------------------------------------------
    def list_subjects(self) -> List[str]:
        """Return a sorted list of subject labels."""

        return list(self._subjects)

    def list_sessions(self, subject: str) -> List[Optional[str]]:
        """Return a list of session labels for a given subject.

        If the dataset does not use sessions, a list with one element
        ``[None]`` is returned.
        """

        if subject not in self._index:
            raise KeyError(f"Subject {subject} not found in dataset")
        return list(self._index[subject].keys())

    def get_files(
        self,
        datatype: str,
        subject: str,
        session: Optional[str] | None = None,
    ) -> List[BIDSFile]:
        """Retrieve files for a datatype and subject."""

        if subject not in self._index:
            raise KeyError(f"Subject {subject} not found in dataset")
        sessions = [session] if session is not None else list(self._index[subject].keys())
        files: List[BIDSFile] = []
        for ses in sessions:
            if ses not in self._index[subject]:
                raise KeyError(f"Session {ses} not found for subject {subject}")
            ses_files = [f for f in self._index[subject][ses] if f.datatype == datatype]
            files.extend(ses_files)
        return files

    def get_functional_runs(
        self, subject: str, session: Optional[str] | None = None
    ) -> List[BIDSFile]:
        """Retrieve functional runs for a subject (and optional session)."""

        return self.get_files("func", subject, session=session)


class DatasetManager:
    """Combine dataset indexing with participant metadata parsing."""

    def __init__(self, root: str) -> None:
        self.root = os.path.abspath(root)
        self.index = DatasetIndex(self.root)
        self.patients: Dict[str, Patient] = {}
        participants_path = os.path.join(self.root, "participants.tsv")
        if os.path.exists(participants_path):
            try:
                self.patients = parse_participants_tsv(participants_path)
            except Exception as exc:  # pragma: no cover - logging side effect
                logger.error("Failed to parse participants.tsv: %s", exc)
        else:  # pragma: no cover - logging side effect
            logger.warning("participants.tsv not found at %s", participants_path)

    @staticmethod
    def fetch_from_openneuro(dataset_id: str, target_dir: str | None = None) -> str:
        """Download a dataset from OpenNeuro and return the local path.

        Parameters
        ----------
        dataset_id:
            The OpenNeuro dataset identifier (e.g. ``ds000114``).
        target_dir:
            Directory where the dataset should be stored. If omitted, a
            directory named after ``dataset_id`` in the current working
            directory will be used.

        Returns
        -------
        str
            Absolute path to the downloaded dataset root.

        Raises
        ------
        RuntimeError
            If the optional ``openneuro`` dependency is not available.
        """

        try:  # pragma: no cover - external dependency
            import openneuro
        except ImportError as exc:  # pragma: no cover - informative error
            raise RuntimeError(
                "The 'openneuro' package is required to fetch datasets"
            ) from exc

        target_dir = os.path.abspath(target_dir or dataset_id)
        # The openneuro library exposes a convenience download function.
        openneuro.download(dataset=dataset_id, target_dir=target_dir)
        return target_dir


__all__ = [
    "BIDSFile",
    "Patient",
    "DatasetIndex",
    "DatasetManager",
    "parse_participants_tsv",
]

