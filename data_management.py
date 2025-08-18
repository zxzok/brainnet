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

The implementation focuses on the functional data contained under
``func/`` directories in a BIDS dataset. It supports optional session
hierarchies (``ses-*``) and will attempt to locate corresponding JSON
sidecar files containing sequence parameters. Additional datatypes (such
as ``anat`` or ``dwi``) could be added by extending the
``_discover_files`` method.

Example
-------
>>> from brainnet.data_management import DatasetIndex
>>> index = DatasetIndex('/path/to/bids_dataset', datatypes=['func', 'anat'])
>>> subjects = index.list_subjects()
>>> runs = index.get_functional_runs(subjects[0])
>>> t1w = index.get_files(subjects[0], datatype='anat')
>>> for run in runs:
...     print(run.path, run.metadata.get('RepetitionTime'))

>>> for img in t1w:
...     print(img.path, img.suffix)

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
from typing import Dict, List, Optional

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
    """Index a BIDS‑like dataset."""

    def __init__(self, root: str, datatypes: Optional[List[str]] | None = None) -> None:
        self.root = os.path.abspath(root)
        if not os.path.exists(self.root):
            raise FileNotFoundError(f"Dataset path does not exist: {self.root}")

        self.datatypes = datatypes or ["func"]
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
        datatypes: Optional[List[str]] | None = None,
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
                task = run = suffix = None
                for tok in tokens:
                    if tok.startswith("task-"):
                        task = tok[len("task-") :]
                    elif tok.startswith("run-"):
                        run = tok[len("run-") :]
                    elif tok.startswith("sub-") or tok.startswith("ses-"):
                        continue
                    else:
                        suffix = tok
                if suffix is None:
                    suffix = "bold" if dtype == "func" else dtype
                fpath = os.path.join(dtype_dir, fname)
                json_path = fpath.replace(".nii.gz", ".json").replace(".nii", ".json")
                metadata: Dict = {}
                if os.path.exists(json_path):
                    try:
                        with open(json_path, "r") as jf:
                            metadata = json.load(jf)
                    except Exception:  # pragma: no cover - best effort
                        metadata = {}
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


__all__ = [
    "BIDSFile",
    "Patient",
    "DatasetIndex",
    "DatasetManager",
    "parse_participants_tsv",
]

