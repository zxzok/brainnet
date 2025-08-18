"""
data_management
================

This module implements a simple indexer for BIDS‑like datasets.  The aim
is to provide a lightweight way to enumerate subjects, sessions and
functional runs without forcing users to load large NIfTI images up
front.  Each run is represented by a :class:`BIDSFile` object which
stores the path to the imaging file along with parsed metadata.  The
dataset indexer can retrieve lists of runs and associated metadata for
subsequent processing.

In addition to the low‑level :class:`DatasetIndex`, a high‑level
convenience wrapper :class:`DatasetManager` is provided.  It augments the
index with demographic information loaded from a ``participants.tsv``
file, exposing helper methods to query patient metadata alongside run
information.

The implementation focuses on the functional data contained under
``func/`` directories in a BIDS dataset.  It supports optional session
hierarchies (``ses-*``) and will attempt to locate corresponding JSON
sidecar files containing sequence parameters.  Additional datatypes
(such as ``anat`` or ``dwi``) could be added by extending the
``_discover_files`` method.

Example
-------
>>> from brainnet.data_management import DatasetIndex
>>> index = DatasetIndex('/path/to/bids_dataset')
>>> subjects = index.list_subjects()
>>> runs = index.get_functional_runs(subjects[0])
>>> for run in runs:
...     print(run.path, run.metadata.get('RepetitionTime'))

Note
----
This module does not attempt to fully validate BIDS compliance.  If a
directory structure does not follow BIDS conventions the resulting
index may be incomplete or incorrect.  Users may want to employ the
official BIDS validator during dataset preparation.
"""

from __future__ import annotations

import os
import json
import csv
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
import csv


@dataclass
class BIDSFile:
    """Represent a single imaging file and its metadata.

    Parameters
    ----------
    subject : str
        Subject identifier (e.g. ``'01'`` for ``sub-01``).
    session : str | None
        Session identifier if present (e.g. ``'01'`` for ``ses-01``),
        otherwise ``None``.
    task : str | None
        Task name for functional data (e.g. ``'rest'`` for
        ``task-rest``).  ``None`` if not applicable.
    run : str | None
        Run identifier (e.g. ``'01'`` for ``run-01``).  ``None`` if not
        provided.
    suffix : str
        Datatype-specific suffix (e.g. ``'bold'``, ``'T1w'``, ``'dwi'``).
        In BIDS this appears after any key‑value entities in the filename.
    datatype : str
        Top‑level BIDS datatype folder (e.g. ``'func'``, ``'anat'``, ``'dwi'``).
    path : str
        Absolute path to the file on disk.
    metadata : dict
        Dictionary of metadata loaded from the JSON sidecar, if any.
    """

    subject: str
    session: Optional[str]
    task: Optional[str]
    run: Optional[str]
    suffix: str
    datatype: str
    path: str
    metadata: Dict = field(default_factory=dict)

    def __repr__(self) -> str:
        return (
            f"BIDSFile(subject={self.subject!r}, session={self.session!r}, "
            f"task={self.task!r}, run={self.run!r}, suffix={self.suffix!r}, "
            f"datatype={self.datatype!r}, path={self.path!r})"
        )


@dataclass
class Patient:
    """Store basic demographic and diagnostic information for a subject.

    The dataclass captures the most common fields found in a BIDS
    ``participants.tsv`` file and allows arbitrary additional metadata to
    be stored in the ``attributes`` dictionary.  Utility methods are
    provided to assist with serialisation when exporting patient records
    to other formats.
    """

    id: str
    age: Optional[float] = None
    sex: Optional[str] = None
    diagnosis: Optional[str] = None
    attributes: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Serialise the patient to a dictionary."""
        data = {
            "id": self.id,
            "age": self.age,
            "sex": self.sex,
            "diagnosis": self.diagnosis,
        }
        data.update(self.attributes)
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Patient":
        """Create a :class:`Patient` instance from a dictionary."""
        known = {k: data.get(k) for k in ["id", "age", "sex", "diagnosis"]}
        extras = {
            k: v
            for k, v in data.items()
            if k not in {"id", "age", "sex", "diagnosis"}
        }
        return cls(attributes=extras, **known)


class DatasetIndex:
    """Index a BIDS‑like dataset.

    The dataset is parsed at initialisation.  Subjects and sessions are
    discovered by scanning directories that match the BIDS naming
    conventions (``sub-<label>`` and ``ses-<label>``).  Functional
    runs are identified by filenames containing the ``task-`` and
    ``_bold`` patterns.

    Parameters
    ----------
    root : str
        Path to the root of the BIDS dataset.
    """

    def __init__(self, root: str, datatypes: Optional[List[str]] | None = None) -> None:
        self.root = os.path.abspath(root)
        if not os.path.exists(self.root):
            raise FileNotFoundError(f"Dataset path does not exist: {self.root}")
        # datatypes to index (e.g. ['func', 'anat', 'dwi'])
        self.datatypes = datatypes or ['func']
        self._subjects: List[str] = []
        # index: subject -> session -> list of BIDSFile
        self._index: Dict[str, Dict[Optional[str], List[BIDSFile]]] = {}
        self._parse_dataset()

    # -- discovery ---------------------------------------------------------
    def _parse_dataset(self) -> None:
        """Discover subjects, sessions and functional runs."""
        # identify subject directories
        for entry in sorted(os.listdir(self.root)):
            if entry.startswith('sub-') and os.path.isdir(os.path.join(self.root, entry)):
                subject_label = entry[len('sub-'):]
                self._subjects.append(subject_label)
                self._index[subject_label] = {}
                subj_path = os.path.join(self.root, entry)
                # find sessions or direct datatypes
                has_session = False
                for ses_entry in sorted(os.listdir(subj_path)):
                    ses_path = os.path.join(subj_path, ses_entry)
                    if ses_entry.startswith('ses-') and os.path.isdir(ses_path):
                        has_session = True
                        session_label = ses_entry[len('ses-'):]
                        self._index[subject_label][session_label] = []
                        self._discover_files(
                            subject_label, session_label, ses_path, self.datatypes
                        )
                if not has_session:
                    # no sessions: treat subject directory as session None
                    self._index[subject_label][None] = []
                    self._discover_files(subject_label, None, subj_path, self.datatypes)

    def _discover_files(
        self,
        subject: str,
        session: Optional[str],
        base_path: str,
        datatypes: List[str],
    ) -> None:
        """Populate index with imaging files located under ``base_path``.

        Parameters
        ----------
        subject : str
            Subject identifier without ``sub-`` prefix.
        session : str | None
            Session identifier without ``ses-`` prefix or ``None`` if no
            sessions exist.
        base_path : str
            Path to search for datatype directories (should be the subject or
            ``ses-`` directory).
        datatypes : list[str]
            List of datatype directory names to search (e.g. ``['func', 'anat']``).
        """
        for dtype in datatypes:
            dtype_dir = os.path.join(base_path, dtype)
            if not os.path.isdir(dtype_dir):
                continue
            for fname in sorted(os.listdir(dtype_dir)):
                if not (fname.endswith('.nii') or fname.endswith('.nii.gz')):
                    continue
                name, _ = os.path.splitext(fname)
                if name.endswith('.nii'):  # handle .nii.gz
                    name, _ = os.path.splitext(name)
                tokens = name.split('_')
                task = None
                run = None
                suffix = None
                for tok in tokens:
                    if tok.startswith('task-'):
                        task = tok[len('task-'):]
                    elif tok.startswith('run-'):
                        run = tok[len('run-'):]
                    elif tok.startswith('sub-') or tok.startswith('ses-'):
                        continue
                    else:
                        suffix = tok
                if suffix is None:
                    suffix = 'bold' if dtype == 'func' else dtype
                fpath = os.path.join(dtype_dir, fname)
                json_path = fpath.replace('.nii.gz', '.json').replace('.nii', '.json')
                metadata: Dict = {}
                if os.path.exists(json_path):
                    try:
                        with open(json_path, 'r') as jf:
                            metadata = json.load(jf)
                    except Exception:
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
        self, datatype: str, subject: str, session: Optional[str] | None = None
    ) -> List[BIDSFile]:
        """Retrieve files of a given datatype for a subject and optional session."""
        if subject not in self._index:
            raise KeyError(f"Subject {subject} not found in dataset")
        if session is None:
            files: List[BIDSFile] = []
            for ses_files in self._index[subject].values():
                files.extend(f for f in ses_files if f.datatype == datatype)
            return files
        if session not in self._index[subject]:
            raise KeyError(f"Session {session} not found for subject {subject}")
        return [f for f in self._index[subject][session] if f.datatype == datatype]

    def get_functional_runs(
        self, subject: str, session: Optional[str] | None = None
    ) -> List[BIDSFile]:
        """Retrieve functional runs for a subject (and optional session)."""
        return self.get_files('func', subject, session)





__all__ = [
    'BIDSFile',
    'Patient',
    'DatasetIndex',
    'DatasetManager',
]