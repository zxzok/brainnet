import os
import sys
import json
from pathlib import Path
from types import SimpleNamespace

# Ensure package import path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import data_management as dm
from data_management import DatasetIndex


def _touch(path: os.PathLike) -> None:
    """Create an empty file at ``path``."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb"):
        pass


def test_index_anatomical_files(tmp_path):
    dataset = tmp_path / "bids"
    anat_dir = dataset / "sub-01" / "anat"
    anat_file = anat_dir / "sub-01_acq-highres_T1w.nii.gz"
    _touch(anat_file)
    json_file = Path(str(anat_file).replace(".nii.gz", ".json").replace(".nii", ".json"))
    with open(json_file, "w") as f:
        json.dump({"FlipAngle": 7}, f)

    index = DatasetIndex(str(dataset), datatypes="anat")
    files = index.get_files("anat", "01")

    assert len(files) == 1
    f = files[0]
    assert f.datatype == "anat"
    assert f.suffix == "T1w"
    assert f.metadata["acq"] == "highres"
    assert f.metadata["FlipAngle"] == 7


def test_index_diffusion_files(tmp_path):
    dataset = tmp_path / "bids"
    dwi_dir = dataset / "sub-01" / "dwi"
    dwi_file = dwi_dir / "sub-01_dir-AP_run-1_dwi.nii.gz"
    _touch(dwi_file)
    json_file = Path(str(dwi_file).replace(".nii.gz", ".json").replace(".nii", ".json"))
    with open(json_file, "w") as f:
        json.dump({"PhaseEncodingDirection": "j-"}, f)

    index = DatasetIndex(str(dataset), datatypes=["dwi"])
    files = index.get_files("dwi", "01")

    assert len(files) == 1
    f = files[0]
    assert f.datatype == "dwi"
    assert f.suffix == "dwi"
    assert f.run == "1"
    assert f.metadata["dir"] == "AP"
    assert f.metadata["PhaseEncodingDirection"] == "j-"


def test_index_from_openneuro(monkeypatch, tmp_path):
    calls: list[tuple[str, str]] = []

    def fake_download_dataset(*, dataset: str, target_dir: str, **kwargs):
        calls.append((dataset, target_dir))
        ds_dir = Path(target_dir) / dataset / "sub-01" / "anat"
        ds_dir.mkdir(parents=True)
        _touch(ds_dir / "sub-01_T1w.nii.gz")

    fake_client = SimpleNamespace(download_dataset=fake_download_dataset)
    monkeypatch.setattr(dm, "openneuro_client", fake_client)
    monkeypatch.setenv("HOME", str(tmp_path))

    index = DatasetIndex("ds999999", datatypes="anat", source="openneuro")
    files = index.get_files("anat", "01")
    assert len(files) == 1
    assert calls
    # Subsequent initialisation should use cache
    calls.clear()
    DatasetIndex("ds999999", datatypes="anat", source="openneuro")
    assert calls == []

