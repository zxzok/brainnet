import os
import sys
from pathlib import Path

# Ensure package import path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from data_management import DatasetIndex


def _touch(path: os.PathLike) -> None:
    """Create an empty file at ``path``."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb"):
        pass


def test_index_anatomical_files(tmp_path):
    dataset = tmp_path / "bids"
    anat_file = dataset / "sub-01" / "anat" / "sub-01_T1w.nii.gz"
    _touch(anat_file)

    index = DatasetIndex(str(dataset), datatypes=["anat"])
    files = index.get_files("anat", "01")

    assert len(files) == 1
    assert files[0].datatype == "anat"
    assert files[0].suffix == "T1w"


def test_index_diffusion_files(tmp_path):
    dataset = tmp_path / "bids"
    dwi_file = dataset / "sub-01" / "dwi" / "sub-01_dwi.nii.gz"
    _touch(dwi_file)

    index = DatasetIndex(str(dataset), datatypes=["dwi"])
    files = index.get_files("dwi", "01")

    assert len(files) == 1
    assert files[0].datatype == "dwi"
    assert files[0].suffix == "dwi"

