import json
import sys
import subprocess
import importlib.util
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2] / "brainnet" / "data_management.py"
spec = importlib.util.spec_from_file_location("data_management", ROOT)
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
assert spec and spec.loader
spec.loader.exec_module(module)  # type: ignore[attr-defined]
DatasetManager = module.DatasetManager


def _create_mock_dataset(root: Path) -> None:
    # Subject 01 with two runs and no sessions
    func1 = root / "sub-01" / "func"
    func1.mkdir(parents=True)
    (func1 / "sub-01_task-rest_run-01_bold.nii.gz").touch()
    (func1 / "sub-01_task-rest_run-02_bold.nii.gz").touch()

    # Subject 02 with two sessions each containing one run
    for ses in ["ses-01", "ses-02"]:
        func_dir = root / "sub-02" / ses / "func"
        func_dir.mkdir(parents=True)
        fname = f"sub-02_{ses}_task-rest_run-01_bold.nii.gz"
        (func_dir / fname).touch()


def test_summarize(tmp_path: Path) -> None:
    _create_mock_dataset(tmp_path)
    mgr = DatasetManager(str(tmp_path))
    summary = mgr.summarize()
    assert summary["total_subjects"] == 2
    assert summary["sessions_per_subject"] == {"01": 1, "02": 2}
    assert summary["runs_per_session"] == {"01": {None: 2}, "02": {"01": 1, "02": 1}}


def test_cli_summary(tmp_path: Path) -> None:
    _create_mock_dataset(tmp_path)
    env = {**os.environ, "PYTHONPATH": str(Path(__file__).resolve().parents[2])}
    result = subprocess.check_output(
        [sys.executable, "-m", "brainnet.data_management", "summary", str(tmp_path)],
        env=env,
    )
    data = json.loads(result.decode("utf-8"))
    assert data["total_subjects"] == 2
    assert data["sessions_per_subject"] == {"01": 1, "02": 2}
    assert data["runs_per_session"]["01"]["null"] == 2
