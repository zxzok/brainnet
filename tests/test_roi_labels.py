import numpy as np
import pytest
import sys
from pathlib import Path
import importlib.util

preproc_path = Path(__file__).resolve().parents[1] / 'preprocessing.py'
spec = importlib.util.spec_from_file_location('preprocessing', preproc_path)
preproc_module = importlib.util.module_from_spec(spec)
sys.modules['preprocessing'] = preproc_module
spec.loader.exec_module(preproc_module)
PreprocConfig = preproc_module.PreprocConfig
Preprocessor = preproc_module.Preprocessor


def test_roi_extraction_with_named_labels(tmp_path, monkeypatch):
    nib = pytest.importorskip('nibabel')
    # create simple atlas with two regions
    atlas_data = np.array([[[1]], [[2]]])  # shape (2,1,1)
    atlas_img = nib.Nifti1Image(atlas_data, np.eye(4))
    atlas_path = tmp_path / 'atlas.nii.gz'
    nib.save(atlas_img, atlas_path)
    # create functional data with two voxels over four time points
    func_data = np.zeros((2,1,1,4), dtype=float)
    func_data[0,0,0,:] = [1,2,3,4]
    func_data[1,0,0,:] = [5,6,7,8]
    func_img = nib.Nifti1Image(func_data, np.eye(4))
    func_path = tmp_path / 'func.nii.gz'
    nib.save(func_img, func_path)
    cfg = PreprocConfig(extract_roi=False, retain_4d=False)
    pre = Preprocessor(cfg)
    pre._atlas_data = {'dummy': atlas_data}
    pre._atlas_ids = {'dummy': [1, 2]}
    pre._atlas_labels = {'dummy': ['left', 'right']}
    pre.config.extract_roi = True
    out = pre.preprocess(str(func_path))
    assert list(out.roi_labels['dummy'].keys()) == [1, 2]
    assert list(out.roi_labels['dummy'].values()) == ['left', 'right']
    df = out.get_roi_dataframe('dummy')
    assert list(df.columns) == ['left', 'right']
    assert np.allclose(df['left'].values, [1,2,3,4])
