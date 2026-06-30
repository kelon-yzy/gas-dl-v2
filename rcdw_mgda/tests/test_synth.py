"""测试合成数据模块。"""
import numpy as np
import pytest
from rcdw.data.synth import synth_timeseries, make_windows, make_splits, WindowedDataset


def test_timeseries_shape_and_sum():
    X, Y = synth_timeseries(100, seed=0)
    assert X.shape == (100, 6)
    assert Y.shape == (100, 3)
    np.testing.assert_allclose(Y.sum(axis=1), 1.0, atol=0.02)


def test_window_shape():
    X, Y = synth_timeseries(20, seed=0)
    Xw, Yw = make_windows(X, Y, L=8)
    assert Xw.shape == (13, 8, 6)
    assert Yw.shape == (13, 3)


def test_splits_no_overlap():
    splits = make_splits(n_train=100, n_val=20, n_test=20, L=8, seed=0)
    assert splits["train"][0].shape[0] == 100
    assert splits["val"][0].shape[0] == 20
    assert splits["test"][0].shape[0] == 20


def test_windowed_dataset():
    splits = make_splits(n_train=50, n_val=10, n_test=10, L=8, seed=0)
    ds = WindowedDataset(*splits["train"])
    x, y = ds[0]
    assert x.shape == (8, 6)
    assert y.shape == (3,)
    assert y.sum().item() == pytest.approx(1.0, abs=0.02)
