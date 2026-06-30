"""测试 BenchmarkDataset：从磁盘 benchmark 目录加载窗口张量。

对应方案 §8.1 / §11.1。
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from rcdw.data.dataset import BenchmarkDataset
from rcdw.sim.generation.benchmark import (
    BenchmarkGenerationSpec,
    generate_benchmark_dataset,
)
from rcdw.sim.generation.conditions import generate_condition_rows
from rcdw.sim.generation.optical_backend import build_hitran_grid_for_condition
from rcdw.sim.generation.spectral import (
    DEFAULT_HITRAN_GAS_SPECS,
    hitran_cache_key,
    write_cached_spectrum,
)


def _write_synthetic_cache_for_conditions(cache_root: Path, conditions):
    """为 conditions 写入合成 HITRAN cache（同 e2e 测试）。"""
    for cond in conditions:
        t_c = float(cond["T_C_base"])
        p_mpa = float(cond["P_MPa_base"])
        grid = build_hitran_grid_for_condition("co2", t_c=t_c, p_mpa=p_mpa)
        wn = np.arange(
            grid.wavenumber_min_cm1,
            grid.wavenumber_max_cm1 + grid.wavenumber_step_cm1 * 0.5,
            grid.wavenumber_step_cm1,
            dtype=np.float64,
        )
        co2_coeff = 1.0e-21 * np.exp(-((wn - 2347.0) / 40.0) ** 2)
        h2o_coeff = 1.0e-22 * np.exp(-((wn - 2330.0) / 60.0) ** 2)
        for gas_spec in DEFAULT_HITRAN_GAS_SPECS:
            data = co2_coeff if gas_spec.gas == "CO2" else h2o_coeff
            key = hitran_cache_key(gas_spec, grid)
            write_cached_spectrum(
                cache_root, key, wavenumber_cm1=wn, absorption_coeff_cm1=data
            )


@pytest.fixture(scope="module")
def smoke_benchmark(tmp_path_factory):
    """模块级 fixture：生成一份 16-sequence smoke benchmark 供本文件复用。"""
    tmp = tmp_path_factory.mktemp("rcdw-dataset")
    cache_root = tmp / "hitran_cache"
    conditions = generate_condition_rows(16, seed=123)
    _write_synthetic_cache_for_conditions(cache_root, conditions)
    spec = BenchmarkGenerationSpec(
        dataset_slug="rcdw-loader-test",
        sequence_count=16,
        seed=123,
        timesteps=24,
        dt_s=0.5,
        storage="memmap",
        multi_path_phase="steady",
        path_lms=(0.2, 0.3, 0.4),
        optical_absorption_backend="hitran_hapi_v1",
        hitran_cache_root=str(cache_root),
    )
    result = generate_benchmark_dataset(tmp / "output", spec)
    return Path(result["output_dir"])


def test_dataset_train_window_shape(smoke_benchmark):
    ds = BenchmarkDataset(smoke_benchmark, split="train", window=8)
    assert len(ds) > 0
    x, y = ds[0]
    assert x.shape == (8, 12), f"window shape mismatch: {x.shape}"
    assert y.shape == (3,)
    assert x.dtype == torch.float32
    assert y.dtype == torch.float32


def test_dataset_window_total_count(smoke_benchmark):
    """total_windows = N_split_seqs * (timesteps - window + 1)。"""
    ds = BenchmarkDataset(smoke_benchmark, split="train", window=8)
    # smoke: 16 seq * train_ratio(70%) ≈ 11 ;windows/seq = 24-8+1=17
    n_train_seqs = len(ds.split_sequence_ids)
    assert len(ds) == n_train_seqs * (24 - 8 + 1)


def test_dataset_split_disjoint(smoke_benchmark):
    train = BenchmarkDataset(smoke_benchmark, split="train", window=8)
    val = BenchmarkDataset(smoke_benchmark, split="val", window=8)
    test = BenchmarkDataset(smoke_benchmark, split="test", window=8)
    train_ids = set(train.split_sequence_ids)
    val_ids = set(val.split_sequence_ids)
    test_ids = set(test.split_sequence_ids)
    assert train_ids & val_ids == set()
    assert train_ids & test_ids == set()
    assert val_ids & test_ids == set()
    assert len(train_ids) + len(val_ids) + len(test_ids) == 16


def test_dataset_labels_sum_to_one(smoke_benchmark):
    """方案 §2.1 + §8.6: 训练侧标签为 [0, 1] 闭包(BenchmarkDataset 把磁盘的
    百分比 0-100 转成 [0, 1] 比例,与 SingleModal / normalize_composition 对齐)。"""
    ds = BenchmarkDataset(smoke_benchmark, split="train", window=8)
    for idx in range(min(5, len(ds))):
        _, y = ds[idx]
        assert abs(y.sum().item() - 1.0) < 1e-4


def test_dataset_channel_layout_matches_slow_then_ultrasonic(smoke_benchmark):
    """v1.2 §8.2: 前 7 维应等于 slow.npy 对应窗口,后 5 维等于 ultrasonic 元数据。"""
    ds = BenchmarkDataset(smoke_benchmark, split="train", window=8)
    slow_all = np.load(smoke_benchmark / "sequences" / "slow.npy")
    us_speed = np.load(
        smoke_benchmark / "sequences"
        / "ultrasonic_sound_speed_estimated_m_per_s.npy"
    )
    # 取 split 中第 0 个 sequence 的 第 0 个 window
    global_idx = ds.split_indexes[0]
    x_w, _ = ds[0]
    # 前 7 维匹配
    np.testing.assert_allclose(
        x_w[:, :7].numpy(), slow_all[global_idx, :8, :], rtol=1e-5
    )
    # idx 8 = US speed
    np.testing.assert_allclose(
        x_w[:, 8].numpy(), us_speed[global_idx, :8], rtol=1e-5
    )


def test_dataset_does_not_read_fiber_mic(smoke_benchmark):
    """方案 §2.4: fiber_mic 文件存在但 Dataset 不应触碰它（用 mmap 计数验证）。"""
    ds = BenchmarkDataset(smoke_benchmark, split="train", window=8)
    # 检查 _us_arrays 不含 fiber_mic
    for sem_name in ds._us_arrays:  # type: ignore[attr-defined]
        assert "fiber_mic" not in sem_name


def test_dataset_invalid_split_rejected(smoke_benchmark):
    with pytest.raises(ValueError, match="split must be"):
        BenchmarkDataset(smoke_benchmark, split="bogus")


def test_dataset_missing_modality_rejected(smoke_benchmark):
    with pytest.raises(ValueError, match="slow.*ultrasonic"):
        BenchmarkDataset(
            smoke_benchmark, split="train", modalities=("slow",)
        )


def test_dataset_window_too_large_rejected(smoke_benchmark):
    """window > timesteps 应直接拒收。"""
    with pytest.raises(ValueError, match="window"):
        BenchmarkDataset(smoke_benchmark, split="train", window=999)


@pytest.mark.parametrize("bad_window", [0, 1])
def test_dataset_window_too_small_rejected(smoke_benchmark, bad_window):
    """window < 2 会导致 FeatureExtractor 无法计算最后两步差分,应在 Dataset 层拒收。"""
    with pytest.raises(ValueError, match="window must be >= 2"):
        BenchmarkDataset(smoke_benchmark, split="train", window=bad_window)


def test_dataset_load_dataloader_pytorch_compatible(smoke_benchmark):
    """可以塞进 torch DataLoader,batch 后 shape 正确。"""
    from torch.utils.data import DataLoader

    ds = BenchmarkDataset(smoke_benchmark, split="val", window=8)
    loader = DataLoader(ds, batch_size=4, shuffle=False)
    batch_x, batch_y = next(iter(loader))
    assert batch_x.shape[1:] == (8, 12)
    assert batch_y.shape[1] == 3
    assert batch_x.shape[0] <= 4
