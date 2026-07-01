"""测试 H1 修复：12 维 input scaler 的拟合、落盘、manifest 标志位与 Dataset 应用。

覆盖：
- fit_input_channel_scaler 单元行为（z_score / passthrough / 缺通道报错）。
- 生成侧：input_scaler.json 落盘 + manifest.input_normalization 标志位。
- 消费侧：Dataset 默认应用、精确匹配、标准化后统计接近 0/1、原始回退、
  向后兼容（无标志位不应用）、显式 True 无标志位报错、通道顺序错位报错。
"""
from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np
import pytest

from rcdw.data.dataset import BenchmarkDataset
from rcdw.sim.generation.benchmark import (
    BenchmarkGenerationSpec,
    generate_benchmark_dataset,
)
from rcdw.sim.packaging.scalers import (
    INPUT_CHANNEL_ORDER,
    fit_input_channel_scaler,
)


@pytest.fixture(scope="module")
def norm_benchmark(tmp_path_factory):
    """模块级 fixture：用离线 empirical 后端生成一份含 input scaler 的 benchmark。"""
    tmp = tmp_path_factory.mktemp("rcdw-inscaler")
    spec = BenchmarkGenerationSpec(
        dataset_slug="rcdw-inscaler-test",
        sequence_count=16,
        seed=7,
        timesteps=20,
        dt_s=0.5,
        storage="memmap",
        multi_path_phase="steady",
        path_lms=(0.2, 0.3, 0.4),
        optical_absorption_backend="empirical_v1",
    )
    result = generate_benchmark_dataset(tmp / "out", spec)
    return Path(result["output_dir"])


# ---- fit_input_channel_scaler 单元测试 ----


def test_fit_input_channel_scaler_constant_channel_passthrough():
    rng = np.random.default_rng(0)
    train_values = {name: rng.normal(size=64) for name in INPUT_CHANNEL_ORDER}
    # 非默认-skip 通道人为置常量 → 应自动 passthrough（std<=eps 兜底）。
    train_values["ultrasonic_peak_index"] = np.full(64, 5.0)
    scaler = fit_input_channel_scaler(train_values)
    by = {e["channel"]: e for e in scaler["channel_entries"]}
    assert by["ultrasonic_peak_index"]["strategy"] == "passthrough"
    assert by["ultrasonic_peak_index"]["reason"] == "std<=eps"
    # 默认声明 skip 的布尔门控通道也是 passthrough。
    assert by["ultrasonic_tof_accepted"]["strategy"] == "passthrough"
    # 正常通道走 z_score。
    assert by["V_NDIR_CO2"]["strategy"] == "z_score"
    assert scaler["coverage"] == "input_12ch"
    assert scaler["fit_scope"] == "train_split_only"


def test_fit_input_channel_scaler_missing_channel_raises():
    with pytest.raises(ValueError, match="missing train values"):
        fit_input_channel_scaler({"V_NDIR_CO2": np.zeros(4)})


# ---- 生成侧：产物与 manifest ----


def test_input_scaler_json_written_and_covers_us_main_signal(norm_benchmark):
    scaler = json.loads(
        (norm_benchmark / "scalers" / "input_scaler.json").read_text("utf-8")
    )
    names = scaler["channel_names"]
    assert len(names) == 12
    assert names[8] == "ultrasonic_sound_speed_estimated_m_per_s"
    by = {e["channel"]: e for e in scaler["channel_entries"]}
    # H1 核心：US 主信号（~346 m/s）必须被纳入 z_score，而非停留在原始量级。
    us = by["ultrasonic_sound_speed_estimated_m_per_s"]
    assert us["strategy"] == "z_score"
    assert us["mean"] > 100.0
    # 布尔门控常量通道 → passthrough（零方差，规避除零）。
    assert by["ultrasonic_tof_accepted"]["strategy"] == "passthrough"


def test_manifest_input_normalization_flag(norm_benchmark):
    manifest = json.loads((norm_benchmark / "manifest.json").read_text("utf-8"))
    norm = manifest["input_normalization"]
    assert norm["applied"] is True
    assert norm["coverage"] == "input_12ch"
    assert norm["artifact"] == "scalers/input_scaler.json"


# ---- 消费侧：Dataset 应用 ----


def test_dataset_applies_scaler_exact_match(norm_benchmark):
    """默认应用的结果 == 原始 (apply_input_scaler=False) 逐通道套 scaler 统计。"""
    scaler = json.loads(
        (norm_benchmark / "scalers" / "input_scaler.json").read_text("utf-8")
    )
    mean = np.array(
        [m if m is not None else 0.0 for m in scaler["mean"]], dtype=np.float32
    )
    std = np.array(
        [s if s is not None else 1.0 for s in scaler["std"]], dtype=np.float32
    )

    raw = BenchmarkDataset(
        norm_benchmark, split="train", window=8, apply_input_scaler=False
    )
    scaled = BenchmarkDataset(norm_benchmark, split="train", window=8)  # 默认→应用
    x_raw, _ = raw[0]
    x_scaled, _ = scaled[0]
    expected = (x_raw.numpy() - mean) / std
    np.testing.assert_allclose(x_scaled.numpy(), expected, rtol=1e-5, atol=1e-6)


def test_dataset_scaled_train_stats_near_standard(norm_benchmark):
    """z_score 通道在 train 窗口上标准化后应显著接近 mean≈0 / std≈1（区别于原始）。"""
    scaler = json.loads(
        (norm_benchmark / "scalers" / "input_scaler.json").read_text("utf-8")
    )
    z_idx = [
        i
        for i, e in enumerate(scaler["channel_entries"])
        if e["strategy"] == "z_score"
    ]
    ds = BenchmarkDataset(norm_benchmark, split="train", window=8)  # 应用
    flat = np.stack([ds[i][0].numpy() for i in range(len(ds))]).reshape(-1, 12)
    assert np.isfinite(flat).all()
    m = flat[:, z_idx].mean(axis=0)
    s = flat[:, z_idx].std(axis=0)
    # 窗口重叠导致非精确 0/1，但应远离原始量级（如 US ~346 / T ~26）。
    assert np.all(np.abs(m) < 0.6), f"scaled means not near 0: {m}"
    assert np.all((s > 0.3) & (s < 2.5)), f"scaled stds not near 1: {s}"


def test_dataset_backward_compat_no_flag(norm_benchmark, tmp_path):
    """旧数据集（manifest 无 input_normalization）默认不标准化，返回原始量纲。"""
    dst = tmp_path / "legacy"
    shutil.copytree(norm_benchmark, dst)
    mp = dst / "manifest.json"
    manifest = json.loads(mp.read_text("utf-8"))
    manifest.pop("input_normalization", None)
    mp.write_text(json.dumps(manifest), encoding="utf-8")

    legacy = BenchmarkDataset(dst, split="train", window=8)  # 默认 None + 无标志 → 原始
    raw = BenchmarkDataset(
        norm_benchmark, split="train", window=8, apply_input_scaler=False
    )
    np.testing.assert_allclose(legacy[0][0].numpy(), raw[0][0].numpy(), rtol=1e-6)


def test_dataset_apply_true_without_flag_raises(norm_benchmark, tmp_path):
    dst = tmp_path / "legacy2"
    shutil.copytree(norm_benchmark, dst)
    mp = dst / "manifest.json"
    manifest = json.loads(mp.read_text("utf-8"))
    manifest.pop("input_normalization", None)
    mp.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="input_normalization"):
        BenchmarkDataset(dst, split="train", window=8, apply_input_scaler=True)


def test_dataset_scaler_channel_order_mismatch_raises(norm_benchmark, tmp_path):
    """input scaler 通道顺序与 Dataset 拼接顺序不一致时必须报错（防静默错标准化）。"""
    dst = tmp_path / "badorder"
    shutil.copytree(norm_benchmark, dst)
    sp = dst / "scalers" / "input_scaler.json"
    scaler = json.loads(sp.read_text("utf-8"))
    scaler["channel_names"] = list(reversed(scaler["channel_names"]))
    sp.write_text(json.dumps(scaler), encoding="utf-8")
    with pytest.raises(ValueError, match="channel_names"):
        BenchmarkDataset(dst, split="train", window=8)


# ---- A.8 补充测试 ----


def test_val_test_use_train_fitted_scaler_no_leakage(norm_benchmark):
    """A.8 §2: val/test 使用 train 拟合的 mean/std（确认无泄露）。

    scaler 仅在 train split 上拟合 → val/test 的 z_score 通道 mean 一般不为 0，
    但数值必须有限，且与 train 使用完全相同的 scaler 参数。
    """
    scaler = json.loads(
        (norm_benchmark / "scalers" / "input_scaler.json").read_text("utf-8")
    )
    z_idx = [
        i
        for i, e in enumerate(scaler["channel_entries"])
        if e["strategy"] == "z_score"
    ]
    ds_train = BenchmarkDataset(norm_benchmark, split="train", window=8)
    ds_val = BenchmarkDataset(norm_benchmark, split="val", window=8)

    # 两个 split 加载的 scaler 参数必须完全一致
    np.testing.assert_array_equal(ds_train._scale_mean, ds_val._scale_mean)
    np.testing.assert_array_equal(ds_train._scale_std, ds_val._scale_std)

    # val 窗口数值有限（idx 11 std=0 未引入 NaN/Inf）
    flat_val = np.stack([ds_val[i][0].numpy() for i in range(len(ds_val))]).reshape(
        -1, 12
    )
    assert np.isfinite(flat_val).all(), "val split 存在 NaN/Inf"

    # val z_score 通道 mean 不必为 0（train-only 拟合，非 val 拟合），但应在合理范围
    val_mean = flat_val[:, z_idx].mean(axis=0)
    assert np.all(np.abs(val_mean) < 5.0), f"val means 偏离过远: {val_mean}"


def test_idx11_zero_variance_no_nan_inf(norm_benchmark):
    """A.8 §3: idx 11（ultrasonic_tof_accepted，clean 数据 std=0）标准化后无 NaN/Inf。

    确认 passthrough 机制对零方差通道生效：transform 为恒等，输出值 == 原始值。
    """
    ds_raw = BenchmarkDataset(
        norm_benchmark, split="train", window=8, apply_input_scaler=False
    )
    ds_scaled = BenchmarkDataset(norm_benchmark, split="train", window=8)
    x_raw, _ = ds_raw[0]
    x_scaled, _ = ds_scaled[0]
    col_raw = x_raw[:, 11].numpy()
    col_scaled = x_scaled[:, 11].numpy()
    assert np.isfinite(col_scaled).all(), "idx 11 标准化后出现 NaN/Inf"
    np.testing.assert_array_equal(col_scaled, col_raw)
