"""测试 RCDW benchmark 端到端流程（合成 HITRAN cache，不联网）。

对应方案 §5.9 / §11.1。
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from rcdw.sim.core.schema import SCHEMA_VERSION, SLOW_CHANNELS
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


def _write_synthetic_cache_for_conditions(
    cache_root: Path, conditions: list[dict[str, str]]
) -> None:
    """为所有 condition * (CO2, H2O) 组合写入合成 HITRAN cache。"""
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


def _smoke_spec(tmp_path: Path, dataset_slug: str = "rcdw-test") -> tuple[BenchmarkGenerationSpec, Path]:
    cache_root = tmp_path / "hitran_cache"
    conditions = generate_condition_rows(8, seed=42)
    _write_synthetic_cache_for_conditions(cache_root, conditions)
    spec = BenchmarkGenerationSpec(
        dataset_slug=dataset_slug,
        sequence_count=8,
        seed=42,
        timesteps=16,
        dt_s=0.5,
        storage="memmap",
        multi_path_phase="steady",
        stage_profile="standard_exposure",
        stage_jitter=0.0,
        path_lms=(0.2, 0.3, 0.4),
        optical_absorption_backend="hitran_hapi_v1",
        hitran_cache_root=str(cache_root),
    )
    return spec, cache_root


def test_generate_benchmark_smoke_creates_expected_layout(tmp_path):
    spec, _ = _smoke_spec(tmp_path)
    output_root = tmp_path / "output"
    result = generate_benchmark_dataset(output_root, spec)

    output_dir = Path(result["output_dir"])
    assert output_dir.is_dir()
    assert (output_dir / "manifest.json").is_file()
    assert (output_dir / "condition_grid_sequence.csv").is_file()
    assert (output_dir / "sequence_index.csv").is_file()
    assert (output_dir / "sequence_labels.csv").is_file()
    assert (output_dir / "sequences" / "slow.npy").is_file()
    assert (output_dir / "sequences" / "ultrasonic_int16.npy").is_file()
    assert (output_dir / "sequences" / "fiber_mic_int16.npy").is_file()
    assert (output_dir / "sequences" / "slow_sequence_long.csv").is_file()
    assert (output_dir / "labels" / "y.npy").is_file()
    assert (output_dir / "splits" / "train.csv").is_file()
    assert (output_dir / "splits" / "val.csv").is_file()
    assert (output_dir / "splits" / "test.csv").is_file()
    # 无 extrapolation split (方案 §2.6)
    assert not (output_dir / "splits" / "extrapolation.csv").is_file()
    assert (output_dir / "splits" / "split_summary.json").is_file()
    assert (output_dir / "scalers" / "scaler_slow_sequence.json").is_file()
    assert (output_dir / "scalers" / "scaler_slow_sequence_modal.json").is_file()
    assert (output_dir / "metadata" / "waveform_spec.json").is_file()
    assert (output_dir / "quality" / "validation_summary.json").is_file()


def test_smoke_manifest_fields(tmp_path):
    spec, _ = _smoke_spec(tmp_path)
    result = generate_benchmark_dataset(tmp_path / "output", spec)
    manifest = json.loads(
        (Path(result["output_dir"]) / "manifest.json").read_text(encoding="utf-8")
    )
    # 方案 §6.2 关键字段
    assert manifest["schema_version"] == "rcdw-benchmark-1"
    assert manifest["schema_version"] == SCHEMA_VERSION
    assert manifest["composition_scheme"] == "rcdw_o2_co2_n2"
    assert manifest["dataset_slug"] == "rcdw-test"
    assert manifest["sequence_count"] == 8
    assert manifest["primary_key"] == "mixture_id"
    assert manifest["instance_key"] == "sequence_id"
    assert manifest["slow_channels"] == list(SLOW_CHANNELS)
    assert manifest["labels"] == ["x_O2", "x_CO2", "x_N2"]
    assert manifest["train_modalities"] == ["slow", "ultrasonic"]
    assert manifest["background_fields"] == []
    # 声学模型版本
    assert manifest["model"] == "linear_mixing_v1"  # acoustic_model_metadata 注入
    # HITRAN 后端
    assert manifest["optical_absorption_backend"] == "hitran_hapi_v1"
    # scaler_metadata
    assert manifest["scaler_metadata"]["peak_index_strategy"] == "skip"


def test_smoke_validation_summary_passes(tmp_path):
    spec, _ = _smoke_spec(tmp_path)
    result = generate_benchmark_dataset(tmp_path / "output", spec)
    summary = json.loads(
        (Path(result["output_dir"]) / "quality" / "validation_summary.json").read_text(
            encoding="utf-8"
        )
    )
    assert summary["status"] == "pass"
    assert summary["sequence_count"] == 8
    # SPLIT_NAMES 仅 train/val/test (无 extrapolation)
    assert set(summary["split_counts"].keys()) == {"train", "val", "test"}


def test_smoke_scaler_json_passthrough_marked(tmp_path):
    """方案 v1.2 §6.5: scaler 输出必须显式标记 passthrough 策略。"""
    spec, _ = _smoke_spec(tmp_path)
    result = generate_benchmark_dataset(tmp_path / "output", spec)
    scaler = json.loads(
        (Path(result["output_dir"]) / "scalers" / "scaler_slow_sequence.json").read_text(
            encoding="utf-8"
        )
    )
    # RCDW slow 矩阵不含 ultrasonic_* 通道(它们仅在 Phase 4 拼接张量中出现),
    # 但 skip_channels 字段应保留默认 passthrough 通道清单。
    assert "skip_channels" in scaler
    assert set(scaler["skip_channels"]) == {
        "ultrasonic_peak_index",
        "ultrasonic_tof_quality",
        "ultrasonic_tof_accepted",
    }


def test_smoke_split_csv_no_overlap(tmp_path):
    """切分覆盖完整,各 split 不重叠。"""
    spec, _ = _smoke_spec(tmp_path)
    result = generate_benchmark_dataset(tmp_path / "output", spec)
    output_dir = Path(result["output_dir"])
    train = (output_dir / "splits" / "train.csv").read_text(encoding="utf-8").strip().split("\n")[1:]
    val = (output_dir / "splits" / "val.csv").read_text(encoding="utf-8").strip().split("\n")[1:]
    test = (output_dir / "splits" / "test.csv").read_text(encoding="utf-8").strip().split("\n")[1:]
    train_ids = {row.split(",")[0] for row in train if row}
    val_ids = {row.split(",")[0] for row in val if row}
    test_ids = {row.split(",")[0] for row in test if row}
    # 互不重叠
    assert not (train_ids & val_ids)
    assert not (train_ids & test_ids)
    assert not (val_ids & test_ids)
    # 总数 = 8
    assert len(train_ids) + len(val_ids) + len(test_ids) == 8


def test_smoke_slow_npy_loadable(tmp_path):
    spec, _ = _smoke_spec(tmp_path)
    result = generate_benchmark_dataset(tmp_path / "output", spec)
    slow = np.load(Path(result["output_dir"]) / "sequences" / "slow.npy")
    assert slow.shape == (8, 16, len(SLOW_CHANNELS))
    assert slow.dtype == np.float32


def test_smoke_labels_y_npy_shape(tmp_path):
    spec, _ = _smoke_spec(tmp_path)
    result = generate_benchmark_dataset(tmp_path / "output", spec)
    labels = np.load(Path(result["output_dir"]) / "labels" / "y.npy")
    assert labels.shape == (8, 3)
    # 组分和应为 100
    sums = labels.sum(axis=1)
    np.testing.assert_allclose(sums, 100.0, atol=1e-4)


def test_smoke_fiber_mic_present_but_not_in_scaler(tmp_path):
    """方案 §2.4: fiber_mic 落盘但不进 scaler。"""
    spec, _ = _smoke_spec(tmp_path)
    result = generate_benchmark_dataset(tmp_path / "output", spec)
    output_dir = Path(result["output_dir"])
    # fiber_mic 文件存在
    assert (output_dir / "sequences" / "fiber_mic_int16.npy").is_file()
    # scaler 字段中不应出现 fiber_mic
    scaler = json.loads(
        (output_dir / "scalers" / "scaler_slow_sequence.json").read_text(encoding="utf-8")
    )
    assert "fiber_mic" not in str(scaler)


def test_smoke_invalid_stage_profile_rejected(tmp_path):
    """v1.2 YAGNI: 未实现 stage_profile 应被 _validate_spec 拒绝。"""
    spec, _ = _smoke_spec(tmp_path)
    bad_spec = BenchmarkGenerationSpec(
        dataset_slug="bad",
        sequence_count=4,
        seed=42,
        stage_profile="variable_onset",  # 未注册
    )
    with pytest.raises(ValueError, match="stage_profile"):
        generate_benchmark_dataset(tmp_path / "output", bad_spec)


def test_smoke_reproducibility(tmp_path):
    """同 seed 应产生相同 slow.npy (memmap 落盘)。"""
    spec_a, _ = _smoke_spec(tmp_path, dataset_slug="rcdw-a")
    spec_b, _ = _smoke_spec(tmp_path, dataset_slug="rcdw-b")
    res_a = generate_benchmark_dataset(tmp_path / "output", spec_a)
    res_b = generate_benchmark_dataset(tmp_path / "output", spec_b)
    slow_a = np.load(Path(res_a["output_dir"]) / "sequences" / "slow.npy")
    slow_b = np.load(Path(res_b["output_dir"]) / "sequences" / "slow.npy")
    np.testing.assert_array_equal(slow_a, slow_b)
