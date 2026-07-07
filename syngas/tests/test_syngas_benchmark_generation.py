"""合成气 benchmark 端到端生成测试。

验证：
- syngas slow.py 生成 9 通道 slow 数组
- syngas benchmark 生成完整数据集（manifest / labels / splits / scalers）
- labels 仅含 4 列 (x_H2, x_CH4, x_CO2, x_CO)
- condition_grid 含 x_N2 但 labels 不含
- manifest.composition_scheme == "syngas"
- HITRAN 后端 raise NotImplementedError（Stage 3c 留位）
- 默认 schema 出口不再保留 hg 目标字段
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from sg.sim.core.syngas_schema import (
    BACKGROUND_FIELDS,
    COMPONENT_FIELDS,
    SLOW_CHANNELS,
)
from sg.sim.generation.syngas import (
    SyngasBenchmarkGenerationSpec,
    generate_syngas_benchmark_dataset,
)


@pytest.fixture
def small_spec(tmp_path) -> SyngasBenchmarkGenerationSpec:
    return SyngasBenchmarkGenerationSpec(
        dataset_slug="sg4-test",
        sequence_count=8,
        seed=20260626,
        timesteps=16,
        dt_s=0.5,
        storage="memmap",
        optical_absorption_backend="empirical_v1",
        workers=1,
    )


# ---------------------------------------------------------------------------
# slow.py 单元测试
# ---------------------------------------------------------------------------


def test_syngas_slow_build_arrays_basic():
    """build_sequence_arrays 直接调用，验证 9 通道 slow 输出。"""
    from sg.sim.generation.syngas.conditions import generate_syngas_condition_rows
    from sg.sim.generation.syngas.slow import build_sequence_arrays
    from sg.sim.generation.waveforms import FiberMicSpec, WaveformSpec

    conditions = generate_syngas_condition_rows(4, seed=42)
    arrays = build_sequence_arrays(
        conditions,
        timesteps=16,
        dt_s=0.5,
        seed=42,
        multi_path_phase="steady",
        ultrasonic_spec=WaveformSpec(),
        fiber_mic_spec=FiberMicSpec(),
        path_lms=(0.20, 0.25, 0.30),
        phase_schedule="standard_exposure",
        stage_jitter=0.0,
        optical_absorption_backend="empirical_v1",
    )
    slow = arrays["slow"]
    # (序列数, 时间步, 通道数) = (4, 16, 9)
    assert slow.shape == (4, 16, 9)
    assert len(arrays["slow_rows"]) == 4 * 16


def test_syngas_slow_hitran_backend_requires_cache(tmp_path):
    """HITRAN 后端在缺少 spectra cache 时应抛 MissingHitranCacheError（cache_only_prechecked 策略）。"""
    from sg.sim.generation.spectral import MissingHitranCacheError
    from sg.sim.generation.syngas.conditions import generate_syngas_condition_rows
    from sg.sim.generation.syngas.slow import build_sequence_arrays
    from sg.sim.generation.waveforms import FiberMicSpec, WaveformSpec

    conditions = generate_syngas_condition_rows(2, seed=42)
    with pytest.raises(MissingHitranCacheError):
        build_sequence_arrays(
            conditions,
            timesteps=8,
            dt_s=0.5,
            seed=42,
            multi_path_phase="steady",
            ultrasonic_spec=WaveformSpec(),
            fiber_mic_spec=FiberMicSpec(),
            path_lms=(0.20,),
            phase_schedule="standard_exposure",
            stage_jitter=0.0,
            optical_absorption_backend="hitran_hapi_v1",
            hitran_cache_root=str(tmp_path / "empty-cache"),
        )


# ---------------------------------------------------------------------------
# Benchmark 端到端生成
# ---------------------------------------------------------------------------


def test_syngas_benchmark_generates_directory(tmp_path, small_spec):
    result = generate_syngas_benchmark_dataset(tmp_path, small_spec)
    output_dir = Path(result["output_dir"])
    assert output_dir.exists()
    assert result["composition_scheme"] == "syngas"
    assert result["sequence_count"] == 8


def test_syngas_benchmark_manifest_contents(tmp_path, small_spec):
    generate_syngas_benchmark_dataset(tmp_path, small_spec)
    manifest_path = tmp_path / small_spec.dataset_slug / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["composition_scheme"] == "syngas"
    assert manifest["schema_version"] == "v4-syngas-1"
    assert manifest["labels"] == list(COMPONENT_FIELDS)
    assert manifest["background_fields"] == list(BACKGROUND_FIELDS)
    assert manifest["slow_channels"] == list(SLOW_CHANNELS)
    assert "V_NDIR_CO" in manifest["slow_channels"]
    # sim_revision.l_m_range 应从 spec.path_lms 派生，而非硬编码（防止与 path_lms 矛盾）
    # small_spec 用默认 path_lms=(0.18,0.20,0.22,0.25,0.28)，min/max 为 0.18/0.28
    assert manifest["sim_revision"]["l_m_range"] == [0.18, 0.28]


def test_syngas_benchmark_labels_npy_shape(tmp_path, small_spec):
    generate_syngas_benchmark_dataset(tmp_path, small_spec)
    labels = np.load(tmp_path / small_spec.dataset_slug / "labels" / "y.npy")
    assert labels.shape == (8, 4)  # 4 列预测目标


def test_syngas_benchmark_label_names(tmp_path, small_spec):
    generate_syngas_benchmark_dataset(tmp_path, small_spec)
    label_names = np.load(
        tmp_path / small_spec.dataset_slug / "metadata" / "label_names.npy",
        allow_pickle=True,
    )
    assert list(label_names.astype(str)) == ["x_H2", "x_CH4", "x_CO2", "x_CO"]
    # x_N2 不在 labels 名字里
    assert "x_N2" not in label_names.astype(str)


def test_syngas_benchmark_slow_npy_has_9_channels(tmp_path, small_spec):
    generate_syngas_benchmark_dataset(tmp_path, small_spec)
    slow = np.load(
        tmp_path / small_spec.dataset_slug / "sequences" / "slow.npy", mmap_mode="r"
    )
    # (sequences, timesteps, channels) = (8, 16, 9)
    assert slow.shape == (8, 16, 9)


def test_syngas_benchmark_condition_grid_contains_x_n2(tmp_path, small_spec):
    """condition grid 应含 x_N2（作为物理仿真输入），即使 labels 不含。"""
    generate_syngas_benchmark_dataset(tmp_path, small_spec)
    cond_path = tmp_path / small_spec.dataset_slug / "condition_grid_sequence.csv"
    header = cond_path.read_text(encoding="utf-8").splitlines()[0]
    fields = header.split(",")
    assert "x_CO" in fields
    assert "x_N2" in fields  # 背景气
    # 验证字段顺序（COMPONENT_FIELDS 在前，BACKGROUND_FIELDS 紧随）
    co_idx = fields.index("x_CO")
    n2_idx = fields.index("x_N2")
    assert n2_idx > co_idx


def test_syngas_benchmark_sequence_labels_only_targets(tmp_path, small_spec):
    """sequence_labels.csv 仅含 sequence_id + 4 列目标。"""
    generate_syngas_benchmark_dataset(tmp_path, small_spec)
    labels_csv = tmp_path / small_spec.dataset_slug / "sequence_labels.csv"
    header = labels_csv.read_text(encoding="utf-8").splitlines()[0]
    fields = header.split(",")
    assert fields == ["sequence_id", "x_H2", "x_CH4", "x_CO2", "x_CO"]
    assert "x_N2" not in fields


def test_syngas_benchmark_splits_exist(tmp_path, small_spec):
    generate_syngas_benchmark_dataset(tmp_path, small_spec)
    splits_dir = tmp_path / small_spec.dataset_slug / "splits"
    for split_name in ("train", "val", "test", "extrapolation"):
        assert (splits_dir / f"{split_name}.csv").exists()


def test_syngas_benchmark_scalers_built_on_9_channels(tmp_path, small_spec):
    generate_syngas_benchmark_dataset(tmp_path, small_spec)
    scaler_path = tmp_path / small_spec.dataset_slug / "scalers" / "scaler_slow_sequence.json"
    scaler = json.loads(scaler_path.read_text(encoding="utf-8"))
    # scaler 应该覆盖全部 9 个慢通道
    assert "channels" in scaler or "mean" in scaler
    # 不深究 schema 细节，只验证 V_NDIR_CO 出现在某处
    assert "V_NDIR_CO" in scaler_path.read_text(encoding="utf-8")


def test_syngas_benchmark_validation_summary(tmp_path, small_spec):
    generate_syngas_benchmark_dataset(tmp_path, small_spec)
    summary_path = tmp_path / small_spec.dataset_slug / "quality" / "validation_summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert summary["status"] == "pass"
    assert summary["sequence_count"] == 8


def test_syngas_benchmark_hitran_backend_rejects_missing_cache(tmp_path):
    """HITRAN 后端在 benchmark 入口校验失败时抛 MissingHitranBenchmarkCacheError。"""
    from sg.sim.generation.optical_backend import MissingHitranBenchmarkCacheError

    spec = SyngasBenchmarkGenerationSpec(
        dataset_slug="sg4-hitran-test",
        sequence_count=4,
        seed=42,
        timesteps=8,
        optical_absorption_backend="hitran_hapi_v1",
        hitran_cache_root=str(tmp_path / "empty-cache"),
        workers=1,
    )
    with pytest.raises(MissingHitranBenchmarkCacheError, match="precompute_hitran_benchmark_cache"):
        generate_syngas_benchmark_dataset(tmp_path, spec)
    # 校验失败后不应留下半成品目录
    assert not (tmp_path / "sg4-hitran-test").exists()


def test_syngas_benchmark_hitran_backend_rejects_co_crosstalk(tmp_path):
    """HITRAN 后端已通过光谱积分包含多气体串扰，不能与 empirical 3x3 crosstalk 同时启用。"""
    spec = SyngasBenchmarkGenerationSpec(
        dataset_slug="sg4-hitran-crosstalk-conflict",
        sequence_count=4,
        seed=42,
        timesteps=8,
        optical_absorption_backend="hitran_hapi_v1",
        hitran_cache_root=str(tmp_path / "cache"),
        enable_co_crosstalk=True,
        workers=1,
    )
    with pytest.raises(ValueError, match="enable_co_crosstalk"):
        generate_syngas_benchmark_dataset(tmp_path, spec)


# ---------------------------------------------------------------------------
# 隔离性：默认 schema 不再携带 hg 目标字段
# ---------------------------------------------------------------------------


def test_default_schema_points_to_syngas_fields():
    from sg.sim.core.schema import BACKGROUND_FIELDS as DEFAULT_BACKGROUND_FIELDS
    from sg.sim.core.schema import COMPONENT_FIELDS as DEFAULT_COMPONENT_FIELDS
    from sg.sim.core.schema import SCHEMA_VERSION as DEFAULT_SCHEMA_VERSION

    assert DEFAULT_SCHEMA_VERSION == "v4-syngas-1"
    assert DEFAULT_COMPONENT_FIELDS == COMPONENT_FIELDS
    assert DEFAULT_BACKGROUND_FIELDS == BACKGROUND_FIELDS
