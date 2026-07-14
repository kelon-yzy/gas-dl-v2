"""掘进通风 benchmark 端到端生成测试。

验证：
- tv3 slow.py 生成 7 通道 slow 数组
- tv3 benchmark 生成完整数据集（manifest / labels / splits / scalers）
- labels 含 3 列 (x_CO2, x_O2, x_N2)，N2 在 labels 中
- condition_grid 含 3 列预测目标
- manifest.composition_scheme == "tunnel_ventilation"
- manifest.background_fields == []
- slow_channels 不含 V_NDIR_CH4
- HITRAN 后端被拒绝（阶段 1 未实现）
- 组分总量严格闭包 sum=100%
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from tv3.sim.core.tunnel_ventilation_schema import (
    BACKGROUND_FIELDS,
    COMPONENT_FIELDS,
    SLOW_CHANNELS,
)
from tv3.sim.generation.tunnel_ventilation import (
    TunnelVentilationBenchmarkGenerationSpec,
    generate_tunnel_ventilation_benchmark_dataset,
)


@pytest.fixture
def small_spec(tmp_path) -> TunnelVentilationBenchmarkGenerationSpec:
    return TunnelVentilationBenchmarkGenerationSpec(
        dataset_slug="tv3-test",
        sequence_count=8,
        seed=20260704,
        timesteps=16,
        dt_s=0.5,
        storage="memmap",
        optical_absorption_backend="empirical_v1",
        workers=1,
    )


# ---------------------------------------------------------------------------
# slow.py 单元测试
# ---------------------------------------------------------------------------


def test_tv3_slow_build_arrays_basic():
    """build_sequence_arrays 直接调用，验证 7 通道 slow 输出。"""
    from tv3.sim.generation.tunnel_ventilation.conditions import (
        generate_tunnel_ventilation_condition_rows,
    )
    from tv3.sim.generation.tunnel_ventilation.slow import build_sequence_arrays
    from tv3.sim.generation.waveforms import FiberMicSpec, WaveformSpec

    conditions = generate_tunnel_ventilation_condition_rows(4, seed=42)
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
    # (序列数, 时间步, 通道数) = (4, 16, 7)
    assert slow.shape == (4, 16, 7)
    assert len(arrays["slow_rows"]) == 4 * 16


def test_tv3_slow_rejects_hitran_backend(tmp_path):
    """HITRAN 后端在阶段 1 应被拒绝。"""
    from tv3.sim.generation.tunnel_ventilation.conditions import (
        generate_tunnel_ventilation_condition_rows,
    )
    from tv3.sim.generation.tunnel_ventilation.slow import build_sequence_arrays
    from tv3.sim.generation.waveforms import FiberMicSpec, WaveformSpec

    conditions = generate_tunnel_ventilation_condition_rows(2, seed=42)
    with pytest.raises(ValueError, match="empirical_v1"):
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


def test_tv3_benchmark_generates_directory(tmp_path, small_spec):
    result = generate_tunnel_ventilation_benchmark_dataset(tmp_path, small_spec)
    output_dir = Path(result["output_dir"])
    assert output_dir.exists()
    assert result["composition_scheme"] == "tunnel_ventilation"
    assert result["sequence_count"] == 8


def test_tv3_benchmark_manifest_contents(tmp_path, small_spec):
    generate_tunnel_ventilation_benchmark_dataset(tmp_path, small_spec)
    manifest_path = tmp_path / small_spec.dataset_slug / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["composition_scheme"] == "tunnel_ventilation"
    assert manifest["schema_version"] == "tunnel-ventilation-1"
    assert manifest["labels"] == list(COMPONENT_FIELDS)
    assert manifest["background_fields"] == list(BACKGROUND_FIELDS)
    assert manifest["background_fields"] == []
    assert manifest["slow_channels"] == list(SLOW_CHANNELS)
    assert "V_NDIR_CH4" not in manifest["slow_channels"]
    assert "V_NDIR_CO" not in manifest["slow_channels"]
    # sim_revision.l_m_range 应从 spec.path_lms 派生
    # small_spec 用默认 path_lms=(0.18,0.20,0.22,0.25,0.28)，min/max 为 0.18/0.28
    assert manifest["sim_revision"]["l_m_range"] == [0.18, 0.28]
    # l_m_base_range 是每条序列基准光程采样范围（conditions.py L_M_BASE_RANGE 常量）
    assert manifest["sim_revision"]["l_m_base_range"] == [0.2, 0.3]


def test_tv3_benchmark_labels_npy_shape(tmp_path, small_spec):
    generate_tunnel_ventilation_benchmark_dataset(tmp_path, small_spec)
    labels = np.load(tmp_path / small_spec.dataset_slug / "labels" / "y.npy")
    assert labels.shape == (8, 3)  # 3 列预测目标


def test_tv3_benchmark_label_names(tmp_path, small_spec):
    generate_tunnel_ventilation_benchmark_dataset(tmp_path, small_spec)
    label_names = np.load(
        tmp_path / small_spec.dataset_slug / "metadata" / "label_names.npy",
        allow_pickle=True,
    )
    assert list(label_names.astype(str)) == ["x_CO2", "x_O2", "x_N2"]


def test_tv3_benchmark_slow_npy_has_7_channels(tmp_path, small_spec):
    generate_tunnel_ventilation_benchmark_dataset(tmp_path, small_spec)
    slow = np.load(
        tmp_path / small_spec.dataset_slug / "sequences" / "slow.npy", mmap_mode="r"
    )
    # (sequences, timesteps, channels) = (8, 16, 7)
    assert slow.shape == (8, 16, 7)


def test_tv3_benchmark_condition_grid_contains_all_targets(tmp_path, small_spec):
    """condition grid 应含 3 列预测目标（含 x_N2）。"""
    generate_tunnel_ventilation_benchmark_dataset(tmp_path, small_spec)
    cond_path = tmp_path / small_spec.dataset_slug / "condition_grid_sequence.csv"
    header = cond_path.read_text(encoding="utf-8").splitlines()[0]
    fields = header.split(",")
    assert "x_CO2" in fields
    assert "x_O2" in fields
    assert "x_N2" in fields  # N2 在本场景是预测目标，写入 condition grid
    # 不应含 hg/syngas 字段
    assert "x_H2" not in fields
    assert "x_CH4" not in fields
    assert "x_CO" not in fields


def test_tv3_benchmark_sequence_labels_includes_n2(tmp_path, small_spec):
    """sequence_labels.csv 含 sequence_id + 3 列目标（含 x_N2）。"""
    generate_tunnel_ventilation_benchmark_dataset(tmp_path, small_spec)
    labels_csv = tmp_path / small_spec.dataset_slug / "sequence_labels.csv"
    header = labels_csv.read_text(encoding="utf-8").splitlines()[0]
    fields = header.split(",")
    assert fields == ["sequence_id", "x_CO2", "x_O2", "x_N2"]


def test_tv3_benchmark_splits_exist(tmp_path, small_spec):
    generate_tunnel_ventilation_benchmark_dataset(tmp_path, small_spec)
    splits_dir = tmp_path / small_spec.dataset_slug / "splits"
    for split_name in ("train", "val", "test", "extrapolation"):
        assert (splits_dir / f"{split_name}.csv").exists()


def test_tv3_benchmark_scalers_built_on_7_channels(tmp_path, small_spec):
    generate_tunnel_ventilation_benchmark_dataset(tmp_path, small_spec)
    scaler_path = tmp_path / small_spec.dataset_slug / "scalers" / "scaler_slow_sequence.json"
    scaler = json.loads(scaler_path.read_text(encoding="utf-8"))
    # tv3 7 通道，应含 V_NDIR_CO2 但不含 V_NDIR_CH4 / V_NDIR_CO
    channels = scaler.get("channels", [])
    if channels:
        assert "V_NDIR_CO2" in channels
        assert "V_NDIR_CH4" not in channels
        assert "V_NDIR_CO" not in channels
    else:
        scaler_text = scaler_path.read_text(encoding="utf-8")
        assert '"V_NDIR_CO2"' in scaler_text
        assert '"V_NDIR_CH4"' not in scaler_text
        assert '"V_NDIR_CO"' not in scaler_text


def test_tv3_benchmark_validation_summary(tmp_path, small_spec):
    generate_tunnel_ventilation_benchmark_dataset(tmp_path, small_spec)
    summary_path = tmp_path / small_spec.dataset_slug / "quality" / "validation_summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert summary["status"] == "pass"
    assert summary["sequence_count"] == 8


def test_tv3_benchmark_labels_sum_to_100(tmp_path, small_spec):
    """3 列预测目标严格闭包：x_CO2 + x_O2 + x_N2 = 100%。"""
    generate_tunnel_ventilation_benchmark_dataset(tmp_path, small_spec)
    labels = np.load(tmp_path / small_spec.dataset_slug / "labels" / "y.npy")
    total = labels.sum(axis=1)
    assert np.allclose(total, 100.0, atol=1e-4), f"labels sum not 100: {total}"


def test_tv3_benchmark_hitran_backend_rejected(tmp_path):
    """HITRAN 后端在阶段 1 应被拒绝。"""
    spec = TunnelVentilationBenchmarkGenerationSpec(
        dataset_slug="tv3-hitran-test",
        sequence_count=4,
        seed=42,
        timesteps=8,
        optical_absorption_backend="hitran_hapi_v1",
        hitran_cache_root=str(tmp_path / "empty-cache"),
        workers=1,
    )
    with pytest.raises(ValueError, match="empirical_v1"):
        generate_tunnel_ventilation_benchmark_dataset(tmp_path, spec)
    # 校验失败后不应留下半成品目录
    assert not (tmp_path / "tv3-hitran-test").exists()


def test_tv3_spec_no_co_crosstalk_field():
    """tv3 spec 不应含 enable_co_crosstalk（无 CO 串扰概念）。"""
    import inspect

    sig = inspect.signature(TunnelVentilationBenchmarkGenerationSpec.__init__)
    assert "enable_co_crosstalk" not in sig.parameters


def test_generic_tv3_default_hitran_cache_root_uses_workspace_shared_cache():
    from tv3.sim.generation.benchmark import DEFAULT_HITRAN_CACHE_ROOT

    workspace_root = Path(__file__).resolve().parents[2]

    assert Path(DEFAULT_HITRAN_CACHE_ROOT) == workspace_root / "shared" / "hitran_cache"


def test_generation_cli_maps_formal_arguments(monkeypatch, tmp_path):
    from tv3.pipeline import generate_tunnel_ventilation_benchmark as cli

    captured = {}

    def fake_generate(output_root, spec):
        captured["output_root"] = output_root
        captured["spec"] = spec
        return {"output_dir": str(output_root / spec.dataset_slug), "sequence_count": spec.sequence_count}

    monkeypatch.setattr(cli, "generate_tunnel_ventilation_benchmark_dataset", fake_generate)

    assert cli.main([
        "--output-root", str(tmp_path),
        "--dataset", "tv3-formal-6000",
        "--sequences", "6000",
        "--seed", "20260704",
        "--timesteps", "512",
        "--dt-s", "0.5",
        "--storage", "memmap",
        "--workers", "4",
        "--skip-fiber-mic",
    ]) == 0

    spec = captured["spec"]
    assert captured["output_root"] == tmp_path
    assert spec.dataset_slug == "tv3-formal-6000"
    assert spec.sequence_count == 6000
    assert spec.seed == 20260704
    assert spec.timesteps == 512
    assert spec.dt_s == 0.5
    assert spec.storage == "memmap"
    assert spec.workers == 4
    assert spec.skip_fiber_mic is True
    assert spec.optical_absorption_backend == "empirical_v1"
    assert spec.split_strategy == "random"
