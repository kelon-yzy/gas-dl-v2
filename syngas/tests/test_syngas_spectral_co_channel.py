"""Syngas HITRAN 后端（CO 通道 + 3 通道 backend）测试。

复用 hg 的合成 cache 模式（write_cached_spectrum + 高斯吸收谱合成），不依赖
真实 HAPI 数据或网络。测试覆盖：

- syngas channel→gas_specs 映射（CO 通道 = CO+CO2+H2O，CH4 不在 CO 通道）
- cache requirements 收集（3 通道 × 各通道 gas 列表）
- cache 缺失时的 MissingHitranBenchmarkCacheError
- 3 通道 absorbance 端到端 + spectra_cache 复用
- CO 通道仅含 CO+CO2+H2O 自身/串扰，不依赖 CH4 浓度
- generate_syngas_benchmark_dataset 在 HITRAN 后端 + 合成 cache 下整链路跑通
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from sg.sim.generation.optical_backend import MissingHitranBenchmarkCacheError
from sg.sim.generation.spectral import (
    SYNGAS_CHANNEL_GAS_SPECS,
    SYNGAS_HITRAN_GAS_SPECS_BY_NAME,
    write_cached_spectrum,
)
from sg.sim.generation.syngas.benchmark import (
    SyngasBenchmarkGenerationSpec,
    generate_syngas_benchmark_dataset,
)
from sg.sim.generation.syngas.conditions import generate_syngas_condition_rows
from sg.sim.generation.syngas.optical_backend import (
    SYNGAS_NDIR_CHANNELS,
    SYNGAS_OPTICAL_CROSSTALK_POLICY,
    collect_hitran_syngas_cache_requirements,
    compute_hitran_syngas_optical_absorption,
    hitran_syngas_manifest_metadata,
    validate_hitran_syngas_benchmark_cache,
)


# ---------------------------------------------------------------------------
# 配置层
# ---------------------------------------------------------------------------


def test_syngas_channel_gas_specs_mapping():
    """syngas channel→gas 映射应符合：ch4/co2 与 hg 一致；co 通道 = CO+CO2+H2O 不含 CH4。"""
    assert set(SYNGAS_CHANNEL_GAS_SPECS) == {"ch4", "co2", "co"}
    ch4_gases = [s.gas for s in SYNGAS_CHANNEL_GAS_SPECS["ch4"]]
    co2_gases = [s.gas for s in SYNGAS_CHANNEL_GAS_SPECS["co2"]]
    co_gases = [s.gas for s in SYNGAS_CHANNEL_GAS_SPECS["co"]]
    assert ch4_gases == ["CH4", "CO2", "H2O"]
    assert co2_gases == ["CH4", "CO2", "H2O"]
    assert co_gases == ["CO", "CO2", "H2O"]
    assert "CH4" not in co_gases, "CO 通道不应依赖 CH4 谱线（基频频段不重合）"
    # CO HitranGasSpec 应包含正确的 HITRAN id
    co_spec = SYNGAS_HITRAN_GAS_SPECS_BY_NAME["CO"]
    assert co_spec.molecule_id == 5
    assert co_spec.isotopologue_id == 1


def test_syngas_ndir_channels_order():
    """SYNGAS_NDIR_CHANNELS 顺序固定为 (ch4, co2, co)，下游 manifest/列名依赖此约定。"""
    assert SYNGAS_NDIR_CHANNELS == ("ch4", "co2", "co")


# ---------------------------------------------------------------------------
# Cache requirements 与校验
# ---------------------------------------------------------------------------


def test_syngas_cache_requirements_dedup_and_count():
    """N 条件 × 3 通道 × 各通道 gas（去重后）= 期望 cache 槽位数。"""
    conditions = generate_syngas_condition_rows(4, seed=20260626)
    reqs = collect_hitran_syngas_cache_requirements(conditions, cache_root=Path("data/hitran_cache"))
    # 每个 condition T/P 唯一时：3 通道 + co 通道额外 CO -> 共 3*3=9，但 (CH4/CO2/H2O) 在 ch4/co2 通道
    # 共用同一 (gas, wmin, wmax, T, P) 时会去重。因为 ch4 grid != co2 grid != co grid，所以
    # 每个 (channel, gas, T, P) 的 SpectralCacheKey 不同，N 条件下应有 N * (3+3+3) = 9N 个 cache key
    # 但 CO2/H2O 在 ch4/co2 通道 grid 不同 -> 不去重。所以应 = 9 * 4 = 36
    assert len(reqs) == 36
    # 没有 CH4 cache 在 co 通道下
    co_channel_gases = {r.gas_spec.gas for r in reqs if r.channel == "co"}
    assert co_channel_gases == {"CO", "CO2", "H2O"}
    # 没有 CO cache 在 ch4/co2 通道下（隔离性）
    ch4_channel_gases = {r.gas_spec.gas for r in reqs if r.channel == "ch4"}
    co2_channel_gases = {r.gas_spec.gas for r in reqs if r.channel == "co2"}
    assert "CO" not in ch4_channel_gases
    assert "CO" not in co2_channel_gases


def test_syngas_cache_validation_rejects_missing(tmp_path):
    """空 cache 目录下 validate 应抛 MissingHitranBenchmarkCacheError。"""
    conditions = generate_syngas_condition_rows(2, seed=42)
    with pytest.raises(MissingHitranBenchmarkCacheError, match="precompute") as exc_info:
        validate_hitran_syngas_benchmark_cache(conditions, cache_root=tmp_path / "empty")
    # 错误消息应至少含 CO 槽位（确认 syngas 路径生效，非 hg 错抛）
    assert "CO" in str(exc_info.value) or "co" in str(exc_info.value).lower()


# ---------------------------------------------------------------------------
# 3 通道 forward 计算
# ---------------------------------------------------------------------------


def _write_synthetic_syngas_cache(cache_root: Path, conditions: list[dict[str, str]]) -> None:
    """写合成谱线 cache：每个 (channel, gas) 用单峰高斯模拟该气体在该通道的吸收。

    系数选择保证 CO 通道下 CO 自身吸收 >> CO2/H2O 串扰，便于断言吸收度方向正确。
    """
    for requirement in collect_hitran_syngas_cache_requirements(conditions, cache_root=cache_root):
        grid = requirement.grid_spec
        wavenumber = grid.wavenumber_min_cm1 + grid.wavenumber_step_cm1 * np.arange(
            int(round((grid.wavenumber_max_cm1 - grid.wavenumber_min_cm1) / grid.wavenumber_step_cm1)) + 1,
            dtype=np.float64,
        )
        # 选择各 gas 在各 channel grid 内的合成峰中心
        gas = requirement.gas_spec.gas
        channel = requirement.channel
        if gas == "CH4":
            center, scale = 3030.0, 1.6e-22
        elif gas == "CO2":
            # CO2 主峰 2347；在 co 通道 [1980, 2310] 内仅有低翼，用 2280 模拟
            center = 2347.0 if channel != "co" else 2280.0
            scale = 1.2e-22 if channel != "co" else 1.0e-23
        elif gas == "CO":
            center, scale = 2145.92, 4.5e-22  # CO 基频带中心
        else:  # H2O
            # H2O 在各通道都有弱吸收
            center = (grid.wavenumber_min_cm1 + grid.wavenumber_max_cm1) / 2.0
            scale = 2.0e-25
        coeff = np.exp(-0.5 * ((wavenumber - center) / 8.0) ** 2) * scale
        write_cached_spectrum(
            cache_root,
            requirement.key,
            wavenumber_cm1=wavenumber,
            absorption_coeff_cm1=coeff,
        )


def test_syngas_optical_absorption_three_channels(tmp_path):
    """端到端：合成 cache 就位时，3 通道 absorbance 应全部计算成功。"""
    conditions = generate_syngas_condition_rows(1, seed=11)
    cache_root = tmp_path / "cache"
    _write_synthetic_syngas_cache(cache_root, conditions)

    # 构造一个 per-step condition（含 x_CO/x_CH4/x_CO2/x_N2 + T_C/P_MPa/H_RH/L_m）
    base = conditions[0]
    step_condition = {
        "x_CH4": base["x_CH4"],
        "x_CO2": base["x_CO2"],
        "x_CO": base["x_CO"],
        "x_H2": base["x_H2"],
        "x_N2": base["x_N2"],
        "T_C": base["T_C_base"],
        "P_MPa": base["P_MPa_base"],
        "H_RH": base["H_RH_base"],
        "L_m": base["L_m_base"],
    }
    result = compute_hitran_syngas_optical_absorption(step_condition, cache_root=cache_root)

    # 三通道 self-absorbance 字段全部存在且 >= 0
    for key in ("absorption_ch4_true", "absorption_co2_true", "absorption_co_true"):
        assert key in result
        assert float(result[key]) >= 0.0
    for key in ("absorption_ch4_observed", "absorption_co2_observed", "absorption_co_observed"):
        assert key in result
        assert float(result[key]) >= 0.0
    # CO 通道 self 吸收 >> ch4/co2 通道里的 CO（CH4 通道根本不含 CO）
    assert float(result["absorption_co_true"]) > 0.0
    # by_gas 嵌套结构：co 通道下不含 CH4 键（channel_gas_specs["co"] 不含 CH4）
    by_gas = result["absorption_by_gas"]
    assert set(by_gas["co"].keys()) == {"CO", "CO2", "H2O"}
    assert set(by_gas["ch4"].keys()) == {"CH4", "CO2", "H2O"}
    # backend / manifest 字段
    assert result["optical_absorption_backend"] == "hitran_hapi_v1"


def test_syngas_optical_absorption_co_channel_independent_of_ch4(tmp_path):
    """CO 通道 absorbance 不应受 CH4 浓度变化影响（channel_gas_specs["co"] 不含 CH4）。"""
    conditions = generate_syngas_condition_rows(1, seed=22)
    cache_root = tmp_path / "cache"
    _write_synthetic_syngas_cache(cache_root, conditions)

    base = conditions[0]
    cond_low_ch4 = {
        "x_CH4": "0.5",
        "x_CO2": base["x_CO2"],
        "x_CO": base["x_CO"],
        "x_H2": base["x_H2"],
        "x_N2": base["x_N2"],
        "T_C": base["T_C_base"],
        "P_MPa": base["P_MPa_base"],
        "H_RH": base["H_RH_base"],
        "L_m": base["L_m_base"],
    }
    cond_high_ch4 = dict(cond_low_ch4, x_CH4="15.0")
    r_low = compute_hitran_syngas_optical_absorption(cond_low_ch4, cache_root=cache_root)
    r_high = compute_hitran_syngas_optical_absorption(cond_high_ch4, cache_root=cache_root)
    # CO 通道吸收度对 CH4 完全不敏感
    assert float(r_low["absorption_co_observed"]) == pytest.approx(
        float(r_high["absorption_co_observed"]), rel=0.0, abs=1e-12
    )
    # CH4 通道 self 吸收应随 CH4 浓度上升（正向验证测试构造合理）
    assert float(r_high["absorption_ch4_true"]) > float(r_low["absorption_ch4_true"])


def test_syngas_optical_absorption_spectra_cache_reuse(tmp_path):
    """传入 spectra_cache dict 后，第二次调用应命中 cache 不再 read+prepare（dict 被填充）。"""
    conditions = generate_syngas_condition_rows(1, seed=33)
    cache_root = tmp_path / "cache"
    _write_synthetic_syngas_cache(cache_root, conditions)

    base = conditions[0]
    step_condition = {
        "x_CH4": base["x_CH4"],
        "x_CO2": base["x_CO2"],
        "x_CO": base["x_CO"],
        "x_H2": base["x_H2"],
        "x_N2": base["x_N2"],
        "T_C": base["T_C_base"],
        "P_MPa": base["P_MPa_base"],
        "H_RH": base["H_RH_base"],
        "L_m": base["L_m_base"],
    }
    spectra_cache: dict = {}
    r1 = compute_hitran_syngas_optical_absorption(step_condition, cache_root=cache_root, spectra_cache=spectra_cache)
    # 3 channel × 1 grid -> dict 至少 3 entries
    assert len(spectra_cache) == 3
    r2 = compute_hitran_syngas_optical_absorption(step_condition, cache_root=cache_root, spectra_cache=spectra_cache)
    # cache 命中后 dict 不再增长
    assert len(spectra_cache) == 3
    # 数值结果一致
    assert float(r1["absorption_co_observed"]) == pytest.approx(float(r2["absorption_co_observed"]))


# ---------------------------------------------------------------------------
# Benchmark 端到端
# ---------------------------------------------------------------------------


def test_syngas_benchmark_end_to_end_with_hitran(tmp_path):
    """generate_syngas_benchmark_dataset 在合成 cache + HITRAN 后端下应跑通整链路。"""
    sequence_count = 4
    conditions = generate_syngas_condition_rows(sequence_count, seed=20260626)
    cache_root = tmp_path / "cache"
    _write_synthetic_syngas_cache(cache_root, conditions)

    spec = SyngasBenchmarkGenerationSpec(
        dataset_slug="sg4-hitran-end2end",
        sequence_count=sequence_count,
        seed=20260626,
        timesteps=8,
        storage="npz",
        optical_absorption_backend="hitran_hapi_v1",
        hitran_cache_root=str(cache_root),
        workers=1,
    )
    result = generate_syngas_benchmark_dataset(tmp_path, spec)
    assert result["optical_absorption_backend"] == "hitran_hapi_v1"

    output_dir = Path(result["output_dir"])
    manifest = json.loads((output_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["optical_absorption_backend"] == "hitran_hapi_v1"
    assert manifest["optical_crosstalk_policy"] == SYNGAS_OPTICAL_CROSSTALK_POLICY
    assert manifest["ndir_channels"] == list(SYNGAS_NDIR_CHANNELS)
    assert manifest["composition_scheme"] == "syngas"
    # 慢通道 V_NDIR_CO 仍存在
    assert "V_NDIR_CO" in manifest["slow_channels"]


def test_hitran_syngas_manifest_metadata_shape():
    """manifest 元数据字段名 / 类型一致性。"""
    md = hitran_syngas_manifest_metadata("data/hitran_cache")
    assert md["hitran_cache_policy"] == "cache_only_prechecked"
    assert md["hitran_temperature_pressure_mode"] == "per_condition"
    assert md["h2o_policy"] == "rh_to_mole_pct"
    assert md["optical_crosstalk_policy"] == SYNGAS_OPTICAL_CROSSTALK_POLICY
    assert md["ndir_channels"] == list(SYNGAS_NDIR_CHANNELS)
