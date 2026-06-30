"""测试 RCDW 光学吸收 / optical_crosstalk / spectral 子包。

对应方案 §5.7 / §5.8 / §11.1。

策略：
- spectral 子包内部逻辑：用合成 TabulatedSpectrum 验证（无 HITRAN 依赖）。
- HITRAN backend fetch + cache roundtrip：单独 mark slow，必要时跑（联网）。
"""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest

from rcdw.sim.generation.optical_backend import (
    EMPIRICAL_ABSORPTION_BACKEND,
    HITRAN_ABSORPTION_BACKEND,
    MissingHitranBenchmarkCacheError,
    build_hitran_grid_for_condition,
    collect_hitran_cache_requirements,
    compute_hitran_optical_absorption,
    hitran_manifest_metadata,
    validate_hitran_benchmark_cache,
)
from rcdw.sim.generation.optical_crosstalk import (
    DEFAULT_OPTICAL_CROSSTALK_SPEC,
    apply_optical_crosstalk,
)
from rcdw.sim.generation.spectral import (
    DEFAULT_HITRAN_GAS_SPECS,
    DEFAULT_HITRAN_GRID_SPECS,
    DEFAULT_NDIR_FILTERS,
    NDIRFilter,
    SpectralCacheKey,
    TabulatedSpectrum,
    cache_path,
    compute_tabulated_ndir_absorbance,
    gaussian_filter,
    get_default_hitran_grid,
    get_default_ndir_filter,
    hitran_cache_key,
    prepare_tabulated_spectra,
    read_cached_spectrum,
    write_cached_spectrum,
)


# ---- spectral defaults 范围验证 ----


def test_default_gas_specs_only_co2_and_h2o():
    """RCDW DEFAULT_HITRAN_GAS_SPECS 仅含 (CO2, H2O), 无 CH4。"""
    gases = {spec.gas for spec in DEFAULT_HITRAN_GAS_SPECS}
    assert gases == {"CO2", "H2O"}


def test_default_ndir_filters_only_co2():
    """RCDW NDIR 仅注册 co2 通道。"""
    assert set(DEFAULT_NDIR_FILTERS.keys()) == {"co2"}


def test_default_hitran_grids_only_co2():
    assert set(DEFAULT_HITRAN_GRID_SPECS.keys()) == {"co2"}


def test_get_default_ndir_filter_ch4_raises():
    """方案 §11.1: get_default_ndir_filter('ch4') 应抛 ValueError。"""
    with pytest.raises(ValueError, match="Unknown NDIR channel"):
        get_default_ndir_filter("ch4")


def test_get_default_ndir_filter_co2_returns_expected():
    f = get_default_ndir_filter("co2")
    assert isinstance(f, NDIRFilter)
    assert f.channel == "co2"
    assert math.isclose(f.center_cm1, 2347.0)
    assert math.isclose(f.fwhm_cm1, 93.0)


def test_get_default_hitran_grid_co2():
    grid = get_default_hitran_grid("co2")
    assert math.isclose(grid.wavenumber_min_cm1, 2250.0)
    assert math.isclose(grid.wavenumber_max_cm1, 2445.0)


# ---- gaussian_filter / cache key / write+read roundtrip ----


def test_gaussian_filter_normalization():
    spec = NDIRFilter(channel="co2", center_cm1=2347.0, fwhm_cm1=93.0)
    wn = np.linspace(2250.0, 2445.0, 200)
    resp = gaussian_filter(wn, spec)
    # 峰值约在中心点
    assert wn[int(np.argmax(resp))] == pytest.approx(2347.0, abs=2.0)
    # 半高宽附近响应应约为 0.5
    assert resp[int(np.argmax(resp))] == pytest.approx(1.0, abs=1e-6)


def test_cache_roundtrip(tmp_path: Path):
    key = SpectralCacheKey(
        backend="hitran_hapi_v1",
        gas="CO2",
        source_version="hitran_hapi_v1",
        wavenumber_min_cm1=2250.0,
        wavenumber_max_cm1=2445.0,
        wavenumber_step_cm1=0.1,
        temperature_k=298.15,
        pressure_atm=1.0,
    )
    wn = np.linspace(2250.0, 2445.0, 50)
    coeff = np.random.default_rng(0).random(50)
    path = write_cached_spectrum(
        tmp_path, key, wavenumber_cm1=wn, absorption_coeff_cm1=coeff
    )
    assert path.is_file()
    loaded = read_cached_spectrum(tmp_path, key)
    assert loaded is not None
    wn2, coeff2 = loaded
    np.testing.assert_allclose(wn, wn2)
    np.testing.assert_allclose(coeff, coeff2)


# ---- tabulated backend：合成谱线表，验证 CO2 / H2O 单调性 ----


def _synthetic_spectra(
    *, wn_min: float = 2250.0, wn_max: float = 2445.0, n: int = 200
) -> tuple[TabulatedSpectrum, TabulatedSpectrum, np.ndarray]:
    """合成 CO2 + H2O 谱线表（中心吸收峰均在 2347 cm⁻¹ 附近）。"""
    wn = np.linspace(wn_min, wn_max, n)
    # CO2 强主峰
    co2_coeff = 0.01 * np.exp(-((wn - 2347.0) / 40.0) ** 2)
    # H2O 弱重叠
    h2o_coeff = 0.002 * np.exp(-((wn - 2330.0) / 60.0) ** 2)
    co2 = TabulatedSpectrum(
        gas="CO2",
        wavenumber_cm1=wn,
        absorption_coeff_per_percent_m=co2_coeff,
        source_version="synthetic_v1",
    )
    h2o = TabulatedSpectrum(
        gas="H2O",
        wavenumber_cm1=wn,
        absorption_coeff_per_percent_m=h2o_coeff,
        source_version="synthetic_v1",
    )
    return co2, h2o, wn


def test_tabulated_co2_absorbance_monotonic():
    co2, h2o, _ = _synthetic_spectra()
    filter_spec = get_default_ndir_filter("co2")
    low = compute_tabulated_ndir_absorbance(
        spectra=(co2, h2o),
        concentrations_pct={"CO2": 1.0, "H2O": 0.5},
        path_length_m=0.5,
        filter_spec=filter_spec,
    )
    high = compute_tabulated_ndir_absorbance(
        spectra=(co2, h2o),
        concentrations_pct={"CO2": 15.0, "H2O": 0.5},
        path_length_m=0.5,
        filter_spec=filter_spec,
    )
    assert high["absorbance_observed"] > low["absorbance_observed"]


def test_tabulated_h2o_crosstalk_into_co2_channel():
    """H2O 浓度增加时, CO2 通道观测吸光度应增大（自然光谱重叠）。"""
    co2, h2o, _ = _synthetic_spectra()
    filter_spec = get_default_ndir_filter("co2")
    dry = compute_tabulated_ndir_absorbance(
        spectra=(co2, h2o),
        concentrations_pct={"CO2": 5.0, "H2O": 0.0},
        path_length_m=0.5,
        filter_spec=filter_spec,
    )
    wet = compute_tabulated_ndir_absorbance(
        spectra=(co2, h2o),
        concentrations_pct={"CO2": 5.0, "H2O": 3.0},
        path_length_m=0.5,
        filter_spec=filter_spec,
    )
    assert wet["absorbance_observed"] > dry["absorbance_observed"]


def test_prepare_tabulated_spectra_grid_mismatch_rejected():
    co2, _, _ = _synthetic_spectra(wn_min=2250.0, wn_max=2445.0, n=200)
    co2_b, _, _ = _synthetic_spectra(wn_min=2200.0, wn_max=2500.0, n=200)
    with pytest.raises(ValueError, match="share the same wavenumber grid"):
        prepare_tabulated_spectra(
            spectra=(co2, co2_b), filter_spec=get_default_ndir_filter("co2")
        )


# ---- empirical crosstalk fallback ----


def test_apply_optical_crosstalk_co2_from_h2o():
    result = apply_optical_crosstalk(absorption_co2=0.5, absorption_h2o=0.3)
    assert "absorption_co2_observed" in result
    assert result["absorption_co2_observed"] > result["absorption_co2_true"]
    expected_cross = DEFAULT_OPTICAL_CROSSTALK_SPEC.co2_channel_h2o_response * 0.3
    assert math.isclose(
        result["absorption_co2_observed"], 0.5 + expected_cross, abs_tol=1e-12
    )


def test_apply_optical_crosstalk_no_ch4_fields():
    """RCDW 交叉表不应含 CH4 相关字段。"""
    result = apply_optical_crosstalk(absorption_co2=0.4, absorption_h2o=0.2)
    forbidden = {
        "absorption_ch4_true",
        "absorption_ch4_observed",
        "absorption_co2_cross_from_ch4",
    }
    assert not (forbidden & result.keys())


# ---- optical_backend cache requirement collection ----


def _sample_condition(idx: int = 1) -> dict[str, str]:
    return {
        "sequence_id": f"RCDW-Q{idx:06d}",
        "mixture_id": f"RCDW-M{idx:06d}",
        "x_O2": "10.0",
        "x_CO2": "5.0",
        "x_N2": "85.0",
        "T_C_base": "25.0",
        "P_MPa_base": "0.1",
        "H_RH_base": "40.0",
        "L_m_base": "0.5",
        "status": "synthetic_measurement",
    }


def test_collect_hitran_cache_requirements_only_co2_channel(tmp_path):
    """RCDW 仅 co2 通道, 每条 condition 应需要 (CO2 + H2O) × co2 = 2 个 cache key。"""
    conditions = [_sample_condition(i + 1) for i in range(3)]
    reqs = collect_hitran_cache_requirements(conditions, cache_root=tmp_path)
    channels = {r.channel for r in reqs}
    assert channels == {"co2"}
    gases = {r.gas_spec.gas for r in reqs}
    assert gases == {"CO2", "H2O"}
    # 3 conditions 共享同一 (T_C_base, P_MPa_base) → 去重后仅 2 个 key
    assert len(reqs) == 2


def test_validate_missing_cache_raises(tmp_path):
    conditions = [_sample_condition(1)]
    with pytest.raises(MissingHitranBenchmarkCacheError):
        validate_hitran_benchmark_cache(conditions, cache_root=tmp_path)


def test_hitran_manifest_metadata_keys(tmp_path):
    meta = hitran_manifest_metadata(tmp_path)
    assert meta["hitran_cache_root"] == str(tmp_path)
    assert meta["h2o_policy"] == "rh_to_mole_pct"
    assert meta["hitran_cache_policy"] == "cache_only_prechecked"


# ---- compute_hitran_optical_absorption + 合成 cache ----


def _write_synthetic_cache(cache_root: Path) -> None:
    """根据 RCDW 默认 grid 在 cache_root 下写入 CO2 + H2O 合成谱线表。"""
    grid = build_hitran_grid_for_condition("co2", t_c=25.0, p_mpa=0.1)
    # 拟合 wavenumber_step_cm1 步长
    wn = np.arange(
        grid.wavenumber_min_cm1,
        grid.wavenumber_max_cm1 + grid.wavenumber_step_cm1 * 0.5,
        grid.wavenumber_step_cm1,
        dtype=np.float64,
    )
    # 合成 line absorption coefficient (cm² / molecule), 量级模拟 HAPI 输出
    co2_coeff = 1.0e-21 * np.exp(-((wn - 2347.0) / 40.0) ** 2)
    h2o_coeff = 1.0e-22 * np.exp(-((wn - 2330.0) / 60.0) ** 2)
    for gas_spec, coeff in zip(DEFAULT_HITRAN_GAS_SPECS, (co2_coeff, h2o_coeff)):
        if gas_spec.gas == "CO2":
            data = co2_coeff
        else:
            data = h2o_coeff
        key = hitran_cache_key(gas_spec, grid)
        write_cached_spectrum(
            cache_root, key, wavenumber_cm1=wn, absorption_coeff_cm1=data
        )


def test_compute_hitran_optical_absorption_uses_cache(tmp_path):
    """写入合成 cache 后, compute_hitran_optical_absorption 应能跑通。"""
    _write_synthetic_cache(tmp_path)
    condition = {
        "x_O2": "10.0",
        "x_CO2": "5.0",
        "x_N2": "85.0",
        "T_C": "25.0",
        "P_MPa": "0.1",
        "H_RH": "40.0",
        "L_m": "0.5",
    }
    result = compute_hitran_optical_absorption(condition, cache_root=tmp_path)
    assert result["optical_absorption_backend"] == HITRAN_ABSORPTION_BACKEND
    assert "absorption_co2_true" in result
    assert "absorption_co2_observed" in result
    assert "absorption_h2o_true" in result
    assert result["absorption_co2_observed"] >= result["absorption_co2_true"]
    # 不应出现 CH4 字段
    assert "absorption_ch4_true" not in result
    assert "absorption_ch4_observed" not in result


def test_compute_hitran_co2_monotonic(tmp_path):
    _write_synthetic_cache(tmp_path)
    base = {
        "x_O2": "10.0",
        "x_N2": "85.0",
        "T_C": "25.0",
        "P_MPa": "0.1",
        "H_RH": "40.0",
        "L_m": "0.5",
    }
    low = compute_hitran_optical_absorption(
        {**base, "x_CO2": "1.0"}, cache_root=tmp_path
    )
    high = compute_hitran_optical_absorption(
        {**base, "x_CO2": "15.0"}, cache_root=tmp_path
    )
    assert high["absorption_co2_observed"] > low["absorption_co2_observed"]


def test_hitran_backend_constant():
    assert HITRAN_ABSORPTION_BACKEND == "hitran_hapi_v1"
    assert EMPIRICAL_ABSORPTION_BACKEND == "empirical_v1"
