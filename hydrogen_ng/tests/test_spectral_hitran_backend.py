import pickle

import numpy as np
import pytest

from sim.generation.spectral import (
    BOLTZMANN_J_PER_K,
    NDIRFilter,
    STANDARD_ATMOSPHERE_PA,
    SpectralCacheKey,
    HitranGasSpec,
    HitranGridSpec,
    MissingHitranCacheError,
    MissingHitranTableError,
    compute_hitran_ndir_absorbance,
    convert_hitran_coeff_to_per_percent_m,
    read_cached_spectrum,
    write_cached_spectrum,
)


class FakeHapi:
    def __init__(self):
        self.fetch_calls = []
        self.coefficient_calls = []

    def db_begin(self, path):
        self.db_path = path

    def fetch(self, table_name, molecule_id, isotopologue_id, wavenumber_min_cm1, wavenumber_max_cm1):
        self.fetch_calls.append((table_name, molecule_id, isotopologue_id, wavenumber_min_cm1, wavenumber_max_cm1))

    def absorptionCoefficient_Voigt(self, *, SourceTables, Environment, WavenumberRange, WavenumberStep, HITRAN_units):
        self.coefficient_calls.append((SourceTables, Environment, WavenumberRange, WavenumberStep, HITRAN_units))
        wavenumber = np.linspace(WavenumberRange[0], WavenumberRange[1], int(round((WavenumberRange[1] - WavenumberRange[0]) / WavenumberStep)) + 1)
        scale = 1.1e-22 if SourceTables.startswith("CO2") else 2.0e-23
        center = 2340.0 if SourceTables.startswith("CO2") else 2365.0
        coeff = np.exp(-0.5 * ((wavenumber - center) / 8.0) ** 2) * scale
        return wavenumber, coeff


def test_hitran_unit_conversion_uses_ideal_gas_column_density():
    coeff = np.array([1.0e-22, 2.0e-22], dtype=np.float64)
    temperature_k = 296.0
    pressure_atm = 1.0

    converted = convert_hitran_coeff_to_per_percent_m(
        coeff,
        temperature_k=temperature_k,
        pressure_atm=pressure_atm,
    )

    expected_column_density = pressure_atm * STANDARD_ATMOSPHERE_PA / (BOLTZMANN_J_PER_K * temperature_k) * 1e-6
    np.testing.assert_allclose(converted, coeff * expected_column_density)


def test_hitran_unit_conversion_rejects_invalid_state():
    with pytest.raises(ValueError, match="temperature_k"):
        convert_hitran_coeff_to_per_percent_m(np.array([1.0e-22]), temperature_k=0.0, pressure_atm=1.0)
    with pytest.raises(ValueError, match="pressure_atm"):
        convert_hitran_coeff_to_per_percent_m(np.array([1.0e-22]), temperature_k=296.0, pressure_atm=0.0)


def test_hitran_backend_uses_hapi_then_cache(tmp_path):
    fake_hapi = FakeHapi()
    grid = HitranGridSpec(
        wavenumber_min_cm1=2300.0,
        wavenumber_max_cm1=2400.0,
        wavenumber_step_cm1=1.0,
        temperature_k=296.0,
        pressure_atm=1.0,
    )
    gas_specs = (
        HitranGasSpec("CO2", "CO2", 2, 1),
        HitranGasSpec("CH4", "CH4", 6, 1),
    )

    first = compute_hitran_ndir_absorbance(
        gas_specs=gas_specs,
        concentrations_pct={"CO2": 8.0, "CH4": 60.0},
        path_length_m=0.3,
        filter_spec=NDIRFilter(channel="co2", center_cm1=2340.0, fwhm_cm1=24.0),
        grid_spec=grid,
        cache_root=tmp_path,
        hapi_module=fake_hapi,
    )
    second = compute_hitran_ndir_absorbance(
        gas_specs=gas_specs,
        concentrations_pct={"CO2": 8.0, "CH4": 60.0},
        path_length_m=0.3,
        filter_spec=NDIRFilter(channel="co2", center_cm1=2340.0, fwhm_cm1=24.0),
        grid_spec=grid,
        cache_root=tmp_path,
    )

    assert first["backend"] == "hitran_hapi_v1"
    assert second["absorbance_observed"] == pytest.approx(first["absorbance_observed"], rel=1e-12)
    assert len(fake_hapi.fetch_calls) == 2
    assert fake_hapi.fetch_calls[0][0] == "CO2_2300p0000_2400p0000"
    assert fake_hapi.fetch_calls[1][0] == "CH4_2300p0000_2400p0000"
    assert len(fake_hapi.coefficient_calls) == 2
    assert first["absorbance_by_gas"]["CO2"] > first["absorbance_by_gas"]["CH4"]


def test_hitran_backend_reuses_local_hapi_table_before_fetch(tmp_path):
    fake_hapi = FakeHapi()
    grid = HitranGridSpec(
        wavenumber_min_cm1=2300.0,
        wavenumber_max_cm1=2400.0,
        wavenumber_step_cm1=1.0,
        temperature_k=296.0,
        pressure_atm=1.0,
    )
    (tmp_path / "CO2_2300p0000_2400p0000.data").write_text("local table\n", encoding="utf-8")
    (tmp_path / "CO2_2300p0000_2400p0000.header").write_text("local header\n", encoding="utf-8")

    result = compute_hitran_ndir_absorbance(
        gas_specs=(HitranGasSpec("CO2", "CO2", 2, 1),),
        concentrations_pct={"CO2": 8.0},
        path_length_m=0.3,
        filter_spec=NDIRFilter(channel="co2", center_cm1=2340.0, fwhm_cm1=24.0),
        grid_spec=grid,
        cache_root=tmp_path,
        hapi_module=fake_hapi,
    )

    assert result["backend"] == "hitran_hapi_v1"
    assert fake_hapi.fetch_calls == []
    assert fake_hapi.coefficient_calls[0][0] == "CO2_2300p0000_2400p0000"


def test_hitran_backend_computes_cache_from_local_table_without_fetch(tmp_path):
    fake_hapi = FakeHapi()
    grid = HitranGridSpec(
        wavenumber_min_cm1=2300.0,
        wavenumber_max_cm1=2400.0,
        wavenumber_step_cm1=1.0,
        temperature_k=296.0,
        pressure_atm=1.0,
    )
    (tmp_path / "CO2_2300p0000_2400p0000.data").write_text("local table\n", encoding="utf-8")
    (tmp_path / "CO2_2300p0000_2400p0000.header").write_text("local header\n", encoding="utf-8")

    result = compute_hitran_ndir_absorbance(
        gas_specs=(HitranGasSpec("CO2", "CO2", 2, 1),),
        concentrations_pct={"CO2": 8.0},
        path_length_m=0.3,
        filter_spec=NDIRFilter(channel="co2", center_cm1=2340.0, fwhm_cm1=24.0),
        grid_spec=grid,
        cache_root=tmp_path,
        hapi_module=fake_hapi,
        allow_fetch=False,
    )

    assert result["backend"] == "hitran_hapi_v1"
    assert fake_hapi.fetch_calls == []
    assert len(fake_hapi.coefficient_calls) == 1


def test_hitran_backend_rejects_missing_local_table_when_fetch_is_disabled(tmp_path):
    fake_hapi = FakeHapi()
    grid = HitranGridSpec(
        wavenumber_min_cm1=2300.0,
        wavenumber_max_cm1=2400.0,
        wavenumber_step_cm1=1.0,
        temperature_k=296.0,
        pressure_atm=1.0,
    )

    with pytest.raises(MissingHitranTableError, match="CO2_2300p0000_2400p0000"):
        compute_hitran_ndir_absorbance(
            gas_specs=(HitranGasSpec("CO2", "CO2", 2, 1),),
            concentrations_pct={"CO2": 8.0},
            path_length_m=0.3,
            filter_spec=NDIRFilter(channel="co2", center_cm1=2340.0, fwhm_cm1=24.0),
            grid_spec=grid,
            cache_root=tmp_path,
            hapi_module=fake_hapi,
            allow_fetch=False,
        )

    assert fake_hapi.fetch_calls == []
    assert fake_hapi.coefficient_calls == []


def test_hitran_errors_are_pickle_safe_for_process_workers(tmp_path):
    key = SpectralCacheKey(
        backend="hitran_hapi_v1",
        gas="CO2",
        source_version="hitran_hapi_v1",
        wavenumber_min_cm1=2300.0,
        wavenumber_max_cm1=2400.0,
        wavenumber_step_cm1=1.0,
        temperature_k=296.0,
        pressure_atm=1.0,
    )
    errors = (
        MissingHitranCacheError(key, tmp_path / "missing.npz"),
        MissingHitranTableError("CO2_2300p0000_2400p0000", tmp_path),
    )

    for error in errors:
        restored = pickle.loads(pickle.dumps(error))
        assert type(restored) is type(error)
        assert str(restored) == str(error)


def test_spectral_cache_roundtrip(tmp_path):
    key = SpectralCacheKey(
        backend="hitran_hapi_v1",
        gas="CO2",
        source_version="hitran_hapi_v1",
        wavenumber_min_cm1=2300.0,
        wavenumber_max_cm1=2400.0,
        wavenumber_step_cm1=1.0,
        temperature_k=296.0,
        pressure_atm=1.0,
    )
    wavenumber = np.array([2300.0, 2301.0, 2302.0])
    coeff = np.array([0.1, 0.2, 0.3])

    write_cached_spectrum(tmp_path, key, wavenumber_cm1=wavenumber, absorption_coeff_cm1=coeff)
    cached = read_cached_spectrum(tmp_path, key)

    assert cached is not None
    cached_wavenumber, cached_coeff = cached
    np.testing.assert_allclose(cached_wavenumber, wavenumber)
    np.testing.assert_allclose(cached_coeff, coeff)
