import numpy as np
import pytest

from sim.generation.spectral import (
    NDIRFilter,
    SpectralCacheKey,
    HitranGasSpec,
    HitranGridSpec,
    compute_hitran_ndir_absorbance,
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
        scale = 0.011 if SourceTables == "CO2" else 0.002
        center = 2340.0 if SourceTables == "CO2" else 2365.0
        coeff = np.exp(-0.5 * ((wavenumber - center) / 8.0) ** 2) * scale
        return wavenumber, coeff


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
    assert len(fake_hapi.coefficient_calls) == 2
    assert first["absorbance_by_gas"]["CO2"] > first["absorbance_by_gas"]["CH4"]


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
