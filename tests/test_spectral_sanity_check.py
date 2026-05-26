from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from pipeline.sanity_check_tabulated_spectra import (
    load_cli_spectra,
    parse_channels,
    sanity_check_tabulated_spectra,
)
from sim.generation.spectral import (
    TabulatedSpectrum,
    convert_hitran_coeff_to_per_percent_m,
    get_default_hitran_grid,
)


class FakeHapi:
    def __init__(self):
        self.fetch_calls = []

    def db_begin(self, path):
        self.db_path = path

    def fetch(self, table_name, molecule_id, isotopologue_id, wavenumber_min_cm1, wavenumber_max_cm1):
        self.fetch_calls.append((table_name, molecule_id, isotopologue_id, wavenumber_min_cm1, wavenumber_max_cm1))

    def absorptionCoefficient_Voigt(self, *, SourceTables, Environment, WavenumberRange, WavenumberStep, HITRAN_units):
        wavenumber = np.linspace(
            WavenumberRange[0],
            WavenumberRange[1],
            int(round((WavenumberRange[1] - WavenumberRange[0]) / WavenumberStep)) + 1,
        )
        if SourceTables.startswith("CH4"):
            center, scale = 3030.0, 2.0e-23
        elif SourceTables.startswith("CO2"):
            center, scale = 2347.0, 1.1e-22
        else:
            center, scale = 2500.0, 1.0e-25
        coeff = np.exp(-0.5 * ((wavenumber - center) / 8.0) ** 2) * scale
        return wavenumber, coeff


def test_parse_channels_rejects_unknown_channel():
    with pytest.raises(Exception, match="o2"):
        parse_channels("ch4,o2")


def test_sanity_check_tabulated_spectra_reports_ok_for_matching_shape(tmp_path: Path):
    spectra = _make_matching_spectra()
    fake_hapi = FakeHapi()

    summary = sanity_check_tabulated_spectra(
        cache_root=tmp_path,
        spectra=spectra,
        concentrations_pct={"CH4": 60.0, "CO2": 8.0, "H2O": 0.0},
        path_length_m=0.3,
        channels=("ch4", "co2"),
        hapi_module=fake_hapi,
    )

    assert summary["channels"] == ["ch4", "co2"]
    assert summary["results"]["ch4"]["overall_ok"] is True
    assert summary["results"]["co2"]["overall_ok"] is True
    assert summary["results"]["ch4"]["checks"]["tabulated_target_gas_is_dominant"] is True
    assert summary["results"]["co2"]["checks"]["hitran_target_gas_is_dominant"] is True
    assert not any(call[0].startswith("H2O") for call in fake_hapi.fetch_calls)


def test_sanity_check_tabulated_spectra_rejects_missing_nonzero_gas(tmp_path: Path):
    spectra = (
        TabulatedSpectrum(
            gas="CH4",
            wavenumber_cm1=np.array([2880.0, 3180.0]),
            absorption_coeff_per_percent_m=np.array([0.0, 0.0]),
            source_version="synthetic",
        ),
    )

    with pytest.raises(ValueError, match="Missing quantitative spectra"):
        sanity_check_tabulated_spectra(
            cache_root=tmp_path,
            spectra=spectra,
            concentrations_pct={"CH4": 60.0, "CO2": 8.0, "H2O": 0.0},
            path_length_m=0.3,
            channels=("ch4",),
            hapi_module=FakeHapi(),
        )


def test_sanity_check_tabulated_spectra_rejects_no_positive_concentrations(tmp_path: Path):
    with pytest.raises(ValueError, match="at least one positive gas concentration"):
        sanity_check_tabulated_spectra(
            cache_root=tmp_path,
            spectra=(),
            concentrations_pct={"CH4": 0.0, "CO2": 0.0, "H2O": 0.0},
            path_length_m=0.3,
            channels=("ch4",),
            hapi_module=FakeHapi(),
        )


def test_load_cli_spectra_loads_only_provided_paths(tmp_path: Path):
    ch4_path = tmp_path / "ch4.csv"
    ch4_path.write_text("wavenumber_cm1,absorption_coeff\n2880,0.1\n3180,0.2\n", encoding="utf-8")

    class Args:
        ch4_spectrum = str(ch4_path)
        co2_spectrum = None
        h2o_spectrum = None
        unit = "per_percent_m"
        wavenumber_column = "wavenumber_cm1"
        coeff_column = "absorption_coeff"

    spectra = load_cli_spectra(Args())

    assert len(spectra) == 1
    assert spectra[0].gas == "CH4"


def _make_matching_spectra() -> tuple[TabulatedSpectrum, ...]:
    spectra: list[TabulatedSpectrum] = []
    grids = [get_default_hitran_grid("ch4"), get_default_hitran_grid("co2")]
    wavenumber = np.arange(
        min(grid.wavenumber_min_cm1 for grid in grids),
        max(grid.wavenumber_max_cm1 for grid in grids) + 0.05,
        0.1,
        dtype=np.float64,
    )
    for gas, center, scale in (("CH4", 3030.0, 2.0e-23), ("CO2", 2347.0, 1.1e-22)):
        hitran_coeff = np.exp(-0.5 * ((wavenumber - center) / 8.0) ** 2) * scale
        spectra.append(
            TabulatedSpectrum(
                gas=gas,
                wavenumber_cm1=wavenumber,
                absorption_coeff_per_percent_m=convert_hitran_coeff_to_per_percent_m(
                    hitran_coeff,
                    temperature_k=296.0,
                    pressure_atm=1.0,
                ),
                source_version=f"fake-{gas.lower()}",
            )
        )
    return tuple(spectra)
