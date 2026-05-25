import json
from pathlib import Path

import numpy as np

from pipeline.compare_optical_backends import compare_optical_backends
from pipeline.precompute_hitran_spectra import main as precompute_main
from pipeline.precompute_hitran_spectra import parse_channels, precompute_hitran_spectra
from sim.generation.spectral import DEFAULT_HITRAN_GAS_SPECS, get_default_hitran_grid, get_default_ndir_filter


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
        center = 3030.0 if SourceTables.startswith("CH4") else 2347.0 if SourceTables.startswith("CO2") else 2500.0
        scale = 1.0e-22 if not SourceTables.startswith("H2O") else 1.0e-24
        return wavenumber, np.exp(-0.5 * ((wavenumber - center) / 8.0) ** 2) * scale


def test_default_spectral_specs_cover_ch4_and_co2():
    assert get_default_ndir_filter("ch4").channel == "ch4"
    assert get_default_hitran_grid("co2").wavenumber_min_cm1 < 2347.0
    assert {spec.gas for spec in DEFAULT_HITRAN_GAS_SPECS} == {"CH4", "CO2", "H2O"}


def test_spectral_defaults_config_matches_code_defaults():
    config_path = Path(__file__).resolve().parents[1] / "configs" / "data" / "spectral-defaults.json"
    payload = json.loads(config_path.read_text(encoding="utf-8"))

    assert payload["optical_absorption_backend"] == "hitran_hapi_v1"
    assert payload["filters"]["ch4"]["center_cm1"] == get_default_ndir_filter("ch4").center_cm1
    assert payload["filters"]["co2"]["fwhm_cm1"] == get_default_ndir_filter("co2").fwhm_cm1
    assert payload["hitran_grids"]["co2"]["wavenumber_step_cm1"] == get_default_hitran_grid("co2").wavenumber_step_cm1
    assert {spec["gas"] for spec in payload["gas_specs"]} == {spec.gas for spec in DEFAULT_HITRAN_GAS_SPECS}


def test_parse_channels_rejects_unknown_channel():
    try:
        parse_channels("ch4,o2")
    except Exception as exc:
        assert "o2" in str(exc)
    else:
        raise AssertionError("parse_channels should reject unsupported channels")


def test_precompute_hitran_spectra_uses_real_backend_contract(tmp_path):
    fake_hapi = FakeHapi()
    summary = precompute_hitran_spectra(
        cache_root=tmp_path,
        channels=("ch4", "co2"),
        concentrations_pct={"CH4": 60.0, "CO2": 8.0, "H2O": 1.0},
        path_length_m=0.3,
        hapi_module=fake_hapi,
    )

    assert summary["channels"] == ["ch4", "co2"]
    assert summary["results"]["ch4"]["backend"] == "hitran_hapi_v1"
    assert summary["results"]["co2"]["absorbance_observed"] > 0.0
    assert len(fake_hapi.fetch_calls) == 6
    assert len({call[0] for call in fake_hapi.fetch_calls}) == 6


def test_precompute_cli_prints_json_summary(tmp_path, capsys, monkeypatch):
    fake_hapi = FakeHapi()

    def fake_precompute(**kwargs):
        kwargs["hapi_module"] = fake_hapi
        return precompute_hitran_spectra(**kwargs)

    monkeypatch.setattr("pipeline.precompute_hitran_spectra.precompute_hitran_spectra", fake_precompute)
    exit_code = precompute_main([
        "--cache-root",
        str(tmp_path),
        "--channels",
        "co2",
        "--path-length-m",
        "0.3",
    ])
    payload = json.loads(capsys.readouterr().out)

    assert exit_code == 0
    assert payload["channels"] == ["co2"]
    assert payload["results"]["co2"]["backend"] == "hitran_hapi_v1"


def test_compare_optical_backends_reports_delta(tmp_path):
    condition = {
        "x_H2": "10.0",
        "x_CH4": "60.0",
        "x_CO2": "8.0",
        "x_N2": "22.0",
        "T_C": "25.0",
        "P_MPa": "0.101325",
        "H_RH": "50.0",
        "L_m": "0.3",
    }
    summary = compare_optical_backends(
        cache_root=tmp_path,
        condition=condition,
        seed=20260525,
        hapi_module=FakeHapi(),
    )

    assert summary["empirical_v1"]["absorption_ch4_observed"] > 0.0
    assert summary["hitran_hapi_v1"]["absorption_co2_observed"] > 0.0
    assert "absorption_ch4_observed" in summary["delta_hitran_minus_empirical"]
