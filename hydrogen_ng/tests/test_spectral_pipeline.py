import importlib
import json
from pathlib import Path

import numpy as np
import pytest

from hg.pipeline.compare_optical_backends import compare_optical_backends
from hg.pipeline.precompute_hitran_benchmark_cache import main as precompute_benchmark_main
from hg.pipeline.precompute_hitran_benchmark_cache import default_hitran_precompute_worker_count, precompute_hitran_benchmark_cache
from hg.pipeline.precompute_hitran_spectra import main as precompute_main
from hg.pipeline.precompute_hitran_spectra import parse_channels, precompute_hitran_spectra
from hg.sim.generation.conditions import generate_condition_rows
from hg.sim.generation.optical_backend import collect_hitran_cache_requirements
from hg.sim.generation.spectral import (
    DEFAULT_HITRAN_GAS_SPECS,
    SPECTRAL_DEFAULTS_CONFIG_PATH,
    SPECTRAL_DEFAULTS_PAYLOAD,
    get_default_hitran_grid,
    get_default_ndir_filter,
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
        center = 3030.0 if SourceTables.startswith("CH4") else 2347.0 if SourceTables.startswith("CO2") else 2500.0
        scale = 1.0e-22 if not SourceTables.startswith("H2O") else 1.0e-24
        return wavenumber, np.exp(-0.5 * ((wavenumber - center) / 8.0) ** 2) * scale


def test_default_spectral_specs_cover_ch4_and_co2():
    assert get_default_ndir_filter("ch4").channel == "ch4"
    assert get_default_hitran_grid("co2").wavenumber_min_cm1 < 2347.0
    assert {spec.gas for spec in DEFAULT_HITRAN_GAS_SPECS} == {"CH4", "CO2", "H2O"}


def test_spectral_defaults_are_loaded_from_config_source_of_truth():
    config_path = Path(__file__).resolve().parents[1] / "configs" / "data" / "spectral-defaults.json"
    payload = json.loads(config_path.read_text(encoding="utf-8"))

    assert SPECTRAL_DEFAULTS_CONFIG_PATH == config_path
    assert SPECTRAL_DEFAULTS_PAYLOAD == payload
    assert payload["optical_absorption_backend"] == "hitran_hapi_v1"
    assert payload["filters"]["ch4"]["center_cm1"] == get_default_ndir_filter("ch4").center_cm1
    assert payload["filters"]["co2"]["fwhm_cm1"] == get_default_ndir_filter("co2").fwhm_cm1
    assert payload["hitran_grids"]["ch4"]["wavenumber_min_cm1"] == get_default_hitran_grid("ch4").wavenumber_min_cm1
    assert payload["hitran_grids"]["ch4"]["wavenumber_max_cm1"] == get_default_hitran_grid("ch4").wavenumber_max_cm1
    assert payload["hitran_grids"]["co2"]["wavenumber_min_cm1"] == get_default_hitran_grid("co2").wavenumber_min_cm1
    assert payload["hitran_grids"]["co2"]["wavenumber_max_cm1"] == get_default_hitran_grid("co2").wavenumber_max_cm1
    assert payload["hitran_grids"]["co2"]["wavenumber_step_cm1"] == get_default_hitran_grid("co2").wavenumber_step_cm1
    assert {spec["gas"] for spec in payload["gas_specs"]} == {spec.gas for spec in DEFAULT_HITRAN_GAS_SPECS}


def test_default_hitran_grids_cover_filter_main_lobe():
    for channel in ("ch4", "co2"):
        filter_spec = get_default_ndir_filter(channel)
        grid_spec = get_default_hitran_grid(channel)

        assert grid_spec.wavenumber_min_cm1 <= filter_spec.center_cm1 - filter_spec.fwhm_cm1
        assert grid_spec.wavenumber_max_cm1 >= filter_spec.center_cm1 + filter_spec.fwhm_cm1


def test_parse_channels_rejects_unknown_channel():
    with pytest.raises(Exception, match="o2"):
        parse_channels("ch4,o2")


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


def test_precompute_hitran_benchmark_cache_matches_condition_spec(tmp_path):
    fake_hapi = FakeHapi()
    summary = precompute_hitran_benchmark_cache(
        cache_root=tmp_path,
        sequence_count=2,
        seed=29,
        sampling_strategy="lhs",
        hapi_module=fake_hapi,
    )
    conditions = generate_condition_rows(2, seed=29, sampling_strategy="lhs")
    requirements = collect_hitran_cache_requirements(conditions, cache_root=tmp_path)

    assert summary["required_cache_entries"] == len(requirements)
    assert summary["computed_cache_entries"] == len(requirements)
    assert summary["skipped_cache_entries"] == 0
    assert summary["conditions"] == 2
    assert summary["workers"] == 1
    assert len(fake_hapi.fetch_calls) == len(requirements)
    assert all(requirement.path.is_file() for requirement in requirements)


def test_precompute_hitran_benchmark_cache_skips_existing_entries(tmp_path):
    fake_hapi = FakeHapi()
    first_summary = precompute_hitran_benchmark_cache(
        cache_root=tmp_path,
        sequence_count=2,
        seed=37,
        sampling_strategy="lhs",
        hapi_module=fake_hapi,
    )
    second_summary = precompute_hitran_benchmark_cache(
        cache_root=tmp_path,
        sequence_count=2,
        seed=37,
        sampling_strategy="lhs",
        hapi_module=fake_hapi,
    )

    assert first_summary["computed_cache_entries"] == first_summary["required_cache_entries"]
    assert second_summary["computed_cache_entries"] == 0
    assert second_summary["skipped_cache_entries"] == second_summary["required_cache_entries"]
    assert len(fake_hapi.fetch_calls) == first_summary["required_cache_entries"]


def test_precompute_hitran_benchmark_cache_cli_prints_summary(tmp_path, capsys, monkeypatch):
    fake_hapi = FakeHapi()

    def fake_precompute(**kwargs):
        kwargs["hapi_module"] = fake_hapi
        return precompute_hitran_benchmark_cache(**kwargs)

    monkeypatch.setattr("pipeline.precompute_hitran_benchmark_cache.precompute_hitran_benchmark_cache", fake_precompute)
    exit_code = precompute_benchmark_main([
        "--cache-root",
        str(tmp_path),
        "--sequences",
        "1",
        "--seed",
        "31",
        "--workers",
        "1",
    ])
    payload = json.loads(capsys.readouterr().out)

    assert exit_code == 0
    assert payload["sequence_count"] == 1
    assert payload["required_cache_entries"] > 0
    assert len(fake_hapi.fetch_calls) == payload["required_cache_entries"]


def test_default_hitran_precompute_worker_count_is_memory_conservative(monkeypatch):
    module = importlib.import_module("pipeline.precompute_hitran_benchmark_cache")
    monkeypatch.setattr(module, "default_worker_count", lambda sequence_count: 24)

    assert default_hitran_precompute_worker_count(6000) == 4


def test_parallel_hitran_precompute_bounds_pending_futures(monkeypatch, tmp_path):
    module = importlib.import_module("pipeline.precompute_hitran_benchmark_cache")
    max_pending_seen = 0

    class ImmediateFuture:
        def result(self):
            return "computed"

    class FakeExecutor:
        def __init__(self, max_workers):
            self.max_workers = max_workers

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, traceback):
            return False

        def submit(self, *args):
            return ImmediateFuture()

    def fake_wait(pending, *, return_when):
        nonlocal max_pending_seen
        futures = list(pending)
        max_pending_seen = max(max_pending_seen, len(futures))
        return {futures[0]}, set(futures[1:])

    monkeypatch.setattr(module, "ProcessPoolExecutor", FakeExecutor)
    monkeypatch.setattr(module, "wait", fake_wait)

    results = module._precompute_requirements_parallel(tuple(range(20)), cache_root=tmp_path, workers=4)

    assert results == ["computed"] * 20
    assert max_pending_seen <= 8


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
