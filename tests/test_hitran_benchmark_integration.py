from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import pytest

from sim.generation.benchmark import BenchmarkGenerationSpec, generate_benchmark_dataset
from sim.generation.conditions import generate_condition_rows
from sim.generation.gas_state import h2o_mole_percent_from_rh, hitran_pressure_atm, hitran_temperature_k
from sim.generation.optical_backend import (
    MissingHitranBenchmarkCacheError,
    build_hitran_grid_for_condition,
    collect_hitran_cache_requirements,
)
from sim.generation.spectral import write_cached_spectrum


def test_default_benchmark_uses_hitran_cache_only_backend(tmp_path: Path):
    cache_root = tmp_path / "hitran-cache"
    conditions = generate_condition_rows(3, seed=17, sampling_strategy="lhs")
    _write_synthetic_hitran_cache(cache_root, conditions)

    summary = generate_benchmark_dataset(
        tmp_path,
        BenchmarkGenerationSpec(
            dataset_slug="hitran-default",
            sequence_count=3,
            seed=17,
            timesteps=8,
            storage="npz",
            hitran_cache_root=str(cache_root),
        ),
    )

    dataset_dir = tmp_path / "hitran-default"
    manifest = json.loads((dataset_dir / "manifest.json").read_text(encoding="utf-8"))
    waveform_spec = json.loads((dataset_dir / "metadata" / "waveform_spec.json").read_text(encoding="utf-8"))

    assert summary["optical_absorption_backend"] == "hitran_hapi_v1"
    assert manifest["optical_absorption_backend"] == "hitran_hapi_v1"
    assert manifest["hitran_cache_policy"] == "cache_only_prechecked"
    assert manifest["hitran_temperature_pressure_mode"] == "per_condition"
    assert manifest["h2o_policy"] == "rh_to_mole_pct"
    assert manifest["optical_crosstalk_policy"] == "spectral_multigas_integral"
    assert waveform_spec["hitran_cache_root"] == str(cache_root)


def test_default_benchmark_rejects_missing_hitran_cache_before_writing(tmp_path: Path):
    with pytest.raises(MissingHitranBenchmarkCacheError, match="precompute_hitran_benchmark_cache") as exc_info:
        generate_benchmark_dataset(
            tmp_path,
            BenchmarkGenerationSpec(
                dataset_slug="missing-cache",
                sequence_count=2,
                seed=23,
                timesteps=8,
                storage="npz",
                hitran_cache_root=str(tmp_path / "empty-cache"),
            ),
        )

    assert "CH4" in str(exc_info.value)
    assert not (tmp_path / "missing-cache").exists()


def test_hitran_ndir_changes_with_steady_phase_path_length(tmp_path: Path):
    cache_root = tmp_path / "hitran-cache"
    conditions = generate_condition_rows(1, seed=31, sampling_strategy="lhs")
    _write_synthetic_hitran_cache(cache_root, conditions)

    generate_benchmark_dataset(
        tmp_path,
        BenchmarkGenerationSpec(
            dataset_slug="hitran-path",
            sequence_count=1,
            seed=31,
            timesteps=8,
            storage="npz",
            multi_path_phase="steady",
            path_lms=(0.2, 1.0),
            hitran_cache_root=str(cache_root),
        ),
    )

    rows = _read_csv(tmp_path / "hitran-path" / "sequences" / "slow_sequence_long.csv")
    assert rows[4]["phase_id"] == "steady"
    assert rows[5]["phase_id"] == "steady"
    assert rows[4]["L_m"] == "0.20000"
    assert rows[5]["L_m"] == "1.00000"
    assert abs(float(rows[4]["V_NDIR_CH4"]) - float(rows[5]["V_NDIR_CH4"])) > 1e-6


def test_hitran_sequence_generation_does_not_call_empirical_main_sensor_features(tmp_path: Path, monkeypatch):
    cache_root = tmp_path / "hitran-cache"
    conditions = generate_condition_rows(1, seed=37, sampling_strategy="lhs")
    _write_synthetic_hitran_cache(cache_root, conditions)

    def fail_main_sensor_features(*args, **kwargs):
        raise AssertionError("HITRAN sequence generation must not call empirical main_sensor_features")

    monkeypatch.setattr("sim.generation.slow.main_sensor_features", fail_main_sensor_features)

    generate_benchmark_dataset(
        tmp_path,
        BenchmarkGenerationSpec(
            dataset_slug="hitran-no-empirical",
            sequence_count=1,
            seed=37,
            timesteps=8,
            storage="npz",
            hitran_cache_root=str(cache_root),
        ),
    )


def test_hitran_environment_helpers_are_reproducible():
    grid = build_hitran_grid_for_condition("ch4", t_c=25.12345, p_mpa=0.101325)

    assert hitran_temperature_k(25.12345) == 298.273
    assert hitran_pressure_atm(0.101325) == 1.0
    assert grid.temperature_k == 298.273
    assert grid.pressure_atm == 1.0
    assert h2o_mole_percent_from_rh(25.0, 0.101325, 50.0) == pytest.approx(1.5628100324257261, rel=1e-12)


def test_hitran_sequence_generation_uses_equilibrium_dynamics_not_legacy(tmp_path: Path, monkeypatch):
    cache_root = tmp_path / "hitran-cache"
    conditions = generate_condition_rows(1, seed=37, sampling_strategy="lhs")
    _write_synthetic_hitran_cache(cache_root, conditions)

    def fail_legacy(*args, **kwargs):
        raise AssertionError("HITRAN sequence generation must use equilibrium dynamics, not legacy _dynamic_slow_features")

    monkeypatch.setattr("sim.generation.slow._dynamic_slow_features", fail_legacy)

    # 即使 standard_exposure + jitter=0（empirical 的 legacy 条件），HITRAN 后端也不应回退到单时间常数动力学。
    generate_benchmark_dataset(
        tmp_path,
        BenchmarkGenerationSpec(
            dataset_slug="hitran-no-legacy",
            sequence_count=1,
            seed=37,
            timesteps=8,
            storage="npz",
            stage_profile="standard_exposure",
            stage_jitter=0.0,
            hitran_cache_root=str(cache_root),
        ),
    )


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _write_synthetic_hitran_cache(cache_root: Path, conditions: list[dict[str, str]]) -> None:
    for requirement in collect_hitran_cache_requirements(conditions, cache_root=cache_root):
        grid = requirement.grid_spec
        wavenumber = grid.wavenumber_min_cm1 + grid.wavenumber_step_cm1 * np.arange(
            int(round((grid.wavenumber_max_cm1 - grid.wavenumber_min_cm1) / grid.wavenumber_step_cm1)) + 1,
            dtype=np.float64,
        )
        if requirement.gas_spec.gas == "CH4":
            center, scale = 3030.0, 1.6e-22
        elif requirement.gas_spec.gas == "CO2":
            center, scale = 2347.0, 1.2e-22
        else:
            center, scale = (grid.wavenumber_min_cm1 + grid.wavenumber_max_cm1) / 2.0, 2.0e-25
        coeff = np.exp(-0.5 * ((wavenumber - center) / 8.0) ** 2) * scale
        write_cached_spectrum(
            cache_root,
            requirement.key,
            wavenumber_cm1=wavenumber,
            absorption_coeff_cm1=coeff,
        )
