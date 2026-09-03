from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

import pytest

from gf.pipeline.a2_dynamic_protocol import (
    A2DynamicProtocolError,
    run_a2_dynamic_protocol,
    validate_a2_dynamic_configs,
    validate_a2_dynamic_data_config,
    validate_a2_dynamic_eval_config,
    validate_a2_dynamic_experiment_config,
    validate_a2_dynamic_pilot_config,
    validate_a2_dynamic_records,
)
from gf.pipeline.a2_dynamic_benchmark import run_a2_dynamic_benchmark
from gf.pipeline import (
    run_a2_dynamic_benchmark as public_run_a2_dynamic_benchmark,
    run_a2_dynamic_pilot as public_run_a2_dynamic_pilot,
    validate_a2_dynamic_configs as public_validate_a2_dynamic_configs,
)
from gf.sim import (
    estimate_ultrasonic_tof_series as public_estimate_ultrasonic_tof_series,
    load_dataset_splits,
    ultrasonic_signal_amplitude as public_ultrasonic_signal_amplitude,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_ROOT = PROJECT_ROOT / "configs"


def _load_json(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(value, dict)
    return value


def test_a2_dynamic_protocol_is_frozen_and_source_hashes_recompute() -> None:
    result = run_a2_dynamic_protocol(project_root=PROJECT_ROOT)

    assert result["status"] == "PASS"
    assert result["stage"] == "A2-DYN-0"
    assert result["summary"]["protocol_status"] == "PROTOCOL_FROZEN"
    assert result["summary"]["source_hashes_verified"] is True
    assert result["summary"]["split_totals"]["train"]["groups"] == 2520


def test_a2_dynamic_configs_validate_individually() -> None:
    data = _load_json(CONFIG_ROOT / "data" / "ar_he_co2_a2_dynamic_v1.json")
    evaluation = _load_json(CONFIG_ROOT / "eval" / "a2_dynamic_eval.json")
    experiment = _load_json(CONFIG_ROOT / "experiment" / "a2_dynamic_protocol.json")

    validate_a2_dynamic_data_config(data)
    validate_a2_dynamic_eval_config(evaluation)
    validate_a2_dynamic_experiment_config(experiment)
    assert validate_a2_dynamic_configs(PROJECT_ROOT)["reference_assets_registered"] is True
    assert data["data_version"].endswith("-r4")
    assert data["protocol_revision"] == "a2-dyn-0-r4"
    assert data["physics_reference"]["eos"]["sound_speed_model_id"] == "a2dyn_direct_multifluid_eos_v1"


def test_a2_dynamic_pilot_rejects_gate_mutation() -> None:
    experiment = _load_json(CONFIG_ROOT / "experiment" / "a2_dynamic_protocol.json")
    pilot = deepcopy(experiment["pilot"])
    pilot["dynamic_gate"]["minimum_stress_privileged_probe_improvement_fraction"] = 0.0

    with pytest.raises(A2DynamicProtocolError, match="dynamic_gate is not frozen"):
        validate_a2_dynamic_pilot_config(pilot)


def test_a2_dynamic_data_rejects_legacy_identity_field() -> None:
    data = _load_json(CONFIG_ROOT / "data" / "ar_he_co2_a2_dynamic_v1.json")
    invalid = deepcopy(data)
    invalid["sequence_id"] = "legacy"

    with pytest.raises(A2DynamicProtocolError, match="forbidden legacy key"):
        validate_a2_dynamic_data_config(invalid)


def test_a2_dynamic_data_rejects_unknown_profile_reference() -> None:
    data = _load_json(CONFIG_ROOT / "data" / "ar_he_co2_a2_dynamic_v1.json")
    invalid = deepcopy(data)
    invalid["families"]["D-IID"]["noise_by_split"]["val"] = "NOISE-UNKNOWN"

    with pytest.raises(A2DynamicProtocolError, match="references an unknown profile"):
        validate_a2_dynamic_data_config(invalid)


def test_a2_dynamic_data_rejects_incomplete_hardware_profile() -> None:
    data = _load_json(CONFIG_ROOT / "data" / "ar_he_co2_a2_dynamic_v1.json")
    invalid = deepcopy(data)
    invalid["hardware_profiles"]["thermal"]["profiles"][0].pop("heater_power")

    with pytest.raises(A2DynamicProtocolError, match="missing profile fields"):
        validate_a2_dynamic_data_config(invalid)


def test_a2_dynamic_eval_keeps_steady_baseline_out_of_early_competition() -> None:
    evaluation = _load_json(CONFIG_ROOT / "eval" / "a2_dynamic_eval.json")
    steady = next(item for item in evaluation["baseline_registry"] if item["model_id"] == "B-STEADY")

    assert steady["causal"] is False
    assert steady["allowed_horizons"] == ["P150", "FULL"]


def _minimal_record(*, observation_id: str, mixture_id: str, split: str = "train") -> dict[str, object]:
    return {
        "schema_version": "gf-a2-dynamic-record-1",
        "observation_id": observation_id,
        "mixture_id": mixture_id,
        "split": split,
        "family": "D-IID",
        "composition_region": "interior",
        "protocol_profile_id": "STEP_STANDARD",
        "transport_profile_id": "KIN-TRAIN",
        "ultrasonic_profile_id": "US-BURST-XCORR-1",
        "thermal_profile_id": "TCD-LUMPED-SYNTH-1",
        "ndir_profile_id": "NDIR-HIGHRANGE-SHORTPATH-1",
        "environment_id": "ENV-NOMINAL",
        "calibration_profile_id": "CAL-NOMINAL",
        "noise_profile_id": "NOISE-1X",
        "exposure_onset_s": 30.0,
        "exposure_end_s": 180.0,
        "timesteps": 1200,
        "dt_s": 0.2,
        "status": "generated",
        "x_Ar_pct": 60.0,
        "x_He_pct": 30.0,
        "x_CO2_pct": 10.0,
    }


def test_a2_dynamic_records_reject_duplicate_non_pure_compositions() -> None:
    data = _load_json(CONFIG_ROOT / "data" / "ar_he_co2_a2_dynamic_v1.json")
    records = [
        _minimal_record(observation_id="a2dyn-obs-000001", mixture_id="a2dyn-mix-000001"),
        _minimal_record(observation_id="a2dyn-obs-000002", mixture_id="a2dyn-mix-000002"),
    ]

    with pytest.raises(A2DynamicProtocolError, match="duplicate non-pure composition"):
        validate_a2_dynamic_records(records, data)


def test_a2_dynamic_records_reject_group_crossing_split() -> None:
    data = _load_json(CONFIG_ROOT / "data" / "ar_he_co2_a2_dynamic_v1.json")
    records = [
        _minimal_record(observation_id="a2dyn-obs-000001", mixture_id="a2dyn-mix-000001", split="train"),
        _minimal_record(observation_id="a2dyn-obs-000002", mixture_id="a2dyn-mix-000001", split="val"),
    ]

    with pytest.raises(A2DynamicProtocolError, match="crosses split"):
        validate_a2_dynamic_records(records, data)


def _pure_test_record(*, mixture_id: str) -> dict[str, object]:
    record = _minimal_record(
        observation_id="a2dyn-obs-000900",
        mixture_id=mixture_id,
        split="test",
    )
    record.update(
        {
            "family": "D-JOINT",
            "composition_region": "pure",
            "protocol_profile_id": "INCOMPLETE_RECOVERY",
            "transport_profile_id": "KIN-TEST",
            "environment_id": "ENV-FAR",
            "calibration_profile_id": "CAL-CONFLICT",
            "noise_profile_id": "NOISE-10X",
            "x_Ar_pct": 100.0,
            "x_He_pct": 0.0,
            "x_CO2_pct": 0.0,
        }
    )
    return record


def test_a2_dynamic_records_accept_canonical_test_pure_vertex() -> None:
    data = _load_json(CONFIG_ROOT / "data" / "ar_he_co2_a2_dynamic_v1.json")

    validate_a2_dynamic_records([_pure_test_record(mixture_id="a2dyn-mix-pure-Ar")], data)


def test_a2_dynamic_records_reject_non_canonical_pure_mixture_id() -> None:
    data = _load_json(CONFIG_ROOT / "data" / "ar_he_co2_a2_dynamic_v1.json")

    with pytest.raises(A2DynamicProtocolError, match="canonical mixture_id"):
        validate_a2_dynamic_records(
            [_pure_test_record(mixture_id="a2dyn-mix-0003781")],
            data,
        )


def test_a2_dynamic_benchmark_protocol_and_development_entrypoints_are_real() -> None:
    result = run_a2_dynamic_benchmark("protocol", project_root=str(PROJECT_ROOT))

    assert result["status"] == "PASS"
    assert result["stage"] == "protocol"
    from gf.pipeline import a2_dynamic_benchmark

    assert "generate-development" in a2_dynamic_benchmark.PLANNED_STAGES
    assert callable(a2_dynamic_benchmark.run_a2_dynamic_development_generation)
    assert callable(a2_dynamic_benchmark.run_a2_dynamic_difficulty_audit)


def test_a2_dynamic_public_exports_resolve_lazily() -> None:
    assert public_run_a2_dynamic_benchmark is run_a2_dynamic_benchmark
    assert callable(public_run_a2_dynamic_pilot)
    assert callable(public_validate_a2_dynamic_configs)
    assert callable(public_estimate_ultrasonic_tof_series)
    assert callable(public_ultrasonic_signal_amplitude)
    assert callable(load_dataset_splits)
