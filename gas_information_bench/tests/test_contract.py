import copy

import pytest

from gib.contract import (
    ContractError,
    load_contracts,
    make_mixture_id,
    make_sequence_id,
    validate_deployment_fields,
    validate_dsp_provenance,
    validate_manifest,
    validate_sample_record,
    validate_solver_row,
    validate_split_assignments,
)


HASH_A = "a" * 64
HASH_B = "b" * 64
HASH_C = "c" * 64


def _identity(grade_cell: str) -> dict[str, object]:
    return {
        "candidate_id": "GIB-C4-LR",
        "composition": {"N2": 0.70, "CO2": 0.10, "O2": 0.15, "Ar": 0.05},
        "nuisance": {
            "T": 293.15,
            "P": 101.325,
            "RH": 45.0,
            "L": 0.50,
            "gain_m": 1.0,
            "baseline_m": 0.0,
            "delay_m": 0.0,
            "crosstalk_mn": [[0.0, 0.0], [0.0, 0.0]],
            "q_flow": 1.0,
        },
        "grade": {
            "grid_id": "GIB-S1-3x3-v1",
            "grid_cell_id": grade_cell,
            "information_band": "sufficient",
            "angle_band": "high_collinearity",
        },
        "modality_profile": {
            "profile_id": "GIB-MOD-RAW-FULL-v1",
            "enabled_modalities": ["ndir", "acoustic_raw", "thermal", "slow", "calibration"],
            "raw_dsp_view": "raw",
        },
        "units": {
            "composition": "mol/mol",
            "temperature": "K",
            "pressure": "kPa",
            "relative_humidity": "%RH",
            "path_length": "m",
            "flow": "L/min",
            "waveform": "declared_per_channel",
            "label": "mol/mol",
        },
    }


def _mixture_identity(identity: dict[str, object]) -> dict[str, object]:
    return {
        "candidate_id": identity["candidate_id"],
        "composition": identity["composition"],
    }


def _source_ref(source_id: str) -> dict[str, str]:
    return {
        "source_type": "contract_fixture",
        "source_id": source_id,
        "source_revision": "v1",
        "source_hash": HASH_A,
        "locator": "P2-09-contract",
    }


def _array_descriptor(name: str, layer: dict[str, object]) -> dict[str, object]:
    descriptor = {
        "file_ref": f"artifacts/{name}.npy",
        "dtype": "float64",
        "shape_spec": list(layer["axes"]),
        "unit": "mol/mol" if name in {"labels", "crb", "crb_p90"} else "dimensionless",
        "storage": layer["storage"],
        "derived_from": list(layer["derived_from"]),
    }
    if name == "raw_waveform" or name == "slow_channels":
        descriptor["unit"] = "declared_per_channel"
    if name == "dsp_features":
        descriptor["provenance_ref"] = "dsp_provenance"
    if name == "principal_angle":
        descriptor["shape_spec"] = []
        descriptor["unit"] = "degree"
    if name == "incremental_information":
        descriptor["record_fields"] = list(layer["record_fields"])
        descriptor["shape_spec"] = ["increment"]
    return descriptor


def _sample_record() -> dict[str, object]:
    contracts = load_contracts()
    identity = _identity("GIB-S1-SUF-HIG")
    mixture_id = make_mixture_id(_mixture_identity(identity))
    sequence_index = 0
    sequence_profile_id = "GIB-SEQ-P2-v1"
    sequence_id = make_sequence_id(mixture_id, sequence_index, sequence_profile_id)
    layers = contracts["data"]["array_layers"]
    return {
        "schema_version": "gib-benchmark-1",
        "mixture_id": mixture_id,
        "sequence_id": sequence_id,
        "sequence_index": sequence_index,
        "sequence_profile_id": sequence_profile_id,
        "candidate_id": "GIB-C4-LR",
        "composition": identity["composition"],
        "nuisance": identity["nuisance"],
        "grade": identity["grade"],
        "modality_profile": identity["modality_profile"],
        "units": identity["units"],
        "sources": {category: _source_ref(f"{category}-v1") for category in contracts["data"]["source_ref"]["categories"]},
        "split_assignment": {"split_id": "GIB-SPLIT-01", "partition": "train"},
        "arrays": {name: _array_descriptor(name, layer) for name, layer in layers.items()},
        "dsp_provenance": {
            "source_raw_manifest_id": "GIB-MANIFEST-0123456789ABCDEF",
            "raw_manifest_sha256": HASH_A,
            "dsp_config_sha256": HASH_B,
            "code_sha256": HASH_C,
            "derived_from": ["raw_waveform"],
        },
    }


def _manifest() -> dict[str, object]:
    artifact_types = list(load_contracts()["data"]["array_layers"])
    return {
        "manifest_id": "GIB-MANIFEST-0123456789ABCDEF",
        "schema_version": "gib-benchmark-1",
        "primary_key": "mixture_id",
        "instance_key": "sequence_id",
        "split_group_field": "mixture_id",
        "files": [
            {
                "path": f"artifacts/{artifact_type}.npy",
                "artifact_type": artifact_type,
                "sha256": HASH_A,
                "schema_version": "gib-benchmark-1",
            }
            for artifact_type in artifact_types
        ],
        "source_snapshots": [
            {
                "source_id": "P2-09-contract",
                "source_revision": "v1",
                "sha256": HASH_B,
                "locator": "contract-fixture",
            }
        ],
    }


def _validate_sample(record: dict[str, object], manifest: dict[str, object] | None = None) -> None:
    validate_sample_record(
        record,
        manifest=_manifest() if manifest is None else manifest,
        raw_manifest_sha256=HASH_A,
        dsp_config_sha256=HASH_B,
        code_sha256=HASH_C,
    )


def _split_rows() -> list[dict[str, str]]:
    identities = [_identity(f"GIB-S1-SUF-{angle}") for angle in ("HIG", "MED", "LOW")]
    compositions = (
        {"N2": 0.70, "CO2": 0.10, "O2": 0.15, "Ar": 0.05},
        {"N2": 0.65, "CO2": 0.15, "O2": 0.15, "Ar": 0.05},
        {"N2": 0.60, "CO2": 0.20, "O2": 0.15, "Ar": 0.05},
    )
    for identity, composition in zip(identities, compositions):
        identity["composition"] = composition
    mixture_ids = [make_mixture_id(_mixture_identity(identity)) for identity in identities]
    sequence_ids = [make_sequence_id(mixture_id, 0, "GIB-SEQ-P2-v1") for mixture_id in mixture_ids]
    rows = []
    for split_number in range(1, 6):
        split_id = f"GIB-SPLIT-{split_number:02d}"
        for partition, index in zip(("train", "val", "test"), range(3)):
            rows.append(
                {
                    "mixture_id": mixture_ids[index],
                    "sequence_id": sequence_ids[index],
                    "split_id": split_id,
                    "partition": partition,
                }
            )
    return rows


def _solver_row() -> dict[str, object]:
    sample = _sample_record()
    return {
        "sequence_id": sample["sequence_id"],
        "method_id": "ridge",
        "split_id": "GIB-SPLIT-01",
        "seed": 101,
        "iterations": 4,
        "forward_calls": 5,
        "convergence": True,
        "condition_number": 30.0,
        "final_residual": 1.0e-8,
        "runtime_ns": 1000,
        "hardware_fingerprint": "hardware-fixture-v1",
        "logical_cpu_count": 4,
        "blas_threads": 1,
        "omp_threads": 1,
        "mkl_threads": 1,
        "framework_threads": 1,
        "os_version": "Windows-fixture",
        "python_version": "3.10-fixture",
        "numpy_version": "1.24-fixture",
        "framework_version": "none",
        "method_package_versions": "ridge-builtin",
        "git_commit": "0123456789abcdef0123456789abcdef01234567",
        "repeat_index": 0,
    }


def test_contract_bundle_is_frozen_and_covers_all_required_layers():
    contracts = load_contracts()
    assert contracts["data"]["contract_status"] == "contract_frozen"
    assert set(contracts["data"]["array_layers"]) == {
        "raw_waveform",
        "slow_channels",
        "calibration_channels",
        "dsp_features",
        "labels",
        "sample_fisher",
        "effective_fisher",
        "crb",
        "crb_p90",
        "principal_angle",
        "incremental_information",
    }
    assert contracts["manifest"]["fixed_values"] == {
        "primary_key": "mixture_id",
        "instance_key": "sequence_id",
        "split_group_field": "mixture_id",
    }
    assert contracts["split"]["split_ids"] == [
        "GIB-SPLIT-01",
        "GIB-SPLIT-02",
        "GIB-SPLIT-03",
        "GIB-SPLIT-04",
        "GIB-SPLIT-05",
    ]


def test_ids_are_deterministic_and_keep_mixture_and_sequence_namespaces_separate():
    identity = _identity("GIB-S1-SUF-HIG")
    mixture_identity = _mixture_identity(identity)
    mixture_id = make_mixture_id(mixture_identity)
    assert mixture_id == make_mixture_id(copy.deepcopy(mixture_identity))
    changed_observation = copy.deepcopy(identity)
    changed_observation["nuisance"]["T"] = 333.15
    changed_observation["grade"]["grid_cell_id"] = "GIB-S1-INS-LOW"
    changed_observation["modality_profile"]["profile_id"] = "GIB-MOD-DSP-FULL-v1"
    assert mixture_id == make_mixture_id(_mixture_identity(changed_observation))
    sequence_zero = make_sequence_id(mixture_id, 0, "GIB-SEQ-P2-v1")
    sequence_one = make_sequence_id(mixture_id, 1, "GIB-SEQ-P2-v1")
    assert sequence_zero != sequence_one
    with pytest.raises(ContractError):
        make_sequence_id(sequence_zero, 0, "GIB-SEQ-P2-v1")


def test_sample_record_freezes_arrays_sources_and_dsp_provenance():
    record = _sample_record()
    _validate_sample(record)
    mismatched_mixture = copy.deepcopy(record)
    mismatched_mixture["mixture_id"] = "GIB-M-FFFFFFFFFFFFFFFF"
    with pytest.raises(ContractError, match="does not match candidate_id and composition"):
        _validate_sample(mismatched_mixture)
    mismatched_sequence = copy.deepcopy(record)
    mismatched_sequence["sequence_index"] = 1
    with pytest.raises(ContractError, match="sequence_id does not match"):
        _validate_sample(mismatched_sequence)
    with pytest.raises(ContractError, match="mismatch"):
        validate_dsp_provenance(
            record["dsp_provenance"],
            raw_manifest_sha256=HASH_B,
            dsp_config_sha256=HASH_B,
            code_sha256=HASH_C,
        )


def test_sample_record_rejects_invalid_crosstalk_grade_and_raw_dsp_view():
    invalid_crosstalk = _sample_record()
    invalid_crosstalk["nuisance"]["crosstalk_mn"] = "invalid"
    with pytest.raises(ContractError, match="square numeric matrix"):
        _validate_sample(invalid_crosstalk)

    invalid_grade = _sample_record()
    invalid_grade["grade"]["grid_cell_id"] = "GIB-S1-NOT-A-CELL"
    with pytest.raises(ContractError, match="accessible frozen S1 cell"):
        _validate_sample(invalid_grade)

    inconsistent_grade = _sample_record()
    inconsistent_grade["grade"]["information_band"] = "insufficient"
    with pytest.raises(ContractError, match="information_band"):
        _validate_sample(inconsistent_grade)

    mismatched_view = _sample_record()
    mismatched_view["modality_profile"]["enabled_modalities"] = ["ndir", "acoustic_dsp"]
    mismatched_view["modality_profile"]["raw_dsp_view"] = "raw"
    with pytest.raises(ContractError, match="raw_dsp_view must be dsp"):
        _validate_sample(mismatched_view)


def test_sample_record_requires_manifest_bound_paths_axes_and_external_dsp_hashes():
    escaping_path = _sample_record()
    escaping_path["arrays"]["raw_waveform"]["file_ref"] = "../../outside.npy"
    with pytest.raises(ContractError, match="artifact root"):
        _validate_sample(escaping_path)

    wrong_axes = _sample_record()
    wrong_axes["arrays"]["crb"]["shape_spec"] = ["row", "column"]
    with pytest.raises(ContractError, match="frozen axes"):
        _validate_sample(wrong_axes)

    missing_artifact = _manifest()
    missing_artifact["files"] = [
        item for item in missing_artifact["files"] if item["artifact_type"] != "crb_p90"
    ]
    with pytest.raises(ContractError, match="missing or has the wrong artifact_type"):
        _validate_sample(_sample_record(), missing_artifact)

    with pytest.raises(ContractError, match="raw_manifest_sha256 mismatch"):
        validate_sample_record(
            _sample_record(),
            manifest=_manifest(),
            raw_manifest_sha256=HASH_B,
            dsp_config_sha256=HASH_B,
            code_sha256=HASH_C,
        )


def test_manifest_requires_primary_instance_group_keys_and_each_file_hash():
    manifest = _manifest()
    validate_manifest(manifest)
    broken = copy.deepcopy(manifest)
    del broken["files"][0]["sha256"]
    with pytest.raises(ContractError, match="keys mismatch"):
        validate_manifest(broken)


def test_split_validation_is_group_based_and_rejects_cross_partition_overlap():
    rows = _split_rows()
    second_sequence = dict(rows[0])
    second_sequence["sequence_id"] = make_sequence_id(
        second_sequence["mixture_id"],
        1,
        "GIB-SEQ-P2-v1",
    )
    rows.append(second_sequence)
    summary = validate_split_assignments(rows)
    assert summary == {
        "GIB-SPLIT-01": 3,
        "GIB-SPLIT-02": 3,
        "GIB-SPLIT-03": 3,
        "GIB-SPLIT-04": 3,
        "GIB-SPLIT-05": 3,
    }
    broken = copy.deepcopy(rows)
    broken[1]["mixture_id"] = broken[0]["mixture_id"]
    broken[1]["sequence_id"] = make_sequence_id(
        broken[0]["mixture_id"],
        2,
        "GIB-SEQ-P2-v1",
    )
    with pytest.raises(ContractError, match="crosses partitions"):
        validate_split_assignments(broken)


def test_split_validation_rejects_historical_seed_field_without_allowing_silent_recovery():
    broken = _split_rows()
    legacy_key = "noise_" + "seed"
    broken[0][legacy_key] = 0
    with pytest.raises(ContractError, match="historical field"):
        validate_split_assignments(broken)


def test_oracle_fields_are_separate_from_deployment_loader_fields():
    validate_deployment_fields(
        [
            "mixture_id",
            "sequence_id",
            "modality_profile",
            "nuisance",
            "units",
            "raw_waveform",
            "slow_channels",
            "calibration_channels",
            "dsp_features",
            "dsp_provenance",
        ]
    )
    with pytest.raises(ContractError, match="oracle fields"):
        validate_deployment_fields(["dsp_features", "oracle_features"])


def test_solver_difficulty_row_has_p2_08_metrics_and_runtime_metadata():
    row = _solver_row()
    validate_solver_row(row)
    broken = dict(row)
    del broken["runtime_ns"]
    with pytest.raises(ContractError, match="missing fields"):
        validate_solver_row(broken)
