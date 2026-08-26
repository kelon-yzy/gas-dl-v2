"""Deterministic P3 pilot generation and bundle validation."""

from __future__ import annotations

import hashlib
import json
import math
from collections import defaultdict
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from uuid import uuid4

import numpy as np

from ..audit.forward import AuditConfig, ETA_PRIOR_STD, analyze_point, forward_observation
from ..audit.grid import physical_crb_p90
from ..common.io import (
    atomic_promote_directory,
    atomic_write_json,
    atomic_write_jsonl,
    canonical_json_bytes,
    remove_owned_staging,
    sha256_bytes,
    sha256_file,
)
from ..contract import (
    ContractError,
    load_contracts,
    make_manifest_id,
    make_mixture_id,
    make_sequence_id,
    validate_manifest,
    validate_sample_record,
    validate_split_assignments,
)
from ..pipeline.dataset import load_deployment_records, load_oracle_records
from ..pipeline.raw_dsp import build_dsp_provenance, derive_dsp, dsp_config_sha256
from .packaging.arrays import read_array_artifact, write_array_artifact


LAYER_CODES = {
    "raw_waveform": "rw",
    "slow_channels": "sc",
    "calibration_channels": "cc",
    "dsp_features": "df",
    "labels": "lb",
    "sample_fisher": "sf",
    "effective_fisher": "ef",
    "crb": "cb",
    "crb_p90": "cp",
    "principal_angle": "pa",
    "incremental_information": "ii",
}
LAYER_UNITS = {
    "raw_waveform": "declared_per_channel",
    "slow_channels": "declared_per_channel",
    "calibration_channels": "declared_per_channel",
    "dsp_features": "declared_per_channel",
    "labels": "mol/mol",
    "sample_fisher": "dimensionless",
    "effective_fisher": "dimensionless",
    "crb": "mol/mol",
    "crb_p90": "mol/mol",
    "principal_angle": "degree",
    "incremental_information": "dimensionless",
}


@dataclass(frozen=True)
class GeneratedSample:
    record: dict[str, Any]
    deployment: dict[str, Any]
    oracle: dict[str, Any]
    arrays: dict[str, np.ndarray]


def _stable_rng(plan: Mapping[str, Any], stream_name: str, *identity: object) -> np.random.Generator:
    state = plan["random_state"]
    payload = {
        "master": int(state["master"]),
        "stream": int(state[stream_name]),
        "identity": identity,
    }
    digest = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).digest()
    return np.random.default_rng(int.from_bytes(digest[:8], "big"))


def validate_pilot_plan(plan: Mapping[str, Any], grid: Mapping[str, Any]) -> None:
    if plan.get("schema_version") != "gib-benchmark-1":
        raise ContractError("pilot plan schema_version mismatch")
    if plan.get("pilot_plan_status") != "frozen":
        raise ContractError("pilot plan must be frozen")
    if int(plan.get("grid_cell_count", -1)) != 9 or len(grid.get("cells", [])) != 9:
        raise ContractError("pilot plan requires all 9 grid cells")
    if [str(item) for item in plan["splits"]["split_ids"]] != load_contracts()["split"]["split_ids"]:
        raise ContractError("pilot split IDs do not match the frozen contract")
    if plan["splits"]["group_field"] != "mixture_id":
        raise ContractError("pilot split group must be mixture_id")
    if [int(item) for item in plan["splits"]["nested_train_fractions"]] != [10, 25, 50, 75, 100]:
        raise ContractError("nested train fractions do not match the frozen protocol")
    if plan["splits"].get("stratify_by") not in (None, "grid_cell_id"):
        raise ContractError("unsupported pilot split stratification")
    for scope in ("pilot", "dry_run"):
        if int(plan[scope]["mixtures_per_cell"]) < 3 or int(plan[scope]["sequences_per_mixture"]) < 2:
            raise ContractError(f"{scope} does not cover group split and one-to-many IDs")
    if int(plan["raw"]["channel_count"]) != 8 or int(plan["raw"]["time_length"]) < 16:
        raise ContractError("raw array shape does not match the P3 generator")
    if plan["raw"]["storage"] != "per_sequence_npy" or plan["raw"]["compression"] != "none":
        raise ContractError("pilot storage must preserve per-sequence addressability")
    if int(plan["execution"]["workers"]) != 1:
        raise ContractError("P3 pilot worker count is frozen to one")


def _composition(plan: Mapping[str, Any], cell_id: str, mixture_index: int) -> dict[str, float]:
    rng = _stable_rng(plan, "composition_stream", cell_id, mixture_index)
    baseline = np.array([0.55, 0.20, 0.20, 0.05], dtype=np.float64)
    concentration = float(plan["sampling"]["composition_concentration"])
    values = rng.dirichlet(baseline * concentration)
    values[-1] = 1.0 - float(np.sum(values[:-1]))
    if np.any(values < 0.0) or not math.isclose(float(np.sum(values)), 1.0, rel_tol=0.0, abs_tol=1.0e-12):
        raise RuntimeError("sampled composition violates the simplex")
    return {name: float(value) for name, value in zip(("N2", "CO2", "O2", "Ar"), values)}


def _nuisance(plan: Mapping[str, Any], mixture_id: str, sequence_index: int) -> tuple[np.ndarray, dict[str, Any]]:
    rng = _stable_rng(plan, "nuisance_stream", mixture_id, sequence_index)
    bounds = plan["sampling"]["nuisance_bounds"]
    values = {name: float(rng.uniform(float(pair[0]), float(pair[1]))) for name, pair in bounds.items()}
    eta = np.array(
        [
            values["T_K"],
            values["P_kPa"],
            values["RH_frac"],
            values["L_m"],
            values["gain"],
            values["baseline"],
            values["delay_s"],
            values["crosstalk"],
            values["q_flow"],
        ],
        dtype=np.float64,
    )
    record = {
        "T": values["T_K"],
        "P": values["P_kPa"],
        "RH": values["RH_frac"] * 100.0,
        "L": values["L_m"],
        "gain_m": values["gain"],
        "baseline_m": values["baseline"],
        "delay_m": values["delay_s"] * 1000.0,
        "crosstalk_mn": [[0.0, values["crosstalk"]], [values["crosstalk"], 0.0]],
        "q_flow": values["q_flow"],
    }
    return eta, record


def _raw_waveform(
    plan: Mapping[str, Any],
    sequence_id: str,
    observation_values: np.ndarray,
    observation_noise: np.ndarray,
) -> np.ndarray:
    temporal_rng = _stable_rng(plan, "temporal_stream", sequence_id)
    measurement_rng = _stable_rng(plan, "measurement_stream", sequence_id)
    time_length = int(plan["raw"]["time_length"])
    time_axis = np.linspace(0.0, 2.0 * np.pi, time_length, endpoint=False)
    phases = temporal_rng.uniform(0.0, 2.0 * np.pi, size=observation_values.size)
    modulation = 1.0 + 0.01 * np.sin(time_axis[None, :] + phases[:, None])
    noise = measurement_rng.normal(0.0, observation_noise[:, None] * 0.10, size=modulation.shape)
    return observation_values[:, None] * modulation + noise


def _incremental_information(result: Any) -> np.ndarray:
    dtype = np.dtype(
        [
            ("increment_type", "U32"),
            ("delta_I_vector", "<f8", (3,)),
            ("delta_I_trace", "<f8"),
            ("delta_cost", "<f8"),
            ("delta_I_per_delta_cost", "<f8"),
        ]
    )
    records = np.empty(len(result.modality_effective_information), dtype=dtype)
    for index, (modality, vector) in enumerate(result.modality_effective_information.items()):
        values = np.asarray(vector, dtype=np.float64)
        cost = float(result.modality_blocks[modality].stop - result.modality_blocks[modality].start)
        trace = float(np.sum(values))
        records[index] = (modality, values, trace, cost, trace / cost)
    return records


def _sample_arrays(
    plan: Mapping[str, Any],
    cell: Mapping[str, Any],
    composition: Mapping[str, float],
    mixture_id: str,
    sequence_id: str,
    sequence_index: int,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    eta, nuisance_record = _nuisance(plan, mixture_id, sequence_index)
    theta = np.array([composition[name] for name in ("N2", "CO2", "O2")], dtype=np.float64)
    config = AuditConfig(
        candidate_id=str(plan["candidate_id"]),
        modalities=("ndir", "acoustic_raw", "thermal", "slow", "calibration"),
        noise_scale=float(cell["noise_scale"]),
        coupling_strength=float(cell["coupling_strength"]),
    )
    observation = forward_observation(theta, eta, config)
    raw = _raw_waveform(plan, sequence_id, observation.values[:8], observation.noise_std[:8])
    dsp = derive_dsp(raw, plan["dsp"])
    result = analyze_point(theta, eta, config)
    joint = np.column_stack(
        [result.whitened_j_theta, result.whitened_j_eta * ETA_PRIOR_STD[None, :]]
    )
    sample_fisher = joint.T @ joint
    transform = np.vstack([np.eye(3), -np.ones(3)])
    physical_crb = transform @ result.crb @ transform.T
    physical_crb = 0.5 * (physical_crb + physical_crb.T)
    arrays = {
        "raw_waveform": raw,
        "slow_channels": np.repeat(
            np.array([[nuisance_record[name]] for name in ("T", "P", "RH", "q_flow")], dtype=np.float64),
            int(plan["raw"]["time_length"]),
            axis=1,
        ),
        "calibration_channels": np.array(
            [
                nuisance_record["L"],
                nuisance_record["gain_m"],
                nuisance_record["baseline_m"],
                nuisance_record["delay_m"],
                nuisance_record["crosstalk_mn"][0][1],
            ],
            dtype=np.float64,
        ),
        "dsp_features": dsp,
        "labels": np.array([composition[name] for name in ("N2", "CO2", "O2", "Ar")], dtype=np.float64),
        "sample_fisher": sample_fisher,
        "effective_fisher": result.effective_fisher,
        "crb": physical_crb,
        "crb_p90": physical_crb_p90(result),
        "principal_angle": np.asarray(result.minimum_principal_angle_deg, dtype=np.float64),
        "incremental_information": _incremental_information(result),
    }
    return arrays, nuisance_record


def _partitioned_mixtures(plan: Mapping[str, Any], mixture_ids: list[str], split_id: str) -> dict[str, list[str]]:
    ordered = sorted(
        mixture_ids,
        key=lambda mixture_id: hashlib.sha256(
            f"{plan['random_state']['split_stream']}|{split_id}|{mixture_id}".encode("utf-8")
        ).hexdigest(),
    )
    fractions = plan["splits"]["partition_fractions"]
    train_count = math.floor(len(ordered) * float(fractions["train"]))
    val_count = max(1, math.floor(len(ordered) * float(fractions["val"])))
    if len(ordered) - train_count - val_count < 1:
        train_count -= 1
    if train_count < 1 or val_count < 1 or len(ordered) - train_count - val_count < 1:
        raise ContractError("pilot split cannot populate every partition")
    return {
        "train": ordered[:train_count],
        "val": ordered[train_count : train_count + val_count],
        "test": ordered[train_count + val_count :],
    }


def _split_tables(
    plan: Mapping[str, Any],
    mixture_sequences: Mapping[str, list[str]],
    mixture_cells: Mapping[str, str],
) -> tuple[list[dict[str, str]], dict[str, Any]]:
    mixture_ids = sorted(mixture_sequences)
    rows: list[dict[str, str]] = []
    nested: dict[str, Any] = {}
    cell_ids = sorted(set(mixture_cells.values()))
    for split_id in plan["splits"]["split_ids"]:
        if plan["splits"].get("stratify_by") == "grid_cell_id":
            by_cell = {
                cell_id: _partitioned_mixtures(
                    plan,
                    [mixture_id for mixture_id in mixture_ids if mixture_cells[mixture_id] == cell_id],
                    split_id,
                )
                for cell_id in cell_ids
            }
            partitions = {
                partition: [
                    mixture_id
                    for cell_id in cell_ids
                    for mixture_id in by_cell[cell_id][partition]
                ]
                for partition in ("train", "val", "test")
            }
        else:
            partitions = _partitioned_mixtures(plan, mixture_ids, split_id)
        for partition, partition_mixtures in partitions.items():
            for mixture_id in partition_mixtures:
                for sequence_id in mixture_sequences[mixture_id]:
                    rows.append(
                        {
                            "mixture_id": mixture_id,
                            "sequence_id": sequence_id,
                            "split_id": split_id,
                            "partition": partition,
                        }
                    )
        if plan["splits"].get("stratify_by") == "grid_cell_id":
            train_by_cell = {
                cell_id: [mixture_id for mixture_id in partitions["train"] if mixture_cells[mixture_id] == cell_id]
                for cell_id in cell_ids
            }
            train_order = [
                train_by_cell[cell_id][index]
                for index in range(max(len(values) for values in train_by_cell.values()))
                for cell_id in cell_ids
                if index < len(train_by_cell[cell_id])
            ]
        else:
            train_order = partitions["train"]
        prefixes = {}
        for fraction in plan["splits"]["nested_train_fractions"]:
            count = max(1, math.ceil(len(train_order) * int(fraction) / 100.0))
            if plan["splits"].get("nested_rounding") == "ceil_minimum_one_per_cell":
                count = max(count, len(cell_ids))
            prefixes[str(fraction)] = train_order[:count]
        nested[split_id] = {
            "train_group_order": train_order,
            "train_prefixes": prefixes,
            "val_mixture_ids": partitions["val"],
            "test_mixture_ids": partitions["test"],
        }
    validate_split_assignments(rows)
    return rows, nested


def _source_snapshots(config_root: Path) -> list[dict[str, str]]:
    source_path = config_root / "p2_s5_source_registry.json"
    return [
        {
            "source_id": "GIB-P2-S5-SOURCE-REGISTRY",
            "source_revision": "source_complete",
            "sha256": sha256_file(source_path),
            "locator": "configs/p2_s5_source_registry.json",
        }
    ]


def _manifest(
    plan: Mapping[str, Any],
    scope: str,
    files: list[dict[str, str]],
    source_snapshots: list[dict[str, str]],
) -> dict[str, Any]:
    identity = {
        "plan_id": plan["plan_id"],
        "scope": scope,
        "files": files,
    }
    return {
        "manifest_id": make_manifest_id(identity),
        "schema_version": "gib-benchmark-1",
        "primary_key": "mixture_id",
        "instance_key": "sequence_id",
        "split_group_field": "mixture_id",
        "files": files,
        "source_snapshots": source_snapshots,
    }


def _artifact_descriptor(
    layer_name: str,
    file_ref: str,
    dtype: str,
    contracts: Mapping[str, Any],
) -> dict[str, Any]:
    layer = contracts["data"]["array_layers"][layer_name]
    descriptor = {
        "file_ref": file_ref,
        "dtype": dtype,
        "shape_spec": list(layer["axes"]),
        "unit": LAYER_UNITS[layer_name],
        "storage": layer["storage"],
        "derived_from": list(layer["derived_from"]),
    }
    if "provenance_ref" in layer:
        descriptor["provenance_ref"] = layer["provenance_ref"]
    if "record_fields" in layer:
        descriptor["record_fields"] = list(layer["record_fields"])
    return descriptor


def build_pilot_dataset(
    plan: Mapping[str, Any],
    *,
    config_root: Path,
    output_dir: Path,
    dry_run: bool,
    raw_dsp_code_sha256: str,
) -> dict[str, Any]:
    config_root = Path(config_root)
    grid = json.loads((config_root / str(plan["grid_source"])).read_text(encoding="utf-8"))
    validate_pilot_plan(plan, grid)
    target = Path(output_dir)
    if target.exists():
        raise FileExistsError(f"attempt directory already exists: {target}")
    staging = target.parent / f".{target.name}.staging-{uuid4().hex}"
    staging.mkdir(parents=True)
    try:
        scope = "dry_run" if dry_run else "pilot"
        mixtures_per_cell = int(plan[scope]["mixtures_per_cell"])
        sequences_per_mixture = int(plan[scope]["sequences_per_mixture"])
        samples: list[GeneratedSample] = []
        mixture_sequences: dict[str, list[str]] = defaultdict(list)
        mixture_cells: dict[str, str] = {}
        raw_file_entries: list[dict[str, str]] = []
        all_file_entries: list[dict[str, str]] = []
        sample_material: list[tuple[dict[str, Any], dict[str, Any], dict[str, np.ndarray]]] = []

        for cell in grid["cells"]:
            for mixture_index in range(mixtures_per_cell):
                composition = _composition(plan, str(cell["config_id"]), mixture_index)
                mixture_id = make_mixture_id(
                    {"candidate_id": plan["candidate_id"], "composition": composition}
                )
                previous_cell = mixture_cells.setdefault(mixture_id, str(cell["config_id"]))
                if previous_cell != str(cell["config_id"]):
                    raise RuntimeError("mixture_id maps to multiple grid cells")
                for sequence_index in range(sequences_per_mixture):
                    sequence_id = make_sequence_id(
                        mixture_id,
                        sequence_index,
                        str(plan["sequence_profile_id"]),
                    )
                    if sequence_id in mixture_sequences[mixture_id]:
                        raise RuntimeError("duplicate sequence_id generated")
                    mixture_sequences[mixture_id].append(sequence_id)
                    arrays, nuisance_record = _sample_arrays(
                        plan,
                        cell,
                        composition,
                        mixture_id,
                        sequence_id,
                        sequence_index,
                    )
                    metadata = {
                        "mixture_id": mixture_id,
                        "sequence_id": sequence_id,
                        "sequence_index": sequence_index,
                        "composition": composition,
                        "nuisance": nuisance_record,
                        "cell": cell,
                    }
                    sample_material.append((metadata, {}, arrays))

        if len(mixture_sequences) != 9 * mixtures_per_cell:
            raise RuntimeError("mixture_id collision or missing grid cell")

        split_rows, nested = _split_tables(plan, mixture_sequences, mixture_cells)
        split_lookup = {
            (row["split_id"], row["sequence_id"]): row["partition"] for row in split_rows
        }
        contracts = load_contracts()
        source_snapshots = _source_snapshots(config_root)
        raw_layers = {"raw_waveform", "slow_channels", "calibration_channels"}

        artifact_metadata: dict[tuple[str, str], dict[str, Any]] = {}
        for metadata, _, arrays in sample_material:
            suffix = metadata["sequence_id"].removeprefix("GIB-Q-")
            for layer_name, array in arrays.items():
                relative = Path("a") / LAYER_CODES[layer_name] / f"{suffix}.npy"
                written = write_array_artifact(staging / relative, array)
                entry = {
                    "path": relative.as_posix(),
                    "artifact_type": layer_name,
                    "sha256": str(written["sha256"]),
                    "schema_version": "gib-benchmark-1",
                }
                all_file_entries.append(entry)
                if layer_name in raw_layers:
                    raw_file_entries.append(entry)
                artifact_metadata[(metadata["sequence_id"], layer_name)] = {
                    "file_ref": relative.as_posix(),
                    "dtype": str(written["dtype"]),
                    "shape": written["shape"],
                }

        raw_manifest = _manifest(plan, f"{scope}_raw", raw_file_entries, source_snapshots)
        validate_manifest(raw_manifest)
        atomic_write_json(staging / "raw_manifest.json", raw_manifest)
        raw_manifest_hash = sha256_file(staging / "raw_manifest.json")
        full_manifest = _manifest(plan, f"{scope}_dataset", all_file_entries, source_snapshots)
        validate_manifest(full_manifest)
        atomic_write_json(staging / "manifest.json", full_manifest)
        dsp_hash = dsp_config_sha256(plan["dsp"])
        provenance = build_dsp_provenance(
            source_raw_manifest_id=raw_manifest["manifest_id"],
            raw_manifest_sha256=raw_manifest_hash,
            dsp_config_sha256_value=dsp_hash,
            code_sha256=raw_dsp_code_sha256,
        )

        source_ref = {
            "source_type": "controlled_synthetic",
            "source_id": "GIB-P3-PILOT-v1",
            "source_revision": str(plan["plan_id"]),
            "source_hash": sha256_file(config_root / "p2_s5_source_registry.json"),
            "locator": "configs/p2_s5_source_registry.json",
        }
        units = {
            "composition": "mol/mol",
            "temperature": "K",
            "pressure": "kPa",
            "relative_humidity": "%RH",
            "path_length": "m",
            "flow": "L/min",
            "waveform": "declared_per_channel",
            "label": "mol/mol",
        }
        for metadata, _, arrays in sample_material:
            sequence_id = metadata["sequence_id"]
            descriptors = {
                layer_name: _artifact_descriptor(
                    layer_name,
                    artifact_metadata[(sequence_id, layer_name)]["file_ref"],
                    artifact_metadata[(sequence_id, layer_name)]["dtype"],
                    contracts,
                )
                for layer_name in arrays
            }
            cell = metadata["cell"]
            record = {
                "schema_version": "gib-benchmark-1",
                "mixture_id": metadata["mixture_id"],
                "sequence_id": sequence_id,
                "sequence_index": metadata["sequence_index"],
                "sequence_profile_id": plan["sequence_profile_id"],
                "candidate_id": plan["candidate_id"],
                "composition": metadata["composition"],
                "nuisance": metadata["nuisance"],
                "grade": {
                    "grid_id": grid["grid_id"],
                    "grid_cell_id": cell["config_id"],
                    "information_band": cell["information_band"],
                    "angle_band": cell["angle_band"],
                },
                "modality_profile": {
                    "profile_id": "GIB-MOD-RAW-FULL-v1",
                    "enabled_modalities": ["ndir", "acoustic_raw", "thermal", "slow", "calibration"],
                    "raw_dsp_view": "raw",
                },
                "units": units,
                "sources": {category: dict(source_ref) for category in contracts["data"]["source_ref"]["categories"]},
                "split_assignment": {
                    "split_id": "GIB-SPLIT-01",
                    "partition": split_lookup[("GIB-SPLIT-01", sequence_id)],
                },
                "arrays": descriptors,
                "dsp_provenance": dict(provenance),
            }
            validate_sample_record(
                record,
                manifest=full_manifest,
                raw_manifest_sha256=raw_manifest_hash,
                dsp_config_sha256=dsp_hash,
                code_sha256=raw_dsp_code_sha256,
                raw_manifest_id=raw_manifest["manifest_id"],
            )
            deployment = {
                "mixture_id": metadata["mixture_id"],
                "sequence_id": sequence_id,
                "modality_profile": record["modality_profile"],
                "nuisance": metadata["nuisance"],
                "units": units,
                "raw_waveform": descriptors["raw_waveform"]["file_ref"],
                "slow_channels": descriptors["slow_channels"]["file_ref"],
                "calibration_channels": descriptors["calibration_channels"]["file_ref"],
                "dsp_features": descriptors["dsp_features"]["file_ref"],
                "dsp_provenance": dict(provenance),
            }
            oracle = {
                "mixture_id": metadata["mixture_id"],
                "sequence_id": sequence_id,
                "oracle_results": {
                    "labels": descriptors["labels"]["file_ref"],
                    "sample_fisher": descriptors["sample_fisher"]["file_ref"],
                    "effective_fisher": descriptors["effective_fisher"]["file_ref"],
                    "crb": descriptors["crb"]["file_ref"],
                    "crb_p90": descriptors["crb_p90"]["file_ref"],
                    "principal_angle": descriptors["principal_angle"]["file_ref"],
                    "incremental_information": descriptors["incremental_information"]["file_ref"],
                },
                "truth_nuisance": metadata["nuisance"],
            }
            samples.append(GeneratedSample(record, deployment, oracle, arrays))

        atomic_write_jsonl(staging / "sample_records.jsonl", [sample.record for sample in samples])
        atomic_write_jsonl(staging / "deployment" / "records.jsonl", [sample.deployment for sample in samples])
        atomic_write_jsonl(staging / "oracle" / "records.jsonl", [sample.oracle for sample in samples])
        atomic_write_json(staging / "split_assignments.json", split_rows)
        atomic_write_json(staging / "nested_train_groups.json", nested)
        load_deployment_records(staging / "deployment" / "records.jsonl")
        load_oracle_records(staging / "oracle" / "records.jsonl")

        for entry in full_manifest["files"]:
            if sha256_file(staging / entry["path"]) != entry["sha256"]:
                raise RuntimeError(f"artifact hash mismatch after write: {entry['path']}")
        for sample in samples:
            for layer_name, descriptor in sample.record["arrays"].items():
                actual = read_array_artifact(staging / descriptor["file_ref"])
                if list(actual.shape) != artifact_metadata[(sample.record["sequence_id"], layer_name)]["shape"]:
                    raise RuntimeError(f"artifact shape mismatch: {descriptor['file_ref']}")

        summary = {
            "schema_version": "gib-benchmark-1",
            "plan_id": plan["plan_id"],
            "scope": scope,
            "task_status": "completed",
            "pilot_plan_status": "frozen",
            "mixture_count": len(mixture_sequences),
            "sequence_count": len(samples),
            "grid_cell_count": len(grid["cells"]),
            "split_row_count": len(split_rows),
            "artifact_file_count": len(full_manifest["files"]),
            "raw_manifest_id": raw_manifest["manifest_id"],
            "raw_manifest_sha256": raw_manifest_hash,
            "dataset_manifest_id": full_manifest["manifest_id"],
            "dataset_manifest_sha256": sha256_file(staging / "manifest.json"),
            "dsp_config_sha256": dsp_hash,
            "raw_dsp_code_sha256": raw_dsp_code_sha256,
            "claim_scope": "technical_generation_validation_only" if dry_run else "controlled_synthetic_pilot",
            "pilot_integrity": "not_applicable" if dry_run else "pass",
        }
        cell_counts: dict[str, int] = defaultdict(int)
        for sample in samples:
            cell_counts[sample.record["grade"]["grid_cell_id"]] += 1
        summary["sequence_count_by_grid_cell"] = dict(sorted(cell_counts.items()))
        partition_groups: dict[str, dict[str, set[str]]] = {
            split_id: {partition: set() for partition in ("train", "val", "test")}
            for split_id in plan["splits"]["split_ids"]
        }
        for row in split_rows:
            partition_groups[row["split_id"]][row["partition"]].add(row["mixture_id"])
        summary["split_partition_group_counts"] = {
            split_id: {
                partition: len(groups)
                for partition, groups in partitions.items()
            }
            for split_id, partitions in partition_groups.items()
        }
        if plan["splits"].get("stratify_by") == "grid_cell_id":
            cell_partition_counts: dict[str, dict[str, dict[str, set[str]]]] = {
                split_id: {
                    partition: {cell_id: set() for cell_id in sorted(cell_counts)}
                    for partition in ("train", "val", "test")
                }
                for split_id in plan["splits"]["split_ids"]
            }
            for row in split_rows:
                cell_partition_counts[row["split_id"]][row["partition"]][mixture_cells[row["mixture_id"]]].add(row["mixture_id"])
            summary["split_cell_partition_group_counts"] = {
                split_id: {
                    partition: {cell_id: len(groups) for cell_id, groups in cells.items()}
                    for partition, cells in partitions.items()
                }
                for split_id, partitions in cell_partition_counts.items()
            }
            minimum = int(plan["splits"]["minimum_groups_per_partition_per_cell"])
            if any(
                count < minimum
                for partitions in summary["split_cell_partition_group_counts"].values()
                for cells in partitions.values()
                for count in cells.values()
            ):
                raise RuntimeError("split does not cover every grid cell in every partition")
        estimated_bytes = sum((staging / entry["path"]).stat().st_size for entry in full_manifest["files"])
        summary["artifact_bytes"] = estimated_bytes
        if estimated_bytes > int(plan["execution"]["disk_budget_bytes"]):
            raise RuntimeError("generated artifacts exceed the frozen disk budget")
        atomic_write_json(staging / "generation_summary.json", summary)
        atomic_promote_directory(staging, target)
        return summary
    except Exception:
        remove_owned_staging(staging)
        raise


__all__ = ["build_pilot_dataset", "validate_pilot_plan"]
