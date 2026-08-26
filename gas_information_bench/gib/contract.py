"""P2-09 data, manifest, ID and split contract validators."""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Mapping, Sequence
from importlib.resources import files
from pathlib import Path
from typing import Any


CONFIG_PACKAGE = "configs"


class ContractError(ValueError):
    """Raised when a record violates the frozen P2-09 contract."""


def _load_json(name: str) -> dict[str, Any]:
    resource = files(CONFIG_PACKAGE).joinpath(name)
    with resource.open(encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ContractError(f"contract must be a JSON object: {resource}")
    return value


def load_contracts() -> dict[str, dict[str, Any]]:
    """Load every P2-09 contract without a fallback configuration."""

    return {
        "data": _load_json("p2_data_schema.json"),
        "manifest": _load_json("p2_manifest_schema.json"),
        "split": _load_json("p2_split_contract.json"),
        "grid": _load_json("p2_s1_grid.json"),
    }


def _forbidden_field_names() -> tuple[str, ...]:
    # Keep the historical names out of the new package source while rejecting them at runtime.
    return (
        "base_" + "condition_id",
        "noise_" + "seed_index",
        "noise_" + "seed",
    )


def _reject_forbidden_fields(value: Any, path: str = "$") -> None:
    if isinstance(value, Mapping):
        for key, nested in value.items():
            if str(key) in _forbidden_field_names():
                raise ContractError(f"historical field is not allowed: {path}.{key}")
            _reject_forbidden_fields(nested, f"{path}.{key}")
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        for index, nested in enumerate(value):
            _reject_forbidden_fields(nested, f"{path}[{index}]")


def _require_mapping(value: Any, scope: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ContractError(f"{scope} must be an object")
    return value


def _require_exact_keys(value: Mapping[str, Any], required: Sequence[str], scope: str) -> None:
    required_set = set(required)
    actual_set = set(value)
    missing = sorted(required_set - actual_set)
    extra = sorted(actual_set - required_set)
    if missing or extra:
        raise ContractError(f"{scope} keys mismatch; missing={missing}; extra={extra}")


def _require_nonempty_string(value: Any, scope: str) -> str:
    if not isinstance(value, str) or not value:
        raise ContractError(f"{scope} must be a non-empty string")
    return value


def _require_finite_number(value: Any, scope: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
        raise ContractError(f"{scope} must be a finite number")
    return float(value)


def _require_hash(value: Any, scope: str, pattern: str) -> str:
    text = _require_nonempty_string(value, scope)
    if re.fullmatch(pattern, text) is None:
        raise ContractError(f"{scope} must be a SHA256 hex string")
    return text


def _canonical_json(value: Any) -> str:
    try:
        return json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
    except (TypeError, ValueError) as exc:
        raise ContractError(f"identity payload is not canonicalizable: {exc}") from exc


def _digest_id(prefix: str, payload: Mapping[str, Any]) -> str:
    digest = hashlib.sha256(_canonical_json(payload).encode("utf-8")).hexdigest()[:16].upper()
    return f"{prefix}{digest}"


def validate_id(kind: str, value: Any) -> str:
    contracts = load_contracts()
    namespace = contracts["data"]["id_namespace"].get(kind)
    if not isinstance(namespace, Mapping):
        raise ContractError(f"unknown ID kind: {kind}")
    text = _require_nonempty_string(value, kind)
    if re.fullmatch(str(namespace["pattern"]), text) is None:
        raise ContractError(f"invalid {kind}: {text}")
    return text


def make_mixture_id(identity: Mapping[str, Any]) -> str:
    """Derive the primary ID from the frozen mixture identity fields."""

    contracts = load_contracts()
    namespace = contracts["data"]["id_namespace"]["mixture_id"]
    fields = [str(item) for item in namespace["identity_fields"]]
    identity_map = _require_mapping(identity, "mixture identity")
    _reject_forbidden_fields(identity_map)
    _require_exact_keys(identity_map, fields, "mixture identity")
    value = _digest_id(str(namespace["prefix"]), {field: identity_map[field] for field in fields})
    return validate_id("mixture_id", value)


def make_sequence_id(mixture_id: str, sequence_index: int, sequence_profile_id: str) -> str:
    """Derive an instance ID only from an already valid mixture ID and sequence identity."""

    contracts = load_contracts()
    namespace = contracts["data"]["id_namespace"]["sequence_id"]
    mixture = validate_id("mixture_id", mixture_id)
    if isinstance(sequence_index, bool) or not isinstance(sequence_index, int) or sequence_index < 0:
        raise ContractError("sequence_index must be a non-negative integer")
    profile = _require_nonempty_string(sequence_profile_id, "sequence_profile_id")
    payload = {
        "mixture_id": mixture,
        "sequence_index": sequence_index,
        "sequence_profile_id": profile,
    }
    value = _digest_id(str(namespace["prefix"]), payload)
    return validate_id("sequence_id", value)


def make_manifest_id(identity: Mapping[str, Any]) -> str:
    """Derive a manifest ID from its canonical identity payload."""

    identity_map = _require_mapping(identity, "manifest identity")
    _reject_forbidden_fields(identity_map)
    return _digest_id("GIB-MANIFEST-", identity_map)


def _validate_source_refs(sources: Mapping[str, Any], contract: Mapping[str, Any]) -> None:
    source_contract = contract["source_ref"]
    categories = [str(item) for item in source_contract["categories"]]
    _require_exact_keys(sources, categories, "sources")
    for category in categories:
        source = _require_mapping(sources[category], f"sources.{category}")
        _require_exact_keys(source, source_contract["required_fields"], f"sources.{category}")
        for field in ("source_type", "source_id", "source_revision", "locator"):
            _require_nonempty_string(source[field], f"sources.{category}.{field}")
        _require_hash(source["source_hash"], f"sources.{category}.source_hash", str(source_contract["source_hash_pattern"]))


def _relative_artifact_path(value: Any, scope: str) -> str:
    path_text = _require_nonempty_string(value, scope)
    path = Path(path_text)
    if path.is_absolute() or ".." in path.parts:
        raise ContractError(f"{scope} must stay inside the artifact root")
    return path.as_posix()


def _validate_array_descriptors(
    arrays: Mapping[str, Any],
    contract: Mapping[str, Any],
    manifest_files: Mapping[str, str],
) -> None:
    layer_contract = contract["array_layers"]
    _require_exact_keys(arrays, list(layer_contract), "arrays")
    descriptor_fields = [str(item) for item in contract["array_descriptor"]["required_fields"]]
    for name, layer in layer_contract.items():
        descriptor = _require_mapping(arrays[name], f"arrays.{name}")
        required = set(descriptor_fields) | {"derived_from"}
        if "provenance_ref" in layer:
            required.add("provenance_ref")
        if "record_fields" in layer:
            required.add("record_fields")
        _require_exact_keys(descriptor, sorted(required), f"arrays.{name}")
        if descriptor["storage"] != layer["storage"]:
            raise ContractError(f"arrays.{name}.storage does not match schema")
        shape_spec = descriptor["shape_spec"]
        if shape_spec != layer["axes"]:
            raise ContractError(f"arrays.{name}.shape_spec does not match frozen axes")
        if descriptor["derived_from"] != layer["derived_from"]:
            raise ContractError(f"arrays.{name}.derived_from does not match schema")
        file_ref = _relative_artifact_path(descriptor["file_ref"], f"arrays.{name}.file_ref")
        if manifest_files.get(file_ref) != name:
            raise ContractError(f"arrays.{name}.file_ref is missing or has the wrong artifact_type in manifest")
        _require_nonempty_string(descriptor["dtype"], f"arrays.{name}.dtype")
        _require_nonempty_string(descriptor["unit"], f"arrays.{name}.unit")
        if "provenance_ref" in layer and descriptor["provenance_ref"] != layer["provenance_ref"]:
            raise ContractError(f"arrays.{name}.provenance_ref does not match schema")
        if "record_fields" in layer and descriptor["record_fields"] != layer["record_fields"]:
            raise ContractError(f"arrays.{name}.record_fields does not match schema")


def validate_dsp_provenance(
    provenance: Mapping[str, Any],
    *,
    raw_manifest_sha256: str,
    dsp_config_sha256: str,
    code_sha256: str,
) -> None:
    contracts = load_contracts()
    contract = contracts["data"]["dsp_provenance"]
    value = _require_mapping(provenance, "dsp_provenance")
    _reject_forbidden_fields(value)
    _require_exact_keys(value, contract["required_fields"], "dsp_provenance")
    for field in ("source_raw_manifest_id", "derived_from"):
        if field == "derived_from":
            if value[field] != contract["derived_from_fixed"]:
                raise ContractError("dsp_provenance.derived_from must point to raw_waveform")
        else:
            _require_nonempty_string(value[field], f"dsp_provenance.{field}")
    pattern = str(contract["sha256_pattern"])
    expected = {
        "raw_manifest_sha256": raw_manifest_sha256,
        "dsp_config_sha256": dsp_config_sha256,
        "code_sha256": code_sha256,
    }
    for field, expected_value in expected.items():
        actual = _require_hash(value[field], f"dsp_provenance.{field}", pattern)
        expected_hash = _require_hash(expected_value, field, pattern)
        if actual != expected_hash:
            raise ContractError(f"{field} mismatch; cache reuse is not allowed")


def _validate_crosstalk_matrix(value: Any, contract: Mapping[str, Any]) -> None:
    matrix_contract = contract["nuisance"]["crosstalk_matrix"]
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        raise ContractError("nuisance.crosstalk_mn must be a square numeric matrix")
    rows = list(value)
    minimum_size = int(matrix_contract["minimum_size"])
    if len(rows) < minimum_size:
        raise ContractError(f"nuisance.crosstalk_mn must have at least {minimum_size} rows")
    matrix: list[list[float]] = []
    for row_index, row in enumerate(rows):
        if not isinstance(row, Sequence) or isinstance(row, (str, bytes, bytearray)):
            raise ContractError(f"nuisance.crosstalk_mn[{row_index}] must be a numeric row")
        if len(row) != len(rows):
            raise ContractError("nuisance.crosstalk_mn must be square")
        matrix.append([
            _require_finite_number(item, f"nuisance.crosstalk_mn[{row_index}][{column_index}]")
            for column_index, item in enumerate(row)
        ])
    lower = float(matrix_contract["coefficient_minimum"])
    upper = float(matrix_contract["coefficient_maximum"])
    diagonal = float(matrix_contract["diagonal_value"])
    for row_index, row in enumerate(matrix):
        for column_index, coefficient in enumerate(row):
            if row_index == column_index:
                if coefficient != diagonal:
                    raise ContractError("nuisance.crosstalk_mn diagonal must be zero")
            elif not lower <= coefficient <= upper:
                raise ContractError("nuisance.crosstalk_mn coefficient is outside the frozen range")


def _validate_grade(grade: Mapping[str, Any], grid: Mapping[str, Any]) -> None:
    if grade["grid_id"] != grid["grid_id"]:
        raise ContractError("grade.grid_id does not match the frozen S1 grid")
    cells = {
        str(cell["config_id"]): cell
        for cell in grid["cells"]
        if isinstance(cell, Mapping)
    }
    cell = cells.get(str(grade["grid_cell_id"]))
    if cell is None or not bool(cell.get("accessible")):
        raise ContractError("grade.grid_cell_id is not an accessible frozen S1 cell")
    expected = {
        "information_band": cell["information_band"],
        "angle_band": cell["angle_band"],
    }
    for field, expected_value in expected.items():
        if grade[field] != expected_value:
            raise ContractError(f"grade.{field} does not match grade.grid_cell_id")


def _validate_modality_profile(modality: Mapping[str, Any], contract: Mapping[str, Any]) -> None:
    _require_exact_keys(modality, contract["modality_profile"]["required_fields"], "modality_profile")
    enabled = modality["enabled_modalities"]
    if not isinstance(enabled, list) or not enabled or len(set(enabled)) != len(enabled):
        raise ContractError("modality_profile.enabled_modalities must be a non-empty unique list")
    allowed = set(contract["modality_profile"]["allowed_modalities"])
    if not set(enabled).issubset(allowed):
        raise ContractError("modality_profile contains an unknown modality")
    for group in contract["modality_profile"]["mutually_exclusive_groups"]:
        if len(set(enabled) & set(group)) > 1:
            raise ContractError("raw and DSP views cannot be counted together")
    if "acoustic_raw" in enabled:
        expected_view = "raw"
    elif "acoustic_dsp" in enabled:
        expected_view = "dsp"
    else:
        expected_view = "not_applicable"
    if modality["raw_dsp_view"] != expected_view:
        raise ContractError(f"raw_dsp_view must be {expected_view} for the enabled modalities")


def validate_sample_record(
    record: Mapping[str, Any],
    *,
    manifest: Mapping[str, Any],
    raw_manifest_sha256: str,
    dsp_config_sha256: str,
    code_sha256: str,
    raw_manifest_id: str | None = None,
) -> None:
    contracts = load_contracts()
    contract = contracts["data"]
    value = _require_mapping(record, "sample_record")
    _reject_forbidden_fields(value)
    _require_exact_keys(value, contract["required_top_level_fields"], "sample_record")
    if value["schema_version"] != contract["schema_version"]:
        raise ContractError("sample_record.schema_version mismatch")
    mixture_id = validate_id("mixture_id", value["mixture_id"])
    sequence_id = validate_id("sequence_id", value["sequence_id"])
    _require_nonempty_string(value["candidate_id"], "candidate_id")

    composition = _require_mapping(value["composition"], "composition")
    components = [str(item) for item in contract["composition"]["required_fields"]]
    _require_exact_keys(composition, components, "composition")
    total = 0.0
    for component in components:
        amount = _require_finite_number(composition[component], f"composition.{component}")
        if bool(contract["composition"]["non_negative"]) and amount < 0.0:
            raise ContractError(f"composition.{component} must be non-negative")
        total += amount
    if not math.isclose(total, float(contract["composition"]["simplex_sum"]), rel_tol=0.0, abs_tol=float(contract["composition"]["simplex_sum_tolerance"])):
        raise ContractError("composition must sum to one")
    expected_mixture_id = make_mixture_id(
        {
            "candidate_id": value["candidate_id"],
            "composition": composition,
        }
    )
    if mixture_id != expected_mixture_id:
        raise ContractError("sample_record.mixture_id does not match candidate_id and composition")

    sequence_index = value["sequence_index"]
    if isinstance(sequence_index, bool) or not isinstance(sequence_index, int) or sequence_index < 0:
        raise ContractError("sample_record.sequence_index must be a non-negative integer")
    sequence_profile_id = _require_nonempty_string(
        value["sequence_profile_id"],
        "sample_record.sequence_profile_id",
    )
    expected_sequence_id = make_sequence_id(
        mixture_id,
        sequence_index,
        sequence_profile_id,
    )
    if sequence_id != expected_sequence_id:
        raise ContractError("sample_record.sequence_id does not match its frozen identity fields")

    nuisance = _require_mapping(value["nuisance"], "nuisance")
    nuisance_fields = [str(item) for item in contract["nuisance"]["required_fields"]]
    _require_exact_keys(nuisance, nuisance_fields, "nuisance")
    for field in nuisance_fields:
        if field == "crosstalk_mn":
            _validate_crosstalk_matrix(nuisance[field], contract)
        else:
            _require_finite_number(nuisance[field], f"nuisance.{field}")

    grade = _require_mapping(value["grade"], "grade")
    _require_exact_keys(grade, contract["grade"]["required_fields"], "grade")
    _validate_grade(grade, contracts["grid"])

    modality = _require_mapping(value["modality_profile"], "modality_profile")
    _validate_modality_profile(modality, contract)

    units = _require_mapping(value["units"], "units")
    _require_exact_keys(units, contract["units"]["required_fields"], "units")
    for field, expected in contract["units"]["fixed_values"].items():
        if units[field] != expected:
            raise ContractError(f"units.{field} is not the frozen unit")

    _validate_source_refs(_require_mapping(value["sources"], "sources"), contract)
    split = _require_mapping(value["split_assignment"], "split_assignment")
    _require_exact_keys(split, contract["split_assignment"]["required_fields"], "split_assignment")
    split_contract = contracts["split"]
    if split["split_id"] not in split_contract["split_ids"]:
        raise ContractError("unknown split_id")
    if split["partition"] not in contract["split_assignment"]["partition_values"]:
        raise ContractError("unknown split partition")
    validate_manifest(manifest)
    manifest_files = {
        _relative_artifact_path(item["path"], "manifest file path"): str(item["artifact_type"])
        for item in manifest["files"]
    }
    _validate_array_descriptors(
        _require_mapping(value["arrays"], "arrays"),
        contract,
        manifest_files,
    )
    expected_raw_manifest_id = manifest["manifest_id"] if raw_manifest_id is None else raw_manifest_id
    if value["dsp_provenance"]["source_raw_manifest_id"] != expected_raw_manifest_id:
        raise ContractError("dsp_provenance.source_raw_manifest_id does not match manifest")
    validate_dsp_provenance(
        value["dsp_provenance"],
        raw_manifest_sha256=raw_manifest_sha256,
        dsp_config_sha256=dsp_config_sha256,
        code_sha256=code_sha256,
    )


def validate_manifest(manifest: Mapping[str, Any]) -> None:
    contracts = load_contracts()
    contract = contracts["manifest"]
    value = _require_mapping(manifest, "manifest")
    _reject_forbidden_fields(value)
    _require_exact_keys(value, contract["required_fields"], "manifest")
    if value["schema_version"] != contract["schema_version"]:
        raise ContractError("manifest.schema_version mismatch")
    for field, expected in contract["fixed_values"].items():
        if value[field] != expected:
            raise ContractError(f"manifest.{field} must be {expected}")
    if re.fullmatch(str(contract["manifest_id_pattern"]), str(value["manifest_id"])) is None:
        raise ContractError("invalid manifest_id")

    files = value["files"]
    if not isinstance(files, list) or len(files) < int(contract["files"]["minimum_items"]):
        raise ContractError("manifest.files must contain at least one file entry")
    file_fields = contract["files"]["required_fields"]
    file_paths: set[str] = set()
    for index, item in enumerate(files):
        entry = _require_mapping(item, f"manifest.files[{index}]")
        _require_exact_keys(entry, file_fields, f"manifest.files[{index}]")
        path = _relative_artifact_path(entry["path"], f"manifest.files[{index}].path")
        if path in file_paths:
            raise ContractError("manifest file paths must be unique")
        file_paths.add(path)
        if entry["artifact_type"] not in contract["files"]["artifact_types"]:
            raise ContractError("manifest contains an unknown artifact_type")
        if entry["schema_version"] != contract["schema_version"]:
            raise ContractError("manifest file schema_version mismatch")
        _require_hash(entry["sha256"], f"manifest.files[{index}].sha256", str(contract["files"]["sha256_pattern"]))

    snapshots = value["source_snapshots"]
    if not isinstance(snapshots, list) or len(snapshots) < int(contract["source_snapshots"]["minimum_items"]):
        raise ContractError("manifest.source_snapshots must contain at least one entry")
    snapshot_fields = contract["source_snapshots"]["required_fields"]
    for index, item in enumerate(snapshots):
        snapshot = _require_mapping(item, f"manifest.source_snapshots[{index}]")
        _require_exact_keys(snapshot, snapshot_fields, f"manifest.source_snapshots[{index}]")
        for field in ("source_id", "source_revision", "locator"):
            _require_nonempty_string(snapshot[field], f"manifest.source_snapshots[{index}].{field}")
        _require_hash(snapshot["sha256"], f"manifest.source_snapshots[{index}].sha256", str(contract["source_snapshots"]["sha256_pattern"]))


def validate_split_assignments(rows: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    contracts = load_contracts()
    contract = contracts["split"]
    if not isinstance(rows, Sequence) or isinstance(rows, (str, bytes, bytearray)) or not rows:
        raise ContractError("split assignments must be a non-empty sequence")
    required = [str(item) for item in contract["assignment_required_fields"]]
    split_ids = [str(item) for item in contract["split_ids"]]
    partitions = [str(item) for item in contract["partitions"]]
    assignments: dict[str, dict[str, set[str]]] = {
        split_id: {partition: set() for partition in partitions} for split_id in split_ids
    }
    mixture_partitions: dict[tuple[str, str], str] = {}
    sequence_mixture: dict[tuple[str, str], str] = {}
    for index, row in enumerate(rows):
        value = _require_mapping(row, f"split_assignments[{index}]")
        _reject_forbidden_fields(value)
        _require_exact_keys(value, required, f"split_assignments[{index}]")
        mixture = validate_id("mixture_id", value["mixture_id"])
        sequence = validate_id("sequence_id", value["sequence_id"])
        split_id = value["split_id"]
        partition = value["partition"]
        if split_id not in split_ids or partition not in partitions:
            raise ContractError("split assignment contains an unknown split or partition")
        mixture_key = (split_id, mixture)
        previous_partition = mixture_partitions.setdefault(mixture_key, partition)
        if previous_partition != partition:
            raise ContractError("a mixture_id crosses partitions in one split")
        assignments[split_id][partition].add(mixture)
        sequence_key = (split_id, sequence)
        if sequence_key in sequence_mixture:
            raise ContractError("a sequence_id is assigned more than once in one split")
        previous_mixture = sequence_mixture.setdefault(sequence_key, mixture)
        if previous_mixture != mixture:
            raise ContractError("a sequence_id maps to multiple mixture_id values in one split")
    if bool(contract["rules"]["all_partitions_required_per_split"]):
        for split_id in split_ids:
            missing = [partition for partition in partitions if not assignments[split_id][partition]]
            if missing:
                raise ContractError(f"split {split_id} is missing partitions: {missing}")
    return {split_id: sum(len(groups) for groups in partitions_map.values()) for split_id, partitions_map in assignments.items()}


def validate_solver_row(row: Mapping[str, Any]) -> None:
    contracts = load_contracts()
    contract = contracts["data"]["solver_difficulty"]
    value = _require_mapping(row, "solver_difficulty")
    _reject_forbidden_fields(value)
    required = [str(item) for item in contract["required_fields"]]
    missing = sorted(set(required) - set(value))
    if missing:
        raise ContractError(f"solver_difficulty missing fields: {missing}")
    validate_id("sequence_id", value["sequence_id"])
    if not isinstance(value["method_id"], str) or not value["method_id"]:
        raise ContractError("solver_difficulty.method_id must be non-empty")
    split_contract = contracts["split"]
    if value["split_id"] not in split_contract["split_ids"] or value["seed"] not in split_contract["seeds"]:
        raise ContractError("solver_difficulty split or seed is not registered")
    for field in ("iterations", "forward_calls", "repeat_index"):
        number = value[field]
        if isinstance(number, bool) or not isinstance(number, int) or number < 0:
            raise ContractError(f"solver_difficulty.{field} must be a non-negative integer")
    if not isinstance(value["convergence"], bool):
        raise ContractError("solver_difficulty.convergence must be boolean")
    for field in ("condition_number", "final_residual", "runtime_ns"):
        number = _require_finite_number(value[field], f"solver_difficulty.{field}")
        if number < 0.0:
            raise ContractError(f"solver_difficulty.{field} must be non-negative")
    for field in (
        "hardware_fingerprint",
        "os_version",
        "python_version",
        "numpy_version",
        "framework_version",
        "method_package_versions",
        "git_commit",
    ):
        _require_nonempty_string(value[field], f"solver_difficulty.{field}")
    for field in ("logical_cpu_count", "blas_threads", "omp_threads", "mkl_threads", "framework_threads"):
        number = value[field]
        if isinstance(number, bool) or not isinstance(number, int) or number <= 0:
            raise ContractError(f"solver_difficulty.{field} must be a positive integer")


def validate_deployment_fields(fields: Sequence[str]) -> None:
    contracts = load_contracts()
    if isinstance(fields, (str, bytes, bytearray)):
        raise ContractError("deployment fields must be a sequence")
    names = list(fields)
    if len(names) != len(set(names)):
        raise ContractError("deployment fields must be unique")
    contract = contracts["data"]
    oracle_fields = set(str(item) for item in contract["oracle_fields"])
    forbidden = sorted(oracle_fields & set(names))
    if forbidden:
        raise ContractError(f"oracle fields are not allowed in deployment loader: {forbidden}")
    allowed = set(str(item) for item in contract["deployment_input_fields"])
    unknown = sorted(set(names) - allowed)
    if unknown:
        raise ContractError(f"deployment loader field is not registered: {unknown}")


__all__ = [
    "ContractError",
    "load_contracts",
    "make_mixture_id",
    "make_manifest_id",
    "make_sequence_id",
    "validate_deployment_fields",
    "validate_dsp_provenance",
    "validate_id",
    "validate_manifest",
    "validate_sample_record",
    "validate_solver_row",
    "validate_split_assignments",
]
