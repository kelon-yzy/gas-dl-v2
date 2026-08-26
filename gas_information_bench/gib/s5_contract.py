"""P2-10 source registry and model-discrepancy interface contract."""

from __future__ import annotations

import json
from collections.abc import Mapping
from importlib.resources import files
from typing import Any

import numpy as np


class S5ContractError(ValueError):
    """Raised when an S5 registry or interface contract is malformed."""


class DiscrepancyUnavailable(S5ContractError):
    """Raised when a reserved P5 discrepancy profile is requested in P2."""


_CONFIG_PACKAGE = "configs"
_SOURCE_REGISTRY_NAME = "p2_s5_source_registry.json"
_DISCREPANCY_CONTRACT_NAME = "p2_s5_discrepancy_contract.json"
_SOURCE_TYPES = {"peer_reviewed", "device_manual", "engineering_assumption"}
_VERDICTS = {"source_complete", "blocked_source_missing"}


def _read_json(name: str) -> dict[str, Any]:
    resource = files(_CONFIG_PACKAGE).joinpath(name)
    if not resource.is_file():
        raise S5ContractError(f"missing S5 contract file: {resource}")
    try:
        payload = json.loads(resource.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise S5ContractError(f"cannot read S5 contract file: {resource}") from exc
    if not isinstance(payload, dict):
        raise S5ContractError(f"S5 contract root must be an object: {resource}")
    return payload


def load_s5_contracts() -> tuple[dict[str, Any], dict[str, Any]]:
    """Load the source registry and discrepancy contract without fallback."""

    return _read_json(_SOURCE_REGISTRY_NAME), _read_json(_DISCREPANCY_CONTRACT_NAME)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise S5ContractError(message)


def validate_source_registry(registry: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Validate source classes, inventory coverage and explicit blockers."""

    payload = dict(registry) if registry is not None else load_s5_contracts()[0]
    _require(payload.get("registry_id") == "GIB-S5-SOURCE-v1", "unexpected S5 source registry id")
    _require(payload.get("task_id") == "P2-10", "source registry task_id must be P2-10")
    _require(payload.get("schema_version") == "gib-benchmark-1", "unexpected S5 schema version")
    verdict = payload.get("verdict")
    _require(verdict in _VERDICTS, "source registry verdict is invalid")
    scope = payload.get("scope")
    _require(isinstance(scope, dict), "source registry scope is missing")
    _require(scope.get("benchmark_mode") == "controlled_synthetic", "S5 benchmark mode must be controlled_synthetic")
    _require(scope.get("real_hardware_fidelity_claim_allowed") is False, "synthetic S5 cannot claim hardware fidelity")
    _require(scope.get("relative_algorithm_comparison_allowed") is True, "synthetic S5 must declare its comparison purpose")
    source_classes = payload.get("source_classes")
    _require(set(source_classes or ()) == _SOURCE_TYPES, "source classes must include the three frozen classes")

    catalog = payload.get("source_catalog")
    _require(isinstance(catalog, list) and catalog, "source catalog must be a non-empty list")
    source_ids: set[str] = set()
    for source in catalog:
        _require(isinstance(source, dict), "each source catalog entry must be an object")
        source_id = source.get("source_id")
        _require(isinstance(source_id, str) and source_id not in source_ids, "source ids must be unique strings")
        source_ids.add(source_id)
        _require(source.get("source_type") in _SOURCE_TYPES, f"invalid source type for {source_id}")
        for field in ("title", "path", "locator", "citation", "scope", "source_status"):
            _require(isinstance(source.get(field), str) and source[field], f"missing {field} for {source_id}")
        source_hash = source.get("source_hash_sha256")
        _require(source_hash is None or (isinstance(source_hash, str) and len(source_hash) == 64), f"invalid source hash for {source_id}")

    inventory = payload.get("forward_inventory")
    _require(isinstance(inventory, list) and inventory and all(isinstance(item, str) for item in inventory), "forward inventory is invalid")
    inventory_ids = set(inventory)
    _require(len(inventory_ids) == len(inventory), "forward inventory ids must be unique")
    entries = payload.get("entries")
    _require(isinstance(entries, list), "source registry entries must be a list")
    entry_ids: set[str] = set()
    for entry in entries:
        _require(isinstance(entry, dict), "each source registry entry must be an object")
        inventory_id = entry.get("inventory_id")
        _require(isinstance(inventory_id, str) and inventory_id not in entry_ids, "inventory entry ids must be unique strings")
        entry_ids.add(inventory_id)
        _require(inventory_id in inventory_ids, f"entry is not declared in forward_inventory: {inventory_id}")
        _require(entry.get("source_type") in _SOURCE_TYPES, f"invalid entry source type for {inventory_id}")
        for field in ("category", "citation", "locator", "scope", "verification_status", "unit"):
            _require(isinstance(entry.get(field), str) and entry[field], f"missing {field} for {inventory_id}")
        referenced = entry.get("source_ids")
        _require(isinstance(referenced, list) and referenced, f"source references missing for {inventory_id}")
        _require(set(referenced).issubset(source_ids), f"unknown source reference for {inventory_id}")
        if entry["source_type"] == "engineering_assumption":
            _require(entry["verification_status"] != "verified", f"engineering assumption cannot be verified: {inventory_id}")
    _require(entry_ids == inventory_ids, "every forward inventory item must have exactly one source entry")

    missing = payload.get("missing_key_sources")
    _require(isinstance(missing, list), "missing_key_sources must be a list")
    for blocker in missing:
        _require(isinstance(blocker, dict), "each missing source blocker must be an object")
        _require(isinstance(blocker.get("parameter_id"), str) and blocker["parameter_id"], "missing blocker parameter_id")
        _require(isinstance(blocker.get("reason"), str) and blocker["reason"], "missing blocker reason")
        _require(isinstance(blocker.get("required_action"), str) and blocker["required_action"], "missing blocker action")
    if verdict == "blocked_source_missing":
        _require(bool(missing), "blocked_source_missing requires explicit missing source blockers")
    else:
        _require(not missing, "source_complete cannot retain missing source blockers")

    policy = payload.get("policy")
    _require(isinstance(policy, dict), "source registry policy is missing")
    _require(policy.get("engineering_assumptions_not_verified") is True, "engineering-assumption policy must be explicit")
    _require(policy.get("missing_source_blocks") is True, "missing-source blocking policy must be explicit")
    _require(policy.get("no_neighbor_substitution") is True, "neighbor substitution policy must be explicit")
    _require(policy.get("benchmark_mode") == "controlled_synthetic", "policy benchmark mode must be controlled_synthetic")
    _require(policy.get("cross_method_profile_identity_required") is True, "paired methods must share one synthetic profile")
    _require(policy.get("hardware_performance_claim_allowed") is False, "synthetic benchmark cannot claim hardware performance")
    _require(policy.get("unmodeled_hardware_fields_must_be_explicit_null") is True, "unmodeled hardware fields must remain explicit nulls")
    _require(policy.get("real_hardware_validation_stage") == "P5", "real hardware validation must remain in P5")
    _require(policy.get("discrepancy_allowed_stage") == "P5_only", "discrepancy stage must be P5_only")
    _require(policy.get("p2_self_injection_allowed") is False, "P2 discrepancy self-injection must be disabled")
    return {
        "registry_id": payload["registry_id"],
        "verdict": verdict,
        "inventory_count": len(inventory),
        "source_count": len(catalog),
        "missing_key_source_count": len(missing),
    }


def validate_discrepancy_contract(contract: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Validate the frozen signature, profiles and unit conversion table."""

    payload = dict(contract) if contract is not None else load_s5_contracts()[1]
    _require(payload.get("contract_id") == "GIB-S5-DISCREPANCY-v1", "unexpected discrepancy contract id")
    _require(payload.get("task_id") == "P2-10", "discrepancy contract task_id must be P2-10")
    _require(payload.get("schema_version") == "gib-benchmark-1", "unexpected discrepancy schema version")
    _require(payload.get("signature") == "delta(observation, condition, modality, discrepancy_profile)", "discrepancy signature is not frozen")
    _require(payload.get("default_profile") == "off", "discrepancy default profile must be off")
    fields = payload.get("condition_fields")
    _require(isinstance(fields, dict) and fields, "condition fields are missing")
    _require(all(isinstance(spec, dict) and spec.get("required") is True for spec in fields.values()), "all condition fields must be required")
    modalities = payload.get("modalities")
    _require(isinstance(modalities, list) and modalities and len(set(modalities)) == len(modalities), "modalities are invalid")
    profiles = payload.get("profiles")
    _require(isinstance(profiles, list), "discrepancy profiles must be a list")
    profile_ids = {profile.get("profile_id") for profile in profiles if isinstance(profile, dict)}
    _require({"off", "p5_reserved"}.issubset(profile_ids), "off and p5_reserved profiles are required")
    off = next(profile for profile in profiles if profile.get("profile_id") == "off")
    reserved = next(profile for profile in profiles if profile.get("profile_id") == "p5_reserved")
    _require(off.get("enabled") is False and off.get("p2_allowed") is True, "off profile must be disabled and P2-allowed")
    _require(reserved.get("allowed_stage") == "P5" and reserved.get("p2_allowed") is False, "reserved profile must be P5-only")
    conversions = payload.get("unit_conversions")
    _require(isinstance(conversions, list) and conversions, "unit conversion table is missing")
    conversion_keys: set[tuple[str, str]] = set()
    for conversion in conversions:
        _require(isinstance(conversion, dict), "unit conversion must be an object")
        key = (conversion.get("from"), conversion.get("to"))
        _require(all(isinstance(item, str) and item for item in key), "unit conversion units are invalid")
        _require(key not in conversion_keys, f"duplicate unit conversion: {key}")
        conversion_keys.add(key)
        _require(np.isfinite(float(conversion.get("scale"))), f"invalid scale for conversion: {key}")
        _require(np.isfinite(float(conversion.get("offset"))), f"invalid offset for conversion: {key}")
    policy = payload.get("policy")
    _require(isinstance(policy, dict), "discrepancy policy is missing")
    _require(policy.get("p2_self_injection_allowed") is False, "P2 self-injection must be disabled")
    _require(policy.get("p5_only") is True, "discrepancy must be P5-only")
    _require(policy.get("unknown_profile") == "fail", "unknown profile policy must fail")
    _require(policy.get("missing_profile") == "fail", "missing profile policy must fail")
    _require(policy.get("unsupported_unit_conversion") == "fail", "unsupported conversion policy must fail")
    return {
        "contract_id": payload["contract_id"],
        "default_profile": payload["default_profile"],
        "profile_ids": sorted(profile_ids),
        "conversion_count": len(conversions),
    }


def _validate_condition(condition: Mapping[str, Any], contract: Mapping[str, Any]) -> None:
    fields = contract["condition_fields"]
    for field, spec in fields.items():
        if spec.get("required") and field not in condition:
            raise S5ContractError(f"missing discrepancy condition field: {field}")
        value = condition[field]
        try:
            numeric = float(value)
        except (TypeError, ValueError) as exc:
            raise S5ContractError(f"condition field is not numeric: {field}") from exc
        if not np.isfinite(numeric):
            raise S5ContractError(f"condition field is not finite: {field}")


def convert_unit(value: Any, from_unit: str, to_unit: str) -> float | np.ndarray:
    """Apply only an explicitly registered affine unit conversion."""

    if not isinstance(from_unit, str) or not isinstance(to_unit, str):
        raise S5ContractError("unit names must be strings")
    contract = load_s5_contracts()[1]
    array = np.array(value, dtype=float, copy=True)
    if not np.all(np.isfinite(array)):
        raise S5ContractError("unit conversion input must be finite")
    if from_unit == to_unit:
        converted = array
    else:
        match = next(
            (
                item
                for item in contract["unit_conversions"]
                if item["from"] == from_unit and item["to"] == to_unit
            ),
            None,
        )
        if match is None:
            raise S5ContractError(f"unsupported unit conversion: {from_unit} -> {to_unit}")
        converted = array * float(match["scale"]) + float(match["offset"])
    if converted.ndim == 0:
        return float(converted)
    return converted


def delta(
    observation: Any,
    condition: Mapping[str, Any],
    modality: str,
    discrepancy_profile: str = "off",
) -> np.ndarray:
    """Apply the S5 discrepancy interface; only ``off`` is executable in P2."""

    if not isinstance(condition, Mapping):
        raise S5ContractError("condition must be a mapping")
    contract = load_s5_contracts()[1]
    validate_discrepancy_contract(contract)
    _validate_condition(condition, contract)
    if modality not in contract["modalities"]:
        raise S5ContractError(f"unknown modality: {modality}")
    if not isinstance(discrepancy_profile, str):
        raise S5ContractError("discrepancy_profile must be an explicit string")
    array = np.array(observation, dtype=float, copy=True)
    if not np.all(np.isfinite(array)):
        raise S5ContractError("observation must be finite")
    if discrepancy_profile == "off":
        return array
    if discrepancy_profile == "p5_reserved":
        raise DiscrepancyUnavailable("p5_reserved discrepancy is not executable during P2")
    raise S5ContractError(f"unknown discrepancy profile: {discrepancy_profile}")


__all__ = [
    "DiscrepancyUnavailable",
    "S5ContractError",
    "convert_unit",
    "delta",
    "load_s5_contracts",
    "validate_discrepancy_contract",
    "validate_source_registry",
]
