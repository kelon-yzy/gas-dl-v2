"""F0 load and audit for bidirectional ultrasound parameter registry."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from tv3.sim.core.tunnel_ventilation_bidir_schema import (
    BIDIR_ORACLE_ARRAYS,
    BIDIR_REQUIRED_ARRAYS,
    CONDITION_GRID_FLOW_FIELDS,
    FORMAL_FEATURE_BUILDER,
    SCHEMA_VERSION,
    SIM_REVISION_TAG,
    SLOW_CHANNELS,
    SOURCE_TAGS,
)
from tv3.sim.generation.tunnel_ventilation.flow_physics import MAX_PAIR_INTERVAL_S

# bidir_registry.py → .../tv3/sim/generation/tunnel_ventilation/
# parents[4] = tunnel_ventilation package root (configs/ lives there)
_DEFAULT_CONFIG_DIR = Path(__file__).resolve().parents[4] / "configs" / "tv3_bidir"
_REGISTRY_NAME = "parameter_registry.json"


def default_config_dir() -> Path:
    return _DEFAULT_CONFIG_DIR


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def load_json_registry(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"registry not found: {path}")
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"registry root must be object: {path}")
    return data


def load_f0_registry(config_dir: Path | None = None) -> dict[str, Any]:
    root = Path(config_dir) if config_dir is not None else default_config_dir()
    path = root / _REGISTRY_NAME
    registry = load_json_registry(path)
    return {
        "dir": str(root),
        "path": str(path),
        "registry": registry,
        "sha256": sha256_file(path),
    }


def _require_source(name: str, spec: dict[str, Any], issues: list[str]) -> None:
    source = spec.get("source")
    if source is None:
        issues.append(f"{name}: missing source")
        return
    if source not in SOURCE_TAGS:
        issues.append(f"{name}: invalid source {source!r}")


def _audit_parameters(registry: dict[str, Any], issues: list[str]) -> None:
    params = registry.get("parameters")
    if not isinstance(params, dict) or not params:
        issues.append("parameters: must be non-empty object")
        return
    for name, spec in params.items():
        if not isinstance(spec, dict):
            issues.append(f"parameters.{name}: must be object")
            continue
        _require_source(f"parameters.{name}", spec, issues)


def _audit_topology(registry: dict[str, Any], issues: list[str]) -> None:
    topology = registry.get("topology")
    if not isinstance(topology, dict):
        issues.append("topology: missing object")
        return
    _require_source("topology", topology, issues)
    pair = topology.get("pair_interval_s")
    if not isinstance(pair, dict):
        issues.append("topology.pair_interval_s: missing")
    else:
        _require_source("topology.pair_interval_s", pair, issues)
        value = pair.get("value")
        if not isinstance(value, (int, float)) or value <= 0 or value > MAX_PAIR_INTERVAL_S:
            issues.append(f"topology.pair_interval_s.value: must be in (0, {MAX_PAIR_INTERVAL_S}]")


def _audit_jitter(registry: dict[str, Any], issues: list[str]) -> dict[str, Any]:
    block = registry.get("trigger_jitter_scenarios")
    summary: dict[str, Any] = {
        "has_conservative": False,
        "has_nominal_literature_bound": False,
        "scenario_ids": [],
    }
    if not isinstance(block, dict):
        issues.append("trigger_jitter_scenarios: missing object")
        return summary
    if not block.get("parallel_reporting_required", False):
        issues.append("trigger_jitter_scenarios.parallel_reporting_required: must be true")
    scenarios = block.get("scenarios")
    if not isinstance(scenarios, list) or len(scenarios) < 2:
        issues.append("trigger_jitter_scenarios.scenarios: need >= 2 scenarios")
        return summary
    for sc in scenarios:
        if not isinstance(sc, dict):
            issues.append("trigger_jitter_scenarios.scenarios: entry must be object")
            continue
        sid = sc.get("id")
        summary["scenario_ids"].append(sid)
        _require_source(f"trigger_jitter_scenarios.{sid}", sc, issues)
        std = sc.get("std_s")
        if not isinstance(std, (int, float)) or std <= 0:
            issues.append(f"trigger_jitter_scenarios.{sid}: std_s must be > 0")
        if sid == "conservative_v1":
            summary["has_conservative"] = True
            if abs(float(std) - 3.0e-6) > 1e-12:
                issues.append("conservative_v1: std_s must be 3.0e-6")
            if sc.get("source") != "engineering_scenario":
                issues.append("conservative_v1: source must be engineering_scenario")
        if sid == "nominal_daq_half_sample":
            summary["has_nominal_literature_bound"] = sc.get("source") == "literature_bound"
            if sc.get("source") != "literature_bound":
                issues.append("nominal_daq_half_sample: source must be literature_bound")
            if abs(float(std) - 5.0e-7) > 1e-12:
                issues.append("nominal_daq_half_sample: std_s must be 5.0e-7")
            derivation = sc.get("derivation")
            if not isinstance(derivation, dict):
                issues.append("nominal_daq_half_sample: missing derivation")
            else:
                sample_period = derivation.get("sample_period_s")
                if not isinstance(sample_period, (int, float)) or abs(float(sample_period) - 1.0e-6) > 1e-12:
                    issues.append("nominal_daq_half_sample.derivation.sample_period_s must be 1.0e-6")
    if not summary["has_conservative"]:
        issues.append("missing conservative_v1 scenario")
    if not summary["has_nominal_literature_bound"]:
        issues.append("missing nominal literature_bound scenario")
    hardware = block.get("hardware_anchor")
    if not isinstance(hardware, dict):
        issues.append("trigger_jitter_scenarios.hardware_anchor: missing")
    else:
        if hardware.get("sample_rate_hz") != 1000000:
            issues.append("hardware_anchor.sample_rate_hz must be 1000000")
        if abs(float(hardware.get("timing_resolution_s", -1)) - 1.0e-8) > 1e-15:
            issues.append("hardware_anchor.timing_resolution_s must be 1.0e-8")
    return summary


def _audit_schema_draft(registry: dict[str, Any], issues: list[str]) -> None:
    draft = registry.get("schema_draft")
    if not isinstance(draft, dict):
        issues.append("schema_draft: missing object")
        return
    if draft.get("schema") != SCHEMA_VERSION:
        issues.append(f"schema_draft.schema must be {SCHEMA_VERSION}")
    if draft.get("builder_formal") != FORMAL_FEATURE_BUILDER:
        issues.append(f"schema_draft.builder_formal must be {FORMAL_FEATURE_BUILDER}")
    extra = draft.get("condition_grid_extra_fields")
    if list(extra or []) != list(CONDITION_GRID_FLOW_FIELDS):
        issues.append("schema_draft.condition_grid_extra_fields mismatch with schema module")
    required = set(draft.get("required_array_stems") or [])
    if not set(BIDIR_REQUIRED_ARRAYS).issubset(required):
        missing = sorted(set(BIDIR_REQUIRED_ARRAYS) - required)
        issues.append(f"schema_draft.required_array_stems missing: {missing}")
    oracle = set(draft.get("oracle_only_arrays") or [])
    if set(BIDIR_ORACLE_ARRAYS) != oracle:
        issues.append("schema_draft.oracle_only_arrays mismatch with schema module")
    if registry.get("composition_anchor", {}).get("flow_in_slow_channels") is not False:
        issues.append("composition_anchor.flow_in_slow_channels must be false")
    slow = registry.get("composition_anchor", {}).get("slow_channels")
    if tuple(slow or ()) != SLOW_CHANNELS:
        issues.append("composition_anchor.slow_channels must match base SLOW_CHANNELS")


_FORBIDDEN_MUTABLE_REGISTRY_KEYS = frozenset(
    {
        "allowed_next_stage",
        "f1_status",
        "f2_status",
        "f3_status",
        "stage_status",
    }
)


def audit_f0_gate(config_dir: Path | None = None) -> dict[str, Any]:
    """Audit F0 registry completeness. Pass → F1 allowed; fail → inconclusive.

    Registry must remain frozen parameter evidence only. Stage progress lives in
    ``configs/tv3_bidir/stage_status.json`` or ``outputs/tv3_bidir/f*_verdict.json``.
    """
    bundle = load_f0_registry(config_dir)
    registry = bundle["registry"]
    issues: list[str] = []

    if registry.get("schema_version") != SCHEMA_VERSION:
        issues.append(f"schema_version must be {SCHEMA_VERSION}")
    if registry.get("sim_revision_tag") != SIM_REVISION_TAG:
        issues.append(f"sim_revision_tag must be {SIM_REVISION_TAG}")
    if registry.get("stage") != "F0":
        issues.append("stage must be F0")

    for key in _FORBIDDEN_MUTABLE_REGISTRY_KEYS:
        if key in registry:
            issues.append(
                f"registry must not contain mutable stage key {key!r}; "
                "use configs/tv3_bidir/stage_status.json"
            )

    params = registry.get("parameters")
    if isinstance(params, dict):
        for name, spec in params.items():
            if isinstance(spec, dict) and "f1_status" in spec:
                issues.append(f"parameters.{name}.f1_status must not live in frozen registry")
    prop = registry.get("propagation_model")
    if isinstance(prop, dict) and "f1_status" in prop:
        issues.append("propagation_model.f1_status must not live in frozen registry")

    _audit_parameters(registry, issues)
    _audit_topology(registry, issues)
    jitter_summary = _audit_jitter(registry, issues)
    _audit_schema_draft(registry, issues)

    if not isinstance(prop, dict):
        issues.append("propagation_model: missing")
    else:
        _require_source("propagation_model", prop, issues)

    blocking = list(registry.get("blocking_items_f0") or [])
    if blocking:
        issues.append(f"blocking_items_f0 not empty: {blocking}")

    passed = len(issues) == 0
    gate = registry.get("f0_gate") or {}
    verdict = gate.get("pass_verdict", "f0_registry_frozen") if passed else gate.get(
        "fail_verdict", "inconclusive_parameter_bounds"
    )
    allowed_next = gate.get("allowed_next_stage_on_pass") if passed else None
    return {
        "schema_version": registry.get("schema_version"),
        "stage": "F0",
        "passed": passed,
        "verdict": verdict,
        "allowed_next_stage": allowed_next,
        "issues": issues,
        "jitter_scenarios": jitter_summary,
        "registry_sha256": bundle["sha256"],
        "registry_path": bundle["path"],
        "claim_scope": registry.get("claim_scope", "registered_simulation_domain_only"),
    }
