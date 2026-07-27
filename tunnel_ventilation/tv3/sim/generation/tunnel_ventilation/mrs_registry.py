"""MRS-0 load and audit for multifreq relaxation spectroscopy parameter registry."""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any

SOURCE_TAGS = frozenset(
    {
        "implemented_physics",
        "literature_bound",
        "engineering_scenario",
        "not_represented",
    }
)

SCHEMA_VERSION = "tunnel-ventilation-mrs-1"
SIM_REVISION_TAG = "v8-mrs-dispersion-v1"
STAGE = "MRS-0"
EXPECTED_F_HZ = (
    10000.0,
    16000.0,
    25000.0,
    40000.0,
    63000.0,
    100000.0,
    160000.0,
    200000.0,
)
EXPECTED_COMPONENT_FIELDS = ("x_CO2", "x_O2", "x_N2")
EXPECTED_SLOW_CHANNELS = (
    "V_NDIR_CO2",
    "V_TCS",
    "T_C",
    "P_MPa",
    "H_RH",
    "L_m",
    "piston_position_m",
)

_DEFAULT_CONFIG_DIR = Path(__file__).resolve().parents[4] / "configs" / "tv3_mrs"
_REGISTRY_NAME = "parameter_registry.json"
_FORBIDDEN_MUTABLE_REGISTRY_KEYS = frozenset(
    {
        "allowed_next_stage",
        "stage_status",
        "mrs1_status",
        "mrs2_status",
    }
)

# Top-level blocks that must carry a source tag (Occam: explicit list, no deep walk).
_SOURCED_TOP_LEVEL = (
    "composition_anchor",
    "sampling_domain",
    "bass_relaxation_frequencies",
    "vibrational_heat_capacity",
    "frequency_set",
    "burst_schedule",
    "observation_noise",
    "rh_modulation_arm",
    "pressure_scan_arm",
    "low_frequency_diffraction",
    "dispersion_forward_model",
    "storage_policy",
    "mrs2_gates",
    "mrs5_protocol_draft",
    "schema_draft",
    "doc_errata_checklist",
)


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


def load_mrs0_registry(config_dir: Path | None = None) -> dict[str, Any]:
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


def _audit_identity(registry: dict[str, Any], issues: list[str]) -> None:
    if registry.get("schema_version") != SCHEMA_VERSION:
        issues.append(f"schema_version must be {SCHEMA_VERSION}")
    if registry.get("sim_revision_tag") != SIM_REVISION_TAG:
        issues.append(f"sim_revision_tag must be {SIM_REVISION_TAG}")
    if registry.get("stage") != STAGE:
        issues.append(f"stage must be {STAGE}")
    if registry.get("claim_scope") != "registered_simulation_domain_only":
        issues.append("claim_scope must be registered_simulation_domain_only")
    for key in _FORBIDDEN_MUTABLE_REGISTRY_KEYS:
        if key in registry:
            issues.append(
                f"registry must not contain mutable stage key {key!r}; "
                "use configs/tv3_mrs/stage_status.json"
            )


def _audit_sourced_blocks(registry: dict[str, Any], issues: list[str]) -> None:
    for name in _SOURCED_TOP_LEVEL:
        block = registry.get(name)
        if not isinstance(block, dict):
            issues.append(f"{name}: missing object")
            continue
        _require_source(name, block, issues)


def _audit_composition(registry: dict[str, Any], issues: list[str]) -> None:
    anchor = registry.get("composition_anchor")
    if not isinstance(anchor, dict):
        return
    if tuple(anchor.get("component_fields") or ()) != EXPECTED_COMPONENT_FIELDS:
        issues.append("composition_anchor.component_fields must match tv3 base schema")
    if tuple(anchor.get("slow_channels") or ()) != EXPECTED_SLOW_CHANNELS:
        issues.append("composition_anchor.slow_channels must match tv3 base SLOW_CHANNELS")
    if anchor.get("flow_in_slow_channels") is not False:
        issues.append("composition_anchor.flow_in_slow_channels must be false")
    if abs(float(anchor.get("flow_baseline_m_per_s", 1.0)) - 0.0) > 1e-15:
        issues.append("composition_anchor.flow_baseline_m_per_s must be 0.0")


def _bass_fr_o2(h: float, p_atm: float = 1.0) -> float:
    return p_atm * (24.0 + 40400.0 * h * (0.02 + h) / (0.391 + h))


def _bass_fr_n2(h: float, t_c: float, p_atm: float = 1.0, t0: float = 293.15) -> float:
    t_k = t_c + 273.15
    tr = t_k / t0
    return p_atm * (tr ** -0.5) * (9.0 + 280.0 * h * math.exp(-4.17 * (tr ** (-1.0 / 3.0) - 1.0)))


def _audit_bass(registry: dict[str, Any], issues: list[str]) -> None:
    block = registry.get("bass_relaxation_frequencies")
    if not isinstance(block, dict):
        return
    refs = set(block.get("refs") or [])
    if "doi:10.1121/1.400176" not in refs:
        issues.append("bass_relaxation_frequencies.refs must include doi:10.1121/1.400176")
    if "doi:10.1121/1.412989" not in refs:
        issues.append("bass_relaxation_frequencies.refs must include doi:10.1121/1.412989")

    o2 = block.get("f_r_o2_hz_per_atm")
    n2 = block.get("f_r_n2_hz_per_atm")
    if not isinstance(o2, dict) or not isinstance(n2, dict):
        issues.append("bass_relaxation_frequencies: missing f_r_o2/f_r_n2 blocks")
        return
    _require_source("bass_relaxation_frequencies.f_r_o2_hz_per_atm", o2, issues)
    _require_source("bass_relaxation_frequencies.f_r_n2_hz_per_atm", n2, issues)

    o2_h0 = _bass_fr_o2(0.0)
    o2_h1 = _bass_fr_o2(1.0)
    if abs(o2_h0 - 24.0) > 1e-9:
        issues.append("bass O2 dry anchor self-check failed")
    if abs(o2_h1 - 29649.0) / 29649.0 > 0.01:
        issues.append(f"bass O2 h=1% anchor self-check failed: {o2_h1}")
    n2_h0 = _bass_fr_n2(0.0, 20.0)
    n2_h1 = _bass_fr_n2(1.0, 20.0)
    if abs(n2_h0 - 9.0) > 1e-6:
        issues.append(f"bass N2 dry anchor self-check failed: {n2_h0}")
    if abs(n2_h1 - 289.0) / 289.0 > 0.02:
        issues.append(f"bass N2 h=1% 20C anchor self-check failed: {n2_h1}")

    o2_anchor = (o2.get("anchors") or {}).get("h1_pct") or {}
    if abs(float(o2_anchor.get("expected_hz", -1)) - 29649.0) > 1.0:
        issues.append("f_r_o2 anchors.h1_pct.expected_hz must be ~29649")
    n2_anchor = (n2.get("anchors") or {}).get("h1_20C") or {}
    if abs(float(n2_anchor.get("expected_hz", -1)) - 289.0) > 1e-6:
        issues.append("f_r_n2 anchors.h1_20C.expected_hz must be 289")


def _c_vib_over_r(theta_k: float, t_k: float) -> float:
    x = theta_k / t_k
    ex = math.exp(x)
    return (x * x * ex) / ((ex - 1.0) ** 2)


def _audit_cvib(registry: dict[str, Any], issues: list[str]) -> None:
    block = registry.get("vibrational_heat_capacity")
    if not isinstance(block, dict):
        return
    theta = block.get("theta_vib_K") or {}
    if abs(float(theta.get("O2", -1)) - 2270.0) > 1e-9:
        issues.append("theta_vib_K.O2 must be 2270")
    if abs(float(theta.get("N2", -1)) - 3390.0) > 1e-9:
        issues.append("theta_vib_K.N2 must be 3390")
    if abs(float(theta.get("CO2_bending", -1)) - 960.0) > 1e-9:
        issues.append("theta_vib_K.CO2_bending must be 960")

    c_o2 = _c_vib_over_r(2270.0, 300.0)
    c_n2 = _c_vib_over_r(3390.0, 300.0)
    if abs(c_o2 - 0.029) / 0.029 > 0.05:
        issues.append(f"C_vib/R(O2,300K) self-check out of band: {c_o2}")
    if abs(c_n2 - 0.0016) / 0.0016 > 0.05:
        issues.append(f"C_vib/R(N2,300K) self-check out of band: {c_n2}")

    kk = block.get("kramers_kronig_single_relaxation") or {}
    if "pi" not in str(kk.get("constraint", "")).lower():
        issues.append("kramers_kronig_single_relaxation.constraint must include pi*Delta_c/c")


def _audit_frequency_set(registry: dict[str, Any], issues: list[str]) -> None:
    block = registry.get("frequency_set")
    if not isinstance(block, dict):
        return
    if int(block.get("K", -1)) != 8:
        issues.append("frequency_set.K must be 8")
    f_hz = tuple(float(x) for x in (block.get("f_hz") or []))
    if f_hz != EXPECTED_F_HZ:
        issues.append(f"frequency_set.f_hz must be {EXPECTED_F_HZ}")
    subsets = list(block.get("sensitivity_subsets_K") or [])
    if subsets != [4, 6, 8]:
        issues.append("frequency_set.sensitivity_subsets_K must be [4, 6, 8]")

    burst = registry.get("burst_schedule")
    if not isinstance(burst, dict):
        return
    cycles = burst.get("per_frequency_cycles") or {}
    for f in EXPECTED_F_HZ:
        key = str(int(f)) if float(f).is_integer() else str(f)
        # JSON keys are strings of integer Hz
        key = str(int(f))
        if key not in cycles:
            issues.append(f"burst_schedule.per_frequency_cycles missing {key}")
            continue
        n = int(cycles[key])
        if n < 1:
            issues.append(f"burst_schedule.per_frequency_cycles[{key}] must be >= 1")
        duration_s = n / f
        if duration_s > 5.0e-4:
            issues.append(
                f"burst_schedule: duration at {key} Hz is {duration_s:.4e}s "
                "(must stay well below ~725 us TOF)"
            )


def _audit_noise_and_arms(registry: dict[str, Any], issues: list[str]) -> None:
    noise = registry.get("observation_noise")
    if isinstance(noise, dict):
        jitter = noise.get("trigger_jitter") or {}
        _require_source("observation_noise.trigger_jitter", jitter, issues)
        if abs(float(jitter.get("std_s", -1)) - 3.0e-6) > 1e-15:
            issues.append("observation_noise.trigger_jitter.std_s must be 3e-6")
        if jitter.get("independence") != "per_frequency_independent":
            issues.append("trigger_jitter.independence must be per_frequency_independent")
        tof = noise.get("tof_phase_precision") or {}
        _require_source("observation_noise.tof_phase_precision", tof, issues)
        amp = noise.get("amplitude_calibration") or {}
        _require_source("observation_noise.amplitude_calibration", amp, issues)
        rh = noise.get("rh_measurement") or {}
        _require_source("observation_noise.rh_measurement", rh, issues)

    rh_arm = registry.get("rh_modulation_arm")
    if isinstance(rh_arm, dict):
        if rh_arm.get("mode") != "same_sequence_two_rh_levels":
            issues.append("rh_modulation_arm.mode must be same_sequence_two_rh_levels")
        if float(rh_arm.get("delta_rh_percent", -1)) <= 0:
            issues.append("rh_modulation_arm.delta_rh_percent must be > 0")
        if float(rh_arm.get("settle_time_s", -1)) <= 0:
            issues.append("rh_modulation_arm.settle_time_s must be > 0")

    p_arm = registry.get("pressure_scan_arm")
    if isinstance(p_arm, dict):
        pts = list(p_arm.get("P_MPa_points") or [])
        if len(pts) != 2:
            issues.append("pressure_scan_arm.P_MPa_points must have exactly 2 points")


def _audit_diffraction(registry: dict[str, Any], issues: list[str]) -> None:
    block = registry.get("low_frequency_diffraction")
    if not isinstance(block, dict):
        return
    if block.get("source") != "not_represented":
        issues.append("low_frequency_diffraction.source must be not_represented")
    if block.get("representation") != "not_represented":
        issues.append("low_frequency_diffraction.representation must be not_represented")
    if "G-line" not in str(block.get("upgrade_path", "")) and "COMSOL" not in str(
        block.get("upgrade_path", "")
    ):
        issues.append("low_frequency_diffraction.upgrade_path must point to G-line/COMSOL")


def _audit_gates(registry: dict[str, Any], issues: list[str]) -> None:
    gates = registry.get("mrs2_gates")
    if not isinstance(gates, dict):
        return
    if abs(float(gates.get("target_p90_o2_error_vol_pct", -1)) - 0.4) > 1e-15:
        issues.append("mrs2_gates.target_p90_o2_error_vol_pct must be 0.4")
    if abs(float(gates.get("max_nuisance_fraction_of_signal", -1)) - 0.50) > 1e-15:
        issues.append("mrs2_gates.max_nuisance_fraction_of_signal must be 0.50")
    if abs(float(gates.get("max_rejection_rate", -1)) - 0.05) > 1e-15:
        issues.append("mrs2_gates.max_rejection_rate must be 0.05")
    if int(gates.get("min_joint_fisher_relative_svd_rank", -1)) != 2:
        issues.append("mrs2_gates.min_joint_fisher_relative_svd_rank must be 2")


def _audit_doc_errata(registry: dict[str, Any], issues: list[str], tv3_root: Path) -> dict[str, Any]:
    block = registry.get("doc_errata_checklist")
    summary: dict[str, Any] = {"checked": [], "failed": []}
    if not isinstance(block, dict):
        return summary
    items = block.get("items") or []
    if len(items) < 3:
        issues.append("doc_errata_checklist.items must cover >= 3 documents")
    forbidden = list(block.get("forbidden_legacy_tokens") or [])
    required_forbidden = {"65000", "65 kHz/atm", "10.1121/1.400476"}
    if not required_forbidden.issubset(set(forbidden)):
        issues.append("doc_errata_checklist.forbidden_legacy_tokens incomplete")

    for item in items:
        if not isinstance(item, dict):
            issues.append("doc_errata_checklist.items entry must be object")
            continue
        rel = item.get("path")
        if not isinstance(rel, str):
            issues.append("doc_errata_checklist item missing path")
            continue
        path = tv3_root / rel
        summary["checked"].append(rel)
        if not path.is_file():
            issues.append(f"doc_errata: missing file {rel}")
            summary["failed"].append(rel)
            continue
        text = path.read_text(encoding="utf-8")
        if "10.1121/1.400176" not in text:
            issues.append(f"doc_errata: {rel} missing corrected Bass DOI 10.1121/1.400176")
            summary["failed"].append(rel)
        # Allow mentioning the wrong DOI only as historical errata note.
        if "10.1121/1.400476" in text and "修正" not in text and "旧引" not in text:
            issues.append(f"doc_errata: {rel} still presents 10.1121/1.400476 without errata note")
            summary["failed"].append(rel)
        if rel.endswith("co2_o2_n2_gas_properties.md") or rel.endswith("physics_references.md"):
            if "9 Hz/atm" not in text:
                issues.append(f"doc_errata: {rel} missing N2 9 Hz/atm correction")
                summary["failed"].append(rel)
        if item.get("status") != "verified":
            issues.append(f"doc_errata: {rel} status must be verified at MRS-0")
    return summary


def audit_mrs0_gate(config_dir: Path | None = None) -> dict[str, Any]:
    """Audit MRS-0 registry completeness. Pass → MRS-1 allowed."""
    bundle = load_mrs0_registry(config_dir)
    registry = bundle["registry"]
    issues: list[str] = []
    tv3_root = Path(bundle["dir"]).resolve().parents[1]

    _audit_identity(registry, issues)
    _audit_sourced_blocks(registry, issues)
    _audit_composition(registry, issues)
    _audit_bass(registry, issues)
    _audit_cvib(registry, issues)
    _audit_frequency_set(registry, issues)
    _audit_noise_and_arms(registry, issues)
    _audit_diffraction(registry, issues)
    _audit_gates(registry, issues)
    errata_summary = _audit_doc_errata(registry, issues, tv3_root)

    blocking = list(registry.get("blocking_items_mrs0") or [])
    if blocking:
        issues.append(f"blocking_items_mrs0 not empty: {blocking}")

    # Nested sourced sub-blocks that carry independent physics claims
    nested = [
        ("bass_relaxation_frequencies", "f_r_o2_hz_per_atm"),
        ("bass_relaxation_frequencies", "f_r_n2_hz_per_atm"),
        ("vibrational_heat_capacity", "legacy_v2_empirical_intensity"),
        ("frequency_set", "geometry_floor_hz"),
        ("observation_noise", "trigger_jitter"),
        ("observation_noise", "tof_phase_precision"),
        ("observation_noise", "amplitude_calibration"),
        ("observation_noise", "rh_measurement"),
    ]
    for parent, child in nested:
        block = (registry.get(parent) or {}).get(child)
        if isinstance(block, dict):
            _require_source(f"{parent}.{child}", block, issues)

    passed = len(issues) == 0
    gate = registry.get("mrs0_gate") or {}
    verdict = (
        gate.get("pass_verdict", "mrs0_registry_frozen")
        if passed
        else gate.get("fail_verdict", "inconclusive_parameter_bounds")
    )
    allowed_next = gate.get("allowed_next_stage_on_pass") if passed else None
    return {
        "schema_version": registry.get("schema_version"),
        "stage": STAGE,
        "passed": passed,
        "verdict": verdict,
        "allowed_next_stage": allowed_next,
        "issues": issues,
        "frequency_set_hz": list((registry.get("frequency_set") or {}).get("f_hz") or []),
        "doc_errata": errata_summary,
        "registry_sha256": bundle["sha256"],
        "registry_path": bundle["path"],
        "claim_scope": registry.get("claim_scope", "registered_simulation_domain_only"),
    }
