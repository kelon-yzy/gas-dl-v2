"""Deterministic P2-07 S2 complementarity and S3 switch audit."""

from __future__ import annotations

import json
from dataclasses import replace
from importlib.resources import files

import numpy as np

from .forward import (
    CANDIDATES,
    ETA_DEFAULT,
    AuditConfig,
    forward_observation,
    _component_directions,
)
from .grid import build_grid, TARGET_ERROR_TAU


CONFIG_PACKAGE = "configs"
PROFILE_NAME = "p2_s2_s3_audit.json"


def load_profile() -> dict[str, object]:
    with files(CONFIG_PACKAGE).joinpath(PROFILE_NAME).open(encoding="utf-8") as handle:
        return json.load(handle)


def _target_jacobian(
    config: AuditConfig,
    eta: np.ndarray = ETA_DEFAULT,
) -> tuple[object, np.ndarray]:
    profile = CANDIDATES[config.candidate_id]
    theta = np.array(profile.baseline_composition[:-1], dtype=float)
    base = forward_observation(theta, eta, config)
    step = 1.0e-6
    jacobian = np.column_stack(
        [
            (
                forward_observation(theta + np.eye(theta.size)[index] * step, eta, config).values
                - forward_observation(theta - np.eye(theta.size)[index] * step, eta, config).values
            )
            / (2.0 * step)
            for index in range(theta.size)
        ]
    )
    return base, jacobian / base.noise_std[:, None]


def _component_vectors(whitened_jacobian: np.ndarray, candidate_id: str) -> dict[str, np.ndarray]:
    profile = CANDIDATES[candidate_id]
    return {
        name: whitened_jacobian @ direction
        for name, direction in _component_directions(profile).items()
    }


def _pair_abs_cosine(
    whitened_jacobian: np.ndarray,
    candidate_id: str,
    pair: tuple[str, str],
) -> float:
    vectors = _component_vectors(whitened_jacobian, candidate_id)
    left = vectors[pair[0]]
    right = vectors[pair[1]]
    left_norm = float(np.linalg.norm(left))
    right_norm = float(np.linalg.norm(right))
    if left_norm == 0.0 or right_norm == 0.0:
        raise RuntimeError(f"zero whitened sensitivity for pair {pair}")
    return float(abs(left @ right / (left_norm * right_norm)))


def _row_sensitivities(config: AuditConfig) -> dict[str, dict[str, float]]:
    observation, whitened_jacobian = _target_jacobian(config)
    return {
        label: {
            component: float(abs(row @ direction))
            for component, direction in _component_directions(CANDIDATES[config.candidate_id]).items()
        }
        for label, row in zip(observation.labels, whitened_jacobian)
    }


def _component_norms(config: AuditConfig) -> dict[str, float]:
    _, whitened_jacobian = _target_jacobian(config)
    return {
        name: float(np.linalg.norm(vector))
        for name, vector in _component_vectors(whitened_jacobian, config.candidate_id).items()
    }


def _observation_map(config: AuditConfig, eta: np.ndarray) -> dict[str, float]:
    profile = CANDIDATES[config.candidate_id]
    theta = np.array(profile.baseline_composition[:-1], dtype=float)
    observation = forward_observation(theta, eta, config)
    return {label: float(value) for label, value in zip(observation.labels, observation.values)}


def _changed_labels(left: dict[str, float], right: dict[str, float]) -> list[str]:
    labels = sorted(set(left) | set(right))
    return [
        label
        for label in labels
        if label not in left
        or label not in right
        or not np.isclose(left[label], right[label], rtol=0.0, atol=1.0e-12)
    ]


def _shared_labels_equal(
    left: dict[str, float],
    right: dict[str, float],
    labels: list[str],
) -> bool:
    return all(
        label in left
        and label in right
        and np.isclose(left[label], right[label], rtol=0.0, atol=1.0e-12)
        for label in labels
    )


def _switch_delta(
    name: str,
    profile: dict[str, object],
    baseline: AuditConfig,
    eta: np.ndarray,
) -> dict[str, object]:
    on_value = bool(profile["on"])
    off_value = bool(profile["off"])
    on_config = replace(baseline, **{name: on_value})
    off_config = replace(baseline, **{name: off_value})
    on_observation = _observation_map(on_config, eta)
    off_observation = _observation_map(off_config, eta)
    changed = _changed_labels(on_observation, off_observation)
    expected_changed = sorted(str(label) for label in profile["expected_changed_labels"])
    expected_invariant = [str(label) for label in profile.get("expected_invariant_labels", [])]
    invariant_prefixes = [str(prefix) for prefix in profile.get("invariant_label_prefixes", [])]
    invariant_labels = expected_invariant + sorted(
        label
        for label in set(on_observation) & set(off_observation)
        if any(label.startswith(prefix) for prefix in invariant_prefixes)
    )
    invariant_labels = sorted(set(invariant_labels))
    config_delta = [
        field
        for field in ("fast_waveform", "nonlinear_nuisance_coupling", "cross_sensitivity")
        if getattr(on_config, field) != getattr(off_config, field)
    ]
    changed_pass = changed == expected_changed
    invariant_pass = _shared_labels_equal(on_observation, off_observation, invariant_labels)
    return {
        "on_value": on_value,
        "off_value": off_value,
        "config_delta_fields": config_delta,
        "changed_labels": changed,
        "expected_changed_labels": expected_changed,
        "invariant_labels": invariant_labels,
        "changed_labels_match": changed_pass,
        "invariants_hold": invariant_pass,
        "independent_off_pass": bool(changed) and changed_pass and invariant_pass and config_delta == [name],
        "passed": bool(changed) and changed_pass and invariant_pass and config_delta == [name],
    }


def _s2_audit(profile: dict[str, object]) -> dict[str, object]:
    candidate_id = str(profile["candidate_id"])
    base_modalities = tuple(str(item) for item in profile["baseline_modalities"])
    optical_config = AuditConfig(candidate_id=candidate_id, modalities=("ndir",))
    thermal_config = AuditConfig(candidate_id=candidate_id, modalities=("thermal",))
    near_config = optical_config
    cross_config = AuditConfig(
        candidate_id=candidate_id,
        modalities=tuple(str(item) for item in profile["s2_thresholds"]["cross_modal_disambiguation"]["modalities"]),
    )
    full_config = AuditConfig(candidate_id=candidate_id, modalities=base_modalities)
    thresholds = profile["s2_thresholds"]

    optical_threshold = thresholds["optical_row"]
    optical_rows = _row_sensitivities(optical_config)
    optical_actual = optical_rows[str(optical_threshold["label"])][str(optical_threshold["target_component"])]
    optical = {
        "modality": list(optical_config.modalities),
        "row_label": str(optical_threshold["label"]),
        "target_component": str(optical_threshold["target_component"]),
        "whitened_abs_sensitivity": optical_actual,
        "minimum_whitened_abs_sensitivity": float(optical_threshold["minimum_whitened_abs_sensitivity"]),
        "weak_competitor_sensitivities": {
            component: optical_rows[str(optical_threshold["label"])][component]
            for component in optical_threshold["weak_competitors"]
        },
        "passed": optical_actual >= float(optical_threshold["minimum_whitened_abs_sensitivity"]),
    }

    thermal_threshold = thresholds["thermal_row"]
    thermal_rows = _row_sensitivities(thermal_config)
    thermal_row = thermal_rows[str(thermal_threshold["label"])]
    thermal_target = thermal_row[str(thermal_threshold["target_component"])]
    thermal_competitor = max(thermal_row[component] for component in thermal_threshold["competitors"])
    thermal_ratio = thermal_target / thermal_competitor
    thermal = {
        "modality": list(thermal_config.modalities),
        "row_label": str(thermal_threshold["label"]),
        "target_component": str(thermal_threshold["target_component"]),
        "target_whitened_abs_sensitivity": thermal_target,
        "largest_competitor_whitened_abs_sensitivity": thermal_competitor,
        "target_to_competitor_ratio": thermal_ratio,
        "minimum_target_to_competitor_ratio": float(thermal_threshold["minimum_target_to_competitor_ratio"]),
        "passed": thermal_ratio >= float(thermal_threshold["minimum_target_to_competitor_ratio"]),
    }

    pair = tuple(str(item) for item in thresholds["near_degenerate"]["pair"])
    near_observation, near_jacobian = _target_jacobian(near_config)
    near_cosine = _pair_abs_cosine(near_jacobian, candidate_id, pair)
    near = {
        "modality": list(near_config.modalities),
        "pair": list(pair),
        "abs_cosine": near_cosine,
        "minimum_abs_cosine": float(thresholds["near_degenerate"]["minimum_abs_cosine"]),
        "observation_labels": list(near_observation.labels),
        "passed": near_cosine >= float(thresholds["near_degenerate"]["minimum_abs_cosine"]),
    }

    cross_observation, cross_jacobian = _target_jacobian(cross_config)
    cross_cosine = _pair_abs_cosine(cross_jacobian, candidate_id, pair)
    disambiguation = {
        "modalities": list(cross_config.modalities),
        "pair": list(pair),
        "abs_cosine": cross_cosine,
        "maximum_abs_cosine": float(thresholds["cross_modal_disambiguation"]["maximum_abs_cosine"]),
        "observation_labels": list(cross_observation.labels),
        "passed": cross_cosine <= float(thresholds["cross_modal_disambiguation"]["maximum_abs_cosine"]),
    }

    low_threshold = thresholds["low_concentration"]
    low_cell_id = str(profile["low_concentration_cell_id"])
    low_cell = next(cell for cell in build_grid(candidate_id) if cell["config_id"] == low_cell_id)
    candidate = CANDIDATES[candidate_id]
    low_component = str(low_threshold["component"])
    low_index = candidate.components.index(low_component)
    low_fraction = candidate.baseline_composition[low_index]
    low_tau = TARGET_ERROR_TAU[low_component]
    low_p90 = float(low_cell["crb_p90"][low_index])
    low_ratio = low_p90 / low_tau
    full_norms = _component_norms(full_config)
    low = {
        "component": low_component,
        "baseline_fraction": low_fraction,
        "minimum_fraction": float(low_threshold["minimum_fraction"]),
        "maximum_fraction": float(low_threshold["maximum_fraction"]),
        "full_profile_whitened_sensitivity": full_norms[low_component],
        "minimum_whitened_sensitivity": float(low_threshold["minimum_whitened_sensitivity"]),
        "cell_id": low_cell_id,
        "crb_p90": low_p90,
        "tau": low_tau,
        "crb_p90_over_tau": low_ratio,
        "maximum_crb_p90_over_tau": float(low_threshold["maximum_crb_p90_over_tau"]),
        "passed": (
            float(low_threshold["minimum_fraction"]) <= low_fraction <= float(low_threshold["maximum_fraction"])
            and full_norms[low_component] >= float(low_threshold["minimum_whitened_sensitivity"])
            and low_ratio <= float(low_threshold["maximum_crb_p90_over_tau"])
        ),
    }

    s2_passed = all(item["passed"] for item in (optical, thermal, near, disambiguation, low))
    return {
        "optical_primary": optical,
        "acoustic_or_thermal_primary": thermal,
        "single_modality_near_degeneracy": near,
        "cross_modal_disambiguation": disambiguation,
        "low_concentration_target": low,
        "passed": s2_passed,
        "c4_pre_verdict": "eligible_for_P3_test" if s2_passed else "no_complementarity",
    }


def _s3_audit(profile: dict[str, object]) -> dict[str, object]:
    candidate_id = str(profile["candidate_id"])
    baseline = AuditConfig(
        candidate_id=candidate_id,
        modalities=tuple(str(item) for item in profile["baseline_modalities"]),
    )
    eta = np.asarray(profile["probe_eta"], dtype=float)
    switches = {
        name: _switch_delta(name, spec, baseline, eta)
        for name, spec in profile["s3_switches"].items()
    }
    all_off = replace(
        baseline,
        fast_waveform=False,
        nonlinear_nuisance_coupling=False,
        cross_sensitivity=False,
    )
    all_off_changed = _changed_labels(_observation_map(baseline, eta), _observation_map(all_off, eta))
    return {
        "probe_eta": eta.tolist(),
        "switches": switches,
        "all_off_negative_control": {
            "changed_labels": all_off_changed,
            "negative_control_only": True,
            "passed": bool(all_off_changed),
        },
        "passed": all(item["passed"] for item in switches.values()),
    }


def _round_floats(value: object) -> object:
    if isinstance(value, float):
        if 0.0 < abs(value) < 1.0e-6:
            return value
        return round(value, 6)
    if isinstance(value, list):
        return [_round_floats(item) for item in value]
    if isinstance(value, dict):
        return {key: _round_floats(item) for key, item in value.items()}
    return value


def audit_summary() -> dict[str, object]:
    profile = load_profile()
    s2 = _s2_audit(profile)
    s3 = _s3_audit(profile)
    return _round_floats({
        "audit_id": profile["audit_id"],
        "candidate_id": profile["candidate_id"],
        "input_profile": profile,
        "s2": s2,
        "s3": s3,
        "verdict": "pass" if s2["passed"] and s3["passed"] else "redesign_required",
    })


if __name__ == "__main__":
    print(json.dumps(audit_summary(), ensure_ascii=False, indent=2, sort_keys=True))
