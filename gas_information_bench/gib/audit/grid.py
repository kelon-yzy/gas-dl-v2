"""Deterministic S1 3×3 information/collinearity grid builder."""

from __future__ import annotations

import json
from functools import lru_cache
from typing import Any, Mapping

import numpy as np

from .forward import AuditConfig, CANDIDATES, AuditResult, analyze_candidate


TARGET_ERROR_TAU = {
    "N2": 0.08,
    "CO2": 0.03,
    "O2": 0.10,
    "Ar": 0.05,
}
TAU_SOURCE = {
    "N2": "P2-04 primary composition screening tolerance; MFC label error is the engineering reference",
    "CO2": "P2-04 primary composition screening tolerance; TraceGas-NDIR-CO2 hardware profile is the reference",
    "O2": "P2-04 primary composition screening tolerance; low-risk four-component bench contract",
    "Ar": "P2-04 low-concentration screening tolerance; low-concentration requirement is the reference",
}
INFORMATION_BANDS = {
    "sufficient": (None, 0.5),
    "critical": (0.8, 1.2),
    "insufficient": (2.0, None),
}
ANGLE_TARGETS_DEG = {
    "high_collinearity": 10.0,
    "medium_collinearity": 45.0,
    "low_collinearity": 80.0,
}
ANGLE_TOLERANCE_DEG = 5.0
_FULL_MODALITIES = ("ndir", "acoustic_raw", "thermal", "slow", "calibration")
INFORMATION_PROFILES = {
    "sufficient": {"noise_scale": 0.35, "modalities": _FULL_MODALITIES},
    "critical": {"noise_scale": 1.00, "modalities": _FULL_MODALITIES},
    "insufficient": {"noise_scale": 2.20, "modalities": _FULL_MODALITIES},
}
GRID_RECOMPUTE_RTOL = 1.0e-9
GRID_RECOMPUTE_ATOL = 1.0e-12


def physical_crb_p90(result: AuditResult) -> np.ndarray:
    """Map the three free coordinates back to four reported components."""

    free_count = result.crb.shape[0]
    transform = np.vstack([np.eye(free_count), -np.ones(free_count)])
    physical_crb = transform @ result.crb @ transform.T
    return 1.645 * np.sqrt(np.diag(physical_crb))


def information_ratio(result: AuditResult) -> float:
    components = CANDIDATES[result.candidate_id].components
    p90 = physical_crb_p90(result)
    tau = np.array([TARGET_ERROR_TAU[name] for name in components], dtype=float)
    return float(np.max(p90 / tau))


def classify_information(result: AuditResult) -> str:
    ratio = information_ratio(result)
    for name, (lower, upper) in INFORMATION_BANDS.items():
        if (lower is None or ratio >= lower) and (upper is None or ratio <= upper):
            return name
    return "unclassified"


@lru_cache(maxsize=None)
def _select_coupling_strength(target_angle: float) -> tuple[float, float]:
    search_values = np.concatenate(([0.0], np.geomspace(1.0e-4, 1.0e3, 1200)))
    candidates = []
    for coupling_strength in search_values:
        result = analyze_candidate(AuditConfig(coupling_strength=float(coupling_strength)))
        error = abs(result.minimum_principal_angle_deg - target_angle)
        candidates.append((error, float(coupling_strength), result.minimum_principal_angle_deg))
    error, coupling_strength, actual_angle = min(candidates, key=lambda item: (item[0], item[1]))
    if error > ANGLE_TOLERANCE_DEG:
        raise RuntimeError(
            f"target angle {target_angle} deg is not reachable; closest={actual_angle} deg"
        )
    return coupling_strength, actual_angle


def build_grid(candidate_id: str = "GIB-C4-LR") -> list[dict[str, object]]:
    if candidate_id not in CANDIDATES:
        raise ValueError(f"unknown candidate_id: {candidate_id}")
    selected_coupling = {
        name: _select_coupling_strength(target)
        for name, target in ANGLE_TARGETS_DEG.items()
    }
    cells: list[dict[str, object]] = []
    for information_name, profile in INFORMATION_PROFILES.items():
        for angle_name, target_angle in ANGLE_TARGETS_DEG.items():
            coupling_strength, _ = selected_coupling[angle_name]
            config = AuditConfig(
                candidate_id=candidate_id,
                modalities=tuple(profile["modalities"]),
                noise_scale=float(profile["noise_scale"]),
                nuisance_prior_scale=1.0,
                coupling_strength=coupling_strength,
            )
            result = analyze_candidate(config)
            actual_angle = result.minimum_principal_angle_deg
            ratio = information_ratio(result)
            actual_information = classify_information(result)
            accessible = (
                actual_information == information_name
                and abs(actual_angle - target_angle) <= ANGLE_TOLERANCE_DEG
                and len(set(result.ranks_by_tolerance.values())) == 1
            )
            cells.append(
                {
                    "config_id": f"GIB-S1-{information_name[:3].upper()}-{angle_name[:3].upper()}",
                    "information_band": information_name,
                    "angle_band": angle_name,
                    "target_angle_deg": target_angle,
                    "actual_angle_deg": actual_angle,
                    "coupling_strength": coupling_strength,
                    "noise_scale": config.noise_scale,
                    "modalities": config.modalities,
                    "effective_fisher": result.effective_fisher.tolist(),
                    "crb_p90": physical_crb_p90(result).tolist(),
                    "max_crb_p90_over_tau": ratio,
                    "condition_number": result.condition_number,
                    "actual_information_band": actual_information,
                    "accessible": accessible,
                }
            )
    if len(cells) != 9:
        raise RuntimeError(f"expected 9 grid cells, got {len(cells)}")
    if not all(bool(cell["accessible"]) for cell in cells):
        failed = [cell["config_id"] for cell in cells if not cell["accessible"]]
        raise RuntimeError(f"unreachable S1 cells: {failed}")
    for information_name in INFORMATION_PROFILES:
        ratios = [cell["max_crb_p90_over_tau"] for cell in cells if cell["information_band"] == information_name]
        if max(ratios) / min(ratios) > 2.0:
            raise RuntimeError(f"angle axis changes information scale too much in {information_name}")
    return cells


def grid_summary(candidate_id: str = "GIB-C4-LR") -> dict[str, object]:
    cells = build_grid(candidate_id)
    return {
        "grid_id": "GIB-S1-3x3-v1",
        "candidate_id": candidate_id,
        "target_error_tau": TARGET_ERROR_TAU,
        "tau_source": TAU_SOURCE,
        "information_bands": INFORMATION_BANDS,
        "angle_targets_deg": ANGLE_TARGETS_DEG,
        "angle_tolerance_deg": ANGLE_TOLERANCE_DEG,
        "cells": cells,
    }


def _grid_value_mismatches(
    frozen: object,
    recomputed: object,
    path: str = "$",
) -> tuple[list[str], list[float], list[float]]:
    mismatches: list[str] = []
    absolute_errors: list[float] = []
    relative_errors: list[float] = []
    if isinstance(frozen, dict) and isinstance(recomputed, dict):
        if set(frozen) != set(recomputed):
            return [f"{path}: keys differ"], absolute_errors, relative_errors
        for key in sorted(frozen):
            child_mismatches, child_absolute, child_relative = _grid_value_mismatches(
                frozen[key], recomputed[key], f"{path}.{key}"
            )
            mismatches.extend(child_mismatches)
            absolute_errors.extend(child_absolute)
            relative_errors.extend(child_relative)
        return mismatches, absolute_errors, relative_errors
    if isinstance(frozen, list) and isinstance(recomputed, list):
        if len(frozen) != len(recomputed):
            return [f"{path}: lengths differ"], absolute_errors, relative_errors
        for index, (left, right) in enumerate(zip(frozen, recomputed)):
            child_mismatches, child_absolute, child_relative = _grid_value_mismatches(
                left, right, f"{path}[{index}]"
            )
            mismatches.extend(child_mismatches)
            absolute_errors.extend(child_absolute)
            relative_errors.extend(child_relative)
        return mismatches, absolute_errors, relative_errors
    if (
        isinstance(frozen, (int, float))
        and not isinstance(frozen, bool)
        and isinstance(recomputed, (int, float))
        and not isinstance(recomputed, bool)
    ):
        absolute_error = abs(float(frozen) - float(recomputed))
        relative_error = absolute_error / max(abs(float(frozen)), abs(float(recomputed)), GRID_RECOMPUTE_ATOL)
        absolute_errors.append(absolute_error)
        relative_errors.append(relative_error)
        if not np.isclose(
            float(frozen),
            float(recomputed),
            rtol=GRID_RECOMPUTE_RTOL,
            atol=GRID_RECOMPUTE_ATOL,
        ):
            mismatches.append(f"{path}: numeric values differ")
        return mismatches, absolute_errors, relative_errors
    if frozen != recomputed:
        mismatches.append(f"{path}: values differ")
    return mismatches, absolute_errors, relative_errors


def g3_2_grid_audit(frozen_grid: Mapping[str, Any]) -> dict[str, object]:
    """Recompute and validate every preregistered P3 G3-2 grid condition."""

    recomputed = json.loads(json.dumps(grid_summary(str(frozen_grid["candidate_id"]))))
    frozen_normalized = json.loads(json.dumps(frozen_grid))
    cells = recomputed["cells"]
    config_ids = [str(cell["config_id"]) for cell in cells]
    unique_cells_pass = len(cells) == 9 and len(config_ids) == len(set(config_ids))

    information_records: dict[str, dict[str, object]] = {}
    for band, (lower, upper) in INFORMATION_BANDS.items():
        band_cells = [cell for cell in cells if cell["information_band"] == band]
        ratios = [float(cell["max_crb_p90_over_tau"]) for cell in band_cells]
        passed = len(band_cells) == 3 and all(
            (lower is None or ratio >= lower) and (upper is None or ratio <= upper)
            for ratio in ratios
        )
        information_records[band] = {
            "lower": lower,
            "upper": upper,
            "ratios": ratios,
            "passed": passed,
        }

    angle_records: dict[str, dict[str, object]] = {}
    for band, target in ANGLE_TARGETS_DEG.items():
        band_cells = [cell for cell in cells if cell["angle_band"] == band]
        actual = [float(cell["actual_angle_deg"]) for cell in band_cells]
        couplings = [float(cell["coupling_strength"]) for cell in band_cells]
        passed = (
            len(band_cells) == 3
            and all(abs(value - target) <= ANGLE_TOLERANCE_DEG for value in actual)
            and len(set(couplings)) == 1
        )
        angle_records[band] = {
            "target_angle_deg": target,
            "actual_angles_deg": actual,
            "coupling_strengths": couplings,
            "passed": passed,
        }

    monotonic_records: dict[str, dict[str, object]] = {}
    information_order = tuple(INFORMATION_PROFILES)
    for angle_band in ANGLE_TARGETS_DEG:
        ordered = [
            next(
                cell
                for cell in cells
                if cell["angle_band"] == angle_band and cell["information_band"] == information_band
            )
            for information_band in information_order
        ]
        noise_scales = [float(cell["noise_scale"]) for cell in ordered]
        ratios = [float(cell["max_crb_p90_over_tau"]) for cell in ordered]
        passed = (
            all(left < right for left, right in zip(noise_scales, noise_scales[1:]))
            and all(left < right for left, right in zip(ratios, ratios[1:]))
        )
        monotonic_records[angle_band] = {
            "information_order": list(information_order),
            "noise_scales": noise_scales,
            "max_crb_p90_over_tau": ratios,
            "passed": passed,
        }

    mismatches, absolute_errors, relative_errors = _grid_value_mismatches(
        frozen_normalized,
        recomputed,
    )
    checks = {
        "unique_reachable_cells": {
            "config_ids": config_ids,
            "accessible_count": sum(bool(cell["accessible"]) for cell in cells),
            "passed": unique_cells_pass and all(bool(cell["accessible"]) for cell in cells),
        },
        "information_bands": {
            "bands": information_records,
            "passed": all(record["passed"] for record in information_records.values()),
        },
        "angle_bands": {
            "bands": angle_records,
            "tolerance_deg": ANGLE_TOLERANCE_DEG,
            "passed": all(record["passed"] for record in angle_records.values()),
        },
        "noise_information_monotonicity": {
            "angle_bands": monotonic_records,
            "passed": all(record["passed"] for record in monotonic_records.values()),
        },
        "frozen_json_value_match": {
            "relative_tolerance": GRID_RECOMPUTE_RTOL,
            "absolute_tolerance": GRID_RECOMPUTE_ATOL,
            "maximum_absolute_error": max(absolute_errors, default=0.0),
            "maximum_relative_error": max(relative_errors, default=0.0),
            "mismatches": mismatches,
            "passed": not mismatches,
        },
    }
    passed = all(bool(check["passed"]) for check in checks.values())
    return {
        "schema_version": "gib-benchmark-1",
        "audit_id": "GIB-P3-G3-2-v1",
        "task_id": "P3-02",
        "claim_scope": "controlled_synthetic_relative_comparison_only",
        "checks": checks,
        "recomputed_grid": recomputed,
        "gate_verdict": "pass" if passed else "fail",
        "next_allowed_task": "P3-03" if passed else "P2-06",
    }


if __name__ == "__main__":
    print(json.dumps(grid_summary(), ensure_ascii=False, indent=2, sort_keys=True))
