#!/usr/bin/env python3
"""F4 audit: bidirectional identifiability v2 (dual jitter, flow implemented)."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
from collections import defaultdict
from datetime import datetime, timezone
from itertools import product
from pathlib import Path
from typing import Any

import numpy as np

_TV3_ROOT = Path(__file__).resolve().parents[1]
if str(_TV3_ROOT) not in sys.path:
    sys.path.insert(0, str(_TV3_ROOT))

from tv3.audit.error_budget import (  # noqa: E402
    NORMAL_P90_Z,
    combined_p90_o2_error_percent,
    equivalent_o2_std_percent,
)
from tv3.audit.identifiability_v2 import (  # noqa: E402
    BidirAcousticPoint,
    build_bidir_points,
    fisher_information_bidir,
    local_bidir_tof_sensitivity,
    midpair_tof_std_s,
    observed_bidir_tof_s,
    sound_speed_m_per_s,
)
from tv3.sim.generation.tunnel_ventilation.bidir_registry import (  # noqa: E402
    default_config_dir,
)
from tv3.sim.generation.tunnel_ventilation.conditions import (  # noqa: E402
    COMPOSITION_DOMAIN_WIDE,
)

AUDIT_SCHEMA = "tv3-identifiability-bidir-2"
REPRESENTATION_FIELDS = {
    "name",
    "unit",
    "representation",
    "source",
    "distribution",
    "correlation_group",
    "deployable_observable",
    "v1_representation",
    "blocks_go_verdict",
}
BUSINESS_THRESHOLD_FIELDS = {
    "target_p90_o2_error_percent",
    "max_nuisance_fraction_of_signal",
    "max_rejection_rate",
}
REJECTION_POLICY_FIELDS = {
    "id",
    "source",
    "reject_if_unrepresented_blocking_nuisance",
}


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"required file is missing: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"JSON object required: {path}")
    return payload


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"cannot write empty CSV: {path}")
    fields = sorted({field for row in rows for field in row})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _resolve(project_root: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else project_root / path


def _validate_config(config: dict[str, Any]) -> None:
    if config.get("schema_version") != AUDIT_SCHEMA:
        raise ValueError(f"schema_version must be {AUDIT_SCHEMA}")
    if not isinstance(config.get("jitter_scenarios"), list) or len(config["jitter_scenarios"]) < 2:
        raise ValueError("jitter_scenarios must list at least the F0 dual scenarios")
    ids = [item["id"] for item in config["jitter_scenarios"]]
    if len(ids) != len(set(ids)):
        raise ValueError("jitter_scenarios ids must be unique")
    for item in config["representation_audit"]:
        missing = REPRESENTATION_FIELDS - set(item)
        if missing:
            raise ValueError(f"representation audit is missing fields: {sorted(missing)}")
    flow = next(item for item in config["representation_audit"] if item["name"] == "flow_projection")
    if flow["representation"] != "implemented_physics" or flow["blocks_go_verdict"]:
        raise ValueError("F4 requires flow_projection=implemented_physics with blocks_go_verdict=false")
    thresholds = config.get("business_thresholds")
    if not isinstance(thresholds, dict) or set(thresholds) != BUSINESS_THRESHOLD_FIELDS:
        raise ValueError(f"business_thresholds must contain exactly {sorted(BUSINESS_THRESHOLD_FIELDS)}")
    rejection_policy = config.get("rejection_policy")
    if not isinstance(rejection_policy, dict) or set(rejection_policy) != REJECTION_POLICY_FIELDS:
        raise ValueError(f"rejection_policy must contain exactly {sorted(REJECTION_POLICY_FIELDS)}")


def _verify_prerequisites(project_root: Path, config: dict[str, Any]) -> dict[str, Any]:
    registry_path = _resolve(project_root, config["f0_registry"]["path"])
    registry_sha = _sha256(registry_path)
    expected = config["f0_registry"]["expected_sha256"]
    if registry_sha != expected:
        raise RuntimeError(
            f"F0 registry sha256 mismatch: got {registry_sha}, expected {expected}"
        )
    f3_path = _resolve(project_root, config["f3_prerequisite"]["verdict_path"])
    f3 = _read_json(f3_path)
    if f3.get("verdict") != config["f3_prerequisite"]["expected_verdict"]:
        raise RuntimeError(
            f"F3 prerequisite failed: verdict={f3.get('verdict')!r}, "
            f"expected {config['f3_prerequisite']['expected_verdict']!r}"
        )
    if f3.get("feature_builder") != config["f3_prerequisite"]["feature_builder"]:
        raise RuntimeError("F3 feature_builder does not match F4 config")
    return {
        "f0_registry_sha256": registry_sha,
        "f3_verdict": f3.get("verdict"),
        "f3_feature_builder": f3.get("feature_builder"),
        "f3_delay_calibration_digest": f3.get("delay_calibration_digest"),
    }


def _points_for_window(
    window: dict[str, Any], context_grid: dict[str, list[float]]
) -> list[BidirAcousticPoint]:
    points: list[BidirAcousticPoint] = []
    for co2, t_c, path_length, v_path in product(
        context_grid["co2_percent"],
        context_grid["t_c"],
        context_grid["path_length_m"],
        context_grid["v_path_m_per_s"],
    ):
        points.append(
            BidirAcousticPoint(
                float(co2),
                float(window["center_percent"]),
                float(t_c),
                float(path_length),
                float(v_path),
            )
        )
    return points


def _uncertainty_scenarios_for_jitter(
    config: dict[str, Any],
    *,
    jitter_id: str,
    tof_std_s: float,
) -> list[dict[str, Any]]:
    mid_std = midpair_tof_std_s(tof_std_s)
    n_frames = int(config["observation"]["sequence_frames_for_prior_crosscheck"])
    scenarios = [
        {
            "id": f"trigger_jitter_{jitter_id}_single_frame",
            "parameter": "tof_s",
            "std": mid_std,
            "source": f"independent mid-pair σ=σ_j/√2; jitter_id={jitter_id}",
        },
        {
            "id": f"trigger_jitter_{jitter_id}_n{n_frames}",
            "parameter": "tof_s",
            "std": mid_std / math.sqrt(n_frames),
            "source": f"sequence average of {n_frames} independent mid-pair frames",
        },
    ]
    scenarios.extend(config["shared_uncertainty_scenarios"])
    return scenarios


def _point_rows(
    points: list[BidirAcousticPoint],
    *,
    scope: str,
    window_id: str | None,
    config: dict[str, Any],
    point_offset: int,
    tof_std_s: float,
    uncertainty_scenarios: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], list[str]]:
    fixed_delay_s = float(config["observation"]["fixed_delay_s"])
    t_std = float(config["observation"]["temperature_std_c_for_fisher"])
    bounds = {name: tuple(values) for name, values in config["parameter_bounds"].items()}
    sensitivity_rows: list[dict[str, Any]] = []
    fisher_rows: list[dict[str, Any]] = []
    budget_rows: list[dict[str, Any]] = []
    audit_errors: list[str] = []
    for index, point in enumerate(points, start=point_offset):
        point_id = f"{scope}-{index:04d}"
        derivatives = local_bidir_tof_sensitivity(
            point,
            parameter_steps=config["finite_difference"]["steps"],
            parameter_bounds=bounds,
            fixed_delay_s=fixed_delay_s,
            max_relative_step_disagreement=float(
                config["finite_difference"]["max_relative_step_disagreement"]
            ),
        )
        t_ab, t_ba = observed_bidir_tof_s(point, fixed_delay_s=fixed_delay_s)
        row: dict[str, Any] = {
            "point_id": point_id,
            "scope": scope,
            "window_id": window_id,
            "x_CO2_percent": point.co2_percent,
            "x_O2_percent": point.o2_percent,
            "x_N2_percent": point.n2_percent,
            "T_C": point.t_c,
            "L_m": point.path_length_m,
            "v_path_m_per_s": point.v_path_m_per_s,
            "sound_speed_m_per_s": sound_speed_m_per_s(point),
            "observed_tof_ab_s": t_ab,
            "observed_tof_ba_s": t_ba,
        }
        for parameter, detail in derivatives.items():
            row[f"dtof_mid_d_{parameter}"] = detail["derivative_tof_mid_s_per_unit"]
            row[f"dtof_ab_d_{parameter}"] = detail["derivative_tof_ab_s_per_unit"]
            row[f"dtof_ba_d_{parameter}"] = detail["derivative_tof_ba_s_per_unit"]
            row[f"scheme_{parameter}"] = detail["scheme"]
            row[f"step_disagreement_{parameter}"] = detail["step_disagreement"]
            row[f"stable_{parameter}"] = detail["stable"]
            if not detail["stable"]:
                audit_errors.append(f"{point_id} {parameter} finite-difference stability failed")
        sensitivity_rows.append(row)
        fisher = fisher_information_bidir(
            derivatives,
            tof_std_s=tof_std_s,
            temperature_std_c=t_std,
            parameter_steps=config["finite_difference"]["steps"],
        )
        fisher_rows.append(
            {"point_id": point_id, "scope": scope, "window_id": window_id, **fisher}
        )
        dtof_o2 = float(derivatives["o2_percent"]["derivative_tof_mid_s_per_unit"])
        for scenario in uncertainty_scenarios:
            parameter = scenario["parameter"]
            if parameter == "tof_s":
                nuisance_derivative = 1.0
            else:
                nuisance_derivative = float(
                    derivatives[parameter]["derivative_tof_mid_s_per_unit"]
                )
            equivalent_std = equivalent_o2_std_percent(
                tof_per_o2_s_per_percent=dtof_o2,
                tof_per_nuisance_s_per_unit=nuisance_derivative,
                nuisance_std=float(scenario["std"]),
            )
            budget_rows.append(
                {
                    "point_id": point_id,
                    "scope": scope,
                    "window_id": window_id,
                    "scenario_id": scenario["id"],
                    "parameter": parameter,
                    "source": scenario["source"],
                    "std": scenario["std"],
                    "equivalent_o2_std_percent": equivalent_std,
                }
            )
    return sensitivity_rows, fisher_rows, budget_rows, audit_errors


def _budget_summaries(budget_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Combine only temperature + single-frame jitter for the business P90 gate."""
    gate_scenarios = {
        row["scenario_id"]
        for row in budget_rows
        if row["parameter"] == "t_c" or str(row["scenario_id"]).endswith("_single_frame")
    }
    grouped: dict[tuple[str, str | None, str], list[float]] = defaultdict(list)
    for row in budget_rows:
        if row["scenario_id"] not in gate_scenarios:
            continue
        grouped[(str(row["scope"]), row["window_id"], str(row["point_id"]))].append(
            float(row["equivalent_o2_std_percent"])
        )
    point_rows = []
    for (scope, window_id, point_id), stds in grouped.items():
        point_rows.append(
            {
                "scope": scope,
                "window_id": window_id,
                "point_id": point_id,
                "combined_p90_o2_error_percent": combined_p90_o2_error_percent(stds),
            }
        )
    summaries = []
    for scope, window_id in {(row["scope"], row["window_id"]) for row in point_rows}:
        values = [
            row["combined_p90_o2_error_percent"]
            for row in point_rows
            if row["scope"] == scope and row["window_id"] == window_id
        ]
        summaries.append(
            {
                "scope": scope,
                "window_id": window_id,
                "point_count": len(values),
                "combined_p90_o2_error_percent_min": float(np.min(values)),
                "combined_p90_o2_error_percent_median": float(np.median(values)),
                "combined_p90_o2_error_percent_max": float(np.max(values)),
            }
        )
    return sorted(summaries, key=lambda row: (row["scope"], str(row["window_id"])))


def _nuisance_fraction_summaries(
    budget_rows: list[dict[str, Any]], *, narrow_windows: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    widths = {str(window["id"]): float(window["width_percent"]) for window in narrow_windows}
    grouped: dict[tuple[str, str], list[float]] = defaultdict(list)
    for row in budget_rows:
        if row["scope"] != "narrow_window":
            continue
        window_id = str(row["window_id"])
        grouped[(window_id, str(row["scenario_id"]))].append(float(row["equivalent_o2_std_percent"]))
    summaries = []
    for (window_id, scenario_id), stds in grouped.items():
        signal_width_percent = widths[window_id]
        equivalent_o2_p90_percent = NORMAL_P90_Z * max(stds)
        summaries.append(
            {
                "window_id": window_id,
                "scenario_id": scenario_id,
                "signal_width_percent": signal_width_percent,
                "equivalent_o2_p90_percent_max": equivalent_o2_p90_percent,
                "nuisance_fraction_of_signal_max": equivalent_o2_p90_percent / signal_width_percent,
            }
        )
    return sorted(summaries, key=lambda row: (row["window_id"], row["scenario_id"]))


def _dominant_terms(budget_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Rank mean equivalent O2 σ by scenario across narrow-window points."""
    grouped: dict[str, list[float]] = defaultdict(list)
    for row in budget_rows:
        if row["scope"] != "narrow_window":
            continue
        grouped[str(row["scenario_id"])].append(float(row["equivalent_o2_std_percent"]))
    ranked = []
    for scenario_id, values in grouped.items():
        ranked.append(
            {
                "scenario_id": scenario_id,
                "mean_equivalent_o2_std_percent": float(np.mean(values)),
                "max_equivalent_o2_std_percent": float(np.max(values)),
                "p90_equivalent_o2_percent": NORMAL_P90_Z * float(np.max(values)),
            }
        )
    ranked.sort(key=lambda row: row["mean_equivalent_o2_std_percent"], reverse=True)
    for index, row in enumerate(ranked, start=1):
        row["rank"] = index
    return ranked


def _business_gate_assessment(
    config: dict[str, Any],
    summaries: list[dict[str, Any]],
    nuisance_fraction_summaries: list[dict[str, Any]],
    *,
    evaluated_point_count: int,
) -> dict[str, Any]:
    thresholds = config["business_thresholds"]
    narrow_max = max(
        row["combined_p90_o2_error_percent_max"]
        for row in summaries
        if row["scope"] == "narrow_window"
    )
    # Gate uses temperature + single-frame jitter fractions only (exclude n64).
    gate_frac_rows = [
        row
        for row in nuisance_fraction_summaries
        if row["scenario_id"] == "temperature_1K_scenario"
        or str(row["scenario_id"]).endswith("_single_frame")
    ]
    worst_nuisance_fraction = max(row["nuisance_fraction_of_signal_max"] for row in gate_frac_rows)
    target_p90 = thresholds["target_p90_o2_error_percent"]
    max_nuisance_fraction = thresholds["max_nuisance_fraction_of_signal"]
    max_rejection_rate = thresholds["max_rejection_rate"]
    blocking_nuisances = [
        item["name"] for item in config["representation_audit"] if item["blocks_go_verdict"]
    ]
    rejection_policy = config["rejection_policy"]
    rejected_point_count = 0
    if rejection_policy["reject_if_unrepresented_blocking_nuisance"] and blocking_nuisances:
        rejected_point_count = evaluated_point_count
    observed_rejection_rate = rejected_point_count / evaluated_point_count
    return {
        "target_p90_o2_error_percent": {
            "threshold": target_p90,
            "observed_narrow_window_max": narrow_max,
            "status": "passed" if narrow_max <= target_p90 else "failed",
        },
        "max_nuisance_fraction_of_signal": {
            "threshold": max_nuisance_fraction,
            "observed_worst_case": worst_nuisance_fraction,
            "status": (
                "passed" if worst_nuisance_fraction <= max_nuisance_fraction else "failed"
            ),
        },
        "max_rejection_rate": {
            "threshold": max_rejection_rate,
            "evaluated_point_count": evaluated_point_count,
            "rejected_point_count": rejected_point_count,
            "observed_rejection_rate": observed_rejection_rate,
            "blocking_nuisances": blocking_nuisances,
            "policy_id": rejection_policy["id"],
            "status": (
                "passed" if observed_rejection_rate <= max_rejection_rate else "failed"
            ),
        },
    }


def _choose_verdict(
    config: dict[str, Any],
    assessment: dict[str, Any],
    *,
    acoustic_full_rank: bool,
    nuisance_marginalized: bool,
) -> dict[str, Any]:
    if not acoustic_full_rank:
        return {
            "status": "audit_failed",
            "reason": "Bidirectional acoustic Fisher subsystem is rank-deficient (expected rank>=2).",
            "business_gate_assessment": assessment,
        }
    blocking_nuisances = [
        item["name"] for item in config["representation_audit"] if item["blocks_go_verdict"]
    ]
    if blocking_nuisances:
        return {
            "status": "information_source_upgrade_required",
            "blocking_nuisances": blocking_nuisances,
            "reason": "Unrepresented nuisance mechanisms block a continuous-regression claim.",
            "business_gate_assessment": assessment,
        }
    gates_passed = all(gate["status"] == "passed" for gate in assessment.values())
    if gates_passed and not nuisance_marginalized:
        return {
            "status": "coarse_monitoring_only",
            "reason": (
                "conditional_o2_only_nuisance_not_marginalized: business gates pass, "
                "but joint Fisher cannot marginalize CO2/L/(other) nuisances under the "
                "registered AB/BA(+T) observation model."
            ),
            "business_gate_assessment": assessment,
        }
    if gates_passed and nuisance_marginalized:
        return {
            "status": "continuous_regression_supported",
            "scope": "bidir_registered_simulation",
            "business_gate_assessment": assessment,
        }
    return {
        "status": "coarse_monitoring_only",
        "reason": (
            "Flow is implemented and acoustic Fisher rank>=2, but narrow-window P90 "
            "and/or nuisance fraction still exceed registered gates under declared T/jitter."
        ),
        "business_gate_assessment": assessment,
    }


def _prior_crosscheck(config: dict[str, Any]) -> dict[str, Any]:
    ref = config["prior_crosscheck"]["reference_point"]
    point = BidirAcousticPoint(
        float(ref["co2_percent"]),
        float(ref["o2_percent"]),
        float(ref["t_c"]),
        float(ref["path_length_m"]),
        float(ref["v_path_m_per_s"]),
    )
    bounds = {name: tuple(values) for name, values in config["parameter_bounds"].items()}
    derivatives = local_bidir_tof_sensitivity(
        point,
        parameter_steps=config["finite_difference"]["steps"],
        parameter_bounds=bounds,
        fixed_delay_s=float(config["observation"]["fixed_delay_s"]),
        max_relative_step_disagreement=float(
            config["finite_difference"]["max_relative_step_disagreement"]
        ),
    )
    dtof_o2 = float(derivatives["o2_percent"]["derivative_tof_mid_s_per_unit"])
    dtof_t = float(derivatives["t_c"]["derivative_tof_mid_s_per_unit"])
    expected = config["prior_crosscheck"]["expected_single_frame_o2_vol_percent"]
    tol = float(config["prior_crosscheck"]["relative_tolerance"])

    def _eq_o2(std: float, derivative: float) -> float:
        return equivalent_o2_std_percent(
            tof_per_o2_s_per_percent=dtof_o2,
            tof_per_nuisance_s_per_unit=derivative,
            nuisance_std=std,
        )

    observed = {
        "jitter_3us": _eq_o2(midpair_tof_std_s(3.0e-6), 1.0),
        "jitter_0p5us": _eq_o2(midpair_tof_std_s(5.0e-7), 1.0),
        "temperature_1K": _eq_o2(1.0, dtof_t),
    }
    checks = {}
    for key, exp in expected.items():
        obs = observed[key]
        rel = abs(obs - float(exp)) / max(float(exp), 1e-12)
        checks[key] = {
            "expected_vol_percent": float(exp),
            "observed_vol_percent": obs,
            "relative_error": rel,
            "within_tolerance": rel <= tol,
        }
    return {
        "reference_point": ref,
        "dtof_mid_d_o2": dtof_o2,
        "checks": checks,
        "all_within_tolerance": all(item["within_tolerance"] for item in checks.values()),
        "notes": config["prior_crosscheck"]["notes"],
    }


def run_identifiability_v2(
    config_path: Path,
    *,
    project_root: Path = _TV3_ROOT,
    allow_overwrite: bool = False,
) -> Path:
    config = _read_json(config_path)
    _validate_config(config)
    prerequisites = _verify_prerequisites(project_root, config)
    output_dir = _resolve(project_root, config["output_dir"])
    if output_dir.exists():
        existing = sorted(p.name for p in output_dir.iterdir())
        if existing and not allow_overwrite:
            raise FileExistsError(
                f"identifiability v2 output already exists: {output_dir} ({existing})"
            )
    output_dir.mkdir(parents=True, exist_ok=True)

    scenario_payloads: dict[str, Any] = {}
    all_acoustic_full_rank = True
    for jitter in config["jitter_scenarios"]:
        jitter_id = str(jitter["id"])
        tof_std_s = float(jitter["std_s"])
        uncertainty = _uncertainty_scenarios_for_jitter(
            config, jitter_id=jitter_id, tof_std_s=tof_std_s
        )
        global_points = build_bidir_points(config["global_grid"])
        all_sensitivity, all_fisher, all_budget, audit_errors = _point_rows(
            global_points,
            scope="global",
            window_id=None,
            config=config,
            point_offset=0,
            tof_std_s=tof_std_s,
            uncertainty_scenarios=uncertainty,
        )
        point_offset = len(global_points)
        for window in config["narrow_windows"]:
            points = _points_for_window(window, config["narrow_context_grid"])
            sensitivity, fisher, budget, errors = _point_rows(
                points,
                scope="narrow_window",
                window_id=window["id"],
                config=config,
                point_offset=point_offset,
                tof_std_s=tof_std_s,
                uncertainty_scenarios=uncertainty,
            )
            all_sensitivity.extend(sensitivity)
            all_fisher.extend(fisher)
            all_budget.extend(budget)
            audit_errors.extend(errors)
            point_offset += len(points)
        if audit_errors:
            detail = "\n".join(f"- {error}" for error in audit_errors)
            raise RuntimeError(f"identifiability v2 audit failed:\n{detail}")

        summaries = _budget_summaries(all_budget)
        nuisance_fraction_summaries = _nuisance_fraction_summaries(
            all_budget, narrow_windows=config["narrow_windows"]
        )
        dominant = _dominant_terms(all_budget)
        assessment = _business_gate_assessment(
            config,
            summaries,
            nuisance_fraction_summaries,
            evaluated_point_count=point_offset,
        )
        acoustic_full_rank = all(
            bool(row["acoustic_subsystem_full_rank"]) for row in all_fisher
        )
        nuisance_marginalized = all(
            row["nuisance_marginalized_status"] == "available" for row in all_fisher
        )
        min_rank = int(min(int(row["joint_rank"]) for row in all_fisher))
        all_acoustic_full_rank = all_acoustic_full_rank and acoustic_full_rank
        scenario_dir = output_dir / jitter_id
        scenario_dir.mkdir(parents=True, exist_ok=True)
        _write_csv(scenario_dir / "sensitivity.csv", all_sensitivity)
        _write_csv(scenario_dir / "fisher_information.csv", all_fisher)
        _write_csv(scenario_dir / "error_budget.csv", all_budget)
        _write_csv(scenario_dir / "narrow_window_summary.csv", summaries)
        _write_csv(scenario_dir / "nuisance_fraction_summary.csv", nuisance_fraction_summaries)
        _write_csv(scenario_dir / "dominant_terms.csv", dominant)
        scenario_payloads[jitter_id] = {
            "tof_std_s": tof_std_s,
            "source": jitter.get("source"),
            "ref": jitter.get("ref"),
            "evaluated_point_count": point_offset,
            "min_joint_rank": min_rank,
            "acoustic_subsystem_full_rank": acoustic_full_rank,
            "nuisance_marginalized": nuisance_marginalized,
            "business_gate_assessment": assessment,
            "narrow_window_summaries": [
                row for row in summaries if row["scope"] == "narrow_window"
            ],
            "dominant_terms": dominant,
            "nuisance_fraction_summaries": nuisance_fraction_summaries,
        }

    prior = _prior_crosscheck(config)
    # Verdict uses the stricter of the two jitter arms for go/no-go, but both are reported.
    # Prefer naming the nominal arm explicitly for "at least nominal" continuous claim.
    nominal_id = "nominal_daq_half_sample"
    conservative_id = "conservative_v1"
    if nominal_id not in scenario_payloads or conservative_id not in scenario_payloads:
        raise RuntimeError("F0 dual jitter scenarios must both be present in results")

    verdict_by_scenario = {
        sid: _choose_verdict(
            config,
            payload["business_gate_assessment"],
            acoustic_full_rank=bool(payload["acoustic_subsystem_full_rank"]),
            nuisance_marginalized=bool(payload["nuisance_marginalized"]),
        )
        for sid, payload in scenario_payloads.items()
    }
    # Stage-level business status: continuous only if nominal passes all gates
    # with nuisance marginalization available; otherwise coarse_monitoring_only
    # if flow unblocked and Fisher acoustic rank ok.
    if not all_acoustic_full_rank:
        stage_status = "audit_failed"
        stage_reason = "Acoustic Fisher subsystem rank < 2 under at least one jitter scenario."
    elif verdict_by_scenario[nominal_id]["status"] == "continuous_regression_supported":
        stage_status = "continuous_regression_supported"
        stage_reason = (
            "Nominal jitter scenario passes all three business gates with flow implemented "
            "and nuisance parameters marginalized."
        )
    elif any(
        v["status"] == "information_source_upgrade_required" for v in verdict_by_scenario.values()
    ):
        stage_status = "information_source_upgrade_required"
        stage_reason = "Blocking unrepresented nuisance remains."
    else:
        stage_status = "coarse_monitoring_only"
        stage_reason = verdict_by_scenario[nominal_id].get(
            "reason",
            "Flow decoupled; registered T/jitter budget still exceeds continuous-regression gates.",
        )

    created_at = datetime.now(timezone.utc).isoformat()
    f5_prereg = config["f5_amplitude_gate_preregistration"]
    composition_domain = str(config.get("composition_domain", "narrow"))
    next_stage = (
        "F5_wide_formal_model_protocol"
        if composition_domain == COMPOSITION_DOMAIN_WIDE
        else "F5_formal_model_protocol"
    )
    metrics = {
        "schema_version": AUDIT_SCHEMA,
        "composition_domain": composition_domain,
        "created_at": created_at,
        "prerequisites": prerequisites,
        "jitter_scenarios": scenario_payloads,
        "prior_crosscheck": prior,
        "verdict_by_scenario": {
            sid: {"status": payload["status"], "business_gate_assessment": payload["business_gate_assessment"]}
            for sid, payload in verdict_by_scenario.items()
        },
        "f5_amplitude_gate_preregistration": f5_prereg,
        "joint_fisher_status": (
            "acoustic_subsystem_full_rank_ge_2"
            if all_acoustic_full_rank
            else "acoustic_subsystem_rank_deficient"
        ),
        "v1_comparison": {
            "v1_verdict": "information_source_upgrade_required",
            "v1_blocking": "flow_projection",
            "v2_flow_representation": "implemented_physics",
            "v1_output_dir_untouched": "outputs/tv3_identifiability",
            "narrow_f4_output_dir_untouched": "outputs/tv3_bidir/identifiability_v2",
        },
    }
    verdict = {
        "stage": "F4_wide" if composition_domain == COMPOSITION_DOMAIN_WIDE else "F4",
        "schema_version": AUDIT_SCHEMA,
        "composition_domain": composition_domain,
        "verdict": stage_status,
        "passed": stage_status != "audit_failed",
        "reason": stage_reason,
        "created_at": created_at,
        "verdict_by_scenario": {sid: v["status"] for sid, v in verdict_by_scenario.items()},
        "business_gate_assessment_nominal": scenario_payloads[nominal_id][
            "business_gate_assessment"
        ],
        "business_gate_assessment_conservative": scenario_payloads[conservative_id][
            "business_gate_assessment"
        ],
        "prior_crosscheck_passed": prior["all_within_tolerance"],
        "f5_amplitude_gate_preregistration": f5_prereg,
        "allowed_next_stage_on_pass": next_stage,
        "note": (
            "F4 stage pass means the audit completed with acoustic Fisher rank>=2 and "
            "flow unblocked; continuous_regression_supported additionally requires "
            "nuisance_marginalized_status=available (full joint rank). Current AB/BA(+T) "
            "observation model cannot marginalize five parameters, so continuous is unreachable "
            "until NDIR/TCS (or other) observables enter the Fisher. "
            "Wide-domain F4 does not rewrite narrow F4 / v1 directories or coarse_monitoring_only physics wall."
        ),
    }

    _write_json(output_dir / "representation_audit.json", {"parameters": config["representation_audit"]})
    _write_json(
        output_dir / "manifest.json",
        {
            "schema_version": AUDIT_SCHEMA,
            "composition_domain": composition_domain,
            "config_sha256": _sha256(config_path),
            "prerequisites": prerequisites,
            "observation": config["observation"],
            "uncertainty_model": config["uncertainty_model"],
            "business_thresholds": config["business_thresholds"],
            "rejection_policy": config["rejection_policy"],
            "independent_observables": ["observed_tof_ab_s", "observed_tof_ba_s", "T_C"],
            "shared_observables_excluded_from_fisher": ["sound_speed_m_per_s", "v_hat_from_tof"],
            "jitter_scenario_ids": [item["id"] for item in config["jitter_scenarios"]],
            "narrow_windows": config.get("narrow_windows"),
            "parameter_bounds": config.get("parameter_bounds"),
        },
    )
    _write_json(output_dir / "metrics.json", metrics)
    _write_json(
        output_dir / "audit.json",
        {
            "status": "passed" if stage_status != "audit_failed" else "failed",
            "composition_domain": composition_domain,
            "checks": [
                "f0_registry_sha256",
                "f3_dsp_passed_prerequisite",
                "composition_closure",
                "finite_difference_step_stability",
                "bidir_tof_pair_without_sound_speed_double_counting",
                "flow_implemented_physics_nonblocking",
                "dual_jitter_parallel_reporting",
                "prior_table_crosscheck",
            ],
            "prior_crosscheck_passed": prior["all_within_tolerance"],
        },
    )
    _write_json(output_dir / "verdict.json", verdict)
    _write_json(output_dir / "f4_verdict.json", verdict)
    readme_title = (
        "# tv3 双向可辨识性审计 v2-wide（F4-wide）\n\n"
        if composition_domain == COMPOSITION_DOMAIN_WIDE
        else "# tv3 双向可辨识性审计 v2（F4）\n\n"
    )
    (output_dir / "README.md").write_text(
        readme_title
        + "本目录独立于 `outputs/tv3_identifiability/`（v1 单向）"
        + (
            "与 `outputs/tv3_bidir/identifiability_v2/`（窄域 F4）。\n\n"
            if composition_domain == COMPOSITION_DOMAIN_WIDE
            else "。\n\n"
        )
        + "- 观测：AB/BA TOF + 登记 T（Fisher 附加行）；误差预算用 mid-pair TOF。\n"
        "- flow_projection：`implemented_physics`，不再阻断 verdict。\n"
        "- jitter：`conservative_v1` 与 `nominal_daq_half_sample` 并行子目录。\n"
        "- NDIR/TCS：登记为慢通道可观测，本轮未进入声学 Fisher（无 ∂V/∂x 灵敏度模型）。\n",
        encoding="utf-8",
    )
    _update_stage_status_f4_wide(verdict=verdict, output_dir=output_dir)
    return output_dir


def _update_stage_status_f4_wide(*, verdict: dict[str, Any], output_dir: Path) -> None:
    """Write f4_wide only — never rewrite narrow f4 / allowed_next_stage."""
    if verdict.get("composition_domain") != COMPOSITION_DOMAIN_WIDE or not verdict.get("passed"):
        return
    stage_path = default_config_dir() / "stage_status.json"
    if not stage_path.is_file():
        return
    stage = json.loads(stage_path.read_text(encoding="utf-8"))
    gates_nom = verdict.get("business_gate_assessment_nominal") or {}
    gates_con = verdict.get("business_gate_assessment_conservative") or {}
    stage["f4_wide"] = {
        "verdict": verdict.get("verdict"),
        "stage_passed": True,
        "passed_at": datetime.now(timezone.utc).date().isoformat(),
        "verdict_path": "outputs/tv3_bidir/identifiability_v2_wide/f4_verdict.json",
        "metrics_path": "outputs/tv3_bidir/identifiability_v2_wide/metrics.json",
        "composition_domain": "wide",
        "flow_representation": "implemented_physics",
        "prior_crosscheck_passed": verdict.get("prior_crosscheck_passed"),
        "f5_amplitude_gate_preregistered": True,
        "criterion_c_anchor": (verdict.get("f5_amplitude_gate_preregistration") or {}).get(
            "criterion_c_anchor"
        ),
        "narrow_p90_max_nominal": (gates_nom.get("target_p90_o2_error_percent") or {}).get(
            "observed_narrow_window_max"
        ),
        "narrow_p90_max_conservative": (gates_con.get("target_p90_o2_error_percent") or {}).get(
            "observed_narrow_window_max"
        ),
        "allowed_next_stage": verdict.get("allowed_next_stage_on_pass"),
        "narrow_f4_directory_untouched": True,
        "v1_directory_untouched": True,
    }
    stage_path.write_text(json.dumps(stage, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=_TV3_ROOT / "configs" / "tv3_bidir_identifiability_v2.json",
    )
    parser.add_argument("--allow-overwrite", action="store_true")
    args = parser.parse_args(argv)
    output_dir = run_identifiability_v2(
        args.config, allow_overwrite=bool(args.allow_overwrite)
    )
    verdict = _read_json(output_dir / "f4_verdict.json")
    print(
        json.dumps(
            {
                "status": "passed" if verdict.get("passed") else "failed",
                "verdict": verdict.get("verdict"),
                "verdict_by_scenario": verdict.get("verdict_by_scenario"),
                "prior_crosscheck_passed": verdict.get("prior_crosscheck_passed"),
                "output_dir": str(output_dir),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0 if verdict.get("passed") else 1


if __name__ == "__main__":
    raise SystemExit(main())
