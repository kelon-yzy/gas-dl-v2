"""运行 tv3 冻结 v1 单向 TOF 可辨识性与误差预算审计。"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections import defaultdict
from itertools import product
from pathlib import Path
from typing import Any

import numpy as np

from tv3.audit.error_budget import NORMAL_P90_Z, combined_p90_o2_error_percent, equivalent_o2_std_percent
from tv3.audit.identifiability import (
    AcousticPoint,
    build_points,
    fisher_information,
    local_tof_sensitivity,
    observed_tof_s,
    sound_speed_m_per_s,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
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


def _verify_baseline(project_root: Path, config: dict[str, Any]) -> dict[str, Any]:
    baseline = config["baseline"]
    baseline_dir = _resolve(project_root, baseline["dir"])
    manifest_path = baseline_dir / "manifest.json"
    if _sha256(manifest_path) != baseline["manifest_sha256"]:
        raise RuntimeError("frozen baseline manifest hash does not match the identifiability config")
    manifest = _read_json(manifest_path)
    verdict = _read_json(baseline_dir / "verdict.json")
    if verdict.get("status") != "frozen":
        raise RuntimeError("baseline verdict is not frozen")
    expected_contract = baseline["expected_contract"]
    if manifest.get("contract") != expected_contract:
        raise RuntimeError("frozen baseline contract does not match the identifiability config")
    return {"dir": str(baseline_dir), "manifest": manifest, "manifest_sha256": baseline["manifest_sha256"]}


def _validate_config(config: dict[str, Any]) -> None:
    if config.get("schema_version") != "tv3-identifiability-1":
        raise ValueError("schema_version must be tv3-identifiability-1")
    uncertainty_model = config.get("uncertainty_model")
    if not isinstance(uncertainty_model, dict) or uncertainty_model.get("type") != "diagonal":
        raise ValueError("only an explicit diagonal uncertainty_model is supported")
    if not isinstance(uncertainty_model.get("source"), str) or not uncertainty_model["source"]:
        raise ValueError("uncertainty_model.source must be a non-empty string")
    for item in config["representation_audit"]:
        missing = REPRESENTATION_FIELDS - set(item)
        if missing:
            raise ValueError(f"representation audit is missing fields: {sorted(missing)}")
    for scenario in config["uncertainty_scenarios"]:
        for field in ("id", "parameter", "std", "source"):
            if field not in scenario:
                raise ValueError(f"uncertainty scenario is missing {field}")
        if scenario["parameter"] not in {"tof_s", "o2_percent", "co2_percent", "t_c", "path_length_m"}:
            raise ValueError(f"unsupported uncertainty parameter: {scenario['parameter']!r}")
    thresholds = config.get("business_thresholds")
    if not isinstance(thresholds, dict) or set(thresholds) != BUSINESS_THRESHOLD_FIELDS:
        raise ValueError(f"business_thresholds must contain exactly {sorted(BUSINESS_THRESHOLD_FIELDS)}")
    target_p90 = thresholds["target_p90_o2_error_percent"]
    if target_p90 is not None and (not math.isfinite(float(target_p90)) or float(target_p90) <= 0.0):
        raise ValueError("target_p90_o2_error_percent must be finite and > 0 when configured")
    max_nuisance_fraction = thresholds["max_nuisance_fraction_of_signal"]
    if max_nuisance_fraction is not None and (
        not math.isfinite(float(max_nuisance_fraction)) or not 0.01 <= float(max_nuisance_fraction) <= 1.0
    ):
        raise ValueError("max_nuisance_fraction_of_signal must be within [0.01, 1.0] when configured")
    max_rejection_rate = thresholds["max_rejection_rate"]
    rejection_policy = config.get("rejection_policy")
    if max_rejection_rate is None:
        if rejection_policy is not None:
            raise ValueError("rejection_policy requires a configured max_rejection_rate")
        return
    if not math.isfinite(float(max_rejection_rate)) or not 0.0 <= float(max_rejection_rate) <= 1.0:
        raise ValueError("max_rejection_rate must be within [0.0, 1.0] when configured")
    if not isinstance(rejection_policy, dict) or set(rejection_policy) != REJECTION_POLICY_FIELDS:
        raise ValueError(f"rejection_policy must contain exactly {sorted(REJECTION_POLICY_FIELDS)}")
    if not isinstance(rejection_policy["id"], str) or not rejection_policy["id"]:
        raise ValueError("rejection_policy.id must be a non-empty string")
    if not isinstance(rejection_policy["source"], str) or not rejection_policy["source"]:
        raise ValueError("rejection_policy.source must be a non-empty string")
    if not isinstance(rejection_policy["reject_if_unrepresented_blocking_nuisance"], bool):
        raise ValueError("rejection_policy.reject_if_unrepresented_blocking_nuisance must be boolean")


def _points_for_window(window: dict[str, Any], context_grid: dict[str, list[float]]) -> list[AcousticPoint]:
    points = []
    for co2, t_c, path_length in product(
        context_grid["co2_percent"], context_grid["t_c"], context_grid["path_length_m"]
    ):
        points.append(AcousticPoint(float(co2), float(window["center_percent"]), float(t_c), float(path_length)))
    return points


def _point_rows(
    points: list[AcousticPoint],
    *,
    scope: str,
    window_id: str | None,
    config: dict[str, Any],
    point_offset: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], list[str]]:
    fixed_delay_s = float(config["observation"]["fixed_delay_s"])
    tof_std_s = float(config["observation"]["tof_std_s"])
    bounds = {name: tuple(values) for name, values in config["parameter_bounds"].items()}
    sensitivity_rows: list[dict[str, Any]] = []
    fisher_rows: list[dict[str, Any]] = []
    budget_rows: list[dict[str, Any]] = []
    audit_errors: list[str] = []
    for index, point in enumerate(points, start=point_offset):
        point_id = f"{scope}-{index:04d}"
        derivatives = local_tof_sensitivity(
            point,
            parameter_steps=config["finite_difference"]["steps"],
            parameter_bounds=bounds,
            fixed_delay_s=fixed_delay_s,
            max_relative_step_disagreement=float(config["finite_difference"]["max_relative_step_disagreement"]),
        )
        row: dict[str, Any] = {
            "point_id": point_id,
            "scope": scope,
            "window_id": window_id,
            "x_CO2_percent": point.co2_percent,
            "x_O2_percent": point.o2_percent,
            "x_N2_percent": point.n2_percent,
            "T_C": point.t_c,
            "L_m": point.path_length_m,
            "sound_speed_m_per_s": sound_speed_m_per_s(point),
            "observed_tof_s": observed_tof_s(point, fixed_delay_s=fixed_delay_s),
        }
        for parameter, detail in derivatives.items():
            row[f"dtof_d_{parameter}"] = detail["derivative_tof_s_per_unit"]
            row[f"scheme_{parameter}"] = detail["scheme"]
            row[f"step_disagreement_{parameter}"] = detail["step_disagreement"]
            row[f"stable_{parameter}"] = detail["stable"]
            if not detail["stable"]:
                audit_errors.append(f"{point_id} {parameter} finite-difference stability failed")
        sensitivity_rows.append(row)
        fisher_rows.append({"point_id": point_id, "scope": scope, "window_id": window_id, **fisher_information(derivatives, tof_std_s=tof_std_s)})
        dtof_o2 = float(derivatives["o2_percent"]["derivative_tof_s_per_unit"])
        for scenario in config["uncertainty_scenarios"]:
            parameter = scenario["parameter"]
            nuisance_derivative = 1.0 if parameter == "tof_s" else float(derivatives[parameter]["derivative_tof_s_per_unit"])
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
    grouped: dict[tuple[str, str | None, str], list[float]] = defaultdict(list)
    for row in budget_rows:
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
        values = [row["combined_p90_o2_error_percent"] for row in point_rows if row["scope"] == scope and row["window_id"] == window_id]
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
        if not math.isfinite(signal_width_percent) or signal_width_percent <= 0.0:
            raise ValueError(f"narrow window {window_id!r} must have a finite positive width_percent")
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


def _business_gate_assessment(
    config: dict[str, Any],
    summaries: list[dict[str, Any]],
    nuisance_fraction_summaries: list[dict[str, Any]],
    *,
    evaluated_point_count: int,
) -> dict[str, Any]:
    if evaluated_point_count <= 0:
        raise ValueError("evaluated_point_count must be positive")
    thresholds = config["business_thresholds"]
    narrow_max = max(
        row["combined_p90_o2_error_percent_max"] for row in summaries if row["scope"] == "narrow_window"
    )
    worst_nuisance_fraction = max(
        row["nuisance_fraction_of_signal_max"] for row in nuisance_fraction_summaries
    )
    target_p90 = thresholds["target_p90_o2_error_percent"]
    max_nuisance_fraction = thresholds["max_nuisance_fraction_of_signal"]
    max_rejection_rate = thresholds["max_rejection_rate"]
    blocking_nuisances = [item["name"] for item in config["representation_audit"] if item["blocks_go_verdict"]]
    rejection_policy = config.get("rejection_policy")
    rejected_point_count = 0
    if rejection_policy is not None and rejection_policy["reject_if_unrepresented_blocking_nuisance"] and blocking_nuisances:
        rejected_point_count = evaluated_point_count
    observed_rejection_rate = rejected_point_count / evaluated_point_count
    return {
        "target_p90_o2_error_percent": {
            "threshold": target_p90,
            "observed_narrow_window_max": narrow_max,
            "status": "not_configured" if target_p90 is None else ("passed" if narrow_max <= target_p90 else "failed"),
        },
        "max_nuisance_fraction_of_signal": {
            "threshold": max_nuisance_fraction,
            "observed_worst_case": worst_nuisance_fraction,
            "status": (
                "not_configured"
                if max_nuisance_fraction is None
                else ("passed" if worst_nuisance_fraction <= max_nuisance_fraction else "failed")
            ),
        },
        "max_rejection_rate": {
            "threshold": max_rejection_rate,
            "evaluated_point_count": evaluated_point_count,
            "rejected_point_count": rejected_point_count,
            "observed_rejection_rate": observed_rejection_rate,
            "blocking_nuisances": blocking_nuisances,
            "policy_id": None if rejection_policy is None else rejection_policy["id"],
            "status": (
                "not_configured"
                if max_rejection_rate is None
                else ("passed" if observed_rejection_rate <= max_rejection_rate else "failed")
            ),
        },
    }


def _choose_verdict(
    config: dict[str, Any],
    summaries: list[dict[str, Any]],
    nuisance_fraction_summaries: list[dict[str, Any]],
    *,
    evaluated_point_count: int,
) -> dict[str, Any]:
    thresholds = config["business_thresholds"]
    assessment = _business_gate_assessment(
        config,
        summaries,
        nuisance_fraction_summaries,
        evaluated_point_count=evaluated_point_count,
    )
    missing = [name for name, value in thresholds.items() if value is None]
    if missing:
        return {
            "status": "inconclusive_missing_business_threshold",
            "blocking_fields": missing,
            "reason": "At least one required business field is not configured; no final go/no-go claim is allowed.",
            "business_gate_assessment": assessment,
        }
    blocking_nuisances = [item["name"] for item in config["representation_audit"] if item["blocks_go_verdict"]]
    if blocking_nuisances:
        return {
            "status": "information_source_upgrade_required",
            "blocking_nuisances": blocking_nuisances,
            "reason": "Unrepresented nuisance mechanisms block a continuous-regression claim.",
            "business_gate_assessment": assessment,
        }
    if all(gate["status"] == "passed" for gate in assessment.values()):
        return {"status": "continuous_regression_supported", "business_gate_assessment": assessment}
    return {"status": "coarse_monitoring_only", "business_gate_assessment": assessment}


def run_identifiability(config_path: Path, *, project_root: Path = PROJECT_ROOT) -> Path:
    config = _read_json(config_path)
    _validate_config(config)
    output_dir = _resolve(project_root, config["output_dir"])
    if output_dir.exists():
        raise FileExistsError(f"identifiability output already exists: {output_dir}")
    baseline = _verify_baseline(project_root, config)
    global_points = build_points(config["global_grid"])
    all_sensitivity, all_fisher, all_budget, audit_errors = _point_rows(
        global_points, scope="global", window_id=None, config=config, point_offset=0
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
        )
        all_sensitivity.extend(sensitivity)
        all_fisher.extend(fisher)
        all_budget.extend(budget)
        audit_errors.extend(errors)
        point_offset += len(points)
    if audit_errors:
        detail = "\n".join(f"- {error}" for error in audit_errors)
        raise RuntimeError(f"identifiability audit failed:\n{detail}")
    summaries = _budget_summaries(all_budget)
    nuisance_fraction_summaries = _nuisance_fraction_summaries(
        all_budget, narrow_windows=config["narrow_windows"]
    )
    verdict = _choose_verdict(
        config,
        summaries,
        nuisance_fraction_summaries,
        evaluated_point_count=point_offset,
    )
    output_dir.mkdir(parents=True)
    _write_csv(output_dir / "sensitivity.csv", all_sensitivity)
    _write_csv(output_dir / "fisher_information.csv", all_fisher)
    _write_csv(output_dir / "error_budget.csv", all_budget)
    _write_csv(output_dir / "narrow_window_summary.csv", summaries)
    _write_csv(output_dir / "nuisance_fraction_summary.csv", nuisance_fraction_summaries)
    _write_json(output_dir / "representation_audit.json", {"parameters": config["representation_audit"]})
    _write_json(
        output_dir / "manifest.json",
        {
            "schema_version": "tv3-identifiability-1",
            "config_sha256": _sha256(config_path),
            "baseline": baseline,
            "observation": config["observation"],
            "uncertainty_model": config["uncertainty_model"],
            "business_thresholds": config["business_thresholds"],
            "rejection_policy": config.get("rejection_policy"),
            "independent_observables": ["observed_tof_s"],
            "shared_observables_excluded_from_fisher": ["sound_speed_m_per_s"],
        },
    )
    _write_json(
        output_dir / "metrics.json",
        {
            "global_point_count": len(global_points),
            "narrow_window_count": len(config["narrow_windows"]),
            "narrow_window_summaries": [row for row in summaries if row["scope"] == "narrow_window"],
            "nuisance_fraction_summaries": nuisance_fraction_summaries,
            "business_gate_assessment": verdict["business_gate_assessment"],
            "joint_fisher_status": "unavailable_rank_deficient_with_single_tof_observable",
        },
    )
    _write_json(
        output_dir / "audit.json",
        {
            "status": "passed",
            "checks": [
                "frozen_baseline_hash_and_contract",
                "composition_closure",
                "finite_difference_step_stability",
                "single_tof_observable_without_sound_speed_double_counting",
                "explicit_representation_audit",
            ],
        },
    )
    _write_json(output_dir / "verdict.json", verdict)
    (output_dir / "README.md").write_text(
        "# tv3 可辨识性与误差预算审计\n\n"
        "本产物只审计冻结 v1 单向 TOF 链路。声速由 TOF 派生，未重复计入 Fisher 信息；"
        "未表示 nuisance 不构成已验证物理证据。nuisance_fraction_summary.csv 报告每个已声明情景"
        "相对于 0.8% 窄窗口的最坏等效 O₂ P90 比例。\n",
        encoding="utf-8",
    )
    return output_dir


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="run tv3 identifiability audit")
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args(argv)
    output_dir = run_identifiability(args.config)
    print(f"wrote tv3 identifiability audit: {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
