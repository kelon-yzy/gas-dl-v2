"""MEI-0 registry audit and delta_num recomputation for the MRS-EI line."""
from __future__ import annotations

import hashlib
import json
import math
import multiprocessing
import os
from pathlib import Path
from typing import Any

import numpy as np

from tv3.audit.error_budget import NORMAL_P90_Z
from tv3.audit.identifiability_v2 import DEFAULT_RANK_RELATIVE_TOL
from tv3.audit.identifiability_v3_mrs import MrsPoint, evaluate_point_arm

SCHEMA_VERSION = "tunnel-ventilation-mrs-ei-1"
STAGE = "MEI-0"
CLAIM_SCOPE = "registered_simulation_domain_only"
SOURCE_TAGS = frozenset(
    {
        "implemented_physics",
        "literature_bound",
        "engineering_scenario",
        "not_represented",
    }
)

REGISTRY_FILES = (
    "model_family_registry.json",
    "design_space.json",
    "metric_registry.json",
)

_FORBIDDEN_MUTABLE_KEYS = frozenset(
    {
        "allowed_next_stage",
        "stage_status",
        "mei1_status",
        "mei2_status",
    }
)

_TV3_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_CONFIG_DIR = _TV3_ROOT / "configs" / "tv3_mrs_ei"

_BOUNDS_KEYS = ("co2_percent", "o2_percent", "t_c", "path_length_m", "h_rh")


def default_config_dir() -> Path:
    return _DEFAULT_CONFIG_DIR


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"registry not found: {path}")
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"registry root must be object: {path}")
    return data


def verify_evidence_manifest(
    manifest_path: Path,
    *,
    project_root: Path,
    expected_manifest_sha256: str | None = None,
) -> list[str]:
    issues: list[str] = []
    if not manifest_path.is_file():
        return [f"evidence manifest missing: {manifest_path}"]
    if (
        expected_manifest_sha256 is not None
        and sha256_file(manifest_path) != expected_manifest_sha256
    ):
        issues.append("evidence manifest sha256 mismatch")
        return issues
    manifest = load_json(manifest_path)
    for name, expected in (manifest.get("artifact_sha256") or {}).items():
        path = manifest_path.parent / str(name)
        if not path.is_file():
            issues.append(f"freeze artifact missing: {path}")
        elif sha256_file(path) != expected:
            issues.append(f"freeze artifact sha256 mismatch: {name}")
    for name, evidence in (manifest.get("source_sha256") or {}).items():
        if not isinstance(evidence, dict):
            issues.append(f"source evidence must be object: {name}")
            continue
        path = Path(str(evidence.get("path") or ""))
        if not path.is_absolute():
            path = project_root / path
        if not path.is_file():
            issues.append(f"source evidence missing: {path}")
        elif sha256_file(path) != evidence.get("sha256"):
            issues.append(f"source evidence sha256 mismatch: {name}")
    return issues


def dumps_stable(payload: dict[str, Any]) -> str:
    return json.dumps(payload, indent=2, ensure_ascii=False) + "\n"


def _require_source(name: str, block: dict[str, Any], issues: list[str]) -> None:
    source = block.get("source")
    if source is None:
        issues.append(f"{name}: missing source")
        return
    if source not in SOURCE_TAGS:
        issues.append(f"{name}: invalid source {source!r}")


def _require_refs(name: str, block: dict[str, Any], issues: list[str]) -> None:
    refs = block.get("refs")
    if not isinstance(refs, list) or not refs or not all(
        isinstance(item, str) and item.strip() for item in refs
    ):
        issues.append(f"{name}: quantitative bound requires non-empty refs")


def _audit_identity(name: str, registry: dict[str, Any], issues: list[str]) -> None:
    if registry.get("schema_version") != SCHEMA_VERSION:
        issues.append(f"{name}: schema_version must be {SCHEMA_VERSION}")
    if registry.get("stage") != STAGE:
        issues.append(f"{name}: stage must be {STAGE}")
    if registry.get("claim_scope") != CLAIM_SCOPE:
        issues.append(f"{name}: claim_scope must be {CLAIM_SCOPE}")
    for key in _FORBIDDEN_MUTABLE_KEYS:
        if key in registry:
            issues.append(
                f"{name}: must not contain mutable stage key {key!r}; "
                "use configs/tv3_mrs_ei/stage_status.json"
            )


def build_narrow_points(design: dict[str, Any]) -> list[tuple[str, MrsPoint]]:
    ctx = design["narrow_context_grid"]
    rh_delta = float(ctx["rh_delta_percent"])
    points: list[tuple[str, MrsPoint]] = []
    for window in design["target_direction"]["narrow_windows"]:
        wid = str(window["id"])
        o2 = float(window["center_percent"])
        for co2 in ctx["co2_percent"]:
            for t_c in ctx["t_c"]:
                for l_m in ctx["path_length_m"]:
                    for h_rh in ctx["h_rh"]:
                        for p_mpa in ctx["p_mpa"]:
                            if float(h_rh) + rh_delta > 80.0:
                                continue
                            pt = MrsPoint(
                                float(co2),
                                o2,
                                float(t_c),
                                float(l_m),
                                float(h_rh),
                                float(p_mpa),
                            )
                            points.append((wid, pt))
    return points


def _summarize_p90(values: list[float]) -> dict[str, float]:
    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 0 or not np.isfinite(arr).all():
        raise ValueError("p90 values must be non-empty and finite")
    return {
        "max_p90_o2_percent": float(np.max(arr)),
        "median_p90_o2_percent": float(np.median(arr)),
    }


def _eval_arm_summary(
    points: list[MrsPoint],
    *,
    f_hz: list[float],
    parameter_steps: dict[str, float],
    parameter_bounds: dict[str, list[float]],
    prior_std: dict[str, float],
    jitter_std_s: float,
    relative_amp_std: float,
    fixed_delay_s: float,
    rh_delta: float,
    rank_relative_tol: float,
    max_relative_step_disagreement: float,
) -> dict[str, Any]:
    p90_values: list[float] = []
    joint_ranks: list[int] = []
    unstable_fd_count = 0
    for point in points:
        row = evaluate_point_arm(
            point,
            arm="obs-cfreq",
            f_hz=f_hz,
            parameter_steps=parameter_steps,
            parameter_bounds=parameter_bounds,
            fixed_delay_s=fixed_delay_s,
            rh_delta=rh_delta,
            p_scan_mpa=(0.10, 0.50),
            jitter_std_s=jitter_std_s,
            relative_amp_std=relative_amp_std,
            prior_std=prior_std,
            window_width_percent=0.8,
            max_relative_step_disagreement=max_relative_step_disagreement,
        )
        # rank tol is consumed inside fisher via default; re-evaluate if custom tol needed
        if abs(float(rank_relative_tol) - DEFAULT_RANK_RELATIVE_TOL) > 0.0:
            from tv3.audit.identifiability_v3_mrs import (
                fisher_rank_crb,
                local_mrs_jacobian,
                observation_noise_std,
            )

            loc = local_mrs_jacobian(
                point,
                arm="obs-cfreq",
                f_hz=f_hz,
                parameter_steps=parameter_steps,
                parameter_bounds=parameter_bounds,
                fixed_delay_s=fixed_delay_s,
                rh_delta=rh_delta,
                p_scan_mpa=(0.10, 0.50),
                max_relative_step_disagreement=max_relative_step_disagreement,
            )
            sigmas = observation_noise_std(
                loc["labels"],
                point=point,
                jitter_std_s=jitter_std_s,
                relative_amp_std=relative_amp_std,
            )
            fish = fisher_rank_crb(
                loc["jacobian"],
                row_sigmas=sigmas,
                parameter_steps=parameter_steps,
                prior_std=prior_std,
                rank_relative_tol=float(rank_relative_tol),
            )
            p90 = float(fish["p90_o2_percent"])
            joint_rank = int(fish["joint_rank"])
        else:
            p90 = float(row["p90_o2_percent"])
            joint_rank = int(row["joint_rank"])
        if not math.isfinite(p90):
            raise ValueError("non-finite p90 encountered during delta_num recompute")
        p90_values.append(p90)
        joint_ranks.append(joint_rank)
        if not bool(row["all_stable"]):
            unstable_fd_count += 1
    summary: dict[str, Any] = _summarize_p90(p90_values)
    summary.update(
        {
            "min_joint_rank": min(joint_ranks),
            "max_joint_rank": max(joint_ranks),
            "median_joint_rank": float(np.median(joint_ranks)),
            "rank_vector_sha256": sha256_bytes(
                np.asarray(joint_ranks, dtype=np.int16).tobytes()
            ),
            "unstable_fd_count": unstable_fd_count,
            "process_id": os.getpid(),
            "_joint_ranks": joint_ranks,
        }
    )
    return summary


def _run_delta_summary(payload: dict[str, Any]) -> dict[str, Any]:
    return _eval_arm_summary(**payload)


def _delta_summary_process(
    payload: dict[str, Any],
    connection: Any,
) -> None:
    try:
        connection.send({"result": _run_delta_summary(payload)})
    except BaseException as exc:
        connection.send({"error": f"{type(exc).__name__}: {exc}"})
    finally:
        connection.close()


def _public_summary(summary: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in summary.items() if not key.startswith("_")}


def compute_delta_num(
    design: dict[str, Any],
    metric: dict[str, Any],
) -> dict[str, Any]:
    """Recompute numerical protection band from FD/SVD perturbations and repeats."""
    spec = metric["delta_num"]["recompute_spec"]
    num = metric["numerical_protocol"]
    if spec.get("arm") != "obs-cfreq":
        raise ValueError("MEI-0 delta_num arm must be obs-cfreq")
    if spec.get("noise") != "registered_mrs2":
        raise ValueError("MEI-0 delta_num noise must be registered_mrs2")
    if spec.get("point_set") != "full_narrow_216":
        raise ValueError("MEI-0 delta_num point_set must be full_narrow_216")
    if not math.isclose(
        float(num.get("normal_p90_z", float("nan"))),
        NORMAL_P90_Z,
        rel_tol=1e-15,
    ):
        raise ValueError("numerical_protocol.normal_p90_z mismatch")
    obs = design["observation_baselines"]["registered_mrs2"]

    points_labeled = build_narrow_points(design)
    expected_n = int(design["narrow_context_grid"]["expected_n_points"])
    if len(points_labeled) != expected_n:
        raise ValueError(
            f"narrow point count {len(points_labeled)} != expected {expected_n}"
        )
    points = [pt for _, pt in points_labeled]

    f_hz = [float(x) for x in spec["frequencies_hz"]]
    base_steps = {k: float(v) for k, v in num["finite_difference_steps"].items()}
    bounds = {k: list(map(float, num["parameter_bounds"][k])) for k in _BOUNDS_KEYS}
    prior = {k: float(v) for k, v in obs["prior_std"].items()}
    jitter = float(obs["jitter_std_s"])
    amp = float(obs["relative_amp_std"])
    delay = float(design["observation_baselines"]["fixed_delay_s"])
    rh_delta = float(design["narrow_context_grid"]["rh_delta_percent"])
    base_tol = float(num["svd_rank_relative_tol"])
    max_dis = float(num["max_relative_step_disagreement"])

    base_payload = {
        "points": points,
        "f_hz": f_hz,
        "parameter_bounds": bounds,
        "prior_std": prior,
        "jitter_std_s": jitter,
        "relative_amp_std": amp,
        "fixed_delay_s": delay,
        "rh_delta": rh_delta,
        "max_relative_step_disagreement": max_dis,
    }

    def run(steps: dict[str, float], tol: float) -> dict[str, Any]:
        return _run_delta_summary(
            {
                **base_payload,
                "parameter_steps": steps,
                "rank_relative_tol": tol,
            }
        )

    nominal = run(base_steps, base_tol)
    n_repeat = int(spec["n_repeat"])
    if spec.get("repeat_execution") != "fresh_process":
        raise ValueError("delta_num repeat_execution must be fresh_process")
    if n_repeat < 2:
        raise ValueError("delta_num n_repeat must be >= 2")
    repeat_payload = {
        **base_payload,
        "parameter_steps": base_steps,
        "rank_relative_tol": base_tol,
    }
    context = multiprocessing.get_context("spawn")
    processes: list[Any] = []
    receivers: list[Any] = []
    for _ in range(n_repeat):
        receiver, sender = context.Pipe(duplex=False)
        process = context.Process(
            target=_delta_summary_process,
            args=(repeat_payload, sender),
        )
        process.start()
        sender.close()
        processes.append(process)
        receivers.append(receiver)
    messages = [receiver.recv() for receiver in receivers]
    for receiver in receivers:
        receiver.close()
    for process in processes:
        process.join()
        if process.exitcode != 0:
            raise RuntimeError(
                f"fresh-process delta recompute exited with code {process.exitcode}"
            )
    errors = [message["error"] for message in messages if "error" in message]
    if errors:
        raise RuntimeError(f"fresh-process delta recompute failed: {errors}")
    repeats = [message["result"] for message in messages]

    perturbations: list[dict[str, Any]] = []
    rel_changes: list[float] = []

    for scale in spec["fd_step_scales"]:
        steps = {k: v * float(scale) for k, v in base_steps.items()}
        summary = run(steps, base_tol)
        for key in spec["metrics"]:
            base = nominal[key]
            rel = abs(summary[key] - base) / base if base > 0.0 else float("inf")
            rel_changes.append(rel)
            perturbations.append(
                {
                    "kind": "fd_step_scale",
                    "scale": float(scale),
                    "metric": key,
                    "value": summary[key],
                    "relative_change": rel,
                }
            )

    svd_rank_diagnostics: list[dict[str, Any]] = []
    nominal_ranks = nominal["_joint_ranks"]
    for scale in spec["svd_tol_scales"]:
        tol = base_tol * float(scale)
        summary = run(base_steps, tol)
        ranks = summary["_joint_ranks"]
        svd_rank_diagnostics.append(
            {
                "scale": float(scale),
                "relative_tol": tol,
                "n_rank_changes_vs_nominal": sum(
                    int(left != right) for left, right in zip(nominal_ranks, ranks)
                ),
                "summary": _public_summary(summary),
            }
        )

    repeat_rels: list[float] = []
    for summary in repeats:
        for key in spec["metrics"]:
            base = nominal[key]
            rel = abs(summary[key] - base) / base if base > 0.0 else float("inf")
            repeat_rels.append(rel)

    max_fd = float(max(rel_changes)) if rel_changes else 0.0
    max_repeat = float(max(repeat_rels)) if repeat_rels else 0.0
    floor = float(metric["delta_num"]["floor"])
    delta_num = max(2.0 * max_fd, 2.0 * max_repeat, floor)

    return {
        "delta_num": delta_num,
        "floor": floor,
        "max_relative_change_fd": max_fd,
        "max_relative_change_fresh_process_repeat": max_repeat,
        "two_times_fd": 2.0 * max_fd,
        "two_times_fresh_process_repeat": 2.0 * max_repeat,
        "nominal": _public_summary(nominal),
        "fresh_process_repeats": [_public_summary(item) for item in repeats],
        "repeat_execution": "fresh_process",
        "perturbations": perturbations,
        "svd_rank_diagnostics": svd_rank_diagnostics,
        "svd_tolerance_policy": "rank_diagnostic_only_not_part_of_crb_p90_delta",
        "n_points": len(points),
        "arm": spec["arm"],
        "frequencies_hz": f_hz,
        "noise": spec["noise"],
    }


def _audit_hashed_json_verdict(
    *,
    name: str,
    block: dict[str, Any],
    path_key: str,
    hash_key: str,
    root: Path,
    issues: list[str],
) -> None:
    path = root / str(block.get(path_key) or "")
    expected_hash = str(block.get(hash_key) or "")
    if not path.is_file():
        issues.append(f"{name} missing: {path}")
        return
    got_hash = sha256_file(path)
    if got_hash != expected_hash:
        issues.append(
            f"{name} sha256 mismatch: expected {expected_hash}, got {got_hash}"
        )
        return
    expected_verdict = block.get("expected_verdict")
    if load_json(path).get("verdict") != expected_verdict:
        issues.append(
            f"{name} verdict mismatch: expected {expected_verdict}, "
            f"got {load_json(path).get('verdict')}"
        )


def _audit_lineage(model: dict[str, Any], issues: list[str], root: Path) -> None:
    lineage = model.get("lineage")
    if not isinstance(lineage, dict):
        issues.append("model_family_registry.lineage missing")
        return
    mrs0 = lineage.get("mrs0_registry") or {}
    path = root / str(mrs0.get("path") or "")
    expected = str(mrs0.get("expected_sha256") or "")
    if not path.is_file():
        issues.append(f"mrs0 registry missing: {path}")
    else:
        got = sha256_file(path)
        if got != expected:
            issues.append(
                f"mrs0 sha256 mismatch: expected {expected}, got {got}"
            )

    mrs1 = lineage.get("mrs1_forward") or {}
    module_path = root / str(mrs1.get("module") or "")
    module_hash = str(mrs1.get("expected_module_sha256") or "")
    if not module_path.is_file():
        issues.append(f"mrs1 forward module missing: {module_path}")
    elif sha256_file(module_path) != module_hash:
        issues.append(
            "mrs1 forward module sha256 mismatch: "
            f"expected {module_hash}, got {sha256_file(module_path)}"
        )
    _audit_hashed_json_verdict(
        name="mrs1 verdict",
        block=mrs1,
        path_key="verdict_path",
        hash_key="expected_verdict_sha256",
        root=root,
        issues=issues,
    )
    _audit_hashed_json_verdict(
        name="mrs2 verdict",
        block=lineage.get("mrs2_verdict") or {},
        path_key="path",
        hash_key="expected_sha256",
        root=root,
        issues=issues,
    )
    _audit_hashed_json_verdict(
        name="mrs6 verdict",
        block=lineage.get("mrs6_verdict") or {},
        path_key="path",
        hash_key="expected_sha256",
        root=root,
        issues=issues,
    )

    stage_path = root / str(
        mrs1.get("stage_status")
        or "configs/tv3_mrs/stage_status.json"
    )
    if not stage_path.is_file():
        issues.append(f"MRS stage_status missing: {stage_path}")
        return
    stage = load_json(stage_path)
    mrs1_expected = (lineage.get("mrs1_forward") or {}).get("expected_verdict")
    if (stage.get("mrs1") or {}).get("verdict") != mrs1_expected:
        issues.append(
            f"mrs1 verdict mismatch: expected {mrs1_expected}, "
            f"got {(stage.get('mrs1') or {}).get('verdict')}"
        )
    mrs2_expected = (lineage.get("mrs2_verdict") or {}).get("expected_verdict")
    if (stage.get("mrs2") or {}).get("verdict") != mrs2_expected:
        issues.append(
            f"mrs2 verdict mismatch: expected {mrs2_expected}, "
            f"got {(stage.get('mrs2') or {}).get('verdict')}"
        )
    mrs6_expected = (lineage.get("mrs6_verdict") or {}).get("expected_verdict")
    if (stage.get("mrs6") or {}).get("verdict") != mrs6_expected:
        issues.append(
            f"mrs6 verdict mismatch: expected {mrs6_expected}, "
            f"got {(stage.get('mrs6') or {}).get('verdict')}"
        )


def _audit_model_family(model: dict[str, Any], issues: list[str]) -> None:
    _require_source("model_family_registry", model, issues)
    families = model.get("model_families")
    if not isinstance(families, list) or not families:
        issues.append("model_families must be non-empty list")
        return
    ids = [f.get("id") for f in families if isinstance(f, dict)]
    if "F0_mrs1_baseline" not in ids:
        issues.append("model_families must include F0_mrs1_baseline")
    required = {
        "F1_humid_air_c_eq",
        "F2_h2o_relaxation_params",
        "F3_coupled_relaxation",
        "F4_diffraction_near_field",
        "F5_transducer_response",
    }
    missing = required.difference(ids)
    if missing:
        issues.append(f"model_families missing required ids: {sorted(missing)}")
    for family in families:
        if not isinstance(family, dict):
            issues.append("model_families entries must be objects")
            continue
        family_name = f"model_family_registry.{family.get('id')}"
        _require_source(family_name, family, issues)
        perturbation = family.get("perturbation")
        if isinstance(perturbation, dict) and perturbation.get("relative_bounds") is not None:
            _require_refs(family_name, family, issues)
        if family.get("id") == "F2_h2o_relaxation_params":
            if family.get("status") != "not_represented":
                issues.append("F2 must remain not_represented without a validated bound")
            if (perturbation or {}).get("relative_bounds") is not None:
                issues.append("F2 must not register an unsupported quantitative bound")
    baseline = model.get("baseline_forward")
    if not isinstance(baseline, dict):
        issues.append("baseline_forward missing")
    else:
        _require_source("baseline_forward", baseline, issues)
        if baseline.get("module") != (
            "tv3/sim/generation/tunnel_ventilation/relaxation_spectrum.py"
        ):
            issues.append("baseline_forward.module must point to relaxation_spectrum.py")


def _measurement_cost(
    frequencies_hz: list[float],
    *,
    cycles: dict[str, Any],
    drive_power_relative: float,
    per_frequency_acquire_s: float,
    inter_frequency_switch_s: float,
    control_overhead_s: float = 0.0,
) -> dict[str, float]:
    energy = 0.0
    for frequency_hz in frequencies_hz:
        key = f"{frequency_hz:g}"
        if key not in cycles:
            raise ValueError(f"missing burst cycle count for {key} Hz")
        energy += float(cycles[key]) / frequency_hz * drive_power_relative
    n_frequencies = len(frequencies_hz)
    return {
        "total_acoustic_energy_relative_s": energy,
        "total_measurement_time_s": (
            n_frequencies * per_frequency_acquire_s
            + max(n_frequencies - 1, 0) * inter_frequency_switch_s
            + control_overhead_s
        ),
    }


def _audit_design(design: dict[str, Any], issues: list[str]) -> None:
    _require_source("design_space", design, issues)
    band = design.get("frequency_band") or {}
    k4 = [float(x) for x in (band.get("baseline_k4_hz") or [])]
    if k4 != [25000.0, 63000.0, 100000.0, 200000.0]:
        issues.append("frequency_band.baseline_k4_hz must be {25,63,100,200} kHz")
    if float(band.get("min_hz", -1)) != 25000.0 or float(band.get("max_hz", -1)) != 200000.0:
        issues.append("frequency_band must be 25–200 kHz")

    points = build_narrow_points(design)
    expected_n = int((design.get("narrow_context_grid") or {}).get("expected_n_points", -1))
    if expected_n != 216:
        issues.append("narrow_context_grid.expected_n_points must be 216")
    if len(points) != expected_n:
        issues.append(
            f"narrow points built={len(points)} expected={expected_n}"
        )

    cost = design.get("cost_function") or {}
    _require_source("cost_function", cost, issues)
    required_terms = {
        "n_frequencies",
        "total_acoustic_energy",
        "total_measurement_time_s",
        "rh_switch_time_s",
        "pressure_switch_time_s",
        "settle_wait_time_s",
    }
    terms = set(cost.get("terms") or [])
    if not required_terms.issubset(terms):
        issues.append(f"cost_function.terms missing {sorted(required_terms - terms)}")

    arm_rows = {
        a.get("id"): a
        for a in (design.get("design_arms") or [])
        if isinstance(a, dict)
    }
    arms = set(arm_rows)
    for aid in ("D0", "D1", "D2", "D3", "D4", "D5"):
        if aid not in arms:
            issues.append(f"design_arms missing {aid}")

    calculator = cost.get("cost_calculator") or {}
    ledger = calculator.get("d0_ledger") or {}
    time_model = cost.get("time_model") or {}
    cycles = cost.get("baseline_burst_cycles") or {}
    d0 = arm_rows.get("D0") or {}
    if d0:
        computed = _measurement_cost(
            [float(value) for value in d0.get("design") or []],
            cycles=cycles,
            drive_power_relative=1.0,
            per_frequency_acquire_s=float(time_model.get("per_frequency_acquire_s", 0.0)),
            inter_frequency_switch_s=float(time_model.get("inter_frequency_switch_s", 0.0)),
        )
        for key, value in computed.items():
            if not math.isclose(float(ledger.get(key, -1.0)), value, rel_tol=1e-12):
                issues.append(f"cost_function.cost_calculator.d0_ledger.{key} mismatch")

    d4 = arm_rows.get("D4") or {}
    if d4.get("eligible_for_information_gate") is not False:
        issues.append("D4 must be excluded from the frozen equal-cost information gate")
    if float(d4.get("total_measurement_time_s", -1.0)) <= float(
        ledger.get("total_measurement_time_s", -1.0)
    ):
        issues.append("D4 diagnostic must record its wall-time budget exceedance")

    d5 = arm_rows.get("D5") or {}
    if d5.get("eligible_for_information_gate") is not False:
        issues.append("D5 must remain a cost-ineligible redundancy diagnostic")
    if d5:
        computed_d5 = _measurement_cost(
            [float(value) for value in d5.get("design") or []],
            cycles=cycles,
            drive_power_relative=float(d5.get("drive_power_relative", 0.0)),
            per_frequency_acquire_s=float(time_model.get("per_frequency_acquire_s", 0.0)),
            inter_frequency_switch_s=float(time_model.get("inter_frequency_switch_s", 0.0)),
        )
        for key, value in computed_d5.items():
            if not math.isclose(float(d5.get(key, -1.0)), value, rel_tol=1e-12):
                issues.append(f"D5 {key} mismatch")

    noise = design.get("noise_families") or {}
    for key in ("N0_independent", "N1_low_rank_common_mode", "N2_mixed"):
        if key not in noise:
            issues.append(f"noise_families missing {key}")
    n1 = noise.get("N1_low_rank_common_mode") or {}
    if n1.get("status") != "sensitivity_analysis_only":
        issues.append("N1 common-mode must remain sensitivity_analysis_only at MEI-0")

    holdout = design.get("holdout_conditions") or {}
    defs = holdout.get("definitions") or {}
    for hid in holdout.get("holdout_ids") or []:
        if hid not in defs:
            issues.append(f"holdout definition missing for {hid}")


def _audit_metric(
    metric: dict[str, Any],
    issues: list[str],
    *,
    require_frozen_delta_num: bool,
) -> None:
    _require_source("metric_registry", metric, issues)
    numerical = metric.get("numerical_protocol") or {}
    if not math.isclose(
        float(numerical.get("normal_p90_z", float("nan"))),
        NORMAL_P90_Z,
        rel_tol=1e-15,
    ):
        issues.append("numerical_protocol.normal_p90_z mismatch")
    rank_protocol = numerical.get("rank_reporting_protocol") or {}
    tolerance_grid = [
        float(value) for value in rank_protocol.get("relative_tolerance_grid") or []
    ]
    if tolerance_grid != [1e-7, 1e-6, 1e-5]:
        issues.append("rank reporting tolerance grid must be [1e-7, 1e-6, 1e-5]")
    if "must hold at every registered tolerance" not in str(
        rank_protocol.get("decision_rule") or ""
    ):
        issues.append("rank reporting decision rule must forbid single-tolerance gates")
    delta = metric.get("delta_num") or {}
    if float(delta.get("floor", -1)) != 0.02:
        issues.append("delta_num.floor must be 0.02")
    if not delta.get("recompute_required_at_freeze"):
        issues.append("delta_num.recompute_required_at_freeze must be true")
    if delta.get("formula") != (
        "max(2*rel_change_fd, 2*rel_change_fresh_process_repeat, floor)"
    ):
        issues.append("delta_num.formula does not match the MEI-0 numerical contract")
    if delta.get("svd_tolerance_policy") != (
        "rank_diagnostic_only_not_part_of_crb_p90_delta"
    ):
        issues.append("delta_num.svd_tolerance_policy missing or invalid")
    spec = delta.get("recompute_spec") or {}
    if spec.get("arm") != "obs-cfreq":
        issues.append("delta_num arm must be obs-cfreq")
    if spec.get("noise") != "registered_mrs2":
        issues.append("delta_num noise must be registered_mrs2")
    if spec.get("point_set") != "full_narrow_216":
        issues.append("delta_num point_set must be full_narrow_216")
    if spec.get("repeat_execution") != "fresh_process":
        issues.append("delta_num repeats must run in fresh processes")
    frozen = delta.get("frozen_value")
    if frozen is None and require_frozen_delta_num:
        issues.append("delta_num.frozen_value missing (run freeze recompute first)")
    elif frozen is not None:
        try:
            fv = float(frozen)
        except (TypeError, ValueError):
            issues.append("delta_num.frozen_value must be float")
            fv = None
        if fv is not None and (not math.isfinite(fv) or fv < 0.02):
            issues.append("delta_num.frozen_value must be finite and >= 0.02")
        artifact = delta.get("recompute_artifact")
        if not isinstance(artifact, dict):
            issues.append("delta_num.recompute_artifact missing")
        else:
            if artifact.get("repeat_execution") != "fresh_process":
                issues.append("delta_num artifact lacks fresh-process repeats")
            if artifact.get("svd_tolerance_policy") != (
                "rank_diagnostic_only_not_part_of_crb_p90_delta"
            ):
                issues.append("delta_num artifact lacks SVD rank-only policy")
            expected_delta = max(
                2.0 * float(artifact.get("max_relative_change_fd", -1.0)),
                2.0
                * float(
                    artifact.get("max_relative_change_fresh_process_repeat", -1.0)
                ),
                float(delta.get("floor", -1.0)),
            )
            if fv is not None and not math.isclose(fv, expected_delta, rel_tol=1e-12):
                issues.append("delta_num.frozen_value does not match recompute artifact")

    auth = metric.get("data_generation_authorization") or {}
    if auth.get("formal_waveform_generation") != "forbidden_until_authorized":
        issues.append("formal waveform generation must remain forbidden at MEI-0")
    for key in ("q1_delta_num", "q4_varpro_linear_block", "q7_authorize_new_data"):
        if key not in (metric.get("review_answers") or {}):
            issues.append(f"metric review_answers missing {key}")


def audit_mei0_registries(
    config_dir: Path | None = None,
    *,
    project_root: Path | None = None,
    require_frozen_delta_num: bool = True,
    registry_overrides: dict[str, dict[str, Any]] | None = None,
) -> dict[str, Any]:
    root = Path(project_root) if project_root is not None else _TV3_ROOT
    cfg = Path(config_dir) if config_dir is not None else default_config_dir()
    issues: list[str] = []

    loaded: dict[str, dict[str, Any]] = {}
    digests: dict[str, str] = {}
    overrides = registry_overrides or {}
    for name in REGISTRY_FILES:
        path = cfg / name
        if name in overrides:
            data = overrides[name]
            if not isinstance(data, dict):
                issues.append(f"registry override root must be object: {name}")
                continue
            digests[name] = sha256_bytes(
                dumps_stable(data).encode("utf-8")
            )
        elif not path.is_file():
            issues.append(f"missing registry file: {path}")
            continue
        else:
            data = load_json(path)
            digests[name] = sha256_file(path)
        loaded[name] = data
        _audit_identity(name, data, issues)

    model = loaded.get("model_family_registry.json")
    design = loaded.get("design_space.json")
    metric = loaded.get("metric_registry.json")

    if model is not None:
        _audit_lineage(model, issues, root)
        _audit_model_family(model, issues)
        for key in ("q2_model_family_coverage",):
            if key not in (model.get("review_answers") or {}):
                issues.append(f"model review_answers missing {key}")

    if design is not None:
        _audit_design(design, issues)
        for key in ("q3_cost_balancing", "q5_mei5_without_hardware", "q6_mixture_id_stability"):
            if key not in (design.get("review_answers") or {}):
                issues.append(f"design review_answers missing {key}")

    if metric is not None:
        _audit_metric(
            metric,
            issues,
            require_frozen_delta_num=require_frozen_delta_num,
        )

    passed = len(issues) == 0
    verdict = "mei0_registry_frozen" if passed else "mei0_registry_incomplete"
    return {
        "schema_version": SCHEMA_VERSION,
        "stage": STAGE,
        "passed": passed,
        "verdict": verdict,
        "allowed_next_stage": "MEI-1_forward_envelope" if passed else None,
        "issues": issues,
        "registry_sha256": digests,
        "config_dir": str(cfg.resolve()),
        "claim_scope": CLAIM_SCOPE,
        "formal_waveform_generation": "forbidden_until_authorized",
    }


def metric_with_delta_num(
    metric: dict[str, Any],
    delta_result: dict[str, Any],
) -> dict[str, Any]:
    updated = json.loads(json.dumps(metric))
    updated["delta_num"]["frozen_value"] = float(delta_result["delta_num"])
    updated["delta_num"]["recompute_artifact"] = {
        "max_relative_change_fd": delta_result["max_relative_change_fd"],
        "max_relative_change_fresh_process_repeat": delta_result[
            "max_relative_change_fresh_process_repeat"
        ],
        "nominal": delta_result["nominal"],
        "fresh_process_repeats": delta_result["fresh_process_repeats"],
        "repeat_execution": delta_result["repeat_execution"],
        "svd_rank_diagnostics": delta_result["svd_rank_diagnostics"],
        "svd_tolerance_policy": delta_result["svd_tolerance_policy"],
        "n_points": delta_result["n_points"],
        "arm": delta_result["arm"],
        "frequencies_hz": delta_result["frequencies_hz"],
        "noise": delta_result["noise"],
    }
    return updated


__all__ = [
    "REGISTRY_FILES",
    "SCHEMA_VERSION",
    "audit_mei0_registries",
    "build_narrow_points",
    "compute_delta_num",
    "default_config_dir",
    "dumps_stable",
    "load_json",
    "metric_with_delta_num",
    "sha256_file",
    "verify_evidence_manifest",
]
