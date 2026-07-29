"""MEI-0 registry audit and delta_numerical recomputation for the MRS-EI line."""
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

REGISTRY_SCHEMA_VERSION = "tunnel-ventilation-mrs-ei-registry-2"
RESERVED_BENCHMARK_SCHEMA_VERSION = "tunnel-ventilation-mrs-ei-1"
FREEZE_MANIFEST_SCHEMA_VERSION = "tunnel-ventilation-mrs-ei-freeze-manifest-2"
# Backward-compatible alias used by older imports/tests during migration.
SCHEMA_VERSION = REGISTRY_SCHEMA_VERSION
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
NOISE_PROFILE_REQUIRED_FIELDS = (
    "jitter_std_s",
    "relative_amp_std",
    "covariance_model",
    "prior_std",
    "fixed_delay_s",
    "source",
    "refs",
)
PRIOR_STD_REQUIRED_KEYS = ("t_c", "path_length_m", "h_rh", "co2_percent")
FAMILY_EVIDENCE_FIELDS = (
    "status",
    "source",
    "refs",
    "implementation_or_holdout_path",
    "evidence_path",
    "evidence_sha256",
    "parameter_or_bias_bounds",
    "bound_semantics",
    "can_clear_not_represented",
)
FAMILY_STATUS_ALLOWED = frozenset(
    {
        "represented_traceable",
        "independent_holdout_available",
        "parked_nonblocking",
        "not_represented",
        "implemented",
        "conditional",
    }
)
AUTHORIZATION_FIELDS = (
    "registered_sparse_simulation_generation",
    "formal_waveform_generation",
    "benchmark_packaging",
    "hardware_trial",
)
FORBIDDEN_AUTH_VALUE = "forbidden_until_explicit_authorization"

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
    schema = manifest.get("schema_version") or manifest.get(
        "freeze_manifest_schema_version"
    )
    if schema not in {
        FREEZE_MANIFEST_SCHEMA_VERSION,
        "tunnel-ventilation-mrs-ei-freeze-manifest-1",
    }:
        issues.append(f"unsupported freeze manifest schema: {schema!r}")
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
    parent_path = manifest.get("parent_manifest_path")
    parent_sha = manifest.get("parent_manifest_sha256")
    if parent_path is not None or parent_sha is not None:
        if not parent_path or not parent_sha:
            issues.append("parent_manifest_path and parent_manifest_sha256 must both be set")
        else:
            path = Path(str(parent_path))
            if not path.is_absolute():
                path = project_root / path
            if not path.is_file():
                issues.append(f"parent manifest missing: {path}")
            elif sha256_file(path) != parent_sha:
                issues.append("parent_manifest_sha256 mismatch")
    plan_path = manifest.get("plan_path")
    plan_sha = manifest.get("plan_sha256")
    if plan_path is not None or plan_sha is not None:
        if not plan_path or not plan_sha:
            issues.append("plan_path and plan_sha256 must both be set")
        else:
            path = Path(str(plan_path))
            if not path.is_absolute():
                path = project_root / path
            if not path.is_file():
                issues.append(f"plan missing: {path}")
            elif sha256_file(path) != plan_sha:
                issues.append("plan_sha256 mismatch")
    return issues


def dumps_stable(payload: dict[str, Any]) -> str:
    return json.dumps(payload, indent=2, ensure_ascii=False) + "\n"


def combined_registry_contract_sha256(
    registries: dict[str, dict[str, Any]],
) -> str:
    """Stable combined hash over normalized registry payloads."""
    ordered = {
        name: registries[name]
        for name in REGISTRY_FILES
        if name in registries
    }
    return sha256_bytes(dumps_stable(ordered).encode("utf-8"))


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
    if registry.get("registry_schema_version") != REGISTRY_SCHEMA_VERSION:
        issues.append(
            f"{name}: registry_schema_version must be {REGISTRY_SCHEMA_VERSION}"
        )
    if registry.get("reserved_benchmark_schema_version") != (
        RESERVED_BENCHMARK_SCHEMA_VERSION
    ):
        issues.append(
            f"{name}: reserved_benchmark_schema_version must be "
            f"{RESERVED_BENCHMARK_SCHEMA_VERSION}"
        )
    if registry.get("registry_schema_version") == registry.get(
        "reserved_benchmark_schema_version"
    ):
        issues.append(
            f"{name}: registry schema must be distinct from reserved benchmark schema"
        )
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


def _point_set_axes(design: dict[str, Any], point_set_id: str) -> dict[str, Any]:
    point_sets = design.get("point_sets") or {}
    if point_set_id not in point_sets:
        raise KeyError(f"unknown point_set_id: {point_set_id}")
    spec = point_sets[point_set_id]
    if "union_of" in spec:
        raise ValueError(
            f"{point_set_id} is a union set; use build_formal_mei1_points"
        )
    return spec


def build_named_point_set(
    design: dict[str, Any],
    point_set_id: str,
) -> list[tuple[str, MrsPoint]]:
    """Build labeled points for a named axis point set (not unions)."""
    ctx = _point_set_axes(design, point_set_id)
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
                            point_id = (
                                f"{point_set_id}|{wid}|"
                                f"co2={float(co2):g}|o2={o2:g}|"
                                f"T={float(t_c):g}|L={float(l_m):g}|"
                                f"RH={float(h_rh):g}|P={float(p_mpa):g}"
                            )
                            points.append((point_id, pt))
    expected = int(ctx["expected_n_points"])
    if len(points) != expected:
        raise ValueError(
            f"{point_set_id} built={len(points)} expected={expected}"
        )
    ids = [pid for pid, _ in points]
    if len(set(ids)) != len(ids):
        raise ValueError(f"{point_set_id} point IDs are not unique")
    return points


def build_formal_mei1_points(
    design: dict[str, Any],
) -> list[tuple[str, MrsPoint]]:
    """Union ambient_core_216 and pressure_extension_low_rh_216 (432 unique)."""
    core = build_named_point_set(design, "ambient_core_216")
    pressure = build_named_point_set(design, "pressure_extension_low_rh_216")
    union_spec = (design.get("point_sets") or {}).get("formal_mei1_432") or {}
    expected = int(union_spec.get("expected_n_points", 432))
    by_key: dict[tuple[float, float, float, float, float, float], tuple[str, MrsPoint]] = {}
    for pid, pt in core + pressure:
        key = (
            round(pt.co2_percent, 6),
            round(pt.o2_percent, 6),
            round(pt.t_c, 6),
            round(pt.path_length_m, 6),
            round(pt.h_rh, 6),
            round(pt.p_mpa, 6),
        )
        if key not in by_key:
            by_key[key] = (pid, pt)
    points = list(by_key.values())
    if len(points) != expected:
        raise ValueError(
            f"formal_mei1_432 unique count {len(points)} != expected {expected}"
        )
    return points


def build_narrow_points(design: dict[str, Any]) -> list[tuple[str, MrsPoint]]:
    """Compatibility alias for ambient_core_216."""
    return build_named_point_set(design, "ambient_core_216")


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
            raise ValueError("non-finite p90 encountered during delta_numerical recompute")
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


def _profile_observation(
    design: dict[str, Any],
    profile_id: str,
) -> dict[str, Any]:
    profiles = design.get("noise_profiles") or {}
    if profile_id not in profiles:
        raise KeyError(f"noise profile missing: {profile_id}")
    profile = profiles[profile_id]
    for field in NOISE_PROFILE_REQUIRED_FIELDS:
        if field not in profile:
            raise KeyError(
                f"mei0_registry_incomplete: low_cost_noise_profile_missing_traceable_fields "
                f"({profile_id}.{field})"
            )
    prior = profile.get("prior_std") or {}
    for key in PRIOR_STD_REQUIRED_KEYS:
        if key not in prior:
            raise KeyError(
                f"mei0_registry_incomplete: low_cost_noise_profile_missing_traceable_fields "
                f"({profile_id}.prior_std.{key})"
            )
    return profile


def _recompute_one_profile(
    *,
    points: list[MrsPoint],
    design: dict[str, Any],
    metric: dict[str, Any],
    profile_id: str,
    point_set_id: str,
) -> dict[str, Any]:
    spec = metric["decision_thresholds"]["delta_numerical"]["recompute_spec"]
    num = metric["numerical_protocol"]
    profile = _profile_observation(design, profile_id)
    f_hz = [float(x) for x in spec["frequencies_hz"]]
    base_steps = {k: float(v) for k, v in num["finite_difference_steps"].items()}
    bounds = {k: list(map(float, num["parameter_bounds"][k])) for k in _BOUNDS_KEYS}
    prior = {k: float(v) for k, v in profile["prior_std"].items()}
    jitter = float(profile["jitter_std_s"])
    amp = float(profile["relative_amp_std"])
    delay = float(profile["fixed_delay_s"])
    rh_delta = float(
        ((design.get("point_sets") or {}).get(point_set_id) or {}).get(
            "rh_delta_percent", 20.0
        )
    )
    base_tol = float(num["svd_rank_relative_tol"])
    max_dis = float(num["max_relative_step_disagreement"])
    optimizer_tol = float(
        metric["decision_thresholds"]["delta_numerical"].get(
            "optimizer_relative_tolerance", 0.0
        )
    )

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
        raise ValueError("delta_numerical repeat_execution must be fresh_process")
    if n_repeat < 2:
        raise ValueError("delta_numerical n_repeat must be >= 2")
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
    # No practical 2% floor inside delta_numerical.
    delta_value = max(2.0 * max_fd, 2.0 * max_repeat, optimizer_tol)
    return {
        "noise_profile": profile_id,
        "delta_numerical": delta_value,
        "max_relative_change_fd": max_fd,
        "max_relative_change_fresh_process_repeat": max_repeat,
        "two_times_fd": 2.0 * max_fd,
        "two_times_fresh_process_repeat": 2.0 * max_repeat,
        "optimizer_relative_tolerance": optimizer_tol,
        "nominal": _public_summary(nominal),
        "fresh_process_repeats": [_public_summary(item) for item in repeats],
        "repeat_execution": "fresh_process",
        "perturbations": perturbations,
        "svd_rank_diagnostics": svd_rank_diagnostics,
        "svd_tolerance_policy": "rank_diagnostic_only_not_part_of_crb_p90_delta",
        "n_points": len(points),
        "arm": spec["arm"],
        "frequencies_hz": f_hz,
        "point_set": point_set_id,
    }


def compute_delta_numerical(
    design: dict[str, Any],
    metric: dict[str, Any],
) -> dict[str, Any]:
    """Recompute numerical protection band for each registered noise profile."""
    if "delta_num" in metric:
        raise ValueError("live metric registry must not contain delta_num")
    thresholds = metric["decision_thresholds"]
    delta_block = thresholds["delta_numerical"]
    spec = delta_block["recompute_spec"]
    num = metric["numerical_protocol"]
    if spec.get("arm") != "obs-cfreq":
        raise ValueError("MEI-0 delta_numerical arm must be obs-cfreq")
    expected_profiles = ["low_cost_k4_primary", "registered_mrs2_stress"]
    if list(spec.get("noise_profiles") or []) != expected_profiles:
        raise ValueError(
            "MEI-0 delta_numerical noise_profiles must be "
            f"{expected_profiles}"
        )
    point_set_id = str(spec.get("point_set") or "")
    if point_set_id != "ambient_core_216":
        raise ValueError("MEI-0 delta_numerical point_set must be ambient_core_216")
    if not math.isclose(
        float(num.get("normal_p90_z", float("nan"))),
        NORMAL_P90_Z,
        rel_tol=1e-15,
    ):
        raise ValueError("numerical_protocol.normal_p90_z mismatch")

    points_labeled = build_named_point_set(design, point_set_id)
    points = [pt for _, pt in points_labeled]

    by_profile: dict[str, Any] = {}
    for profile_id in expected_profiles:
        by_profile[profile_id] = _recompute_one_profile(
            points=points,
            design=design,
            metric=metric,
            profile_id=profile_id,
            point_set_id=point_set_id,
        )

    shared_upper_bound = max(
        float(row["delta_numerical"]) for row in by_profile.values()
    )
    return {
        "delta_numerical_by_profile": {
            pid: float(row["delta_numerical"]) for pid, row in by_profile.items()
        },
        "shared_upper_bound": shared_upper_bound,
        "by_noise_profile": by_profile,
        "formula": delta_block["formula"],
        "optimizer_relative_tolerance": float(
            delta_block.get("optimizer_relative_tolerance", 0.0)
        ),
        "svd_tolerance_policy": delta_block["svd_tolerance_policy"],
        "n_points": len(points),
        "point_set": point_set_id,
        "arm": spec["arm"],
        "frequencies_hz": [float(x) for x in spec["frequencies_hz"]],
        "noise_profiles": list(expected_profiles),
    }


# Backward-compatible name used by older scripts during migration.
def compute_delta_num(
    design: dict[str, Any],
    metric: dict[str, Any],
) -> dict[str, Any]:
    return compute_delta_numerical(design, metric)


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


def _audit_traceable_evidence(
    family: dict[str, Any],
    family_name: str,
    issues: list[str],
    root: Path,
) -> None:
    status = family.get("status")
    if status not in {"represented_traceable", "independent_holdout_available"}:
        return
    if family.get("can_clear_not_represented") is not True:
        issues.append(f"{family_name}: traceable evidence must allow clearing not_represented")
    if not family.get("refs"):
        issues.append(f"{family_name}: traceable evidence requires non-empty refs")
    if not family.get("implementation_or_holdout_path"):
        issues.append(
            f"{family_name}: traceable evidence requires implementation_or_holdout_path"
        )
    evidence_path = family.get("evidence_path")
    evidence_sha256 = family.get("evidence_sha256")
    if not isinstance(evidence_path, str) or not evidence_path:
        issues.append(f"{family_name}: traceable evidence requires evidence_path")
        return
    if not isinstance(evidence_sha256, str) or len(evidence_sha256) != 64:
        issues.append(f"{family_name}: traceable evidence requires evidence_sha256")
        return
    path = Path(evidence_path)
    if not path.is_absolute():
        path = root / path
    if not path.is_file():
        issues.append(f"{family_name}: evidence file missing: {path}")
    elif sha256_file(path) != evidence_sha256:
        issues.append(f"{family_name}: evidence_sha256 mismatch")


def _audit_parked_nonblocking(
    family: dict[str, Any],
    family_name: str,
    issues: list[str],
    root: Path,
) -> None:
    if family.get("status") != "parked_nonblocking":
        return
    if family.get("can_clear_not_represented") is not False:
        issues.append(f"{family_name}: parked family must not claim represented evidence")
    if family.get("parameter_or_bias_bounds") is not None:
        issues.append(f"{family_name}: parked family must not forge quantitative bounds")
    if not family.get("refs"):
        issues.append(f"{family_name}: parked family requires literature refs")
    policy = family.get("parking_policy")
    if not isinstance(policy, dict):
        issues.append(f"{family_name}: parked family requires parking_policy")
    else:
        for field in (
            "nonblocking_scope",
            "still_blocks",
            "unresolved_reason",
            "revisit_trigger",
        ):
            value = policy.get(field)
            if not isinstance(value, (str, list)) or not value:
                issues.append(f"{family_name}: parking_policy.{field} must be non-empty")
    evidence_path = family.get("evidence_path")
    evidence_sha256 = family.get("evidence_sha256")
    if not isinstance(evidence_path, str) or not evidence_path:
        issues.append(f"{family_name}: parked family requires evidence_path")
        return
    if not isinstance(evidence_sha256, str) or len(evidence_sha256) != 64:
        issues.append(f"{family_name}: parked family requires evidence_sha256")
        return
    path = Path(evidence_path)
    if not path.is_absolute():
        path = root / path
    if not path.is_file():
        issues.append(f"{family_name}: parking evidence file missing: {path}")
    elif sha256_file(path) != evidence_sha256:
        issues.append(f"{family_name}: parking evidence_sha256 mismatch")


def _audit_pressure_domain_evidence(
    model: dict[str, Any], issues: list[str], root: Path
) -> None:
    evidence = model.get("pressure_domain_evidence")
    if not isinstance(evidence, dict):
        issues.append("pressure_domain_evidence must be an object")
        return
    _require_source("pressure_domain_evidence", evidence, issues)
    required = sorted(float(v) for v in evidence.get("required_pressure_mpa") or [])
    if required != [0.5, 0.709]:
        issues.append(
            "pressure_domain_evidence.required_pressure_mpa must be [0.5, 0.709]"
        )
    status = evidence.get("status")
    if status in {"not_validated", "parked_nonblocking"}:
        for field in ("validated_range_mpa", "evidence_path", "evidence_sha256"):
            if status == "not_validated" and evidence.get(field) is not None:
                issues.append(
                    f"pressure_domain_evidence.{field} must be null while not_validated"
                )
        if status == "parked_nonblocking":
            if evidence.get("decision_scope") != "diagnostic_only_not_primary_gate":
                issues.append(
                    "parked pressure evidence requires diagnostic_only_not_primary_gate"
                )
            path_value = evidence.get("evidence_path")
            digest = evidence.get("evidence_sha256")
            if not isinstance(path_value, str) or not path_value:
                issues.append("parked pressure evidence requires evidence_path")
            elif not isinstance(digest, str) or len(digest) != 64:
                issues.append("parked pressure evidence requires evidence_sha256")
            else:
                path = Path(path_value)
                if not path.is_absolute():
                    path = root / path
                if not path.is_file():
                    issues.append(f"pressure parking evidence file missing: {path}")
                elif sha256_file(path) != digest:
                    issues.append("pressure parking evidence_sha256 mismatch")
        return
    if status != "validated_traceable":
        issues.append(
            "pressure_domain_evidence.status must be not_validated or validated_traceable"
        )
        return
    bounds = evidence.get("validated_range_mpa")
    if not isinstance(bounds, list) or len(bounds) != 2:
        issues.append(
            "validated pressure evidence requires validated_range_mpa [min, max]"
        )
    else:
        lower, upper = (float(bounds[0]), float(bounds[1]))
        if lower > min(required) or upper < max(required):
            issues.append("validated pressure range must cover 0.5 and 0.709 MPa")
    path_value = evidence.get("evidence_path")
    digest = evidence.get("evidence_sha256")
    if not isinstance(path_value, str) or not path_value:
        issues.append("validated pressure evidence requires evidence_path")
        return
    if not isinstance(digest, str) or len(digest) != 64:
        issues.append("validated pressure evidence requires evidence_sha256")
        return
    path = Path(path_value)
    if not path.is_absolute():
        path = root / path
    if not path.is_file():
        issues.append(f"pressure evidence file missing: {path}")
    elif sha256_file(path) != digest:
        issues.append("pressure_domain_evidence.evidence_sha256 mismatch")


def _audit_model_family(
    model: dict[str, Any], issues: list[str], root: Path
) -> None:
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
        fid = family.get("id")
        family_name = f"model_family_registry.{fid}"
        _require_source(family_name, family, issues)
        for field in FAMILY_EVIDENCE_FIELDS:
            if field not in family:
                issues.append(f"{family_name}: missing evidence field {field}")
        status = family.get("status")
        if status not in FAMILY_STATUS_ALLOWED and status is not None:
            issues.append(f"{family_name}: invalid status {status!r}")
        if status == "not_represented":
            if family.get("can_clear_not_represented") is not False:
                issues.append(f"{family_name}: can_clear_not_represented must be false")
            if family.get("parameter_or_bias_bounds") is not None:
                issues.append(f"{family_name}: must not forge parameter_or_bias_bounds")
            if family.get("evidence_path") is not None:
                issues.append(
                    f"{family_name}: evidence_path must be null while not_represented"
                )
            if family.get("evidence_sha256") is not None:
                issues.append(
                    f"{family_name}: evidence_sha256 must be null while not_represented"
                )
        _audit_traceable_evidence(family, family_name, issues, root)
        _audit_parked_nonblocking(family, family_name, issues, root)
        perturbation = family.get("perturbation")
        if isinstance(perturbation, dict) and perturbation.get("relative_bounds") is not None:
            _require_refs(family_name, family, issues)
        if fid == "F1_humid_air_c_eq":
            if status != "represented_traceable":
                issues.append("F1 must be represented_traceable with CoolProp bounds")
            if family.get("parameter_or_bias_bounds") is None:
                issues.append("F1 must register parameter_or_bias_bounds")
            if not family.get("refs"):
                issues.append("F1 quantitative bound requires non-empty refs")
    _audit_pressure_domain_evidence(model, issues, root)
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
        "total_drive_budget_relative_s": energy,
        "total_measurement_time_s": (
            n_frequencies * per_frequency_acquire_s
            + max(n_frequencies - 1, 0) * inter_frequency_switch_s
            + control_overhead_s
        ),
    }


def _audit_noise_profiles(design: dict[str, Any], issues: list[str]) -> None:
    profiles = design.get("noise_profiles")
    if not isinstance(profiles, dict):
        issues.append("design_space.noise_profiles missing")
        return
    required = ("low_cost_k4_primary", "registered_mrs2_stress")
    for pid in required:
        if pid not in profiles:
            issues.append(f"noise_profiles missing {pid}")
            continue
        profile = profiles[pid]
        if not isinstance(profile, dict):
            issues.append(f"noise_profiles.{pid} must be object")
            continue
        for field in NOISE_PROFILE_REQUIRED_FIELDS:
            if field not in profile or profile[field] is None:
                issues.append(
                    f"mei0_registry_incomplete: low_cost_noise_profile_missing_traceable_fields "
                    f"({pid}.{field})"
                )
        prior = profile.get("prior_std")
        if not isinstance(prior, dict):
            issues.append(f"{pid}.prior_std missing")
        else:
            for key in PRIOR_STD_REQUIRED_KEYS:
                if key not in prior or prior[key] is None:
                    issues.append(
                        f"mei0_registry_incomplete: low_cost_noise_profile_missing_traceable_fields "
                        f"({pid}.prior_std.{key})"
                    )
        refs = profile.get("refs")
        if not isinstance(refs, list) or not refs:
            issues.append(f"{pid}: refs must be non-empty")
        # Profiles must own full field sets independently (no shared inheritance object).
        for other_id, other in profiles.items():
            if other_id == pid or not isinstance(other, dict):
                continue
            if profile is other:
                issues.append(f"noise profiles must not share identity: {pid}/{other_id}")

    low = profiles.get("low_cost_k4_primary") or {}
    stress = profiles.get("registered_mrs2_stress") or {}
    if float(low.get("jitter_std_s", -1)) != 5.0e-7:
        issues.append("low_cost_k4_primary.jitter_std_s must be 5e-7")
    if float((low.get("prior_std") or {}).get("t_c", -1)) != 0.1:
        issues.append("low_cost_k4_primary.prior_std.t_c must be 0.1")
    if float(stress.get("jitter_std_s", -1)) != 3.0e-6:
        issues.append("registered_mrs2_stress.jitter_std_s must be 3e-6")
    if float((stress.get("prior_std") or {}).get("t_c", -1)) != 1.0:
        issues.append("registered_mrs2_stress.prior_std.t_c must be 1.0")


def _audit_design(design: dict[str, Any], issues: list[str]) -> None:
    _require_source("design_space", design, issues)
    band = design.get("frequency_band") or {}
    k4 = [float(x) for x in (band.get("baseline_k4_hz") or [])]
    if k4 != [25000.0, 63000.0, 100000.0, 200000.0]:
        issues.append("frequency_band.baseline_k4_hz must be {25,63,100,200} kHz")
    if float(band.get("min_hz", -1)) != 25000.0 or float(band.get("max_hz", -1)) != 200000.0:
        issues.append("frequency_band must be 25–200 kHz")

    if "narrow_context_grid" in design:
        issues.append("narrow_context_grid must be replaced by named point_sets")
    point_sets = design.get("point_sets") or {}
    for required in (
        "ambient_core_216",
        "pressure_extension_low_rh_216",
        "formal_mei1_432",
    ):
        if required not in point_sets:
            issues.append(f"point_sets missing {required}")
    try:
        core = build_named_point_set(design, "ambient_core_216")
        pressure = build_named_point_set(design, "pressure_extension_low_rh_216")
        union = build_formal_mei1_points(design)
        if len(core) != 216 or len(pressure) != 216 or len(union) != 432:
            issues.append(
                f"point_sets counts must be 216/216/432, got "
                f"{len(core)}/{len(pressure)}/{len(union)}"
            )
        high_p = [pt for _, pt in union if float(pt.p_mpa) in {0.5, 0.709}]
        if len(high_p) != 216:
            issues.append("high-pressure points must enter formal_mei1_432 gate")
    except (KeyError, ValueError) as exc:
        issues.append(f"point_sets build failed: {exc}")

    cost = design.get("cost_function") or {}
    _require_source("cost_function", cost, issues)
    if cost.get("actual_incident_acoustic_energy_status") != (
        "unavailable_without_F5_calibration"
    ):
        issues.append(
            "cost_function.actual_incident_acoustic_energy_status must be "
            "unavailable_without_F5_calibration"
        )
    forbidden_energy_names = {
        "total_acoustic_energy",
        "total_acoustic_energy_relative_s",
        "equal_total_energy",
    }
    terms = set(cost.get("terms") or [])
    if forbidden_energy_names.intersection(terms):
        issues.append("drive budget must not be labeled acoustic energy")
    required_terms = {
        "n_frequencies",
        "total_drive_budget",
        "total_measurement_time_s",
        "rh_switch_time_s",
        "pressure_switch_time_s",
        "settle_wait_time_s",
    }
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
    if "total_acoustic_energy_relative_s" in ledger:
        issues.append("d0_ledger must use total_drive_budget_relative_s")
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
    if "total_acoustic_energy_relative_s" in d5:
        issues.append("D5 must use total_drive_budget_relative_s")
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

    balancing = cost.get("cost_balancing_rule") or {}
    if "equal_total_energy" in str(balancing.get("id") or ""):
        issues.append("cost_balancing_rule must use equal_input_drive_budget naming")

    noise = design.get("noise_families") or {}
    for key in ("N0_independent", "N1_low_rank_common_mode", "N2_mixed"):
        if key not in noise:
            issues.append(f"noise_families missing {key}")
    n1 = noise.get("N1_low_rank_common_mode") or {}
    if n1.get("status") != "sensitivity_analysis_only":
        issues.append("N1 common-mode must remain sensitivity_analysis_only at MEI-0")

    _audit_noise_profiles(design, issues)

    holdout = design.get("holdout_conditions") or {}
    defs = holdout.get("definitions") or {}
    for hid in holdout.get("holdout_ids") or []:
        if hid not in defs:
            issues.append(f"holdout definition missing for {hid}")


def _audit_metric(
    metric: dict[str, Any],
    issues: list[str],
    *,
    require_frozen_delta_numerical: bool,
) -> None:
    _require_source("metric_registry", metric, issues)
    if "delta_num" in metric:
        issues.append("live metric registry must not contain delta_num")

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

    thresholds = metric.get("decision_thresholds") or {}
    delta_num = thresholds.get("delta_numerical") or {}
    delta_prac = thresholds.get("delta_practical") or {}
    formula = str(delta_num.get("formula") or "")
    if "floor" in formula.lower() or "0.02" in formula:
        issues.append("delta_numerical must not contain practical 2% floor")
    if formula != (
        "max(2*fd_relative_change, 2*fresh_process_relative_change, "
        "optimizer_relative_tolerance)"
    ):
        issues.append("delta_numerical.formula does not match MEI-0 contract")
    if not delta_num.get("recompute_required_at_freeze"):
        issues.append("delta_numerical.recompute_required_at_freeze must be true")
    if delta_num.get("svd_tolerance_policy") != (
        "rank_diagnostic_only_not_part_of_crb_p90_delta"
    ):
        issues.append("delta_numerical.svd_tolerance_policy missing or invalid")
    if float(delta_num.get("optimizer_relative_tolerance", -1.0)) != 0.0:
        issues.append("optimizer_relative_tolerance must be 0.0 at MEI-0")
    spec = delta_num.get("recompute_spec") or {}
    if spec.get("arm") != "obs-cfreq":
        issues.append("delta_numerical arm must be obs-cfreq")
    if list(spec.get("noise_profiles") or []) != [
        "low_cost_k4_primary",
        "registered_mrs2_stress",
    ]:
        issues.append("delta_numerical must recompute both noise profiles")
    if spec.get("point_set") != "ambient_core_216":
        issues.append("delta_numerical point_set must be ambient_core_216")
    if spec.get("repeat_execution") != "fresh_process":
        issues.append("delta_numerical repeats must run in fresh processes")

    if float(delta_prac.get("value", -1)) != 0.02:
        issues.append("delta_practical.value must be 0.02")
    if delta_prac.get("source") != "pre_registered_practical_equivalence_policy":
        issues.append("delta_practical.source must be pre_registered_practical_equivalence_policy")
    if delta_prac.get("not_a_numerical_error") is not True:
        issues.append("delta_practical.not_a_numerical_error must be true")

    by_profile = delta_num.get("by_noise_profile")
    shared = delta_num.get("shared_upper_bound")
    if require_frozen_delta_numerical:
        if not isinstance(by_profile, dict) or not by_profile:
            issues.append(
                "delta_numerical.by_noise_profile missing (run freeze recompute first)"
            )
        if shared is None:
            issues.append("delta_numerical.shared_upper_bound missing")
        elif isinstance(by_profile, dict) and by_profile:
            expected = max(float(v) for v in by_profile.values())
            if not math.isclose(float(shared), expected, rel_tol=1e-12):
                issues.append(
                    "delta_numerical.shared_upper_bound must equal max over profiles"
                )

    stats = metric.get("statistics_protocols") or {}
    finite = stats.get("finite_registry_information_audit") or {}
    if finite.get("bootstrap") != "forbidden":
        issues.append("finite_registry_information_audit must forbid bootstrap")
    if finite.get("random_unit") != "none":
        issues.append("finite_registry_information_audit.random_unit must be none")
    learning = stats.get("learning_solver_experiment") or {}
    if learning.get("random_unit") != "mixture_id":
        issues.append("learning_solver_experiment.random_unit must be mixture_id")
    if int(learning.get("n_bootstrap_resamples", -1)) != 2000:
        issues.append("learning_solver_experiment.n_bootstrap_resamples must be 2000")
    calib = stats.get("posterior_calibration_experiment") or {}
    if list(calib.get("nominal_coverages") or []) != [0.5, 0.8, 0.9, 0.95]:
        issues.append("posterior_calibration_experiment coverages mismatch")
    for key in (
        "report_marginal_coverage",
        "report_group_conditional_coverage",
        "report_selection_conditional_coverage_after_rejection",
        "report_interval_width",
        "report_rejection_rate",
    ):
        if calib.get(key) is not True:
            issues.append(f"posterior_calibration_experiment.{key} must be true")

    solver_gate = (metric.get("gates") or {}).get("solver") or {}
    if solver_gate.get("require_bootstrap_ci_lb_gt_delta_practical") is not True:
        issues.append("solver gate CI lower bound must exceed delta_practical")
    if "require_bootstrap_ci_lb_positive" in solver_gate:
        issues.append("solver gate must not retain positive-only CI threshold")

    output = (
        ((metric.get("component_reporting") or {}).get("output_contract")) or {}
    )
    point_est = output.get("point_estimate") or {}
    if point_est.get("mode") != "raw3" or int(point_est.get("out_dim", -1)) != 3:
        issues.append("raw3 point_estimate contract missing")
    if point_est.get("silent_normalization") is not False:
        issues.append("raw3 contract forbids silent normalization")
    posterior = output.get("posterior") or {}
    if posterior.get("silent_normalization") is not False:
        issues.append("posterior must forbid silent normalization")
    forbidden = set(output.get("forbidden") or [])
    if "silent_normalization" not in forbidden:
        issues.append("output_contract.forbidden must include silent_normalization")

    varpro = metric.get("varpro_observation_contract") or {}
    required_fields = set(varpro.get("required_fields") or [])
    for field in (
        "raw_tof_s",
        "observation_covariance",
        "frequency_hz",
        "device_profile_id",
        "view_id",
        "T_C",
        "P_MPa",
        "H_RH",
        "L_m",
    ):
        if field not in required_fields:
            issues.append(f"varpro_observation_contract missing {field}")
    if varpro.get("forbid_unconstrained_free_offsets_per_k4_sample") is not True:
        issues.append("varpro must forbid four unconstrained free offsets per K4 sample")
    if varpro.get("c_observed_only_status") != (
        "mei3_varpro_not_applicable_to_c_observed_only"
    ):
        issues.append("c_observed-only VarPro status missing")

    transitions = metric.get("stage_transition_policy") or {}
    if (transitions.get("mei3_varpro_not_applicable") or {}).get("mei4_baseline") != "S1":
        issues.append("stage_transition must select S1 when varpro not applicable")
    if (transitions.get("mei3_varpro_supported") or {}).get("mei4_baseline") != "S2":
        issues.append("stage_transition must select S2 when varpro supported")
    fixed_k4 = transitions.get("mei1_fixed_k4_retained") or {}
    if fixed_k4.get("allowed_next_stage") != "MEI-3_varpro_audit":
        issues.append("fixed K4 decision must advance to MEI-3_varpro_audit")
    if fixed_k4.get("skip_stage") != "MEI-2_robust_design":
        issues.append("fixed K4 decision must explicitly skip MEI-2")

    parking = metric.get("parked_nonblocking_policy") or {}
    if parking.get("forbidden_interpretation") != (
        "parked_nonblocking_is_not_represented_traceable"
    ):
        issues.append("parked_nonblocking policy must forbid represented interpretation")
    if parking.get("hardware_and_waveform_authorizations_remain_forbidden") is not True:
        issues.append("parked_nonblocking must keep hardware and waveform forbidden")

    auths = metric.get("authorizations") or {}
    for field in AUTHORIZATION_FIELDS:
        if field not in auths:
            issues.append(f"authorizations missing independent field {field}")
        elif auths[field] != FORBIDDEN_AUTH_VALUE:
            issues.append(f"authorizations.{field} must be {FORBIDDEN_AUTH_VALUE}")
    if len(set(AUTHORIZATION_FIELDS)) != len(AUTHORIZATION_FIELDS):
        issues.append("authorization fields must be independent")

    auth = metric.get("data_generation_authorization") or {}
    if auth.get("formal_waveform_generation") != FORBIDDEN_AUTH_VALUE:
        issues.append("formal waveform generation must remain forbidden at MEI-0")
    for key in ("q1_delta_numerical", "q4_varpro_linear_block", "q7_authorize_new_data"):
        if key not in (metric.get("review_answers") or {}):
            issues.append(f"metric review_answers missing {key}")


def audit_mei0_registries(
    config_dir: Path | None = None,
    *,
    project_root: Path | None = None,
    require_frozen_delta_numerical: bool = True,
    require_frozen_delta_num: bool | None = None,
    registry_overrides: dict[str, dict[str, Any]] | None = None,
) -> dict[str, Any]:
    root = Path(project_root) if project_root is not None else _TV3_ROOT
    cfg = Path(config_dir) if config_dir is not None else default_config_dir()
    issues: list[str] = []
    if require_frozen_delta_num is not None:
        require_frozen_delta_numerical = require_frozen_delta_num

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
        _audit_model_family(model, issues, root)
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
            require_frozen_delta_numerical=require_frozen_delta_numerical,
        )

    passed = len(issues) == 0
    verdict = "mei0_registry_frozen" if passed else "mei0_registry_incomplete"
    return {
        "registry_schema_version": REGISTRY_SCHEMA_VERSION,
        "reserved_benchmark_schema_version": RESERVED_BENCHMARK_SCHEMA_VERSION,
        "schema_version": REGISTRY_SCHEMA_VERSION,
        "stage": STAGE,
        "passed": passed,
        "verdict": verdict,
        "allowed_next_stage": "MEI-1_forward_envelope" if passed else None,
        "issues": issues,
        "registry_sha256": digests,
        "input_contract_sha256": combined_registry_contract_sha256(loaded)
        if len(loaded) == len(REGISTRY_FILES)
        else None,
        "config_dir": str(cfg.resolve()),
        "claim_scope": CLAIM_SCOPE,
        "formal_waveform_generation": FORBIDDEN_AUTH_VALUE,
        "authorizations": {field: FORBIDDEN_AUTH_VALUE for field in AUTHORIZATION_FIELDS},
    }


def metric_with_delta_numerical(
    metric: dict[str, Any],
    delta_result: dict[str, Any],
) -> dict[str, Any]:
    updated = json.loads(json.dumps(metric))
    if "delta_num" in updated:
        del updated["delta_num"]
    block = updated["decision_thresholds"]["delta_numerical"]
    block["by_noise_profile"] = {
        pid: float(value)
        for pid, value in delta_result["delta_numerical_by_profile"].items()
    }
    block["shared_upper_bound"] = float(delta_result["shared_upper_bound"])
    block["recompute_artifact"] = {
        "delta_numerical_by_profile": delta_result["delta_numerical_by_profile"],
        "shared_upper_bound": delta_result["shared_upper_bound"],
        "by_noise_profile": {
            pid: {
                "max_relative_change_fd": row["max_relative_change_fd"],
                "max_relative_change_fresh_process_repeat": row[
                    "max_relative_change_fresh_process_repeat"
                ],
                "nominal": row["nominal"],
                "fresh_process_repeats": row["fresh_process_repeats"],
                "svd_rank_diagnostics": row["svd_rank_diagnostics"],
                "n_points": row["n_points"],
            }
            for pid, row in delta_result["by_noise_profile"].items()
        },
        "repeat_execution": "fresh_process",
        "svd_tolerance_policy": delta_result["svd_tolerance_policy"],
        "n_points": delta_result["n_points"],
        "arm": delta_result["arm"],
        "frequencies_hz": delta_result["frequencies_hz"],
        "noise_profiles": delta_result["noise_profiles"],
        "point_set": delta_result["point_set"],
    }
    return updated


def metric_with_delta_num(
    metric: dict[str, Any],
    delta_result: dict[str, Any],
) -> dict[str, Any]:
    return metric_with_delta_numerical(metric, delta_result)


__all__ = [
    "AUTHORIZATION_FIELDS",
    "FORBIDDEN_AUTH_VALUE",
    "FREEZE_MANIFEST_SCHEMA_VERSION",
    "REGISTRY_FILES",
    "REGISTRY_SCHEMA_VERSION",
    "RESERVED_BENCHMARK_SCHEMA_VERSION",
    "SCHEMA_VERSION",
    "audit_mei0_registries",
    "build_formal_mei1_points",
    "build_named_point_set",
    "build_narrow_points",
    "combined_registry_contract_sha256",
    "compute_delta_num",
    "compute_delta_numerical",
    "default_config_dir",
    "dumps_stable",
    "load_json",
    "metric_with_delta_num",
    "metric_with_delta_numerical",
    "sha256_bytes",
    "sha256_file",
    "verify_evidence_manifest",
]
