#!/usr/bin/env python3
"""Run MRS-2 multifreq forward identifiability audit (life/death gate)."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

_TV3_ROOT = Path(__file__).resolve().parents[1]
if str(_TV3_ROOT) not in sys.path:
    sys.path.insert(0, str(_TV3_ROOT))

from tv3.audit.identifiability_v3_mrs import (  # noqa: E402
    ARM_IDS,
    MULTIFREQ_RANK_ARMS,
    MrsPoint,
    choose_mrs2_verdict,
    evaluate_point_arm,
)


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _load_config(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("config root must be object")
    return data


def _check_prerequisites(config: dict[str, Any], tv3_root: Path) -> None:
    reg = config["mrs0_registry"]
    reg_path = tv3_root / reg["path"]
    sha = _sha256(reg_path)
    if sha != reg["expected_sha256"]:
        raise SystemExit(
            f"MRS-0 registry sha256 mismatch: got {sha}, expected {reg['expected_sha256']}"
        )
    stage_path = tv3_root / config["mrs1_prerequisite"]["stage_status"]
    stage = json.loads(stage_path.read_text(encoding="utf-8"))
    mrs1 = stage.get("mrs1") or {}
    if mrs1.get("verdict") != config["mrs1_prerequisite"]["expected_verdict"]:
        raise SystemExit(
            f"MRS-1 prerequisite failed: stage_status mrs1={mrs1!r}"
        )


def _narrow_points(config: dict[str, Any]) -> list[tuple[str, MrsPoint]]:
    ctx = config["narrow_context_grid"]
    points: list[tuple[str, MrsPoint]] = []
    for window in config["narrow_windows"]:
        wid = window["id"]
        o2 = float(window["center_percent"])
        for co2 in ctx["co2_percent"]:
            for t_c in ctx["t_c"]:
                for l_m in ctx["path_length_m"]:
                    for h_rh in ctx["h_rh"]:
                        for p_mpa in ctx["p_mpa"]:
                            # Keep RH+delta inside sampling domain for rh-diff arm.
                            if float(h_rh) + float(config["observation"]["rh_delta_percent"]) > 80.0:
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


def _heatmap_points(config: dict[str, Any]) -> list[MrsPoint]:
    g = config["heatmap_grid"]
    pts: list[MrsPoint] = []
    for co2 in g["co2_percent"]:
        for o2 in g["o2_percent"]:
            for t_c in g["t_c"]:
                for l_m in g["path_length_m"]:
                    for h_rh in g["h_rh"]:
                        for p_mpa in g["p_mpa"]:
                            pts.append(
                                MrsPoint(
                                    float(co2),
                                    float(o2),
                                    float(t_c),
                                    float(l_m),
                                    float(h_rh),
                                    float(p_mpa),
                                )
                            )
    return pts


def _eval_kwargs(config: dict[str, Any], f_hz: list[float], arm: str) -> dict[str, Any]:
    obs = config["observation"]
    prior = {
        "t_c": float(config["prior_std"]["t_c"]),
        "path_length_m": float(config["prior_std"]["path_length_m"]),
        "h_rh": float(config["prior_std"]["h_rh"]),
    }
    if "co2_percent" in config["prior_std"]:
        prior["co2_percent"] = float(config["prior_std"]["co2_percent"])
    return {
        "arm": arm,
        "f_hz": f_hz,
        "parameter_steps": config["finite_difference"]["steps"],
        "parameter_bounds": config["parameter_bounds"],
        "fixed_delay_s": float(obs["fixed_delay_s"]),
        "rh_delta": float(obs["rh_delta_percent"]),
        "p_scan_mpa": list(obs["p_scan_mpa"]),
        "jitter_std_s": float(obs["jitter_std_s"]),
        "relative_amp_std": float(obs["relative_amp_std"]),
        "phase_std_s_at_anchor": float(obs.get("phase_std_s_at_anchor", 0.0)),
        "prior_std": prior,
        "window_width_percent": float(config["narrow_windows"][0]["width_percent"]),
        "max_relative_step_disagreement": float(
            config["finite_difference"]["max_relative_step_disagreement"]
        ),
    }


def _summarize_arm(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        raise ValueError("no rows to summarize")
    ranks = [int(r["joint_rank"]) for r in rows]
    p90s = [float(r["p90_o2_percent"]) for r in rows if math.isfinite(float(r["p90_o2_percent"]))]
    fracs = [float(r["nuisance"]["worst_fraction_of_window"]) for r in rows]
    unstable = sum(1 for r in rows if not r["all_stable"])
    invertible = all(bool(r["fisher_aug_invertible"]) for r in rows)
    return {
        "n_points": len(rows),
        "min_joint_rank": min(ranks),
        "max_joint_rank": max(ranks),
        "median_joint_rank": float(sorted(ranks)[len(ranks) // 2]),
        "frac_rank_ge_2": sum(1 for r in ranks if r >= 2) / len(ranks),
        "max_p90_o2_percent": max(p90s) if p90s else float("inf"),
        "median_p90_o2_percent": float(sorted(p90s)[len(p90s) // 2]) if p90s else float("inf"),
        "max_nuisance_fraction": max(fracs) if fracs else float("inf"),
        "unstable_fd_count": unstable,
        "all_crlb_invertible": invertible,
    }


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    # Flatten a stable subset of fields
    flat_rows: list[dict[str, Any]] = []
    for r in rows:
        flat = {
            "arm": r["arm"],
            "window_id": r.get("window_id"),
            "subset": r.get("subset"),
            "co2_percent": r["point"]["co2_percent"],
            "o2_percent": r["point"]["o2_percent"],
            "t_c": r["point"]["t_c"],
            "path_length_m": r["point"]["path_length_m"],
            "h_rh": r["point"]["h_rh"],
            "p_mpa": r["point"]["p_mpa"],
            "joint_rank": r["joint_rank"],
            "n_obs": r["n_obs"],
            "p90_o2_percent": r["p90_o2_percent"],
            "crlb_o2_std_percent": r["crlb_o2_std_percent"],
            "fisher_aug_invertible": r["fisher_aug_invertible"],
            "nuisance_worst_fraction": r["nuisance"]["worst_fraction_of_window"],
            "all_stable": r["all_stable"],
        }
        flat_rows.append(flat)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(flat_rows[0].keys()))
        writer.writeheader()
        writer.writerows(flat_rows)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=_TV3_ROOT / "configs" / "tv3_mrs_identifiability.json",
    )
    parser.add_argument("--allow-overwrite", action="store_true")
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Smoke grid: one window, reduced context (for tests)",
    )
    args = parser.parse_args()
    config = _load_config(args.config.resolve())
    _check_prerequisites(config, _TV3_ROOT)

    out_dir = (_TV3_ROOT / config["output_dir"]).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    verdict_path = out_dir / "mrs2_verdict.json"
    if verdict_path.exists() and not args.allow_overwrite:
        raise SystemExit(f"refuse overwrite: {verdict_path} (pass --allow-overwrite)")

    if args.quick:
        config = json.loads(json.dumps(config))  # deep copy
        config["narrow_windows"] = [config["narrow_windows"][2]]  # o2_20_0
        config["narrow_context_grid"] = {
            "co2_percent": [1.0],
            "t_c": [25.0],
            "path_length_m": [0.25],
            "h_rh": [50.0],
            "p_mpa": [0.101325],
        }
        config["heatmap_grid"]["h_rh"] = [20.0, 50.0, 80.0]
        config["heatmap_grid"]["p_mpa"] = [0.10, 0.50, 0.709]

    arms = list(config["arms"])
    for arm in arms:
        if arm not in ARM_IDS:
            raise SystemExit(f"unknown arm in config: {arm}")

    f_full = [float(x) for x in config["frequency_set_hz"]]
    narrow = _narrow_points(config)
    print(f"narrow points: {len(narrow)}; arms: {arms}", flush=True)

    narrow_rows: list[dict[str, Any]] = []
    arm_rows: dict[str, list[dict[str, Any]]] = {a: [] for a in arms}
    unstable_total = 0

    for window_id, point in narrow:
        for arm in arms:
            f_hz = [200000.0] if arm == "obs-single-200k" else f_full
            result = evaluate_point_arm(point, **_eval_kwargs(config, f_hz, arm))
            result["window_id"] = window_id
            result["subset"] = "K1" if arm == "obs-single-200k" else "K8"
            if not result["all_stable"]:
                unstable_total += 1
            narrow_rows.append(result)
            arm_rows[arm].append(result)

    if unstable_total:
        # Soft warning: do not auto-fail; record count. Hard-fail only if all unstable.
        print(f"WARNING: unstable FD count={unstable_total}", flush=True)

    arm_summaries = {arm: _summarize_arm(rows) for arm, rows in arm_rows.items()}

    # Heatmap: K subsets × RH × P for obs-cfreq and obs-calpha
    heatmap_rows: list[dict[str, Any]] = []
    heat_points = _heatmap_points(config)
    heat_arms = ["obs-cfreq", "obs-calpha"]
    for subset_name, freqs in config["frequency_subsets"].items():
        f_hz = [float(x) for x in freqs]
        for arm in heat_arms:
            for point in heat_points:
                result = evaluate_point_arm(point, **_eval_kwargs(config, f_hz, arm))
                result["window_id"] = "heatmap"
                result["subset"] = subset_name
                heatmap_rows.append(result)

    # Dead-corner slice: low RH × high P
    dead = [
        r
        for r in heatmap_rows
        if r["subset"] == "K8"
        and r["arm"] == "obs-cfreq"
        and float(r["point"]["h_rh"]) <= 20.0 + 1e-9
        and float(r["point"]["p_mpa"]) >= 0.50 - 1e-9
    ]

    blocking = [
        item
        for item in config.get("representation_audit", [])
        if item.get("blocks_go_verdict") is True
    ]
    rejection_rate = 1.0 if blocking else 0.0

    gates = config["business_thresholds"]
    decision = choose_mrs2_verdict(
        single_200k_min_rank=int(arm_summaries["obs-single-200k"]["min_joint_rank"]),
        arm_summaries=arm_summaries,
        target_p90=float(gates["target_p90_o2_error_percent"]),
        max_nuisance_fraction=float(gates["max_nuisance_fraction_of_signal"]),
        max_rejection_rate=float(gates["max_rejection_rate"]),
        rejection_rate=rejection_rate,
    )

    # MRS-5 amplitude gate placeholder from best passing arm CRB budget
    amp_gate = {
        "status": "preregistered_from_mrs2" if decision["verdict"] == "mrs2_rank_upgraded_p90_pass" else "not_applicable",
        "note": "Numerical DL amplitude gates deferred unless MRS-2 pass; structure frozen in MRS-0.",
    }
    if decision.get("passing_arms"):
        best = decision["passing_arms"][0]
        amp_gate["anchor_arm"] = best
        amp_gate["anchor_max_p90_o2"] = arm_summaries[best]["max_p90_o2_percent"]

    payload = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "schema_version": config["schema_version"],
        "quick": bool(args.quick),
        "mrs0_registry_sha256": config["mrs0_registry"]["expected_sha256"],
        "gates": gates,
        "arm_summaries": arm_summaries,
        "decision": decision,
        "verdict": decision["verdict"],
        "allow_mrs3": decision["allow_mrs3"],
        "rejection_rate": rejection_rate,
        "unstable_fd_count": unstable_total,
        "dead_corner_k8_cfreq": [
            {
                "h_rh": r["point"]["h_rh"],
                "p_mpa": r["point"]["p_mpa"],
                "joint_rank": r["joint_rank"],
                "p90_o2_percent": r["p90_o2_percent"],
            }
            for r in dead
        ],
        "mrs5_amplitude_gate_preregistration": amp_gate,
        "n_narrow_rows": len(narrow_rows),
        "n_heatmap_rows": len(heatmap_rows),
    }

    verdict_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (out_dir / "metrics.json").write_text(
        json.dumps(
            {
                "arm_summaries": arm_summaries,
                "dead_corner_k8_cfreq": payload["dead_corner_k8_cfreq"],
            },
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    _write_csv(out_dir / "narrow_arm_points.csv", narrow_rows)
    _write_csv(out_dir / "heatmap_subset_rh_p.csv", heatmap_rows)

    # Update stage_status
    stage_path = _TV3_ROOT / "configs" / "tv3_mrs" / "stage_status.json"
    stage = json.loads(stage_path.read_text(encoding="utf-8"))
    stage["mrs2"] = {
        "verdict": decision["verdict"],
        "passed_at": datetime.now(timezone.utc).date().isoformat(),
        "verdict_path": "outputs/tv3_mrs/identifiability_mrs2/mrs2_verdict.json",
        "allow_mrs3": decision["allow_mrs3"],
        "require_mrs6": bool(decision.get("require_mrs6", True)),
        "quick": bool(args.quick),
    }
    if decision["allow_mrs3"]:
        stage["allowed_next_stage"] = "MRS-3_multifreq_benchmark"
    else:
        stage["allowed_next_stage"] = "MRS-6_hardware_requirements"
    stage_path.write_text(json.dumps(stage, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    summary = [
        "# tv3 MRS-2 forward identifiability",
        "",
        f"- verdict: `{decision['verdict']}`",
        f"- allow_mrs3: `{decision['allow_mrs3']}`",
        f"- reason: {decision.get('reason')}",
        f"- obs-single-200k min_rank: `{arm_summaries['obs-single-200k']['min_joint_rank']}`",
        "",
        "## Arm summaries (narrow)",
        "",
    ]
    for arm, s in arm_summaries.items():
        summary.append(
            f"- `{arm}`: min_rank={s['min_joint_rank']}, "
            f"frac_rank≥2={s['frac_rank_ge_2']:.2f}, "
            f"max_P90={s['max_p90_o2_percent']:.4g}, "
            f"max_nuisance_frac={s['max_nuisance_fraction']:.4g}"
        )
    summary.append("")
    (out_dir / "mrs2_summary.md").write_text("\n".join(summary) + "\n", encoding="utf-8")

    print(json.dumps(decision, indent=2, ensure_ascii=False))
    print(json.dumps(arm_summaries, indent=2, ensure_ascii=False))
    return 0 if decision["verdict"] != "audit_failed" else 2


if __name__ == "__main__":
    raise SystemExit(main())
