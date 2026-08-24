#!/usr/bin/env python3
"""Run MRS-6 hardware-requirements noise-budget scan (spec derivation only).

Does NOT rerun MRS-2 and does NOT alter its frozen verdict/gates. Reuses the
MRS-2 narrow grid and Jacobian machinery to quantify which noise budgets
(sigma_TOF, T prior, alpha calibration) would be needed to reach the frozen
0.4 vol% O2 spec target, per observation arm and frequency subset.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

_TV3_ROOT = Path(__file__).resolve().parents[1]
if str(_TV3_ROOT) not in sys.path:
    sys.path.insert(0, str(_TV3_ROOT))

from tv3.audit.identifiability_v3_mrs import MrsPoint  # noqa: E402
from tv3.audit.mrs6_noise_budget import (  # noqa: E402
    NoiseBudget,
    evaluate_budget,
    pareto_passing_combos,
    precompute_arm_jacobians,
    required_budget_from_scan,
)

MRS6_VERDICT = "mrs6_hardware_requirements_delivered"


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _load_json(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"config root must be object: {path}")
    return data


def _check_prerequisites(config: dict[str, Any]) -> None:
    reg = config["mrs0_registry"]
    sha = _sha256(_TV3_ROOT / reg["path"])
    if sha != reg["expected_sha256"]:
        raise SystemExit(f"MRS-0 registry sha256 mismatch: got {sha}")
    stage = _load_json(_TV3_ROOT / config["mrs2_prerequisite"]["stage_status"])
    mrs2 = stage.get("mrs2") or {}
    if mrs2.get("verdict") != config["mrs2_prerequisite"]["expected_verdict"]:
        raise SystemExit(f"MRS-2 prerequisite failed: stage_status mrs2={mrs2!r}")
    if mrs2.get("allow_mrs3"):
        raise SystemExit("MRS-2 allow_mrs3=true is inconsistent with the MRS-6 fail path")


def _narrow_points(mrs2_config: dict[str, Any]) -> list[MrsPoint]:
    ctx = mrs2_config["narrow_context_grid"]
    rh_delta = float(mrs2_config["observation"]["rh_delta_percent"])
    points: list[MrsPoint] = []
    for window in mrs2_config["narrow_windows"]:
        o2 = float(window["center_percent"])
        for co2 in ctx["co2_percent"]:
            for t_c in ctx["t_c"]:
                for l_m in ctx["path_length_m"]:
                    for h_rh in ctx["h_rh"]:
                        for p_mpa in ctx["p_mpa"]:
                            if float(h_rh) + rh_delta > 80.0:
                                continue
                            points.append(
                                MrsPoint(
                                    float(co2), o2, float(t_c),
                                    float(l_m), float(h_rh), float(p_mpa),
                                )
                            )
    return points


def _precompute_kwargs(mrs2_config: dict[str, Any]) -> dict[str, Any]:
    obs = mrs2_config["observation"]
    return {
        "parameter_steps": mrs2_config["finite_difference"]["steps"],
        "parameter_bounds": mrs2_config["parameter_bounds"],
        "fixed_delay_s": float(obs["fixed_delay_s"]),
        "rh_delta": float(obs["rh_delta_percent"]),
        "p_scan_mpa": list(obs["p_scan_mpa"]),
        "max_relative_step_disagreement": float(
            mrs2_config["finite_difference"]["max_relative_step_disagreement"]
        ),
    }


def _registered_prior(mrs2_config: dict[str, Any]) -> dict[str, float]:
    prior = {
        "t_c": float(mrs2_config["prior_std"]["t_c"]),
        "path_length_m": float(mrs2_config["prior_std"]["path_length_m"]),
        "h_rh": float(mrs2_config["prior_std"]["h_rh"]),
    }
    if "co2_percent" in mrs2_config["prior_std"]:
        prior["co2_percent"] = float(mrs2_config["prior_std"]["co2_percent"])
    return prior


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    flat_rows = []
    for r in rows:
        flat = dict(r)
        prior = flat.pop("prior_std")
        flat["prior_t_c_k"] = prior["t_c"]
        flat["prior_h_rh_percent"] = prior["h_rh"]
        flat["prior_path_length_m"] = prior["path_length_m"]
        flat["prior_co2_percent"] = prior.get("co2_percent")
        flat_rows.append(flat)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(flat_rows[0].keys()))
        writer.writeheader()
        writer.writerows(flat_rows)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path, default=_TV3_ROOT / "configs" / "tv3_mrs6_hardware.json"
    )
    parser.add_argument("--allow-overwrite", action="store_true")
    parser.add_argument(
        "--quick", action="store_true", help="Reduced grid for smoke/tests"
    )
    args = parser.parse_args()
    config = _load_json(args.config.resolve())
    _check_prerequisites(config)
    mrs2_config = _load_json(_TV3_ROOT / config["mrs2_config"])

    out_dir = (_TV3_ROOT / config["output_dir"]).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    verdict_path = out_dir / "mrs6_verdict.json"
    if verdict_path.exists() and not args.allow_overwrite:
        raise SystemExit(f"refuse overwrite: {verdict_path} (pass --allow-overwrite)")

    if args.quick:
        mrs2_config = json.loads(json.dumps(mrs2_config))
        mrs2_config["narrow_windows"] = [mrs2_config["narrow_windows"][2]]
        mrs2_config["narrow_context_grid"] = {
            "co2_percent": [1.0],
            "t_c": [25.0],
            "path_length_m": [0.25],
            "h_rh": [50.0],
            "p_mpa": [0.101325],
        }

    points = _narrow_points(mrs2_config)
    f_full = [float(x) for x in mrs2_config["frequency_set_hz"]]
    pre_kwargs = _precompute_kwargs(mrs2_config)
    steps = mrs2_config["finite_difference"]["steps"]
    window_width = float(mrs2_config["narrow_windows"][0]["width_percent"])
    registered_prior = _registered_prior(mrs2_config)
    registered_amp = float(mrs2_config["observation"]["relative_amp_std"])
    targets = config["spec_targets"]
    target_p90 = float(targets["target_p90_o2_error_percent"])
    max_nuis = float(targets["max_nuisance_fraction_of_signal"])
    statistic = str(targets["statistic"])

    arms = list(config["arms"])
    jitter_scan_s = [u * 1e-6 for u in config["jitter_scan_us"]]
    t_prior_scan = [float(t) for t in config["t_prior_scan_k"]]
    amp_scan = [float(a) for a in config["amp_scan_rel"]]

    print(f"narrow points: {len(points)}; precomputing Jacobians for arms {arms}", flush=True)
    jac_by_arm = {
        arm: precompute_arm_jacobians(points, arm=arm, f_hz=f_full, **pre_kwargs)
        for arm in arms
    }

    def _budget(jitter_s: float, amp: float, t_prior: float, tag: str) -> NoiseBudget:
        prior = dict(registered_prior)
        prior["t_c"] = t_prior
        return NoiseBudget(
            budget_id=tag,
            jitter_std_s=jitter_s,
            relative_amp_std=amp,
            prior_std=prior,
        )

    # Table A+B: jitter x T-prior per arm (registered amp/RH/L/CO2 priors).
    scan_rows: list[dict[str, Any]] = []
    for arm in arms:
        for t_prior in t_prior_scan:
            for jitter_s in jitter_scan_s:
                tag = f"{arm}|jit{jitter_s*1e6:g}us|T{t_prior:g}K"
                row = evaluate_budget(
                    jac_by_arm[arm],
                    budget=_budget(jitter_s, registered_amp, t_prior, tag),
                    parameter_steps=steps,
                    window_width_percent=window_width,
                )
                scan_rows.append(row)
                print(
                    f"  {tag}: maxP90={row['max_p90_o2_percent']:.3f} "
                    f"medP90={row['median_p90_o2_percent']:.3f} "
                    f"maxNuis={row['max_nuisance_fraction']:.3f}",
                    flush=True,
                )

    # Per-arm required jitter at registered T prior (loose -> tight order).
    required_jitter: dict[str, Any] = {}
    for arm in arms:
        rows = [
            r
            for r in scan_rows
            if r["arm"] == arm and r["prior_std"]["t_c"] == registered_prior["t_c"]
        ]
        rows.sort(key=lambda r: -float(r["jitter_std_s"]))
        hit = required_budget_from_scan(
            rows, target_p90=target_p90, max_nuisance_fraction=max_nuis, statistic=statistic
        )
        required_jitter[arm] = (
            {"jitter_std_us": hit["jitter_std_s"] * 1e6, "max_p90": hit["max_p90_o2_percent"]}
            if hit
            else None
        )

    # Pareto (jitter, T prior) passing combos per arm.
    pareto: dict[str, list[dict[str, Any]]] = {}
    for arm in arms:
        combos = pareto_passing_combos(
            [r for r in scan_rows if r["arm"] == arm],
            target_p90=target_p90,
            max_nuisance_fraction=max_nuis,
            statistic=statistic,
        )
        pareto[arm] = [
            {
                "jitter_std_us": c["jitter_std_s"] * 1e6,
                "prior_t_c_k": c["prior_std"]["t_c"],
                "max_p90": c["max_p90_o2_percent"],
                "max_nuisance_fraction": c["max_nuisance_fraction"],
            }
            for c in combos
        ]

    # Table C: K-subset scan for the designated arm across jitter values.
    k_rows: list[dict[str, Any]] = []
    k_arm = str(config["k_subset_arm"])
    for subset_name, freqs in mrs2_config["frequency_subsets"].items():
        jac = precompute_arm_jacobians(
            points, arm=k_arm, f_hz=[float(x) for x in freqs], **pre_kwargs
        )
        for t_prior in (registered_prior["t_c"], min(t_prior_scan)):
            for jitter_s in jitter_scan_s:
                tag = f"{k_arm}|{subset_name}|jit{jitter_s*1e6:g}us|T{t_prior:g}K"
                row = evaluate_budget(
                    jac,
                    budget=_budget(jitter_s, registered_amp, t_prior, tag),
                    parameter_steps=steps,
                    window_width_percent=window_width,
                )
                row["subset"] = subset_name
                k_rows.append(row)

    # Table D: alpha calibration scan for the designated arm.
    amp_rows: list[dict[str, Any]] = []
    amp_arm = str(config["amp_scan_arm"])
    for amp in amp_scan:
        for t_prior in (registered_prior["t_c"], min(t_prior_scan)):
            for jitter_s in jitter_scan_s:
                tag = f"{amp_arm}|amp{amp:g}|jit{jitter_s*1e6:g}us|T{t_prior:g}K"
                row = evaluate_budget(
                    jac_by_arm[amp_arm],
                    budget=_budget(jitter_s, amp, t_prior, tag),
                    parameter_steps=steps,
                    window_width_percent=window_width,
                )
                amp_rows.append(row)

    # Table E: floor diagnosis — which remaining prior binds once TOF/T are tight.
    diag_cfg = config["floor_diagnosis"]
    diag_arm = str(diag_cfg["arm"])
    diag_jitter_s = float(diag_cfg["jitter_std_us"]) * 1e-6
    diag_rows: list[dict[str, Any]] = []
    from tv3.audit.mrs6_noise_budget import budget_passes  # noqa: E402

    for variant in diag_cfg["variants"]:
        prior = dict(registered_prior)
        prior["t_c"] = float(diag_cfg["t_prior_k"])
        prior.update({k: float(v) for k, v in variant["prior_overrides"].items()})
        row = evaluate_budget(
            jac_by_arm[diag_arm],
            budget=NoiseBudget(
                budget_id=f"diag|{variant['id']}",
                jitter_std_s=diag_jitter_s,
                relative_amp_std=float(variant["relative_amp_std"]),
                prior_std=prior,
            ),
            parameter_steps=steps,
            window_width_percent=window_width,
        )
        row["variant_id"] = variant["id"]
        row["passes_spec_target"] = budget_passes(
            row, target_p90=target_p90, max_nuisance_fraction=max_nuis, statistic=statistic
        )
        diag_rows.append(row)
        print(
            f"  diag {variant['id']}: maxP90={row['max_p90_o2_percent']:.3f} "
            f"medP90={row['median_p90_o2_percent']:.3f} pass={row['passes_spec_target']}",
            flush=True,
        )

    payload = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "schema_version": config["schema_version"],
        "quick": bool(args.quick),
        "verdict": MRS6_VERDICT,
        "mrs2_verdict_unchanged": "mrs2_rank_upgraded_p90_fail",
        "spec_targets": targets,
        "n_narrow_points": len(points),
        "registered_noise": {
            "jitter_std_us": float(mrs2_config["observation"]["jitter_std_s"]) * 1e6,
            "relative_amp_std": registered_amp,
            "prior_std": registered_prior,
        },
        "required_jitter_at_registered_priors": required_jitter,
        "pareto_jitter_tprior": pareto,
        "floor_diagnosis": [
            {
                "variant_id": r["variant_id"],
                "jitter_std_us": r["jitter_std_s"] * 1e6,
                "prior_std": r["prior_std"],
                "relative_amp_std": r["relative_amp_std"],
                "max_p90": r["max_p90_o2_percent"],
                "median_p90": r["median_p90_o2_percent"],
                "passes_spec_target": r["passes_spec_target"],
            }
            for r in diag_rows
        ],
        "n_scan_rows": len(scan_rows),
        "n_k_subset_rows": len(k_rows),
        "n_amp_rows": len(amp_rows),
    }
    verdict_path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    _write_csv(out_dir / "budget_scan.csv", scan_rows)
    _write_csv(out_dir / "k_subset_scan.csv", k_rows)
    _write_csv(out_dir / "amp_scan.csv", amp_rows)
    _write_csv(out_dir / "floor_diagnosis.csv", diag_rows)

    stage_path = _TV3_ROOT / "configs" / "tv3_mrs" / "stage_status.json"
    stage = _load_json(stage_path)
    stage["mrs6"] = {
        "verdict": MRS6_VERDICT,
        "delivered_at": datetime.now(timezone.utc).date().isoformat(),
        "verdict_path": "outputs/tv3_mrs/mrs6_hardware/mrs6_verdict.json",
        "spec_doc": "docs/archive/completed/tv3_mrs6_hardware_requirements.md",
        "quick": bool(args.quick),
    }
    stage["allowed_next_stage"] = "MRS_line_closed"
    stage_path.write_text(json.dumps(stage, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    print(json.dumps(payload["required_jitter_at_registered_priors"], indent=2))
    print(json.dumps(payload["pareto_jitter_tprior"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
