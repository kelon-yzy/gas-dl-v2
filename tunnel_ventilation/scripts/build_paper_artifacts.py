#!/usr/bin/env python3
"""Build the tv3 manuscript tables and figure data from frozen artifacts.

The manuscript contains no hand-transcribed numbers. Every table and figure
series is produced here by reading frozen outputs, and every output records the
source path and its SHA-256 in ``provenance.json``.

Fails loudly when a source artifact is missing: a manuscript number with no
reachable source is a defect, not a warning.
"""
from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

_TV3_ROOT = Path(__file__).resolve().parents[1]
if str(_TV3_ROOT) not in sys.path:
    sys.path.insert(0, str(_TV3_ROOT))

from tv3.audit.mrs_ei_registry import load_json, sha256_file  # noqa: E402

_C2_FREEZE = Path(
    "outputs/runs/tv3_mrs_ei/mei4_posterior_calibration/freezes"
    "/20260730T071532806157Z_76811228bcea"
)
_B4_FREEZE = Path(
    "outputs/runs/tv3_mrs_ei/mei3_varpro_audit/freezes"
    "/20260729T120958962354Z_cf7ed57312d9"
)
_O2_NARROW_RANGE_PP = 3.20
_INTERVAL_LEVELS = ("0.5", "0.8", "0.9", "0.95")

BUILDERS: dict[str, Callable[["BuildContext"], Any]] = {}


def register_builder(name: str) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
        if name in BUILDERS:
            raise ValueError(f"duplicate builder: {name}")
        BUILDERS[name] = func
        return func

    return decorator


@dataclass
class BuildContext:
    """Reads sources and records provenance for every file it touches."""

    project_root: Path
    sources: dict[str, str] = field(default_factory=dict)

    def _register(self, path: Path) -> Path:
        if not path.is_file():
            raise FileNotFoundError(f"required source artifact is missing: {path}")
        relative = path.relative_to(self.project_root).as_posix()
        self.sources.setdefault(relative, sha256_file(path))
        return path

    def json_at(self, relative: str | Path) -> Any:
        return load_json(self._register(self.project_root / relative))

    def register_source(self, relative: str | Path) -> None:
        """Record a source file whose contents are consumed by import, not parsed."""
        self._register(self.project_root / relative)

    def rows_at(self, relative: str | Path) -> list[dict[str, str]]:
        path = self._register(self.project_root / relative)
        return list(csv.DictReader(path.read_text(encoding="utf-8").splitlines()))


def _fmt(value: float, digits: int = 4) -> float:
    return round(float(value), digits)


@register_builder("table1_six_measurements")
def _build_table1(ctx: BuildContext) -> list[dict[str, Any]]:
    """Six mutually independent measurements of the narrow-window limit."""
    rows: list[dict[str, Any]] = []

    oracle = ctx.json_at("outputs/tv3_d0/oracle_ridge/metrics.json")
    bins = oracle["evaluations"]["val"]["conditional_metrics"]["o2_bins"]["bins"]
    bin_r2 = [b["component_metrics"]["x_O2"]["r2"] for b in bins.values()]
    rows.append(
        {
            "id": "d0_oracle_bins",
            "method": "Oracle-feature ridge on 0.8 vol% target bins",
            "quantity": "bin-wise R^2 range (validation)",
            "low": _fmt(min(bin_r2), 2),
            "high": _fmt(max(bin_r2), 2),
            "unit": "R^2",
            "reading": "negative in every bin even with ground-truth features",
            "source": "outputs/tv3_d0/oracle_ridge/metrics.json",
        }
    )

    v1 = ctx.json_at("outputs/tv3_identifiability/metrics.json")
    windows = v1["narrow_window_summaries"]
    rows.append(
        {
            "id": "identifiability_v1",
            "method": "One-way TOF Fisher rank + nuisance propagation",
            "quantity": "narrow-window equivalent target error P90",
            "low": _fmt(min(w["combined_p90_o2_error_percent_min"] for w in windows), 2),
            "high": _fmt(max(w["combined_p90_o2_error_percent_max"] for w in windows), 2),
            "unit": "vol%",
            "reading": "joint Fisher rank 1; all 189 points rejected (flow unrepresented)",
            "source": "outputs/tv3_identifiability/metrics.json",
        }
    )

    f4 = ctx.json_at("outputs/tv3_bidir/identifiability_v2/f4_verdict.json")
    nominal = f4["business_gate_assessment_nominal"]
    conservative = f4["business_gate_assessment_conservative"]
    rows.append(
        {
            "id": "bidirectional_f4",
            "method": "Flow-decoupled bidirectional Fisher",
            "quantity": "narrow-window max P90 (nominal to conservative scenario)",
            "low": _fmt(nominal["target_p90_o2_error_percent"]["observed_narrow_window_max"], 2),
            "high": _fmt(
                conservative["target_p90_o2_error_percent"]["observed_narrow_window_max"], 2
            ),
            "unit": "vol%",
            "reading": (
                "rejection rate fell to "
                f"{nominal['max_rejection_rate']['rejected_point_count']}"
                f"/{nominal['max_rejection_rate']['evaluated_point_count']}; "
                f"verdict {f4['verdict']}"
            ),
            "source": "outputs/tv3_bidir/identifiability_v2/f4_verdict.json",
        }
    )

    mrs2 = ctx.json_at("outputs/tv3_mrs/identifiability_mrs2/mrs2_verdict.json")
    arms = {k: v for k, v in mrs2["arm_summaries"].items() if k != "obs-single-200k"}
    best = min(arms.items(), key=lambda kv: kv[1]["median_p90_o2_percent"])
    rows.append(
        {
            "id": "mrs2_multifreq",
            "method": "Multi-frequency forward relative SVD rank + CRLB",
            "quantity": f"best-arm median P90 ({best[0]})",
            "low": _fmt(best[1]["median_p90_o2_percent"], 2),
            "high": _fmt(best[1]["median_p90_o2_percent"], 2),
            "unit": "vol%",
            "reading": (
                "rank rose from 1 to "
                f"{best[1]['min_joint_rank']}-{best[1]['max_joint_rank']}; "
                f"verdict {mrs2['verdict']}"
            ),
            "source": "outputs/tv3_mrs/identifiability_mrs2/mrs2_verdict.json",
        }
    )

    scan = ctx.rows_at("outputs/tv3_mrs/mrs6_hardware/budget_scan.csv")
    tightest_jitter = min(float(r["jitter_std_s"]) for r in scan)
    tightest_prior = min(float(r["prior_t_c_k"]) for r in scan)
    floor = [
        float(r["max_p90_o2_percent"])
        for r in scan
        if float(r["jitter_std_s"]) == tightest_jitter
        and float(r["prior_t_c_k"]) == tightest_prior
    ]
    rows.append(
        {
            "id": "mrs6_hardware",
            "method": "Hardware requirement inversion scan",
            "quantity": "saturation floor across multi-frequency arms at the tightest budget",
            "low": _fmt(min(floor), 2),
            "high": _fmt(max(floor), 2),
            "unit": "vol%",
            "reading": (
                f"tightest budget = jitter {tightest_jitter * 1e6:.2f} us, "
                f"T prior {tightest_prior:.1f} K; binding constraint is the path-length prior"
            ),
            "source": "outputs/tv3_mrs/mrs6_hardware/budget_scan.csv",
        }
    )

    widths = _o2_interval_width_medians(ctx, domain="test", method="M1")
    s1_p90 = _s1_test_p90(ctx)
    rows.append(
        {
            "id": "mei3_b4_mei4_c2",
            "method": "VarPro solver point estimate + Laplace posterior",
            "quantity": "S1 test P90; 95% interval median width vs working range",
            "low": _fmt(s1_p90, 2),
            "high": _fmt(widths[-1], 2),
            "unit": "vol% ; pp",
            "reading": (
                f"interval median {widths[-1]:.2f} pp is "
                f"{widths[-1] / _O2_NARROW_RANGE_PP:.2f}x the {_O2_NARROW_RANGE_PP:.2f} pp range"
            ),
            "source": f"{_B4_FREEZE.as_posix()} ; {_C2_FREEZE.as_posix()}",
        }
    )
    return rows


def _s1_test_p90(ctx: BuildContext) -> float:
    """S1 absolute-error P90 on the test domain, recomputed from paired solutions."""
    rows = ctx.rows_at(_B4_FREEZE / "solver_comparison.csv")
    errors = sorted(
        float(r["abs_err_o2"])
        for r in rows
        if r["method"] == "S1" and r["split"] == "test" and r["abs_err_o2"]
    )
    if not errors:
        raise KeyError("no S1 test rows in solver_comparison.csv")
    return _quantile(errors, 0.9)


def _quantile(sorted_values: list[float], q: float) -> float:
    if not sorted_values:
        raise ValueError("empty sample")
    position = q * (len(sorted_values) - 1)
    lower = int(position)
    upper = min(lower + 1, len(sorted_values) - 1)
    weight = position - lower
    return sorted_values[lower] * (1 - weight) + sorted_values[upper] * weight


def _o2_interval_width_medians(
    ctx: BuildContext, *, domain: str, method: str
) -> list[float]:
    rows = ctx.rows_at(_C2_FREEZE / f"posterior_intervals_{domain}.csv")
    accepted = [
        r
        for r in rows
        if r["method"] == method and r["rejected"].strip().lower() == "false"
    ]
    return [
        statistics.median(
            float(r[f"O2_upper_{level}"]) - float(r[f"O2_lower_{level}"])
            for r in accepted
            if r[f"O2_upper_{level}"]
        )
        for level in _INTERVAL_LEVELS
    ]


def _o2_r2_by_split(
    ctx: BuildContext, relative: str, *, splits: tuple[str, ...] = ("train", "val", "test", "extrapolation")
) -> dict[str, Any]:
    """Per-split target R^2 plus the run provenance every table must carry."""
    metrics = ctx.json_at(relative)
    evaluations = metrics["evaluations"]
    row: dict[str, Any] = {
        "dataset_dir": metrics.get("dataset_dir"),
        "modalities": ",".join(metrics.get("modalities") or []) or None,
        "feature_builder": metrics.get("feature_builder"),
    }
    for split in splits:
        value = (
            evaluations.get(split, {})
            .get("component_metrics", {})
            .get("x_O2", {})
            .get("r2")
        )
        row[f"o2_r2_{split}"] = _fmt(value) if value is not None else None
    return row


def _mean_o2_r2_from_summary(
    ctx: BuildContext, relative: str, *, group: str
) -> dict[str, Any]:
    """Seed-averaged target R^2 from a multi-seed summary."""
    summary = ctx.json_at(relative)
    records = [r for r in summary["records"] if r.get("group") == group]
    if not records:
        raise KeyError(f"no records for group {group!r} in {relative}")
    row: dict[str, Any] = {
        "dataset_dir": summary.get("dataset_dir"),
        "n_seeds": len(records),
    }
    for split in ("val", "test", "extrapolation"):
        values = [r["o2_r2"][split] for r in records if split in (r.get("o2_r2") or {})]
        row[f"o2_r2_{split}"] = _fmt(statistics.mean(values)) if values else None
        row[f"o2_r2_{split}_std"] = (
            _fmt(statistics.pstdev(values)) if len(values) > 1 else 0.0
        )
    return row


@register_builder("table2_discriminative_heads")
def _build_table2(ctx: BuildContext) -> list[dict[str, Any]]:
    """Regression heads on the same feature matrices, and what each may claim."""
    single_runs = [
        ("D0_observed_ridge", "observed features, ridge", "measurement-level linear baseline",
         "outputs/tv3_d0/observed_ridge/metrics.json"),
        ("B1_rawdsp_ridge", "RawDSP features, ridge", "deployable linear baseline",
         "outputs/tv3_d2b/raw_dsp_ridge_provenance/metrics.json"),
        ("R5_mlp", "observed features, MLP without target scaling", "fails on the training set",
         "outputs/tv3_r5/mlp_observed/metrics.json"),
        ("R5T_mlp_target_scaled", "observed features, MLP with per-target scaling", "same model, scaled targets",
         "outputs/tv3_r5/mlp_observed_target_scaled/metrics.json"),
        ("R5p_tabpfn", "observed features, TabPFN", "upper-bound probe, not deployable",
         "outputs/tv3_r5/tabpfn_observed/metrics.json"),
        ("R7_extratrees", "observed features, ExtraTrees", "generalisation gap",
         "outputs/tv3_r7/extratrees_observed/metrics.json"),
    ]
    rows = []
    for run_id, label, role, path in single_runs:
        rows.append({"id": run_id, "label": label, "role": role, "n_seeds": 1, **_o2_r2_by_split(ctx, path), "source": path})

    multi = [
        ("B6_rawdsp_mlp", "RawDSP features, target-scaled MLP", "multi-seed stable",
         "outputs/tv3_r5t_b6_multiseed/summary.json", "b6"),
        ("B7_oof_ridge_residual_mlp", "out-of-fold ridge + residual MLP", "frozen default head",
         "outputs/tv3_d2b/b7_oof_ridge_residual_mlp/summary.json", "b7"),
    ]
    for run_id, label, role, path, group in multi:
        rows.append({"id": run_id, "label": label, "role": role, **_mean_o2_r2_from_summary(ctx, path, group=group), "source": path})
    return rows


@register_builder("table3_b7_protocol")
def _build_table3(ctx: BuildContext) -> list[dict[str, Any]]:
    """Four splitters x three split seeds x three training seeds."""
    rows = ctx.rows_at("outputs/tv3_b7_protocol/result_matrix.csv")
    hashes = ctx.json_at("outputs/tv3_baseline_freeze/split_hashes.json")
    unique_ood = {}
    for entry in hashes["derived_splits"]:
        unique_ood.setdefault(entry["protocol_id"], set()).add(entry["ood_set_hash"])

    out = []
    for protocol in ("R", "L", "S-Y", "S-L"):
        subset = [r for r in rows if r["protocol_id"] == protocol]
        if not subset:
            raise KeyError(f"protocol {protocol} missing from the B7 matrix")

        def stats(column: str) -> tuple[float, float]:
            values = [float(r[column]) for r in subset]
            return statistics.mean(values), statistics.pstdev(values)

        test_mean, test_std = stats("b7_test_o2_r2")
        ood_mean, ood_std = stats("b7_extrapolation_o2_r2")
        out.append(
            {
                "protocol_id": protocol,
                "is_ood_evidence": subset[0]["is_ood_evidence"],
                "n_rows": len(subset),
                "n_unique_ood_sets": len(unique_ood.get(protocol, set())),
                "b7_test_mean": _fmt(test_mean),
                "b7_test_std": _fmt(test_std),
                "b7_ood_mean": _fmt(ood_mean),
                "b7_ood_std": _fmt(ood_std),
                "delta_test_vs_b1": _fmt(statistics.mean(float(r["delta_o2_r2_test"]) for r in subset)),
                "delta_ood_vs_b1": _fmt(statistics.mean(float(r["delta_o2_r2_extrapolation"]) for r in subset)),
            }
        )
    return out


@register_builder("table4_end_to_end")
def _build_table4(ctx: BuildContext) -> list[dict[str, Any]]:
    """Five distinct end-to-end waveform structures, reported on every split."""
    runs = [
        ("R1a_minirocket_tof", "MiniRocket on the TOF scalar sequence", "random convolution",
         "outputs/tv3_rocket/r1a/metrics.json"),
        ("R1b_minirocket_waveform", "MiniRocket on raw waveforms", "random convolution",
         "outputs/tv3_rocket/r1b/metrics.json"),
        ("D2_tof_phasenet", "fixed I/Q filter bank + TCN", "learned front end",
         "outputs/tv3_d2/tof_phasenet_s20260704/metrics.json"),
        ("fusion_v3_l2", "CNN1D + TCN multimodal fusion", "learned front end",
         "outputs/archive/legacy_dl/tv3_tcn_multimodal_v3_l2/metrics.json"),
        ("E1_ec_msw", "multi-scale learned encoder with gating", "learned front end",
         "outputs/tv3_ec_msw/e1_s20260704/metrics.json"),
        ("E1r_template_anchored", "frozen template anchor + learned sequence head", "learned sequence head",
         "outputs/tv3_ec_msw/e1r_s20260704/metrics.json"),
    ]
    return [
        {"id": run_id, "label": label, "family": family, **_o2_r2_by_split(ctx, path), "source": path}
        for run_id, label, family, path in runs
    ]


@register_builder("table5_module_c_grouping")
def _build_table5(ctx: BuildContext) -> list[dict[str, Any]]:
    """Physical grouping against an equal-capacity random permutation control."""
    rows = ctx.rows_at("outputs/tv3_module_c_grouped_bottleneck/result_matrix.csv")
    out = []
    for variant in sorted({r["variant"] for r in rows}):
        for protocol in ("R", "L", "S-Y", "S-L"):
            subset = [r for r in rows if r["variant"] == variant and r["protocol_id"] == protocol]
            if not subset:
                continue
            out.append(
                {
                    "variant": variant,
                    "protocol_id": protocol,
                    "n_rows": len(subset),
                    "parameter_count": subset[0]["parameter_count"],
                    "test_o2_r2": _fmt(statistics.mean(float(r["c1c2_test_o2_r2"]) for r in subset)),
                    "delta_test_vs_b7": _fmt(statistics.mean(float(r["delta_vs_c0_test"]) for r in subset)),
                    "ood_o2_r2": _fmt(statistics.mean(float(r["c1c2_extrapolation_o2_r2"]) for r in subset)),
                    "delta_ood_vs_b7": _fmt(statistics.mean(float(r["delta_vs_c0_extrapolation"]) for r in subset)),
                }
            )
    return out


@register_builder("table6_solver_efficiency")
def _build_table6(ctx: BuildContext) -> list[dict[str, Any]]:
    """Solver comparison gated on the bootstrap confidence lower bound."""
    bootstrap = ctx.json_at(_B4_FREEZE / "bootstrap_report.json")
    rows = ctx.rows_at(_B4_FREEZE / "solver_comparison.csv")
    out = []
    for domain, key in (("test", "test"), ("ood", "ood")):
        report = bootstrap[key]
        entry = {
            "domain": domain,
            "n_resamples": report["n_resamples"],
            "ci_level": report["ci_level"],
            "relative_improvement_point": _fmt(report["point_estimate"]),
            "relative_improvement_ci_lower": _fmt(report["ci_lower"]),
            "relative_improvement_ci_upper": _fmt(report["ci_upper"]),
            "practical_equivalence_band": 0.02,
            "clears_band": report["ci_lower"] > 0.02,
        }
        for method in ("S1", "S2"):
            errors = sorted(
                float(r["abs_err_o2"])
                for r in rows
                if r["method"] == method and r["split"] == domain and r["abs_err_o2"]
            )
            entry[f"{method}_p90_abs_err_o2"] = _fmt(_quantile(errors, 0.9)) if errors else None
        out.append(entry)
    return out


@register_builder("table7_structural_verification")
def _build_table7(ctx: BuildContext) -> list[dict[str, Any]]:
    """Solver structural admissibility, verified before the adoption comparison."""
    phase_a = ctx.json_at(
        "outputs/runs/tv3_mrs_ei/mei3_varpro_audit/freezes"
        "/20260728T080522165154Z_7cd8443230fa/mei3_structure_audit.json"
    )
    equivalence = phase_a["numerical_equivalence"]
    b2 = ctx.json_at(
        "outputs/runs/tv3_mrs_ei/mei3_varpro_audit/freezes"
        "/20260729T081421139186Z_c0ade3f5df14/projected_jacobian_report.json"
    )
    return [
        {
            "check": "variable_projection_vs_joint_reference_parameters",
            "quantity": "max absolute parameter difference",
            "value": equivalence["max_parameter_difference_vs_joint_reference"],
            "n_linear_parameters": equivalence["n_linear_parameters"],
            "source": (
                "outputs/runs/tv3_mrs_ei/mei3_varpro_audit/freezes"
                "/20260728T080522165154Z_7cd8443230fa/mei3_structure_audit.json"
            ),
        },
        {
            "check": "variable_projection_vs_joint_reference_residuals",
            "quantity": "max projected residual difference",
            "value": equivalence["max_projected_residual_difference_vs_joint_reference"],
            "n_linear_parameters": equivalence["n_linear_parameters"],
            "source": (
                "outputs/runs/tv3_mrs_ei/mei3_varpro_audit/freezes"
                "/20260728T080522165154Z_7cd8443230fa/mei3_structure_audit.json"
            ),
        },
        {
            "check": "projected_jacobian",
            "quantity": "max relative error against finite differences",
            "value": b2["max_relative_error"],
            "n_linear_parameters": equivalence["n_linear_parameters"],
            "source": (
                "outputs/runs/tv3_mrs_ei/mei3_varpro_audit/freezes"
                "/20260729T081421139186Z_c0ade3f5df14/projected_jacobian_report.json"
            ),
        },
    ]


@register_builder("table8_f5_flow_decoupling")
def _build_table8(ctx: BuildContext) -> list[dict[str, Any]]:
    """Pre-registered criteria for the flow-decoupling upgrade, per head.

    Criterion (a) is the adoption gate; criterion (e) records how well the
    mechanism itself worked. Reporting both is the point: the mechanism
    succeeded and the gate still failed.

    Criterion (d) is not a per-head check: it is evaluated per
    selector/seed/split cell and is emitted as a separate protocol-level row so
    that the manuscript can quote it from a frozen artifact like the others.
    """
    verdict = ctx.json_at("outputs/tv3_bidir/model_protocol_wide/f5_verdict.json")
    rows: list[dict[str, Any]] = []
    for head, payload in verdict["gates"].items():
        for check_id, check in payload["checks"].items():
            detail = check.get("detail") or {}
            row: dict[str, Any] = {
                "head": head,
                "check": check_id,
                "value": _fmt(check["value"], 5) if "value" in check else None,
                "threshold_min": check.get("threshold_min"),
                "threshold_max": check.get("threshold_max"),
                "passed": check["passed"],
            }
            # reciprocity is a time in seconds, of order 1e-5; six decimal
            # places would round it to two significant figures and make the
            # value quoted in the manuscript unreproducible from this file.
            detail_digits = {
                "a1_o2_mae": 6,
                "a2_o2_mae": 6,
                "a3_o2_mae": 6,
                "ab_mean_abs_seq_bias": 6,
                "pair_mean_abs_seq_bias": 6,
                "reciprocity_p95_of_seq_p95": 11,
            }
            for key, digits in detail_digits.items():
                if key in detail:
                    row[key] = _fmt(detail[key], digits)
            if "ab_mean_abs_seq_bias" in detail and detail["ab_mean_abs_seq_bias"]:
                row["paired_bias_reduction"] = _fmt(
                    1.0 - detail["pair_mean_abs_seq_bias"] / detail["ab_mean_abs_seq_bias"], 4
                )
            rows.append(row)
        rows.append(
            {
                "head": head,
                "check": "_head_summary",
                "value": None,
                "passed": payload["core_gates_passed"],
                "a1_test_o2_r2": _fmt(payload["a1_test_o2_r2"]),
                "a3_test_o2_r2": _fmt(payload["a3_test_o2_r2"]),
            }
        )
    selector_d = verdict["selector_gate_d"]
    deltas = [float(cell["delta_r2_o2"]) for cell in selector_d["cells"]]
    rows.append(
        {
            "head": "_protocol",
            "check": "d_selector_paired_noninferiority",
            "value": _fmt(min(deltas), 5),
            "threshold_min": selector_d["threshold_min"],
            "threshold_max": None,
            "passed": selector_d["passed"],
            "n_cells": selector_d["n_cells"],
            "delta_r2_o2_max": _fmt(max(deltas), 5),
        }
    )
    rows.append(
        {
            "head": "_verdict",
            "check": verdict["verdict"],
            "value": None,
            "passed": verdict["stage_passed"],
        }
    )
    return rows


@register_builder("table9_bands_and_frame_fidelity")
def _build_table9(ctx: BuildContext) -> list[dict[str, Any]]:
    """Which band a decision actually turned on, and the frame-fidelity gate.

    The design-ranking rows exist because the two bands are easy to confuse: the
    span here is above the numerical protection band and below the practical
    band, and the frozen decision reason names the practical one.
    """
    mei1 = ctx.json_at(
        "outputs/runs/tv3_mrs_ei/mei1_forward_envelope/freezes"
        "/20260728T064100731550Z_1b55aa2e09cb/mei1_verdict.json"
    )["audit"]
    stage = ctx.json_at("configs/tv3_mrs_ei/stage_status.json")
    delta_numerical = stage["mei0"]["delta_numerical_shared_upper_bound"]
    delta_numerical_by_profile = stage["mei0"]["delta_numerical_by_profile"]
    delta_practical = stage["mei0"]["delta_practical"]
    comparison = ctx.json_at(
        "outputs/runs/tv3_mrs_ei/mei1_forward_envelope/freezes"
        "/20260728T064100731550Z_1b55aa2e09cb/noise_profile_comparison.json"
    )

    rows: list[dict[str, Any]] = []
    for profile, payload in comparison.items():
        meta = payload["f0_ranking_meta"]
        span = meta["ranking_span_relative"]
        rows.append(
            {
                "item": f"design_ranking_span[{profile}]",
                "value": _fmt(span, 6),
                "delta_numerical": delta_numerical,
                "delta_numerical_this_profile": delta_numerical_by_profile.get(profile),
                "delta_practical": delta_practical,
                "above_numerical_band": span > delta_numerical,
                "below_practical_band": span < delta_practical,
                "span_over_delta_numerical": _fmt(span / delta_numerical, 2),
                "resolvable_numerical": meta["ranking_resolvable_numerical"],
                "resolvable_practical": meta["ranking_resolvable_practical"],
                "distinguishable_rank_levels": meta["distinguishable_rank_levels"],
            }
        )
    rows.append(
        {
            "item": "mei1_decision_reason",
            "value": mei1["decision_reason"],
            "resolvable_numerical": mei1["f0_ranking_meta"]["ranking_resolvable_numerical"],
            "resolvable_practical": mei1["f0_ranking_meta"]["ranking_resolvable_practical"],
            "distinguishable_rank_levels": mei1["f0_ranking_meta"]["distinguishable_rank_levels"],
        }
    )

    fidelity = ctx.json_at(
        "outputs/tv3_ec_msw/e1r_attach_e1d_sb_s20260704/frame_fidelity.json"
    )
    flat: dict[str, Any] = {}

    def _collect(node: Any) -> None:
        if isinstance(node, dict):
            for key, value in node.items():
                if isinstance(value, (int, float)) and not isinstance(value, bool):
                    flat.setdefault(key, value)
                else:
                    _collect(value)
        elif isinstance(node, list):
            for item in node:
                _collect(item)

    _collect(fidelity)
    rows.append(
        {
            "item": "e1r_frame_peak_mae_samples",
            "value": _fmt(flat["peak_mae_samples"], 6),
            "gate_max": flat.get("peak_mae_samples_max"),
            "passed": flat["peak_mae_samples"] <= flat["peak_mae_samples_max"],
        }
    )
    rows.append(
        {
            "item": "e1r_frame_peak_p95_samples",
            "value": _fmt(flat["peak_p95_abs_error_samples"], 6),
            "gate_max": flat.get("peak_p95_abs_error_samples_max"),
            "passed": flat["peak_p95_abs_error_samples"]
            <= flat["peak_p95_abs_error_samples_max"],
        }
    )
    return rows


@register_builder("table10_forward_model_constants")
def _build_table10(ctx: BuildContext) -> list[dict[str, Any]]:
    """Registered forward-model and acquisition constants, read from the code."""
    from tv3.sim.core import tunnel_ventilation_schema as schema
    from tv3.sim.generation import waveforms
    from tv3.sim.generation.tunnel_ventilation import acoustic_physics, conditions

    for path in (
        "tv3/sim/core/tunnel_ventilation_schema.py",
        "tv3/sim/generation/waveforms.py",
        "tv3/sim/generation/tunnel_ventilation/acoustic_physics.py",
        "tv3/sim/generation/tunnel_ventilation/conditions.py",
    ):
        ctx.register_source(path)

    spec = waveforms.WaveformSpec(sample_rate_hz=waveforms.SAMPLE_RATE_HZ)
    ranges = conditions.TUNNEL_VENTILATION_RANGES
    relaxation = acoustic_physics.PROCESSING_PARAMS_V2
    rows: list[dict[str, Any]] = []

    def add(group: str, name: str, value: Any, unit: str = "") -> None:
        rows.append({"group": group, "name": name, "value": value, "unit": unit})

    for component in schema.COMPONENT_FIELDS:
        key = component.removeprefix("x_")
        add("gas_property", f"molar_mass[{key}]", acoustic_physics._GAS_M[key], "kg/mol")
        add("gas_property", f"heat_capacity[{key}]", acoustic_physics._GAS_CP[key], "J/(mol K)")
        add("gas_property", f"thermal_conductivity[{key}]", acoustic_physics._GAS_LAMBDA[key], "W/(m K)")
        add("gas_property", f"viscosity[{key}]", acoustic_physics._GAS_ETA[key], "Pa s")
        add("gas_property", f"conductivity_T_exponent[{key}]", acoustic_physics._LAMBDA_T_EXPONENT[key], "")

    for species in ("co2", "n2", "o2", "h2o"):
        add("relaxation", f"f_relax[{species}]", relaxation[f"f_relax_{species}_per_atm"], "Hz/atm")
        add("relaxation", f"alpha_lambda_max[{species}]", relaxation[f"alpha_lambda_max_{species}"], "")
    add("relaxation", "h2o_catalysis_coefficient_co2", relaxation["k_h2o_to_f_relax_co2"], "1/%")
    add("relaxation", "classical_absorption_K_ref", relaxation["alpha_classical_K_ref"], "")

    for name, value, unit in (
        ("carrier_frequency", spec.center_frequency_hz, "Hz"),
        ("burst_cycles", spec.burst_cycles, ""),
        ("sample_rate", spec.sample_rate_hz, "Hz"),
        ("measurement_window", spec.measurement_window_s, "s"),
        ("daq_bits", spec.daq_bits, "bit"),
        ("daq_full_scale", spec.daq_full_scale_v, "V"),
        ("system_delay", spec.system_delay_s, "s"),
        ("cable_delay", spec.cable_delay_s, "s"),
        ("delay_correction", spec.delay_correction_s, "s"),
        ("trigger_jitter_std", spec.trigger_jitter_std_s, "s"),
        ("transducer_bandwidth", spec.transducer_bandwidth_hz, "Hz"),
        ("transducer_ringdown", spec.transducer_ringdown_cycles, "cycles"),
    ):
        add("acquisition", name, value, unit)

    add("composition_domain", "co2_range", list(ranges.co2), "%")
    add("composition_domain", "o2_range", list(ranges.o2), "%")
    add("composition_domain", "n2_range", [ranges.n2_min, ranges.n2_max], "%")
    add("composition_domain", "o2_working_range", _fmt(ranges.o2[1] - ranges.o2[0], 2), "pp")
    add("composition_domain", "path_length_base_range", list(conditions.L_M_BASE_RANGE), "m")
    add("schema", "component_fields", list(schema.COMPONENT_FIELDS), "")
    add("schema", "background_fields", list(schema.BACKGROUND_FIELDS), "")
    add("schema", "slow_channels", list(schema.SLOW_CHANNELS), "")
    add("schema", "n_slow_channels", len(schema.SLOW_CHANNELS), "")
    return rows


@register_builder("table11_front_end_parameters")
def _build_table11(ctx: BuildContext) -> list[dict[str, Any]]:
    """Signal-processing front-end parameters and the frame acceptance gates."""
    from tv3.ml import raw_dsp_features as front_end
    from tv3.ml import rocket_features
    from tv3.sim.generation import waveforms

    ctx.register_source("tv3/ml/raw_dsp_features.py")
    ctx.register_source("tv3/ml/rocket_features.py")
    config = front_end.RawDSPConfig(sample_rate_hz=waveforms.SAMPLE_RATE_HZ)
    rows = [
        {"group": "search_window", "name": "sound_speed_min", "value": config.sound_speed_min_m_per_s, "unit": "m/s"},
        {"group": "search_window", "name": "sound_speed_max", "value": config.sound_speed_max_m_per_s, "unit": "m/s"},
        {"group": "search_window", "name": "delay_min", "value": config.delay_min_s, "unit": "s"},
        {"group": "search_window", "name": "delay_max", "value": config.delay_max_s, "unit": "s"},
        {"group": "frame_gate", "name": "min_correlation_peak", "value": config.min_corr_peak, "unit": ""},
        {"group": "frame_gate", "name": "min_peak_to_sidelobe_ratio", "value": config.min_peak_to_sidelobe_ratio, "unit": ""},
        {"group": "frame_gate", "name": "min_snr", "value": config.min_snr_db, "unit": "dB"},
        {"group": "frame_gate", "name": "max_peak_width", "value": config.max_peak_width_samples, "unit": "samples"},
        {"group": "frame_gate", "name": "sidelobe_exclusion", "value": config.sidelobe_exclusion_samples, "unit": "samples"},
        {"group": "calibration", "name": "min_baseline_frames", "value": config.calibration_min_frames, "unit": "frames"},
        {"group": "calibration", "name": "fresh_air_composition", "value": list(front_end.FRESH_AIR_COMPOSITION), "unit": "%"},
        {"group": "robust_noise", "name": "mad_to_sigma_factor", "value": 1.4826, "unit": ""},
    ]
    for name, arrays, builder in (
        ("observed", rocket_features.D0_OBSERVED_PHYSICS_ARRAYS, rocket_features.D0_OBSERVED_FEATURE_BUILDER),
        ("raw_dsp", rocket_features.RAW_DSP_PHYSICS_ARRAYS, rocket_features.RAW_DSP_FEATURE_BUILDER),
    ):
        rows.append(
            {
                "group": "feature_builder",
                "name": f"n_physics_arrays[{name}]",
                "value": len(arrays),
                "unit": builder,
            }
        )
    return rows


@register_builder("table12_sensitivity_scale")
def _build_table12(ctx: BuildContext) -> list[dict[str, Any]]:
    """The scale of the target signal against the scale of one nuisance parameter.

    Computed from the registered forward operator rather than quoted, because the
    ratio in the last row is the quantitative statement of why the scheme is hard.
    """
    from tv3.sim.generation.tunnel_ventilation import conditions
    from tv3.sim.generation.tunnel_ventilation.acoustic_physics import (
        hidden_sound_speed_v2,
    )

    ctx.register_source("tv3/sim/generation/tunnel_ventilation/acoustic_physics.py")
    ranges = conditions.TUNNEL_VENTILATION_RANGES
    reference_t_c = 26.85  # 300.0 K

    def speed(co2: float, o2: float, t_c: float) -> float:
        return hidden_sound_speed_v2(0.0, 0.0, co2, 100.0 - co2 - o2, t_c, x_o2=o2)

    rows: list[dict[str, Any]] = []
    pure = {
        "CO2": hidden_sound_speed_v2(0.0, 0.0, 100.0, 0.0, reference_t_c, x_o2=0.0),
        "O2": hidden_sound_speed_v2(0.0, 0.0, 0.0, 0.0, reference_t_c, x_o2=100.0),
        "N2": hidden_sound_speed_v2(0.0, 0.0, 0.0, 100.0, reference_t_c, x_o2=0.0),
    }
    for name, value in pure.items():
        rows.append(
            {
                "item": f"pure_component_sound_speed[{name}]",
                "value": _fmt(value, 2),
                "unit": "m/s",
                "note": f"at {reference_t_c + 273.15:.1f} K",
            }
        )
    contrast = pure["N2"] - pure["O2"]
    rows.append(
        {
            "item": "sound_speed_contrast[N2-O2]",
            "value": _fmt(contrast, 2),
            "unit": "m/s",
            "note": (
                f"{contrast / pure['N2'] * 100:.2f}% of the N2 speed, "
                f"{contrast / pure['O2'] * 100:.2f}% of the O2 speed"
            ),
        }
    )

    o2_low, o2_high = ranges.o2
    spans = []
    for co2 in (ranges.co2[0], ranges.co2[1]):
        low = speed(co2, o2_low, reference_t_c)
        high = speed(co2, o2_high, reference_t_c)
        spans.append(abs(high - low))
        rows.append(
            {
                "item": f"target_range_sound_speed_span[co2={co2:.2f}]",
                "value": _fmt(abs(high - low), 4),
                "unit": "m/s",
                "note": (
                    f"O2 {o2_low:.2f} -> {o2_high:.2f} vol% moves the speed "
                    f"{low:.3f} -> {high:.3f} m/s ({(high - low) / low * 100:+.3f}%)"
                ),
            }
        )

    mid_co2 = (ranges.co2[0] + ranges.co2[1]) / 2.0
    mid_o2 = (o2_low + o2_high) / 2.0
    per_kelvin = abs(
        speed(mid_co2, mid_o2, 26.0) - speed(mid_co2, mid_o2, 25.0)
    )
    rows.append(
        {
            "item": "sound_speed_change_per_kelvin",
            "value": _fmt(per_kelvin, 4),
            "unit": "m/s",
            "note": f"at CO2={mid_co2:.2f} vol%, O2={mid_o2:.2f} vol%, 25 -> 26 C",
        }
    )
    rows.append(
        {
            "item": "one_kelvin_as_fraction_of_full_target_range",
            "value": _fmt(per_kelvin / max(spans), 4),
            "unit": "",
            "note": (
                "a 1 K temperature error moves the sound speed by this fraction of "
                "the change produced by traversing the entire target working range"
            ),
        }
    )
    return rows


@register_builder("fig2_d0_information")
def _build_fig2(ctx: BuildContext) -> list[dict[str, Any]]:
    """Where the target information sits, per feature configuration."""
    configs = [
        ("oracle_ridge", "oracle (ground-truth features)", "upper bound probe"),
        ("observed_ridge", "observed (measurement level)", "linear baseline"),
        ("no_tcs_ridge", "observed minus thermal conductivity", "ablation"),
        ("tof_only_ridge", "observed minus estimated sound speed", "ablation"),
        ("slow_only_ridge", "slow channels only", "ablation"),
        ("no_tof_ridge", "slow channels only (duplicate config)", "duplicate of previous"),
    ]
    rows = []
    for run, label, role in configs:
        metrics = ctx.json_at(f"outputs/tv3_d0/{run}/metrics.json")
        evaluations = metrics["evaluations"]
        row = {
            "run": run,
            "label": label,
            "role": role,
            "feature_builder": metrics["feature_builder"],
            "feature_count": metrics["feature_count"],
        }
        for split in ("val", "test", "extrapolation"):
            value = (
                evaluations.get(split, {})
                .get("component_metrics", {})
                .get("x_O2", {})
                .get("r2")
            )
            row[f"o2_r2_{split}"] = _fmt(value) if value is not None else None
        rows.append(row)

    oracle = next(r for r in rows if r["run"] == "oracle_ridge")["o2_r2_val"]
    observed = next(r for r in rows if r["run"] == "observed_ridge")["o2_r2_val"]
    for row in rows:
        row["oracle_inflation_val"] = _fmt(oracle - observed)
    return rows


@register_builder("fig3_e1d_ladder")
def _build_fig3(ctx: BuildContext) -> list[dict[str, Any]]:
    """Incremental feature ladder that localises the information gap."""
    summary = ctx.json_at("outputs/tv3_ec_msw/e1d_s20260704/summary.json")
    rows = []
    for stage, entries in summary["stages"].items():
        for entry in entries:
            evaluation = entry["eval"]
            rows.append(
                {
                    "stage": stage,
                    "feature_set": entry["name"],
                    "feature_count": entry["feature_count"],
                    "o2_r2_val": _fmt(evaluation["val"]["x_O2_r2"]),
                    "o2_r2_test": _fmt(evaluation["test"]["x_O2_r2"]),
                    "o2_r2_extrapolation": _fmt(evaluation["extrapolation"]["x_O2_r2"]),
                    "parity_passed_test": evaluation["test"]["parity_passed"],
                    "delta_o2_vs_control_test": _fmt(
                        evaluation["test"]["delta_vs_control"]["x_O2"]
                    ),
                }
            )
    return rows


@register_builder("fig4_mrs2_rank_vs_p90")
def _build_fig4(ctx: BuildContext) -> list[dict[str, Any]]:
    """Rank rises while the precision criterion does not follow."""
    verdict = ctx.json_at("outputs/tv3_mrs/identifiability_mrs2/mrs2_verdict.json")
    gate = verdict["gates"]["target_p90_o2_error_percent"]
    return [
        {
            "arm": arm,
            "is_negative_control": arm == "obs-single-200k",
            "n_points": summary["n_points"],
            "min_joint_rank": summary["min_joint_rank"],
            "max_joint_rank": summary["max_joint_rank"],
            "median_p90_vol_percent": _fmt(summary["median_p90_o2_percent"]),
            "max_p90_vol_percent": _fmt(summary["max_p90_o2_percent"]),
            "gate_vol_percent": gate,
            "unstable_fd_count": summary["unstable_fd_count"],
        }
        for arm, summary in verdict["arm_summaries"].items()
    ]


@register_builder("fig5_mrs6_hardware_floor")
def _build_fig5(ctx: BuildContext) -> list[dict[str, Any]]:
    """Precision as a function of jitter and temperature prior."""
    scan = ctx.rows_at("outputs/tv3_mrs/mrs6_hardware/budget_scan.csv")
    return [
        {
            "arm": row["arm"],
            "jitter_us": _fmt(float(row["jitter_std_s"]) * 1e6, 4),
            "t_prior_k": _fmt(float(row["prior_t_c_k"]), 3),
            "max_p90_vol_percent": _fmt(float(row["max_p90_o2_percent"])),
            "median_p90_vol_percent": _fmt(float(row["median_p90_o2_percent"])),
            "min_joint_rank": int(row["min_joint_rank"]),
        }
        for row in scan
    ]


@register_builder("fig6_c2_posterior_diagnostics")
def _build_fig6(ctx: BuildContext) -> dict[str, Any]:
    """Rejection composition, both coverage statistics, and width versus range."""
    diagnostics = ctx.json_at(_C2_FREEZE / "laplace_diagnostics.json")
    coverage = ctx.json_at(_C2_FREEZE / "coverage_report.json")

    rejection = [
        {
            "method": method,
            "domain": domain,
            "n": group["n"],
            "rejected": group["rejected"],
            "rejection_rate": _fmt(group["rejected"] / group["n"]),
            "reasons": dict(group["rejection_reasons"]),
        }
        for method, by_domain in diagnostics["methods"].items()
        for domain, group in by_domain.items()
    ]

    bands = []
    for band in coverage["primary_bands"]:
        if band["component"] != "O2":
            continue
        accepted = band["n"] - band["rejected"]
        bands.append(
            {
                "method": band["method"],
                "domain": band["domain"],
                "nominal_level": band["nominal_level"],
                "n": band["n"],
                "n_accepted": accepted,
                "unconditional": _fmt(band["covered"] / band["n"]),
                "selection_conditional": (
                    _fmt(band["covered"] / accepted) if accepted else None
                ),
                "within_acceptance_band": bool(band["within_acceptance_band"]),
            }
        )

    widths = {
        f"{method}_{domain}": [
            _fmt(w) for w in _o2_interval_width_medians(ctx, domain=domain, method=method)
        ]
        for method in ("M1", "M1b", "M2")
        for domain in ("test", "ood")
    }

    probes = diagnostics["complete_hessian"]
    return {
        "rejection_composition": rejection,
        "o2_coverage_bands": bands,
        "o2_interval_width_medians_pp": {
            "levels": list(_INTERVAL_LEVELS),
            "working_range_pp": _O2_NARROW_RANGE_PP,
            "series": widths,
        },
        "curvature_correctness": {
            method: {
                domain: group["o2_laplace_to_crb_ratio"].get("median")
                for domain, group in by_domain.items()
            }
            for method, by_domain in diagnostics["methods"].items()
        },
        "estimate_outside_domain": {
            "n_probes": len(probes),
            "n_unavailable": sum(1 for p in probes if p.get("status") != "computed"),
            "mixture_ids": sorted(
                p["mixture_id"] for p in probes if p.get("status") != "computed"
            ),
        },
    }


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"refuse to write an empty table: {path}")
    # Rows may carry different keys (e.g. single-run versus seed-averaged entries).
    # Use the union in first-seen order so a table never silently drops a column.
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, restval="")
        writer.writeheader()
        writer.writerows(rows)


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()
    output_dir = (
        args.output_dir.resolve()
        if args.output_dir is not None
        else _TV3_ROOT / "docs" / "paper" / "artifacts"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    context = BuildContext(project_root=_TV3_ROOT)
    written: dict[str, str] = {}
    for name, builder in BUILDERS.items():
        payload = builder(context)
        if isinstance(payload, list):
            target = output_dir / f"{name}.csv"
            _write_csv(target, payload)
        else:
            target = output_dir / f"{name}.json"
            _write_json(target, payload)
        written[name] = target.relative_to(_TV3_ROOT).as_posix()

    provenance = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "generator": "scripts/build_paper_artifacts.py",
        "outputs": written,
        "source_sha256": dict(sorted(context.sources.items())),
    }
    _write_json(output_dir / "provenance.json", provenance)

    print(
        json.dumps(
            {
                "output_dir": output_dir.relative_to(_TV3_ROOT).as_posix(),
                "n_outputs": len(written),
                "n_sources": len(context.sources),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
