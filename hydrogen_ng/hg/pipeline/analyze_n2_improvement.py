from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Sequence

from sim.core.schema import COMPONENT_FIELDS


DEFAULT_COMPARISONS = (
    ("ridge_all_modalities", "ridge_alr_ch4_all_modalities"),
    ("ridge_all_modalities", "ridge_ilr_n2_first_all_modalities"),
    ("cnn1d_tcn_fusion", "cnn1d_tcn_fusion_ilr"),
)

DEFAULT_PHASE_AWARE_COMPARISONS = (
    ("ml", "phase", "exposure", "ridge_all_modalities", "ridge_all_modalities_phase_exposure"),
    ("ml", "phase", "recovery", "ridge_all_modalities", "ridge_all_modalities_phase_recovery"),
    ("ml", "early", "0.50", "ridge_all_modalities", "ridge_all_modalities_early_050"),
    ("ml", "early", "0.75", "ridge_all_modalities", "ridge_all_modalities_early_075"),
    ("dl", "phase", "exposure", "cnn1d_tcn_fusion", "cnn1d_tcn_fusion_phase_exposure"),
    ("dl", "phase", "recovery", "cnn1d_tcn_fusion", "cnn1d_tcn_fusion_phase_recovery"),
    ("dl", "early", "0.50", "cnn1d_tcn_fusion", "cnn1d_tcn_fusion_early_050"),
    ("dl", "early", "0.75", "cnn1d_tcn_fusion", "cnn1d_tcn_fusion_early_075"),
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Analyze N2 gains for compositional target experiments.")
    parser.add_argument("--run-root", type=Path, required=True, help="Experiment run root, e.g. outputs/runs/formal_full.")
    parser.add_argument("--split", type=str, default="test", help="Evaluation split to compare.")
    parser.add_argument(
        "--extrapolation-split",
        type=str,
        default="extrapolation",
        help="Extrapolation split required by --phase-aware.",
    )
    parser.add_argument("--phase-aware", action="store_true", default=False, help="Analyze phase-aware N2 runs.")
    parser.add_argument("--n2-min-gain", type=float, default=0.10, help="Required x_N2 R2 gain.")
    parser.add_argument(
        "--other-component-max-drop",
        type=float,
        default=None,
        help="Maximum allowed R2 drop for H2/CH4/CO2.",
    )
    parser.add_argument(
        "--macro-rmse-max-regression",
        type=float,
        default=0.0,
        help="Maximum allowed macro RMSE regression.",
    )
    parser.add_argument("--json", action="store_true", default=False, help="Print JSON instead of Markdown.")
    parser.add_argument("--output-path", type=Path, default=None, help="Optional path to write the analysis report.")
    parser.add_argument("--json-output-path", type=Path, default=None, help="Optional path to write the analysis payload as JSON.")
    return parser


def analyze_n2_improvement(
    run_root: Path | str,
    *,
    split: str = "test",
    comparisons: tuple[tuple[str, str], ...] = DEFAULT_COMPARISONS,
    n2_min_gain: float = 0.10,
    other_component_max_drop: float = 0.02,
    macro_rmse_max_regression: float = 0.0,
) -> dict[str, Any]:
    run_root = Path(run_root)
    results = []
    for baseline_run, candidate_run in comparisons:
        baseline_payload = _load_metrics_payload(run_root, baseline_run)
        candidate_payload = _load_metrics_payload(run_root, candidate_run)
        baseline_eval = _payload_split_eval(baseline_payload, run_name=baseline_run, split=split, window="full")
        candidate_eval = _payload_split_eval(candidate_payload, run_name=candidate_run, split=split, window="full")
        baseline_components = baseline_eval["component_metrics"]
        candidate_components = candidate_eval["component_metrics"]

        n2_gain = _component_r2(candidate_components, "x_N2") - _component_r2(baseline_components, "x_N2")
        other_drops = _other_component_r2_drops(baseline_components, candidate_components)
        macro_rmse_regression = float(candidate_eval["metrics"]["rmse"]) - float(baseline_eval["metrics"]["rmse"])
        pass_flags = _pass_flags(
            n2_gain=n2_gain,
            other_component_r2_drops=other_drops,
            macro_rmse_regression=macro_rmse_regression,
            n2_min_gain=n2_min_gain,
            other_component_max_drop=other_component_max_drop,
            macro_rmse_max_regression=macro_rmse_max_regression,
        )

        results.append(
            {
                "baseline_run": baseline_run,
                "candidate_run": candidate_run,
                "split": split,
                "baseline_n2_r2": _component_r2(baseline_components, "x_N2"),
                "candidate_n2_r2": _component_r2(candidate_components, "x_N2"),
                "n2_r2_gain": n2_gain,
                "baseline_rmse": float(baseline_eval["metrics"]["rmse"]),
                "candidate_rmse": float(candidate_eval["metrics"]["rmse"]),
                "macro_rmse_regression": macro_rmse_regression,
                "other_component_r2_drops": other_drops,
                "candidate_aitchison_mean": _optional_aitchison_mean(candidate_eval),
                "conditional_bins": _conditional_bin_comparisons(
                    baseline_eval,
                    candidate_eval,
                    n2_min_gain=n2_min_gain,
                    other_component_max_drop=other_component_max_drop,
                    macro_rmse_max_regression=macro_rmse_max_regression,
                ),
                "protocol_windows": _protocol_window_comparisons(
                    baseline_payload,
                    candidate_payload,
                    baseline_run=baseline_run,
                    candidate_run=candidate_run,
                    split=split,
                    n2_min_gain=n2_min_gain,
                    other_component_max_drop=other_component_max_drop,
                    macro_rmse_max_regression=macro_rmse_max_regression,
                ),
                **pass_flags,
            }
        )
    return {
        "run_root": str(run_root),
        "split": split,
        "thresholds": {
            "n2_min_gain": n2_min_gain,
            "other_component_max_drop": other_component_max_drop,
            "macro_rmse_max_regression": macro_rmse_max_regression,
        },
        "comparisons": results,
    }


def analyze_phase_aware_n2(
    run_root: Path | str,
    *,
    split: str = "test",
    extrapolation_split: str = "extrapolation",
    comparisons: tuple[tuple[str, str, str, str, str], ...] = DEFAULT_PHASE_AWARE_COMPARISONS,
    n2_min_gain: float = 0.10,
    other_component_max_drop: float = 0.05,
    macro_rmse_max_regression: float = 0.0,
) -> dict[str, Any]:
    run_root = Path(run_root)
    results = []
    for kind, window_kind, window_value, baseline_run, candidate_run in comparisons:
        baseline_payload = _load_metrics_payload(run_root, baseline_run)
        candidate_payload = _load_metrics_payload(run_root, candidate_run)
        test_comparison = _compare_payload_split(
            baseline_payload,
            candidate_payload,
            baseline_run=baseline_run,
            candidate_run=candidate_run,
            split=split,
            n2_min_gain=n2_min_gain,
            other_component_max_drop=other_component_max_drop,
            macro_rmse_max_regression=macro_rmse_max_regression,
        )
        extrapolation_comparison = _compare_payload_split(
            baseline_payload,
            candidate_payload,
            baseline_run=baseline_run,
            candidate_run=candidate_run,
            split=extrapolation_split,
            n2_min_gain=n2_min_gain,
            other_component_max_drop=other_component_max_drop,
            macro_rmse_max_regression=macro_rmse_max_regression,
        )
        results.append(
            {
                "kind": kind,
                "window_kind": window_kind,
                "window_value": window_value,
                "baseline_run": baseline_run,
                "candidate_run": candidate_run,
                "test": test_comparison,
                "extrapolation": extrapolation_comparison,
                "extrapolation_n2_r2_margin": (
                    extrapolation_comparison["candidate_n2_r2"] - test_comparison["candidate_n2_r2"]
                ),
                "passed_strong_extrapolation": (
                    extrapolation_comparison["candidate_n2_r2"] >= test_comparison["candidate_n2_r2"] - 0.10
                ),
                "conditional_bins": test_comparison["conditional_bins"],
            }
        )
    return {
        "analysis_type": "phase_aware_n2",
        "run_root": str(run_root),
        "split": split,
        "extrapolation_split": extrapolation_split,
        "thresholds": {
            "n2_min_gain": n2_min_gain,
            "other_component_max_drop": other_component_max_drop,
            "macro_rmse_max_regression": macro_rmse_max_regression,
        },
        "comparisons": results,
        "groups": _phase_aware_groups(results),
    }


def format_markdown_report(payload: dict[str, Any]) -> str:
    lines = [
        "# N2 Improvement Analysis",
        "",
        f"- run_root: `{payload['run_root']}`",
        f"- split: `{payload['split']}`",
        "",
        "| baseline | candidate | N2 R2 gain | RMSE regression | max other R2 drop | Aitchison mean | pass |",
        "|---|---|---:|---:|---:|---:|---|",
    ]
    for item in payload["comparisons"]:
        max_other_drop = max(item["other_component_r2_drops"].values())
        aitchison = item["candidate_aitchison_mean"]
        aitchison_text = "" if aitchison is None else f"{aitchison:.6f}"
        pass_text = "yes" if item["passed_overall"] else "no"
        lines.append(
            f"| {item['baseline_run']} | {item['candidate_run']} | {item['n2_r2_gain']:.6f} | "
            f"{item['macro_rmse_regression']:.6f} | {max_other_drop:.6f} | {aitchison_text} | {pass_text} |"
        )
    window_rows = [
        (item, group_name, window_name, window)
        for item in payload["comparisons"]
        for group_name, group in item["protocol_windows"].items()
        for window_name, window in group.items()
    ]
    if window_rows:
        lines.extend(
            [
                "",
                "## Protocol Windows",
                "",
                "| baseline | candidate | group | window | N2 R2 gain | RMSE regression | max other R2 drop | Aitchison mean | pass |",
                "|---|---|---|---|---:|---:|---:|---:|---|",
            ]
        )
        for item, group_name, window_name, window in window_rows:
            aitchison = window["candidate_aitchison_mean"]
            aitchison_text = "" if aitchison is None else f"{aitchison:.6f}"
            max_other_drop = max(window["other_component_r2_drops"].values())
            pass_text = "yes" if window["passed_overall"] else "no"
            lines.append(
                f"| {item['baseline_run']} | {item['candidate_run']} | {group_name} | {window_name} | "
                f"{window['n2_r2_gain']:.6f} | {window['macro_rmse_regression']:.6f} | "
                f"{max_other_drop:.6f} | {aitchison_text} | {pass_text} |"
            )
    bin_rows = [
        (item, group_name, bin_name, bin_payload)
        for item in payload["comparisons"]
        for group_name, group in item["conditional_bins"].items()
        for bin_name, bin_payload in group.items()
    ]
    if bin_rows:
        lines.extend(
            [
                "",
                "## Conditional Bins",
                "",
                "| baseline | candidate | group | bin | count | range | N2 R2 gain | RMSE regression | max other R2 drop | pass |",
                "|---|---|---|---|---:|---|---:|---:|---:|---|",
            ]
        )
        for item, group_name, bin_name, bin_payload in bin_rows:
            range_text = f"{bin_payload['range'][0]:.6g}-{bin_payload['range'][1]:.6g}"
            max_other_drop = max(bin_payload["other_component_r2_drops"].values())
            pass_text = "yes" if bin_payload["passed_overall"] else "no"
            lines.append(
                f"| {item['baseline_run']} | {item['candidate_run']} | {group_name} | {bin_name} | "
                f"{bin_payload['count']} | {range_text} | {bin_payload['n2_r2_gain']:.6f} | "
                f"{bin_payload['macro_rmse_regression']:.6f} | {max_other_drop:.6f} | {pass_text} |"
            )
    protocol_bin_rows = [
        (item, window_group, window_name, bin_group, bin_name, bin_payload)
        for item in payload["comparisons"]
        for window_group, windows in item["protocol_windows"].items()
        for window_name, window_payload in windows.items()
        for bin_group, bins in window_payload["conditional_bins"].items()
        for bin_name, bin_payload in bins.items()
    ]
    if protocol_bin_rows:
        lines.extend(
            [
                "",
                "## Protocol Conditional Bins",
                "",
                "| baseline | candidate | window group | window | bin group | bin | count | range | N2 R2 gain | RMSE regression | max other R2 drop | pass |",
                "|---|---|---|---|---|---|---:|---|---:|---:|---:|---|",
            ]
        )
        for item, window_group, window_name, bin_group, bin_name, bin_payload in protocol_bin_rows:
            range_text = f"{bin_payload['range'][0]:.6g}-{bin_payload['range'][1]:.6g}"
            max_other_drop = max(bin_payload["other_component_r2_drops"].values())
            pass_text = "yes" if bin_payload["passed_overall"] else "no"
            lines.append(
                f"| {item['baseline_run']} | {item['candidate_run']} | {window_group} | {window_name} | "
                f"{bin_group} | {bin_name} | {bin_payload['count']} | {range_text} | "
                f"{bin_payload['n2_r2_gain']:.6f} | {bin_payload['macro_rmse_regression']:.6f} | "
                f"{max_other_drop:.6f} | {pass_text} |"
            )
    return "\n".join(lines).rstrip() + "\n"


def format_phase_aware_markdown_report(payload: dict[str, Any]) -> str:
    lines = [
        "# Phase-aware N2 Analysis",
        "",
        f"- run_root: `{payload['run_root']}`",
        f"- test split: `{payload['split']}`",
        f"- extrapolation split: `{payload['extrapolation_split']}`",
        "",
        "| kind | window | baseline | candidate | test N2 R2 gain | test RMSE regression | max other R2 drop | extrap N2 R2 | extrap margin | pass |",
        "|---|---|---|---|---:|---:|---:|---:|---:|---|",
    ]
    for item in payload["comparisons"]:
        test = item["test"]
        max_other_drop = max(test["other_component_r2_drops"].values())
        pass_text = "yes" if test["passed_overall"] and item["passed_strong_extrapolation"] else "no"
        lines.append(
            f"| {item['kind']} | {item['window_kind']}:{item['window_value']} | {item['baseline_run']} | "
            f"{item['candidate_run']} | {test['n2_r2_gain']:.6f} | {test['macro_rmse_regression']:.6f} | "
            f"{max_other_drop:.6f} | {item['extrapolation']['candidate_n2_r2']:.6f} | "
            f"{item['extrapolation_n2_r2_margin']:.6f} | {pass_text} |"
        )

    for kind in ("ml", "dl"):
        rows = [item for item in payload["comparisons"] if item["kind"] == kind]
        if not rows:
            continue
        lines.extend(["", f"## {kind.upper()}", ""])
        for window_kind in ("phase", "early"):
            window_rows = [item for item in rows if item["window_kind"] == window_kind]
            if not window_rows:
                continue
            lines.extend(
                [
                    f"### {window_kind}",
                    "",
                    "| window | candidate | test N2 baseline | test N2 candidate | gain | macro RMSE regression | max other drop | test pass | extrap pass |",
                    "|---|---|---:|---:|---:|---:|---:|---|---|",
                ]
            )
            for item in window_rows:
                test = item["test"]
                max_other_drop = max(test["other_component_r2_drops"].values())
                lines.append(
                    f"| {item['window_value']} | {item['candidate_run']} | {test['baseline_n2_r2']:.6f} | "
                    f"{test['candidate_n2_r2']:.6f} | {test['n2_r2_gain']:.6f} | "
                    f"{test['macro_rmse_regression']:.6f} | {max_other_drop:.6f} | "
                    f"{'yes' if test['passed_overall'] else 'no'} | "
                    f"{'yes' if item['passed_strong_extrapolation'] else 'no'} |"
                )
            lines.append("")

    bin_rows = [
        (item, group_name, bin_name, bin_payload)
        for item in payload["comparisons"]
        for group_name, group in item["conditional_bins"].items()
        for bin_name, bin_payload in group.items()
    ]
    if bin_rows:
        lines.extend(
            [
                "## Conditional Bins",
                "",
                "| kind | window | group | bin | count | range | N2 R2 gain | RMSE regression | max other R2 drop | pass |",
                "|---|---|---|---|---:|---|---:|---:|---:|---|",
            ]
        )
        for item, group_name, bin_name, bin_payload in bin_rows:
            range_text = f"{bin_payload['range'][0]:.6g}-{bin_payload['range'][1]:.6g}"
            max_other_drop = max(bin_payload["other_component_r2_drops"].values())
            lines.append(
                f"| {item['kind']} | {item['window_kind']}:{item['window_value']} | {group_name} | {bin_name} | "
                f"{bin_payload['count']} | {range_text} | {bin_payload['n2_r2_gain']:.6f} | "
                f"{bin_payload['macro_rmse_regression']:.6f} | {max_other_drop:.6f} | "
                f"{'yes' if bin_payload['passed_overall'] else 'no'} |"
            )
    return "\n".join(lines).rstrip() + "\n"


def _load_metrics_payload(run_root: Path, run_name: str) -> dict[str, Any]:
    metrics_path = run_root / run_name / "metrics.json"
    if not metrics_path.is_file():
        raise FileNotFoundError(f"metrics.json not found for run {run_name!r}: {metrics_path}")
    return json.loads(metrics_path.read_text(encoding="utf-8"))


def _compare_payload_split(
    baseline_payload: dict[str, Any],
    candidate_payload: dict[str, Any],
    *,
    baseline_run: str,
    candidate_run: str,
    split: str,
    n2_min_gain: float,
    other_component_max_drop: float,
    macro_rmse_max_regression: float,
) -> dict[str, Any]:
    baseline_eval = _payload_split_eval(baseline_payload, run_name=baseline_run, split=split, window="full")
    candidate_eval = _payload_split_eval(candidate_payload, run_name=candidate_run, split=split, window="full")
    _ensure_finite_eval(baseline_eval, run_name=baseline_run, split=split)
    _ensure_finite_eval(candidate_eval, run_name=candidate_run, split=split)
    baseline_components = baseline_eval["component_metrics"]
    candidate_components = candidate_eval["component_metrics"]
    n2_gain = _component_r2(candidate_components, "x_N2") - _component_r2(baseline_components, "x_N2")
    other_drops = _other_component_r2_drops(baseline_components, candidate_components)
    macro_rmse_regression = float(candidate_eval["metrics"]["rmse"]) - float(baseline_eval["metrics"]["rmse"])
    return {
        "split": split,
        "baseline_n2_r2": _component_r2(baseline_components, "x_N2"),
        "candidate_n2_r2": _component_r2(candidate_components, "x_N2"),
        "n2_r2_gain": n2_gain,
        "baseline_rmse": float(baseline_eval["metrics"]["rmse"]),
        "candidate_rmse": float(candidate_eval["metrics"]["rmse"]),
        "macro_rmse_regression": macro_rmse_regression,
        "other_component_r2_drops": other_drops,
        "candidate_aitchison_mean": _optional_aitchison_mean(candidate_eval),
        "conditional_bins": _conditional_bin_comparisons(
            baseline_eval,
            candidate_eval,
            n2_min_gain=n2_min_gain,
            other_component_max_drop=other_component_max_drop,
            macro_rmse_max_regression=macro_rmse_max_regression,
        ),
        **_pass_flags(
            n2_gain=n2_gain,
            other_component_r2_drops=other_drops,
            macro_rmse_regression=macro_rmse_regression,
            n2_min_gain=n2_min_gain,
            other_component_max_drop=other_component_max_drop,
            macro_rmse_max_regression=macro_rmse_max_regression,
        ),
    }


def _payload_split_eval(payload: dict[str, Any], *, run_name: str, split: str, window: str) -> dict[str, Any]:
    if window == "full":
        evaluations = payload["full"]["evaluations"] if "full" in payload else payload["evaluations"]
    elif window.startswith("per_phase:"):
        phase = window.removeprefix("per_phase:")
        evaluations = payload["per_phase"][phase]["evaluations"]
    elif window.startswith("early:"):
        fraction = window.removeprefix("early:")
        evaluations = payload["early"][fraction]["evaluations"]
    else:
        raise ValueError(f"Unknown protocol window: {window!r}")
    if split not in evaluations:
        raise KeyError(f"split {split!r} not found in run {run_name!r}")
    return evaluations[split]


def _ensure_finite_eval(split_eval: dict[str, Any], *, run_name: str, split: str) -> None:
    values = {
        "metrics.rmse": split_eval["metrics"]["rmse"],
        "metrics.r2": split_eval["metrics"]["r2"],
    }
    for component, metrics in split_eval["component_metrics"].items():
        values[f"component_metrics.{component}.rmse"] = metrics["rmse"]
        values[f"component_metrics.{component}.r2"] = metrics["r2"]
    for name, value in values.items():
        if not math.isfinite(float(value)):
            raise ValueError(f"non-finite metric in run {run_name!r}, split {split!r}: {name}={value!r}")


def _component_r2(component_metrics: dict[str, Any], component: str) -> float:
    return float(component_metrics[component]["r2"])


def _other_component_r2_drops(
    baseline_components: dict[str, Any],
    candidate_components: dict[str, Any],
) -> dict[str, float]:
    return {
        component: _component_r2(baseline_components, component) - _component_r2(candidate_components, component)
        for component in COMPONENT_FIELDS
        if component != "x_N2"
    }


def _pass_flags(
    *,
    n2_gain: float,
    other_component_r2_drops: dict[str, float],
    macro_rmse_regression: float,
    n2_min_gain: float,
    other_component_max_drop: float,
    macro_rmse_max_regression: float,
) -> dict[str, bool]:
    passed_n2_gain = n2_gain >= n2_min_gain
    passed_other_components = all(drop <= other_component_max_drop for drop in other_component_r2_drops.values())
    passed_macro_rmse = macro_rmse_regression <= macro_rmse_max_regression
    return {
        "passed_n2_gain": passed_n2_gain,
        "passed_other_components": passed_other_components,
        "passed_macro_rmse": passed_macro_rmse,
        "passed_overall": passed_n2_gain and passed_other_components and passed_macro_rmse,
    }


def _optional_aitchison_mean(split_eval: dict[str, Any]) -> float | None:
    metrics = split_eval.get("compositional_metrics")
    if metrics is None:
        return None
    return float(metrics["aitchison_mean"])


def _conditional_bin_comparisons(
    baseline_eval: dict[str, Any],
    candidate_eval: dict[str, Any],
    *,
    n2_min_gain: float,
    other_component_max_drop: float,
    macro_rmse_max_regression: float,
) -> dict[str, dict[str, Any]]:
    baseline_groups = baseline_eval.get("conditional_metrics")
    candidate_groups = candidate_eval.get("conditional_metrics")
    if not isinstance(baseline_groups, dict) or not isinstance(candidate_groups, dict):
        return {}

    comparisons: dict[str, dict[str, Any]] = {}
    for group_name in ("n2_bins", "ch4_bins"):
        baseline_bins = _conditional_bins(baseline_groups, group_name)
        candidate_bins = _conditional_bins(candidate_groups, group_name)
        shared_bins = sorted(set(baseline_bins) & set(candidate_bins))
        group_comparisons: dict[str, Any] = {}
        for bin_name in shared_bins:
            baseline_bin = baseline_bins[bin_name]
            candidate_bin = candidate_bins[bin_name]
            if "metrics" not in baseline_bin or "metrics" not in candidate_bin:
                continue
            baseline_components = baseline_bin["component_metrics"]
            candidate_components = candidate_bin["component_metrics"]
            n2_gain = _component_r2(candidate_components, "x_N2") - _component_r2(baseline_components, "x_N2")
            other_drops = _other_component_r2_drops(baseline_components, candidate_components)
            macro_rmse_regression = float(candidate_bin["metrics"]["rmse"]) - float(baseline_bin["metrics"]["rmse"])
            group_comparisons[bin_name] = {
                "range": candidate_bin.get("range", baseline_bin.get("range")),
                "count": int(candidate_bin.get("count", baseline_bin.get("count", 0))),
                "baseline_count": int(baseline_bin.get("count", 0)),
                "candidate_count": int(candidate_bin.get("count", 0)),
                "baseline_n2_r2": _component_r2(baseline_components, "x_N2"),
                "candidate_n2_r2": _component_r2(candidate_components, "x_N2"),
                "n2_r2_gain": n2_gain,
                "baseline_rmse": float(baseline_bin["metrics"]["rmse"]),
                "candidate_rmse": float(candidate_bin["metrics"]["rmse"]),
                "macro_rmse_regression": macro_rmse_regression,
                "other_component_r2_drops": other_drops,
                **_pass_flags(
                    n2_gain=n2_gain,
                    other_component_r2_drops=other_drops,
                    macro_rmse_regression=macro_rmse_regression,
                    n2_min_gain=n2_min_gain,
                    other_component_max_drop=other_component_max_drop,
                    macro_rmse_max_regression=macro_rmse_max_regression,
                ),
            }
        if group_comparisons:
            comparisons[group_name] = group_comparisons
    return comparisons


def _conditional_bins(groups: dict[str, Any], group_name: str) -> dict[str, Any]:
    group = groups.get(group_name)
    if not isinstance(group, dict):
        return {}
    bins = group.get("bins", {})
    return bins if isinstance(bins, dict) else {}


def _protocol_window_comparisons(
    baseline_payload: dict[str, Any],
    candidate_payload: dict[str, Any],
    *,
    baseline_run: str,
    candidate_run: str,
    split: str,
    n2_min_gain: float,
    other_component_max_drop: float,
    macro_rmse_max_regression: float,
) -> dict[str, dict[str, Any]]:
    if "full" not in baseline_payload or "full" not in candidate_payload:
        return {"per_phase": {}, "early": {}}
    return {
        "per_phase": _window_group_comparisons(
            baseline_payload,
            candidate_payload,
            baseline_run=baseline_run,
            candidate_run=candidate_run,
            split=split,
            group="per_phase",
            n2_min_gain=n2_min_gain,
            other_component_max_drop=other_component_max_drop,
            macro_rmse_max_regression=macro_rmse_max_regression,
        ),
        "early": _window_group_comparisons(
            baseline_payload,
            candidate_payload,
            baseline_run=baseline_run,
            candidate_run=candidate_run,
            split=split,
            group="early",
            n2_min_gain=n2_min_gain,
            other_component_max_drop=other_component_max_drop,
            macro_rmse_max_regression=macro_rmse_max_regression,
        ),
    }


def _window_group_comparisons(
    baseline_payload: dict[str, Any],
    candidate_payload: dict[str, Any],
    *,
    baseline_run: str,
    candidate_run: str,
    split: str,
    group: str,
    n2_min_gain: float,
    other_component_max_drop: float,
    macro_rmse_max_regression: float,
) -> dict[str, Any]:
    baseline_group = baseline_payload.get(group, {})
    candidate_group = candidate_payload.get(group, {})
    if not isinstance(baseline_group, dict) or not isinstance(candidate_group, dict):
        return {}
    shared_windows = sorted(set(baseline_group) & set(candidate_group))
    comparisons = {}
    for window_name in shared_windows:
        window_key = f"{group}:{window_name}"
        baseline_eval = _payload_split_eval(baseline_payload, run_name=baseline_run, split=split, window=window_key)
        candidate_eval = _payload_split_eval(candidate_payload, run_name=candidate_run, split=split, window=window_key)
        baseline_components = baseline_eval["component_metrics"]
        candidate_components = candidate_eval["component_metrics"]
        n2_gain = _component_r2(candidate_components, "x_N2") - _component_r2(baseline_components, "x_N2")
        other_drops = _other_component_r2_drops(baseline_components, candidate_components)
        macro_rmse_regression = float(candidate_eval["metrics"]["rmse"]) - float(baseline_eval["metrics"]["rmse"])
        comparisons[window_name] = {
            "baseline_n2_r2": _component_r2(baseline_components, "x_N2"),
            "candidate_n2_r2": _component_r2(candidate_components, "x_N2"),
            "n2_r2_gain": n2_gain,
            "baseline_rmse": float(baseline_eval["metrics"]["rmse"]),
            "candidate_rmse": float(candidate_eval["metrics"]["rmse"]),
            "macro_rmse_regression": macro_rmse_regression,
            "other_component_r2_drops": other_drops,
            "candidate_aitchison_mean": _optional_aitchison_mean(candidate_eval),
            "conditional_bins": _conditional_bin_comparisons(
                baseline_eval,
                candidate_eval,
                n2_min_gain=n2_min_gain,
                other_component_max_drop=other_component_max_drop,
                macro_rmse_max_regression=macro_rmse_max_regression,
            ),
            **_pass_flags(
                n2_gain=n2_gain,
                other_component_r2_drops=other_drops,
                macro_rmse_regression=macro_rmse_regression,
                n2_min_gain=n2_min_gain,
                other_component_max_drop=other_component_max_drop,
                macro_rmse_max_regression=macro_rmse_max_regression,
            ),
        }
    return comparisons


def _phase_aware_groups(results: list[dict[str, Any]]) -> dict[str, dict[str, list[dict[str, Any]]]]:
    groups: dict[str, dict[str, list[dict[str, Any]]]] = {}
    for item in results:
        groups.setdefault(item["kind"], {}).setdefault(item["window_kind"], []).append(item)
    return groups


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    other_component_max_drop = args.other_component_max_drop
    if other_component_max_drop is None:
        other_component_max_drop = 0.05 if args.phase_aware else 0.02
    if args.phase_aware:
        payload = analyze_phase_aware_n2(
            args.run_root,
            split=args.split,
            extrapolation_split=args.extrapolation_split,
            n2_min_gain=args.n2_min_gain,
            other_component_max_drop=other_component_max_drop,
            macro_rmse_max_regression=args.macro_rmse_max_regression,
        )
    else:
        payload = analyze_n2_improvement(
            args.run_root,
            split=args.split,
            n2_min_gain=args.n2_min_gain,
            other_component_max_drop=other_component_max_drop,
            macro_rmse_max_regression=args.macro_rmse_max_regression,
        )
    if args.json:
        output = json.dumps(payload, ensure_ascii=False, indent=2) + "\n"
    else:
        output = format_phase_aware_markdown_report(payload) if args.phase_aware else format_markdown_report(payload)
    if args.output_path is not None:
        args.output_path.parent.mkdir(parents=True, exist_ok=True)
        args.output_path.write_text(output, encoding="utf-8")
    if args.json_output_path is not None:
        args.json_output_path.parent.mkdir(parents=True, exist_ok=True)
        args.json_output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(output, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
