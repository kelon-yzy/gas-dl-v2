from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any
from xml.sax.saxutils import escape

import numpy as np

from gf.ml.baselines import BaselineSuiteResult, run_baseline_suite
from gf.sim.a1_audit import run_information_audit
from gf.sim.a1_dataset import (
    A1PhysicsConfig,
    A1Dataset,
    TARGET_NAMES,
    generate_dataset,
)


EXPECTED_COUNTS = {
    "pilot": {"binary_per_pair": 20, "ternary_count": 180, "sample_count": 240},
    "formal": {"binary_per_pair": 100, "ternary_count": 900, "sample_count": 1200},
}


def run_a1(
    *,
    project_root: str | Path,
    mode: str = "all",
    pilot_config_path: str | Path | None = None,
    formal_config_path: str | Path | None = None,
    eval_config_path: str | Path | None = None,
) -> dict[str, Any]:
    root = Path(project_root).resolve()
    if mode not in {"pilot", "formal", "all"}:
        raise ValueError(f"mode must be pilot, formal, or all, got {mode!r}")
    config_dir = root / "configs"
    pilot_config = _read_json(
        Path(pilot_config_path)
        if pilot_config_path is not None
        else config_dir / "data" / "ar_he_co2_a1_pilot.json"
    )
    formal_config = _read_json(
        Path(formal_config_path)
        if formal_config_path is not None
        else config_dir / "data" / "ar_he_co2_a1_v1.json"
    )
    eval_config = _read_json(
        Path(eval_config_path)
        if eval_config_path is not None
        else config_dir / "eval" / "a1_baselines.json"
    )

    pilot_result: dict[str, Any] | None = None
    if mode in {"pilot", "all"}:
        pilot_result = _run_stage(
            stage="pilot",
            config=pilot_config,
            eval_config=eval_config,
            root=root,
            include_mlp=False,
            bootstrap_samples=0,
        )
    if mode == "pilot":
        return {"pilot": pilot_result}
    if mode == "formal" and pilot_result is None:
        pilot_result = _load_pilot_gate(root)
    if pilot_result["stage_gate"]["status"] != "PASS":
        raise RuntimeError(
            "A1 pilot gate failed; formal v1 generation is blocked. "
            f"See {root / 'outputs' / 'reports' / 'a1_pilot' / 'stage_gate.json'}"
        )

    formal_result = _run_stage(
        stage="formal",
        config=formal_config,
        eval_config=eval_config,
        root=root,
        include_mlp=True,
        bootstrap_samples=int(eval_config["bootstrap_samples"]),
    )
    return {"pilot": pilot_result, "formal": formal_result}


def _run_stage(
    *,
    stage: str,
    config: dict[str, Any],
    eval_config: dict[str, Any],
    root: Path,
    include_mlp: bool,
    bootstrap_samples: int,
) -> dict[str, Any]:
    _validate_data_config(stage, config)
    physics = A1PhysicsConfig.from_mapping(config["physics"])
    data_dir = root / "data" / f"a1_{stage}"
    summary_dir = root / "outputs" / "summary" / f"a1_{stage}"
    report_dir = root / "outputs" / "reports" / f"a1_{stage}"
    data_dir.mkdir(parents=True, exist_ok=True)
    summary_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)

    dataset = generate_dataset(
        data_dir,
        binary_per_pair=int(config["binary_per_pair"]),
        ternary_count=int(config["ternary_count"]),
        generation_seed=int(config["generation_seed"]),
        split_seed=int(config["split_seed"]),
        data_version=str(config["data_version"]),
        physics=physics,
    )
    _write_json(data_dir / "config_snapshot.json", config)
    audit = run_information_audit(dataset, physics)
    baseline = run_baseline_suite(
        dataset,
        training_seed=int(eval_config["deterministic_training_seed"]),
        include_mlp=include_mlp,
        mlp_seeds=tuple(int(seed) for seed in eval_config["formal_training_seeds"]),
        bootstrap_seed=int(eval_config["bootstrap_seed"]),
        bootstrap_samples=bootstrap_samples,
    )
    manifest = json.loads((data_dir / "manifest.json").read_text(encoding="utf-8"))
    stage_gate = {
        "status": (
            "PASS"
            if audit["gate"]["status"] == "PASS" and baseline.summary["gate"]["status"] == "PASS"
            else "FAIL"
        ),
        "audit_status": audit["gate"]["status"],
        "baseline_status": baseline.summary["gate"]["status"],
        "data_version": config["data_version"],
        "content_sha256": manifest["content_sha256"],
        "sample_count": len(dataset.conditions),
    }
    _write_json(summary_dir / "information_audit.json", audit)
    _write_json(summary_dir / "baseline_summary.json", dict(baseline.summary))
    _write_json(summary_dir / "stage_gate.json", stage_gate)
    _write_predictions_csv(summary_dir / "predictions.csv", dataset, baseline)
    _write_simplex_error_svg(summary_dir / "simplex_error_map.svg", dataset, baseline)
    _write_json(report_dir / "stage_gate.json", stage_gate)
    _write_stage_report(
        report_dir / "A1评审报告.md",
        stage=stage,
        dataset=dataset,
        manifest=manifest,
        audit=audit,
        baseline=baseline.summary,
        stage_gate=stage_gate,
    )
    return {
        "stage": stage,
        "data_dir": str(data_dir),
        "summary_dir": str(summary_dir),
        "report_dir": str(report_dir),
        "stage_gate": stage_gate,
        "content_sha256": manifest["content_sha256"],
        "audit": audit,
        "baseline": dict(baseline.summary),
    }


def _validate_data_config(stage: str, config: dict[str, Any]) -> None:
    if config.get("schema_version") != "gf-a1-data-1":
        raise ValueError(f"{stage} config has unsupported schema_version")
    if config.get("dataset_id") != "ar_he_co2":
        raise ValueError(f"{stage} config has unsupported dataset_id")
    expected = EXPECTED_COUNTS[stage]
    actual = {
        "binary_per_pair": int(config["binary_per_pair"]),
        "ternary_count": int(config["ternary_count"]),
        "sample_count": 3 * int(config["binary_per_pair"]) + int(config["ternary_count"]),
    }
    if actual != expected:
        raise ValueError(f"{stage} config counts must be {expected}, got {actual}")
    if not config.get("data_version"):
        raise ValueError(f"{stage} config data_version must be non-empty")


def _load_pilot_gate(root: Path) -> dict[str, Any]:
    gate_path = root / "outputs" / "reports" / "a1_pilot" / "stage_gate.json"
    if not gate_path.exists():
        raise FileNotFoundError(
            f"formal mode requires a completed pilot gate at {gate_path}"
        )
    return {
        "stage": "pilot",
        "stage_gate": json.loads(gate_path.read_text(encoding="utf-8")),
    }


def _write_predictions_csv(path: Path, dataset: A1Dataset, baseline: BaselineSuiteResult) -> None:
    keys = sorted(baseline.predictions)
    fieldnames = [
        "mixture_id",
        "split",
        *TARGET_NAMES,
        *[f"{key}__{target}" for key in keys for target in TARGET_NAMES],
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for index, condition in enumerate(dataset.conditions):
            row: dict[str, Any] = {
                "mixture_id": condition.mixture_id,
                "split": condition.split,
                **dict(zip(TARGET_NAMES, condition.composition, strict=True)),
            }
            for key in keys:
                row.update(
                    {
                        f"{key}__{target}": float(baseline.predictions[key][index, target_index])
                        for target_index, target in enumerate(TARGET_NAMES)
                    }
                )
            writer.writerow(row)


def _write_simplex_error_svg(path: Path, dataset: A1Dataset, baseline: BaselineSuiteResult) -> None:
    best_single_key = str(baseline.summary["best_single"]["key"])
    best_full_key = str(baseline.summary["best_full_fusion"]["key"])
    test_indices = [
        index for index, condition in enumerate(dataset.conditions) if condition.split == "test"
    ]
    if not test_indices:
        raise ValueError("simplex error map requires a non-empty test split")
    errors = np.array(
        [
            np.abs(baseline.predictions[best_full_key][index] - baseline.targets[index]).mean()
            - np.abs(baseline.predictions[best_single_key][index] - baseline.targets[index]).mean()
            for index in test_indices
        ],
        dtype=np.float64,
    )
    width, height = 800, 460
    plot_left, plot_top, plot_width, plot_height = 70, 55, 660, 330
    shapes = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        '<text x="70" y="25" font-family="sans-serif" font-size="16">A1 test error: best fusion minus best single (mol%)</text>',
        f'<line x1="{plot_left}" y1="{plot_top + plot_height}" x2="{plot_left + plot_width}" y2="{plot_top + plot_height}" stroke="#333"/>',
        f'<line x1="{plot_left}" y1="{plot_top}" x2="{plot_left}" y2="{plot_top + plot_height}" stroke="#333"/>',
        f'<text x="{plot_left + plot_width / 2}" y="{plot_top + plot_height + 38}" text-anchor="middle" font-family="sans-serif">x_Ar (mol%)</text>',
        f'<text x="18" y="{plot_top + plot_height / 2}" transform="rotate(-90 18 {plot_top + plot_height / 2})" text-anchor="middle" font-family="sans-serif">x_He (mol%)</text>',
        '<text x="70" y="430" font-family="sans-serif" font-size="12" fill="#555">blue: fusion lower error; red: fusion higher error</text>',
    ]
    for condition_index, error in zip(test_indices, errors, strict=True):
        condition = dataset.conditions[condition_index]
        x = plot_left + plot_width * condition.x_ar_pct / 100.0
        y = plot_top + plot_height * (1.0 - condition.x_he_pct / 100.0)
        color = _error_color(error)
        shapes.append(
            f'<circle cx="{x:.2f}" cy="{y:.2f}" r="5" fill="{color}" fill-opacity="0.8">'
            f'<title>{escape(condition.mixture_id)} delta={error:.5f} mol%</title></circle>'
        )
    shapes.append("</svg>")
    path.write_text("\n".join(shapes) + "\n", encoding="utf-8")


def _error_color(error: float) -> str:
    scaled = float(np.clip(error / 10.0, -1.0, 1.0))
    if scaled >= 0.0:
        red = 220
        green = int(220 * (1.0 - scaled))
        blue = int(220 * (1.0 - scaled))
    else:
        red = int(220 * (1.0 + scaled))
        green = int(220 * (1.0 + scaled))
        blue = 220
    return f"rgb({red},{green},{blue})"


def _write_stage_report(
    path: Path,
    *,
    stage: str,
    dataset: A1Dataset,
    manifest: dict[str, Any],
    audit: dict[str, Any],
    baseline: dict[str, Any],
    stage_gate: dict[str, Any],
) -> None:
    counts = {
        split: sum(condition.split == split for condition in dataset.conditions)
        for split in ("train", "val", "test")
    }
    lines = [
        f"# A1 {stage} 评审报告",
        "",
        f"- stage gate: {stage_gate['status']}",
        f"- data_version: {manifest['data_version']}",
        f"- content_sha256: {manifest['content_sha256']}",
        f"- samples: {len(dataset.conditions)}；split: {counts}",
        f"- audit gate: {audit['gate']['status']}",
        f"- baseline gate: {baseline['gate']['status']}",
        "",
        "## 信息审计",
        "",
        f"- Jacobian full-rank fraction: {audit['jacobian']['full_rank_fraction']:.6f}",
        f"- Jacobian condition number P95: {audit['jacobian']['condition_number_p95']:.6f}",
        f"- maximum acoustic/TCS direction cosine: {audit['degeneration_directions']['pair_abs_cosine_max']['ultrasonic_tof__thermal_conductivity_voltage']:.6f}",
        f"- maximum NDIR direction cosine: {audit['degeneration_directions']['ndir_max_abs_cosine']:.6f}",
        "",
        "## 基线",
        "",
        f"- best single: {baseline['best_single']['key']}, val macro_RNMAE={baseline['best_single']['validation_macro_RNMAE']:.8f}",
        f"- best full fusion: {baseline['best_full_fusion']['key']}, val macro_RNMAE={baseline['best_full_fusion']['validation_macro_RNMAE']:.8f}",
        f"- best overall full-input baseline: {baseline['best_overall_full_input']['key']}, val macro_RNMAE={baseline['best_overall_full_input']['validation_macro_RNMAE']:.8f}",
        f"- relative improvement: {baseline['gate']['relative_improvement']:.6%}",
        "",
        "## 边界",
        "",
        "本阶段使用 HITRAN 参考光学系数和固定稳态仿真，不等同于目标 TraceGas 模块的实测标定；实际硬件滤光片、光程和 ADC 标定仍需在后续物理迁移阶段替换并复核。",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON config must be an object: {path}")
    return value


def _write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the general_fusion A1 benchmark.")
    parser.add_argument("--mode", choices=("pilot", "formal", "all"), default="all")
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[3])
    parser.add_argument("--pilot-config", type=Path)
    parser.add_argument("--formal-config", type=Path)
    parser.add_argument("--eval-config", type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    result = run_a1(
        project_root=args.project_root,
        mode=args.mode,
        pilot_config_path=args.pilot_config,
        formal_config_path=args.formal_config,
        eval_config_path=args.eval_config,
    )
    brief = {
        stage: {
            "stage_gate": payload["stage_gate"],
            "content_sha256": payload["content_sha256"],
        }
        for stage, payload in result.items()
    }
    print(json.dumps(brief, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
