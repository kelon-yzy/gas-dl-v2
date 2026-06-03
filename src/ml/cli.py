from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from ml.evaluation_protocol import BaselineProtocolResult, run_baseline_protocol
from ml.features import MLFeatureConfig
from ml.models import MeanRegressor, RidgeRegressor
from ml.training import train_regressor_on_dataset

MODALITY_CHOICES = ("slow", "ultrasonic", "fiber_mic")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train a traditional ML regressor on a v4 benchmark dataset.",
    )
    parser.add_argument("--dataset-dir", type=Path, required=True, help="v4 benchmark dataset root directory.")
    parser.add_argument(
        "--model",
        choices=["ridge", "mean"],
        default="ridge",
        help="Regressor type (default: ridge).",
    )
    parser.add_argument("--alpha", type=float, default=1.0, help="Ridge alpha (default: 1.0).")
    parser.add_argument(
        "--modalities",
        type=str,
        default="slow",
        help="Comma-separated modalities: slow,ultrasonic,fiber_mic (default: slow).",
    )
    parser.add_argument("--sequence-statistics", type=str, default="mean,std,min,max,last,delta,slope")
    parser.add_argument(
        "--waveform-frame-features",
        type=str,
        default="mean,std,mean_abs,max_abs,energy,peak_index",
    )
    parser.add_argument("--scaler-path", type=Path, default=None, help="Path to z-score scaler JSON for slow channels.")
    parser.add_argument(
        "--protocol",
        action="store_true",
        default=False,
        help="Run the full/per-phase/early baseline protocol instead of a single full-window baseline.",
    )
    parser.add_argument(
        "--phases",
        type=str,
        default="baseline,exposure,steady,recovery",
        help="Comma-separated phase windows for --protocol.",
    )
    parser.add_argument(
        "--early-fractions",
        type=str,
        default="0.25,0.5,0.75,1.0",
        help="Comma-separated early fractions for --protocol.",
    )
    parser.add_argument(
        "--report-path",
        type=Path,
        default=None,
        help="Write a Markdown protocol report to this path.",
    )
    parser.add_argument("--json", action="store_true", default=False, help="Output results as JSON.")
    return parser


def _parse_comma(value: str) -> tuple[str, ...]:
    return tuple(s.strip() for s in value.split(",") if s.strip())


def _parse_float_comma(value: str) -> tuple[float, ...]:
    values = tuple(float(s) for s in _parse_comma(value))
    for fraction in values:
        if not 0.0 < fraction <= 1.0:
            raise ValueError(f"early fractions must be in (0, 1], got {fraction}")
    return values


def run(args: argparse.Namespace) -> None:
    dataset_dir = args.dataset_dir
    if not dataset_dir.is_dir() or not (dataset_dir / "labels" / "y.npy").is_file():
        parser = build_parser()
        parser.error(f"dataset-dir must be a v4 benchmark root: {dataset_dir}")

    modalities = _parse_comma(args.modalities)
    unknown = set(modalities) - set(MODALITY_CHOICES)
    if unknown:
        parser = build_parser()
        parser.error(f"Unknown modalities: {sorted(unknown)}. Available: {MODALITY_CHOICES}")

    feature_config = MLFeatureConfig(
        modalities=modalities,
        sequence_statistics=_parse_comma(args.sequence_statistics),
        waveform_frame_features=_parse_comma(args.waveform_frame_features),
        slow_scaler_path=args.scaler_path,
    )

    model_config: str | dict[str, Any]
    if args.model == "mean":
        model_config = "mean"
    else:
        model_config = {"name": "ridge", "alpha": args.alpha}

    if args.protocol:
        try:
            early_fractions = _parse_float_comma(args.early_fractions)
        except ValueError as exc:
            parser = build_parser()
            parser.error(str(exc))

        result = run_baseline_protocol(
            dataset_dir,
            model_config=model_config,
            feature_config=feature_config,
            phases=_parse_comma(args.phases),
            early_fractions=early_fractions,
        )
        if args.report_path is not None:
            args.report_path.parent.mkdir(parents=True, exist_ok=True)
            args.report_path.write_text(_protocol_markdown(result), encoding="utf-8")
        if args.json:
            print(json.dumps(_protocol_payload(result), indent=2))
        elif args.report_path is None:
            print(_protocol_markdown(result))
        else:
            print(f"wrote protocol report: {args.report_path}")
        return

    result = train_regressor_on_dataset(dataset_dir, model_config=model_config, feature_config=feature_config)

    if args.json:
        _print_json(result)
    else:
        _print_table(result)


def _print_json(result: object) -> None:
    print(json.dumps(_training_payload(result), indent=2))


def _training_payload(result: object) -> dict[str, Any]:
    from dataclasses import asdict

    from ml.training import MLTrainingResult

    assert isinstance(result, MLTrainingResult)
    payload: dict[str, Any] = {
        "feature_config": {
            "modalities": result.feature_config.modalities,
            "sequence_statistics": result.feature_config.sequence_statistics,
            "waveform_frame_features": result.feature_config.waveform_frame_features,
        },
        "feature_names": result.feature_names,
        "label_names": result.label_names,
        "train_split": result.train_split,
        "evaluations": {},
    }
    for split_name, split_eval in result.evaluations.items():
        payload["evaluations"][split_name] = {
            "metrics": asdict(split_eval.metrics),
            "component_metrics": {k: asdict(v) for k, v in split_eval.component_metrics.items()},
        }
    return payload


def _protocol_payload(result: BaselineProtocolResult) -> dict[str, Any]:
    return {
        "full": _training_payload(result.full),
        "per_phase": {phase: _training_payload(value) for phase, value in result.per_phase.items()},
        "early": {str(fraction): _training_payload(value) for fraction, value in result.early.items()},
    }


def _protocol_markdown(result: BaselineProtocolResult) -> str:
    lines = [
        "# Baseline Evaluation Protocol",
        "",
        "## Full Window",
        "",
        _metrics_markdown_table(result.full),
        "",
        "## Per Phase",
        "",
    ]
    for phase, phase_result in result.per_phase.items():
        lines.extend([f"### {phase}", "", _metrics_markdown_table(phase_result), ""])
    lines.extend(["## Early Windows", ""])
    for fraction, early_result in result.early.items():
        lines.extend([f"### first {fraction:g}", "", _metrics_markdown_table(early_result), ""])
    return "\n".join(lines).rstrip() + "\n"


def _metrics_markdown_table(result: object) -> str:
    from ml.training import MLTrainingResult

    assert isinstance(result, MLTrainingResult)
    lines = [
        "| split | MAE | RMSE | R2 |",
        "|---|---:|---:|---:|",
    ]
    for split_name, split_eval in result.evaluations.items():
        m = split_eval.metrics
        lines.append(f"| {split_name} | {m.mae:.6f} | {m.rmse:.6f} | {m.r2:.6f} |")
    return "\n".join(lines)


def _print_table(result: object) -> None:
    from ml.training import MLTrainingResult

    assert isinstance(result, MLTrainingResult)

    print(f"model          {'ridge' if isinstance(result.model, RidgeRegressor) else 'mean'}")
    print(f"modalities     {', '.join(result.feature_config.modalities)}")
    print(f"features       {len(result.feature_names)}")
    print(f"train split    {result.train_split} ({len(result.evaluations)} evaluated)")
    print("-" * 72)
    header = f"{'split':>14s} {'MAE':>8s} {'RMSE':>8s} {'R2':>8s}"
    for i, comp in enumerate(result.label_names):
        header += f" {comp:>8s}"
    print(header)
    print("-" * 72)
    for split_name, split_eval in result.evaluations.items():
        m = split_eval.metrics
        comp_str = " ".join(f"{split_eval.component_metrics[c].mae:8.4f}" for c in result.label_names)
        print(f"{split_name:>14s} {m.mae:8.4f} {m.rmse:8.4f} {m.r2:8.4f} {comp_str}")
    print("-" * 72)


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
