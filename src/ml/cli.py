from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

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
    parser.add_argument("--json", action="store_true", default=False, help="Output results as JSON.")
    return parser


def _parse_comma(value: str) -> tuple[str, ...]:
    return tuple(s.strip() for s in value.split(",") if s.strip())


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

    result = train_regressor_on_dataset(dataset_dir, model_config=model_config, feature_config=feature_config)

    if args.json:
        _print_json(result)
    else:
        _print_table(result)


def _print_json(result: object) -> None:
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
    print(json.dumps(payload, indent=2))


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
