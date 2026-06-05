from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any, Sequence

import torch
from torch import nn
from torch.utils.data import DataLoader

from dl.data.dataset import MODALITY_OPTIONS, V4BenchmarkDataset
from dl.models.registry import MODEL_REGISTRY, build_model
from dl.training.losses import LOSS_REGISTRY, build_loss
from dl.training.trainer import OPTIMIZER_REGISTRY, Trainer, build_optimizer

DEFAULT_EVAL_SPLITS = ("val", "test", "extrapolation")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train a DL regressor on a v4 benchmark dataset.")
    parser.add_argument("--dataset-dir", type=Path, required=True, help="v4 benchmark dataset root directory.")
    parser.add_argument("--output-dir", type=Path, required=True, help="Directory for checkpoint and run metrics.")
    parser.add_argument("--model", choices=sorted(MODEL_REGISTRY), default="cnn1d", help="Model registry name.")
    parser.add_argument(
        "--model-kwargs",
        type=str,
        default="{}",
        help="JSON object merged into the inferred model config.",
    )
    parser.add_argument(
        "--modalities",
        type=str,
        default="slow",
        help="Comma-separated modalities: slow,ultrasonic,fiber_mic (default: slow).",
    )
    parser.add_argument(
        "--input-format",
        choices=["NTC", "NCT"],
        default=None,
        help="Override model input format; by default the model class contract is used.",
    )
    parser.add_argument("--scaler-path", type=Path, default=None, help="Optional z-score scaler JSON for slow channels.")
    parser.add_argument("--epochs", type=int, default=50, help="Training epochs.")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size.")
    parser.add_argument("--num-workers", type=int, default=0, help="DataLoader worker count.")
    parser.add_argument("--seed", type=int, default=42, help="Torch and DataLoader shuffle seed.")
    parser.add_argument("--device", type=str, default="cpu", help="Torch device string, e.g. cpu or cuda.")
    parser.add_argument("--loss", choices=sorted(LOSS_REGISTRY), default="mse", help="Loss function.")
    parser.add_argument("--optimizer", choices=sorted(OPTIMIZER_REGISTRY), default="adamw", help="Optimizer.")
    parser.add_argument("--lr", type=float, default=1e-3, help="Optimizer learning rate.")
    parser.add_argument("--weight-decay", type=float, default=0.0, help="Optimizer weight decay.")
    parser.add_argument(
        "--eval-splits",
        type=str,
        default=",".join(DEFAULT_EVAL_SPLITS),
        help="Comma-separated splits evaluated after training.",
    )
    parser.add_argument("--checkpoint-name", type=str, default="checkpoint.pt", help="Checkpoint filename.")
    parser.add_argument("--json", action="store_true", default=False, help="Print run metrics JSON to stdout.")
    return parser


def run(args: argparse.Namespace) -> dict[str, Any]:
    _validate_run_args(args)
    torch.manual_seed(args.seed)

    modalities = _parse_modalities(args.modalities)
    input_format = args.input_format or _model_input_format(args.model)
    train_dataset = V4BenchmarkDataset(
        args.dataset_dir,
        split="train",
        modalities=modalities,
        input_format=input_format,
        scaler_path=args.scaler_path,
        lazy=True,
    )
    sample_x, sample_y = train_dataset[0]
    in_channels, timesteps = _infer_input_shape(sample_x, input_format)

    model_config = _build_model_config(args.model, args.model_kwargs, in_channels, sample_y.shape[-1], timesteps)
    model = build_model(model_config)
    loss_fn = build_loss(args.loss)
    optimizer = build_optimizer(
        model,
        {
            "name": args.optimizer,
            "lr": args.lr,
            "weight_decay": args.weight_decay,
        },
    )
    trainer = Trainer(model=model, optimizer=optimizer, loss_fn=loss_fn, device=args.device)

    train_loader = _build_loader(train_dataset, args.batch_size, args.num_workers, shuffle=True, seed=args.seed)
    val_loader = _optional_loader(
        args.dataset_dir,
        "val",
        modalities,
        input_format,
        args.scaler_path,
        args.batch_size,
        args.num_workers,
    )
    history = trainer.fit(train_loader, val_loader=val_loader, epochs=args.epochs)

    evaluations: dict[str, Any] = {}
    for split in _parse_comma(args.eval_splits):
        loader = _optional_loader(
            args.dataset_dir,
            split,
            modalities,
            input_format,
            args.scaler_path,
            args.batch_size,
            args.num_workers,
        )
        if loader is not None:
            evaluations[split] = _evaluation_payload(trainer.evaluate(loader))

    args.output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = args.output_dir / args.checkpoint_name
    trainer.save_checkpoint(checkpoint_path)

    payload = {
        "dataset_dir": str(args.dataset_dir),
        "output_dir": str(args.output_dir),
        "checkpoint_path": str(checkpoint_path),
        "model_config": model_config,
        "input_format": input_format,
        "modalities": modalities,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "loss": args.loss,
        "optimizer": {
            "name": args.optimizer,
            "lr": args.lr,
            "weight_decay": args.weight_decay,
        },
        "history": [_epoch_payload(epoch) for epoch in history.epochs],
        "best_epoch": _epoch_payload(history.best_epoch) if history.best_epoch is not None else None,
        "evaluations": evaluations,
    }
    (args.output_dir / "metrics.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    (args.output_dir / "run_config.json").write_text(
        json.dumps(_run_config_payload(args, model_config, input_format, modalities), indent=2),
        encoding="utf-8",
    )

    if args.json:
        print(json.dumps(payload, indent=2))
    else:
        _print_summary(payload)
    return payload


def _validate_run_args(args: argparse.Namespace) -> None:
    parser = build_parser()
    if not args.dataset_dir.is_dir() or not (args.dataset_dir / "labels" / "y.npy").is_file():
        parser.error(f"dataset-dir must be a v4 benchmark root: {args.dataset_dir}")
    if args.epochs < 1:
        parser.error(f"epochs must be >= 1, got {args.epochs}")
    if args.batch_size < 1:
        parser.error(f"batch-size must be >= 1, got {args.batch_size}")
    if args.num_workers < 0:
        parser.error(f"num-workers must be >= 0, got {args.num_workers}")
    if args.lr <= 0.0:
        parser.error(f"lr must be > 0, got {args.lr}")
    if args.weight_decay < 0.0:
        parser.error(f"weight-decay must be >= 0, got {args.weight_decay}")


def _parse_modalities(value: str) -> tuple[str, ...]:
    modalities = _parse_comma(value)
    unknown = set(modalities) - set(MODALITY_OPTIONS)
    if unknown:
        parser = build_parser()
        parser.error(f"Unknown modalities: {sorted(unknown)}. Available: {MODALITY_OPTIONS}")
    if not modalities:
        parser = build_parser()
        parser.error("modalities must not be empty")
    return modalities


def _parse_comma(value: str) -> tuple[str, ...]:
    return tuple(s.strip() for s in value.split(",") if s.strip())


def _model_input_format(model_name: str) -> str:
    entry = MODEL_REGISTRY[model_name]
    input_format = getattr(entry, "input_format", None)
    if input_format not in {"NTC", "NCT"}:
        raise ValueError(f"Model {model_name!r} does not declare input_format; pass --input-format explicitly.")
    return str(input_format)


def _infer_input_shape(sample_x: torch.Tensor, input_format: str) -> tuple[int, int]:
    if sample_x.ndim != 2:
        raise ValueError(f"Expected one sample shaped (T, C) or (C, T), got {tuple(sample_x.shape)}")
    if input_format == "NTC":
        return int(sample_x.shape[1]), int(sample_x.shape[0])
    if input_format == "NCT":
        return int(sample_x.shape[0]), int(sample_x.shape[1])
    raise ValueError(f"input_format must be NTC or NCT, got {input_format!r}")


def _build_model_config(
    model_name: str,
    model_kwargs_json: str,
    in_channels: int,
    out_dim: int,
    timesteps: int,
) -> dict[str, Any]:
    try:
        model_kwargs = json.loads(model_kwargs_json)
    except json.JSONDecodeError as exc:
        raise ValueError(f"model-kwargs must be a JSON object: {exc}") from exc
    if not isinstance(model_kwargs, dict):
        raise ValueError("model-kwargs must be a JSON object")

    config: dict[str, Any] = {
        "name": model_name,
        "in_channels": in_channels,
        "out_dim": out_dim,
    }
    if model_name == "tcn" and "target_timesteps" not in model_kwargs and "channels" not in model_kwargs:
        config["target_timesteps"] = timesteps
    config.update(model_kwargs)
    return config


def _build_loader(
    dataset: V4BenchmarkDataset,
    batch_size: int,
    num_workers: int,
    *,
    shuffle: bool,
    seed: int,
) -> DataLoader:
    generator = torch.Generator()
    generator.manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        generator=generator if shuffle else None,
    )


def _optional_loader(
    dataset_dir: Path,
    split: str,
    modalities: tuple[str, ...],
    input_format: str,
    scaler_path: Path | None,
    batch_size: int,
    num_workers: int,
) -> DataLoader | None:
    dataset = V4BenchmarkDataset(
        dataset_dir,
        split=split,
        modalities=modalities,
        input_format=input_format,
        scaler_path=scaler_path,
        lazy=True,
    )
    if len(dataset) == 0:
        return None
    return _build_loader(dataset, batch_size, num_workers, shuffle=False, seed=0)


def _evaluation_payload(result: dict[str, Any]) -> dict[str, Any]:
    return {
        "loss": result["loss"],
        "metrics": asdict(result["metrics"]),
        "component_metrics": {key: asdict(value) for key, value in result["component_metrics"].items()},
    }


def _epoch_payload(epoch: Any) -> dict[str, Any]:
    payload = {
        "epoch": epoch.epoch,
        "train_loss": epoch.train_loss,
        "val_loss": epoch.val_loss,
        "train_metrics": asdict(epoch.train_metrics) if epoch.train_metrics is not None else None,
        "val_metrics": asdict(epoch.val_metrics) if epoch.val_metrics is not None else None,
        "val_component_metrics": (
            {key: asdict(value) for key, value in epoch.val_component_metrics.items()}
            if epoch.val_component_metrics is not None
            else None
        ),
    }
    return payload


def _run_config_payload(
    args: argparse.Namespace,
    model_config: dict[str, Any],
    input_format: str,
    modalities: tuple[str, ...],
) -> dict[str, Any]:
    return {
        "dataset_dir": str(args.dataset_dir),
        "output_dir": str(args.output_dir),
        "model_config": model_config,
        "input_format": input_format,
        "modalities": modalities,
        "scaler_path": str(args.scaler_path) if args.scaler_path is not None else None,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "seed": args.seed,
        "device": args.device,
        "loss": args.loss,
        "optimizer": {
            "name": args.optimizer,
            "lr": args.lr,
            "weight_decay": args.weight_decay,
        },
        "eval_splits": _parse_comma(args.eval_splits),
    }


def _print_summary(payload: dict[str, Any]) -> None:
    print(f"model          {payload['model_config']['name']}")
    print(f"modalities     {', '.join(payload['modalities'])}")
    print(f"input format   {payload['input_format']}")
    print(f"checkpoint     {payload['checkpoint_path']}")
    print("-" * 72)
    print(f"{'split':>14s} {'loss':>10s} {'MAE':>10s} {'RMSE':>10s} {'R2':>10s}")
    print("-" * 72)
    for split, result in payload["evaluations"].items():
        metrics = result["metrics"]
        print(
            f"{split:>14s} {result['loss']:10.6f} "
            f"{metrics['mae']:10.6f} {metrics['rmse']:10.6f} {metrics['r2']:10.6f}"
        )
    print("-" * 72)


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    run(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
