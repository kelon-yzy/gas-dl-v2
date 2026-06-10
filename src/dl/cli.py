from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from common.composition import (
    TARGET_TRANSFORM_OPTIONS,
    replace_zeros_multiplicative,
    resolve_target_transform_spec,
    resolve_target_transform_for_training,
)
from common.metrics import conditional_metrics_to_payload
from common.splits import load_splits, resolve_split_indices
from dl.data.dataset import MODALITY_OPTIONS, V4BenchmarkDataset
from dl.models.registry import MODEL_REGISTRY, build_model
from dl.training.losses import LOSS_REGISTRY, build_loss, validate_loss_target_transform
from dl.training.trainer import AmpConfig, EarlyStoppingConfig, OPTIMIZER_REGISTRY, Trainer, build_optimizer

DEFAULT_EVAL_SPLITS = ("val", "test", "extrapolation")
DEFAULT_DL_CONFIG: dict[str, Any] = {
    "model": "cnn1d",
    "model_kwargs": {},
    "modalities": "slow",
    "input_format": None,
    "scaler_path": None,
    "target_transform": None,
    "epochs": 50,
    "batch_size": 32,
    "num_workers": 0,
    "pin_memory": False,
    "persistent_workers": False,
    "prefetch_factor": None,
    "seed": 42,
    "device": "cpu",
    "loss": "mse",
    "optimizer": "adamw",
    "lr": 1e-3,
    "weight_decay": 0.0,
    "eval_splits": ",".join(DEFAULT_EVAL_SPLITS),
    "checkpoint_name": "checkpoint.pt",
    "early_stopping": {"enabled": False, "monitor": "val_loss", "patience": 20, "min_delta": 0.0, "mode": "min"},
    "scheduler": {"name": "none"},
    "amp": {"enabled": False, "dtype": "float16"},
    "progress": {"enabled": True, "stdout": True, "jsonl": True, "jsonl_name": "metrics_live.jsonl"},
    "json": False,
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train a DL regressor on a v4 benchmark dataset.")
    parser.add_argument("--config", type=Path, default=None, help="JSON config file. Explicit CLI args override it.")
    parser.add_argument("--dataset-dir", type=Path, default=None, help="v4 benchmark dataset root directory.")
    parser.add_argument("--output-dir", type=Path, default=None, help="Directory for checkpoint and run metrics.")
    parser.add_argument("--model", choices=sorted(MODEL_REGISTRY), default=None, help="Model registry name.")
    parser.add_argument(
        "--model-kwargs",
        type=str,
        default=None,
        help="JSON object merged into the inferred model config.",
    )
    parser.add_argument(
        "--modalities",
        type=str,
        default=None,
        help="Comma-separated modalities: slow,ultrasonic,fiber_mic (default: slow).",
    )
    parser.add_argument(
        "--input-format",
        choices=["NTC", "NCT"],
        default=None,
        help="Override model input format; by default the model class contract is used.",
    )
    parser.add_argument("--scaler-path", type=Path, default=None, help="Optional z-score scaler JSON for slow channels.")
    parser.add_argument(
        "--target-transform",
        choices=("none", *TARGET_TRANSFORM_OPTIONS),
        default=None,
        help="Optional compositional target transform for DL training.",
    )
    parser.add_argument("--epochs", type=int, default=None, help="Training epochs.")
    parser.add_argument("--batch-size", type=int, default=None, help="Batch size.")
    parser.add_argument("--num-workers", type=int, default=None, help="DataLoader worker count.")
    parser.add_argument("--seed", type=int, default=None, help="Torch and DataLoader shuffle seed.")
    parser.add_argument("--device", type=str, default=None, help="Torch device string, e.g. cpu or cuda.")
    parser.add_argument("--loss", choices=sorted(LOSS_REGISTRY), default=None, help="Loss function.")
    parser.add_argument("--optimizer", choices=sorted(OPTIMIZER_REGISTRY), default=None, help="Optimizer.")
    parser.add_argument("--lr", type=float, default=None, help="Optimizer learning rate.")
    parser.add_argument("--weight-decay", type=float, default=None, help="Optimizer weight decay.")
    parser.add_argument(
        "--eval-splits",
        type=str,
        default=None,
        help="Comma-separated splits evaluated after training.",
    )
    parser.add_argument("--checkpoint-name", type=str, default=None, help="Checkpoint filename.")
    parser.add_argument("--json", action="store_true", default=False, help="Print run metrics JSON to stdout.")
    return parser


def run(args: argparse.Namespace) -> dict[str, Any]:
    args = _resolve_args(args)
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
    target_transform = resolve_target_transform_for_training(
        args.target_transform,
        _split_labels(args.dataset_dir, "train"),
    )
    out_dim = 3 if target_transform is not None else int(sample_y.shape[-1])
    target_transform_audits = _target_transform_audits(
        args.dataset_dir,
        target_transform,
        splits=("train", *_parse_comma(args.eval_splits)),
    )

    model_config = _build_model_config(args.model, args.model_kwargs, in_channels, out_dim, timesteps)
    if target_transform is not None and int(model_config["out_dim"]) != 3:
        raise ValueError("DL target_transform requires model out_dim=3")
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
    trainer = Trainer(model=model, optimizer=optimizer, loss_fn=loss_fn, device=args.device, target_transform=target_transform)

    train_loader = _build_loader(
        train_dataset,
        args.batch_size,
        args.num_workers,
        shuffle=True,
        seed=args.seed,
        pin_memory=args.pin_memory,
        persistent_workers=args.persistent_workers,
        prefetch_factor=args.prefetch_factor,
    )
    val_loader = _optional_loader(
        args.dataset_dir,
        "val",
        modalities,
        input_format,
        args.scaler_path,
        args.batch_size,
        args.num_workers,
        args.pin_memory,
        args.persistent_workers,
        args.prefetch_factor,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = args.output_dir / args.checkpoint_name
    best_checkpoint_path = args.output_dir / "best_checkpoint.pt"
    scheduler = _build_scheduler(optimizer, args.scheduler)
    progress = _progress_config(args.progress)
    amp = _amp_config(args.amp)
    progress_log_path = args.output_dir / progress["jsonl_name"] if progress["enabled"] and progress["jsonl"] else None
    if progress_log_path is not None:
        progress_log_path.write_text("", encoding="utf-8")
    epoch_callback = _build_epoch_progress_callback(
        model_name=args.model,
        progress=progress,
        progress_log_path=progress_log_path,
        stdout_enabled=not args.json,
    )
    history = trainer.fit(
        train_loader,
        val_loader=val_loader,
        epochs=args.epochs,
        early_stopping=_early_stopping_config(args.early_stopping),
        scheduler=scheduler,
        best_checkpoint_path=best_checkpoint_path,
        epoch_callback=epoch_callback,
        amp=amp,
        non_blocking=bool(args.pin_memory and torch.device(args.device).type == "cuda"),
    )
    _write_training_progress_event(args.model, history, progress_log_path)
    if best_checkpoint_path.is_file():
        trainer.load_checkpoint(best_checkpoint_path)
        trainer.history = history

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
            args.pin_memory,
            args.persistent_workers,
            args.prefetch_factor,
        )
        if loader is not None:
            evaluations[split] = _evaluation_payload(
                trainer.evaluate(
                    loader,
                    non_blocking=bool(args.pin_memory and torch.device(args.device).type == "cuda"),
                    amp=amp,
                )
            )

    trainer.save_checkpoint(checkpoint_path)

    payload = {
        "dataset_dir": str(args.dataset_dir),
        "output_dir": str(args.output_dir),
        "checkpoint_path": str(checkpoint_path),
        "model_config": model_config,
        "input_format": input_format,
        "modalities": modalities,
        "target_transform": asdict(target_transform) if target_transform is not None else None,
        "target_transform_audits": (
            {split: asdict(audit) for split, audit in target_transform_audits.items()}
            if target_transform_audits is not None
            else None
        ),
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "pin_memory": args.pin_memory,
        "persistent_workers": args.persistent_workers,
        "prefetch_factor": args.prefetch_factor,
        "amp": args.amp,
        "loss": args.loss,
        "optimizer": {
            "name": args.optimizer,
            "lr": args.lr,
            "weight_decay": args.weight_decay,
        },
        "history": [_epoch_payload(epoch) for epoch in history.epochs],
        "best_epoch": _epoch_payload(history.best_epoch) if history.best_epoch is not None else None,
        "stopped_early": history.stopped_early,
        "stop_reason": history.stop_reason,
        "best_checkpoint_path": str(best_checkpoint_path) if best_checkpoint_path.is_file() else None,
        "learning_rates": [epoch.learning_rate for epoch in history.epochs],
        "progress_log_path": str(progress_log_path) if progress_log_path is not None else None,
        "evaluations": evaluations,
    }
    (args.output_dir / "metrics.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    (args.output_dir / "run_config.json").write_text(
        json.dumps(_run_config_payload(args, model_config, input_format, modalities, target_transform), indent=2),
        encoding="utf-8",
    )

    if args.json:
        print(json.dumps(payload, indent=2))
    else:
        _print_summary(payload)
    return payload


def _resolve_args(args: argparse.Namespace) -> argparse.Namespace:
    config = dict(DEFAULT_DL_CONFIG)
    if args.config is not None:
        config.update(_load_config(args.config))
    for key, value in vars(args).items():
        if key == "config":
            continue
        if value is not None and value is not False:
            config[key] = value
    config["dataset_dir"] = Path(config["dataset_dir"]) if config.get("dataset_dir") is not None else None
    config["output_dir"] = Path(config["output_dir"]) if config.get("output_dir") is not None else None
    if config.get("scaler_path") is not None:
        config["scaler_path"] = Path(config["scaler_path"])
    if config.get("prefetch_factor") is not None:
        config["prefetch_factor"] = int(config["prefetch_factor"])
    if isinstance(config.get("eval_splits"), list):
        config["eval_splits"] = ",".join(str(item) for item in config["eval_splits"])
    if isinstance(config.get("modalities"), list):
        config["modalities"] = ",".join(str(item) for item in config["modalities"])
    return argparse.Namespace(**config)


def _load_config(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"DL config must be valid JSON: {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError("DL config must be a JSON object")
    allowed = set(DEFAULT_DL_CONFIG) | {"dataset_dir", "output_dir"}
    unknown = set(payload) - allowed
    if unknown:
        raise ValueError(f"Unknown DL config keys: {sorted(unknown)}")
    return payload


def _validate_run_args(args: argparse.Namespace) -> None:
    parser = build_parser()
    if args.dataset_dir is None:
        parser.error("dataset-dir is required")
    if args.output_dir is None:
        parser.error("output-dir is required")
    if not args.dataset_dir.is_dir() or not (args.dataset_dir / "labels" / "y.npy").is_file():
        parser.error(f"dataset-dir must be a v4 benchmark root: {args.dataset_dir}")
    if args.epochs < 1:
        parser.error(f"epochs must be >= 1, got {args.epochs}")
    if args.batch_size < 1:
        parser.error(f"batch-size must be >= 1, got {args.batch_size}")
    if args.num_workers < 0:
        parser.error(f"num-workers must be >= 0, got {args.num_workers}")
    if args.prefetch_factor is not None and args.prefetch_factor < 1:
        parser.error(f"prefetch-factor must be >= 1, got {args.prefetch_factor}")
    if args.lr <= 0.0:
        parser.error(f"lr must be > 0, got {args.lr}")
    if args.weight_decay < 0.0:
        parser.error(f"weight-decay must be >= 0, got {args.weight_decay}")
    if args.scheduler["name"] not in {"none", "reduce_on_plateau"}:
        parser.error("scheduler.name must be one of ['none', 'reduce_on_plateau']")
    try:
        target_transform = resolve_target_transform_spec(args.target_transform)
        validate_loss_target_transform(args.loss, None if target_transform is None else target_transform.name)
    except ValueError as exc:
        parser.error(str(exc))


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


def _target_transform_audits(
    dataset_dir: Path,
    target_transform: object | None,
    *,
    splits: tuple[str, ...],
):
    if target_transform is None:
        return None
    split_rows = load_splits(dataset_dir / "splits")
    sequence_ids = _load_str_array(dataset_dir / "metadata" / "sequence_ids.npy")
    label_names = tuple(_load_str_array(dataset_dir / "metadata" / "label_names.npy"))
    labels = np.load(dataset_dir / "labels" / "y.npy").astype(np.float32)
    split_indices = resolve_split_indices(split_rows, sequence_ids)

    audits = {}
    for split in dict.fromkeys(splits):
        _unused_values, audit = replace_zeros_multiplicative(
            labels[split_indices[split]],
            epsilon=target_transform.epsilon,
            component_names=label_names,
        )
        audits[split] = audit
    return audits


def _split_labels(dataset_dir: Path, split: str) -> np.ndarray:
    split_rows = load_splits(dataset_dir / "splits")
    sequence_ids = _load_str_array(dataset_dir / "metadata" / "sequence_ids.npy")
    labels = np.load(dataset_dir / "labels" / "y.npy").astype(np.float32)
    split_indices = resolve_split_indices(split_rows, sequence_ids)
    return labels[split_indices[split]]


def _load_str_array(path: Path) -> list[str]:
    values = np.load(path, allow_pickle=True)
    return [str(value) for value in values.tolist()]


def _build_model_config(
    model_name: str,
    model_kwargs_json: str | dict[str, Any],
    in_channels: int,
    out_dim: int,
    timesteps: int,
) -> dict[str, Any]:
    if isinstance(model_kwargs_json, str):
        try:
            model_kwargs = json.loads(model_kwargs_json)
        except json.JSONDecodeError as exc:
            raise ValueError(f"model-kwargs must be a JSON object: {exc}") from exc
    else:
        model_kwargs = dict(model_kwargs_json)
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


def _early_stopping_config(value: dict[str, Any]) -> EarlyStoppingConfig:
    return EarlyStoppingConfig(
        enabled=bool(value.get("enabled", False)),
        monitor=str(value.get("monitor", "val_loss")),
        patience=int(value.get("patience", 20)),
        min_delta=float(value.get("min_delta", 0.0)),
        mode=str(value.get("mode", "min")),
    )


def _amp_config(value: dict[str, Any]) -> AmpConfig:
    if not isinstance(value, dict):
        raise ValueError("amp must be a JSON object")
    enabled = bool(value.get("enabled", False))
    dtype = str(value.get("dtype", "float16"))
    if dtype not in {"float16", "bfloat16"}:
        raise ValueError("amp.dtype must be one of ['bfloat16', 'float16']")
    return AmpConfig(enabled=enabled, dtype=dtype)


def _progress_config(value: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError("progress must be a JSON object")
    config = dict(DEFAULT_DL_CONFIG["progress"])
    config.update(value)
    jsonl_name = str(config["jsonl_name"])
    if not jsonl_name or Path(jsonl_name).name != jsonl_name:
        raise ValueError("progress.jsonl_name must be a filename")
    return {
        "enabled": bool(config["enabled"]),
        "stdout": bool(config["stdout"]),
        "jsonl": bool(config["jsonl"]),
        "jsonl_name": jsonl_name,
    }


def _build_epoch_progress_callback(
    *,
    model_name: str,
    progress: dict[str, Any],
    progress_log_path: Path | None,
    stdout_enabled: bool,
):
    if not progress["enabled"]:
        return None

    def callback(epoch: Any, history: Any, epochs: int) -> None:
        best_epoch = history.best_epoch.epoch if history.best_epoch is not None else None
        event = {
            "event": "epoch_end",
            "model": model_name,
            "epoch": epoch.epoch,
            "epochs": epochs,
            "train_loss": epoch.train_loss,
            "val_loss": epoch.val_loss,
            "learning_rate": epoch.learning_rate,
            "epoch_seconds": epoch.epoch_seconds,
            "train_seconds": epoch.train_seconds,
            "val_seconds": epoch.val_seconds,
            "train_samples_per_second": epoch.train_samples_per_second,
            "gpu_memory_allocated_mb": epoch.gpu_memory_allocated_mb,
            "gpu_memory_reserved_mb": epoch.gpu_memory_reserved_mb,
            "best_epoch": best_epoch,
            "stopped_early": history.stopped_early,
            "stop_reason": history.stop_reason,
        }
        if progress["jsonl"] and progress_log_path is not None:
            _append_jsonl(progress_log_path, event)
        if progress["stdout"] and stdout_enabled:
            _print_epoch_progress(event)

    return callback


def _write_training_progress_event(model_name: str, history: Any, progress_log_path: Path | None) -> None:
    if progress_log_path is None:
        return
    event = {
        "event": "training_stopped" if history.stopped_early else "training_completed",
        "model": model_name,
        "epochs_ran": len(history.epochs),
        "best_epoch": history.best_epoch.epoch if history.best_epoch is not None else None,
        "stopped_early": history.stopped_early,
        "stop_reason": history.stop_reason,
    }
    _append_jsonl(progress_log_path, event)


def _append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False) + "\n")


def _print_epoch_progress(event: dict[str, Any]) -> None:
    val_loss = "none" if event["val_loss"] is None else f"{float(event['val_loss']):.6f}"
    seconds = "none" if event["epoch_seconds"] is None else f"{float(event['epoch_seconds']):.2f}"
    samples_per_second = (
        "none" if event["train_samples_per_second"] is None else f"{float(event['train_samples_per_second']):.1f}"
    )
    gpu_memory = (
        "none" if event["gpu_memory_allocated_mb"] is None else f"{float(event['gpu_memory_allocated_mb']):.0f}MB"
    )
    print(
        f"[epoch] model={event['model']} epoch={event['epoch']}/{event['epochs']} "
        f"train_loss={float(event['train_loss']):.6f} val_loss={val_loss} "
        f"lr={float(event['learning_rate']):.6g} best_epoch={event['best_epoch']} "
        f"sec={seconds} samples/s={samples_per_second} gpu_mem={gpu_memory}",
        flush=True,
    )


def _build_scheduler(optimizer: torch.optim.Optimizer, config: dict[str, Any]):
    name = str(config.get("name", "none"))
    if name == "none":
        return None
    if name == "reduce_on_plateau":
        return torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=float(config.get("factor", 0.5)),
            patience=int(config.get("patience", 8)),
            min_lr=float(config.get("min_lr", 1e-6)),
        )
    raise ValueError(f"Unknown scheduler: {name!r}")


def _build_loader(
    dataset: V4BenchmarkDataset,
    batch_size: int,
    num_workers: int,
    *,
    shuffle: bool,
    seed: int,
    pin_memory: bool,
    persistent_workers: bool,
    prefetch_factor: int | None,
) -> DataLoader:
    generator = torch.Generator()
    generator.manual_seed(seed)
    kwargs: dict[str, Any] = {
        "batch_size": batch_size,
        "shuffle": shuffle,
        "num_workers": num_workers,
        "generator": generator if shuffle else None,
        "pin_memory": pin_memory,
    }
    if num_workers > 0:
        kwargs["persistent_workers"] = persistent_workers
        if prefetch_factor is not None:
            kwargs["prefetch_factor"] = prefetch_factor
    return DataLoader(dataset, **kwargs)


def _optional_loader(
    dataset_dir: Path,
    split: str,
    modalities: tuple[str, ...],
    input_format: str,
    scaler_path: Path | None,
    batch_size: int,
    num_workers: int,
    pin_memory: bool,
    persistent_workers: bool,
    prefetch_factor: int | None,
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
    return _build_loader(
        dataset,
        batch_size,
        num_workers,
        shuffle=False,
        seed=0,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
        prefetch_factor=prefetch_factor,
    )


def _evaluation_payload(result: dict[str, Any]) -> dict[str, Any]:
    return {
        "loss": result["loss"],
        "metrics": asdict(result["metrics"]),
        "component_metrics": {key: asdict(value) for key, value in result["component_metrics"].items()},
        "compositional_metrics": (
            asdict(result["compositional_metrics"]) if result["compositional_metrics"] is not None else None
        ),
        "conditional_metrics": conditional_metrics_to_payload(result["conditional_metrics"]),
        "sum_abs_error": result["sum_abs_error"],
    }


def _epoch_payload(epoch: Any) -> dict[str, Any]:
    payload = {
        "epoch": epoch.epoch,
        "train_loss": epoch.train_loss,
        "learning_rate": epoch.learning_rate,
        "val_loss": epoch.val_loss,
        "epoch_seconds": epoch.epoch_seconds,
        "train_seconds": epoch.train_seconds,
        "val_seconds": epoch.val_seconds,
        "train_samples_per_second": epoch.train_samples_per_second,
        "gpu_memory_allocated_mb": epoch.gpu_memory_allocated_mb,
        "gpu_memory_reserved_mb": epoch.gpu_memory_reserved_mb,
        "train_metrics": asdict(epoch.train_metrics) if epoch.train_metrics is not None else None,
        "val_metrics": asdict(epoch.val_metrics) if epoch.val_metrics is not None else None,
        "val_component_metrics": (
            {key: asdict(value) for key, value in epoch.val_component_metrics.items()}
            if epoch.val_component_metrics is not None
            else None
        ),
        "val_compositional_metrics": (
            asdict(epoch.val_compositional_metrics) if epoch.val_compositional_metrics is not None else None
        ),
        "val_sum_abs_error": epoch.val_sum_abs_error,
    }
    return payload


def _run_config_payload(
    args: argparse.Namespace,
    model_config: dict[str, Any],
    input_format: str,
    modalities: tuple[str, ...],
    target_transform: object | None,
) -> dict[str, Any]:
    return {
        "dataset_dir": str(args.dataset_dir),
        "output_dir": str(args.output_dir),
        "model_config": model_config,
        "input_format": input_format,
        "modalities": modalities,
        "scaler_path": str(args.scaler_path) if args.scaler_path is not None else None,
        "target_transform": args.target_transform,
        "resolved_target_transform": asdict(target_transform) if target_transform is not None else None,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "pin_memory": args.pin_memory,
        "persistent_workers": args.persistent_workers,
        "prefetch_factor": args.prefetch_factor,
        "seed": args.seed,
        "device": args.device,
        "loss": args.loss,
        "optimizer": {
            "name": args.optimizer,
            "lr": args.lr,
            "weight_decay": args.weight_decay,
        },
        "early_stopping": args.early_stopping,
        "scheduler": args.scheduler,
        "amp": args.amp,
        "progress": args.progress,
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
