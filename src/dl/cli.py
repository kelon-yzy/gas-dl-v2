from __future__ import annotations

import argparse
import json
import random
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
from common.windows import resolve_window_config, window_to_payload
from common.splits import load_splits, resolve_split_indices
from dl.data.augmentation import TimeSeriesAugmentConfig
from dl.data.dataset import MODALITY_OPTIONS, V4BenchmarkDataset
from dl.data.feature_dataset import V4FeatureMatrixDataset
from dl.models.registry import MODEL_REGISTRY, build_model
from dl.training.losses import (
    LOSS_REGISTRY,
    build_loss,
    validate_loss_composition_scheme,
    validate_loss_model_output,
    validate_loss_target_transform,
)
from dl.training.trainer import AmpConfig, EarlyStoppingConfig, OPTIMIZER_REGISTRY, Trainer, build_optimizer

DEFAULT_EVAL_SPLITS = ("val", "test", "extrapolation")
DEFAULT_DL_CONFIG: dict[str, Any] = {
    "model": "cnn1d",
    "model_kwargs": {},
    "modalities": "slow",
    "slow_channels": None,
    "input_format": None,
    "scaler_path": None,
    "window": None,
    "phase_windows": None,
    "phase_stats_path": None,
    "dequantize_waveforms": False,
    "augment": None,
    "augment_seed": 0,
    "resume_from": None,
    "target_transform": None,
    "epochs": 50,
    "batch_size": 32,
    "num_workers": 0,
    "pin_memory": False,
    "persistent_workers": False,
    "prefetch_factor": None,
    "drop_last": False,
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
    "performance": {"cudnn_benchmark": False, "tf32": False, "compile": False, "compile_mode": "default"},
    "progress": {"enabled": True, "stdout": True, "jsonl": True, "jsonl_name": "metrics_live.jsonl"},
    "grad_clip_norm": 0.0,
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
        "--slow-channels",
        type=str,
        default=None,
        help="Comma-separated slow channel names to keep (default: all). For channel ablation, e.g. drop V_NDIR_CO.",
    )
    parser.add_argument(
        "--input-format",
        choices=["NTC", "NCT", "FEATURES"],
        default=None,
        help="Override model input format; by default the model class contract is used.",
    )
    parser.add_argument("--scaler-path", type=Path, default=None, help="Optional z-score scaler JSON for slow channels.")
    parser.add_argument("--resume-from", type=Path, default=None, help="Resume model/optimizer/history from checkpoint.")
    parser.add_argument(
        "--window",
        type=str,
        default=None,
        help='Optional JSON window, e.g. {"kind":"phase","value":"exposure"} or {"kind":"early","value":0.5}.',
    )
    parser.add_argument(
        "--phase-windows",
        type=str,
        default=None,
        help='Optional JSON array of DL phase windows, e.g. [null, {"kind":"phase","value":"exposure"}].',
    )
    parser.add_argument(
        "--phase-stats-path",
        type=str,
        default=None,
        help="Optional precomputed phase statistics path. Use 'auto' for dataset_dir/features/phase_stats.npy.",
    )
    parser.add_argument(
        "--dequantize-waveforms",
        action="store_true",
        default=False,
        help="Load waveform inputs as int16 * scale instead of raw int16 values.",
    )
    parser.add_argument(
        "--augment",
        type=str,
        default=None,
        help='Optional JSON augment config applied to train split only. '
             'Keys: jitter_std, window_fraction, max_shift, amplitude_scale_range '
             '(list [lo, hi]), amplitude_apply_from_channel, gaussian_noise_std, apply_prob.',
    )
    parser.add_argument(
        "--augment-seed",
        type=int,
        default=None,
        help="Seed for per-worker augmentation RNG (default 0).",
    )
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
        "--grad-clip-norm",
        type=float,
        default=None,
        help="Gradient clipping max norm (0=disabled).",
    )
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
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    composition_scheme, component_names = _load_composition_scheme(args.dataset_dir)
    validate_loss_composition_scheme(args.loss, composition_scheme)
    performance = _performance_config(args.performance)
    _apply_performance_settings(performance, torch.device(args.device))

    modalities = _parse_modalities(args.modalities)
    slow_channels = _parse_slow_channels(args.slow_channels)
    input_format = args.input_format or _model_input_format(args.model)

    phase_stats_path = _resolve_phase_stats_path(args.phase_stats_path, args.dataset_dir)
    phase_scaler_path = phase_stats_path.with_name("phase_stats_scaler.json") if phase_stats_path is not None else None

    augment_config = _build_augment_config(args.augment, modalities)
    train_dataset = _build_dataset(
        args.dataset_dir,
        split="train",
        modalities=modalities,
        input_format=input_format,
        scaler_path=args.scaler_path,
        window=args.window,
        phase_windows=args.phase_windows,
        dequantize_waveforms=args.dequantize_waveforms,
        lazy=True,
        phase_stats_path=phase_stats_path,
        augment_config=augment_config,
        augment_seed=int(args.augment_seed),
        slow_channels=slow_channels,
    )
    sample_x, sample_y = train_dataset[0]
    in_channels, timesteps = _infer_input_shape(sample_x, input_format)
    train_labels = _split_labels(args.dataset_dir, "train")
    target_transform = resolve_target_transform_for_training(
        args.target_transform,
        train_labels,
    )
    out_dim = 3 if target_transform is not None else int(sample_y.shape[-1])
    target_transform_audits = _target_transform_audits(
        args.dataset_dir,
        target_transform,
        splits=("train", *_parse_comma(args.eval_splits)),
    )

    model_config = _build_model_config(args.model, args.model_kwargs, in_channels, out_dim, timesteps)
    if phase_stats_path is not None and train_dataset.has_phase_stats:
        model_config["phase_stat_dim"] = train_dataset.phase_stat_dim
        if phase_scaler_path is not None and phase_scaler_path.is_file():
            scaler = json.loads(phase_scaler_path.read_text(encoding="utf-8"))
            model_config["phase_stat_norm_mean"] = scaler["mean"]
            model_config["phase_stat_norm_std"] = scaler["std"]
    validate_loss_model_output(
        args.loss,
        model_name=args.model,
        model_kwargs=model_config,
        composition_scheme=composition_scheme,
    )
    if target_transform is not None and int(model_config["out_dim"]) != 3:
        raise ValueError("DL target_transform requires model out_dim=3")
    model = build_model(model_config)
    loss_fn = build_loss(args.loss, train_targets=train_labels)
    optimizer = build_optimizer(
        model,
        {
            "name": args.optimizer,
            "lr": args.lr,
            "weight_decay": args.weight_decay,
        },
    )
    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        loss_fn=loss_fn,
        device=args.device,
        target_transform=target_transform,
        component_names=component_names,
        composition_scheme=composition_scheme,
    )
    if performance["compile"]:
        trainer.model = torch.compile(trainer.model, mode=performance["compile_mode"])

    train_loader = _build_loader(
        train_dataset,
        args.batch_size,
        args.num_workers,
        shuffle=True,
        seed=args.seed,
        pin_memory=args.pin_memory,
        persistent_workers=args.persistent_workers,
        prefetch_factor=args.prefetch_factor,
        drop_last=args.drop_last,
    )
    val_loader = _optional_loader(
        args.dataset_dir,
        "val",
        modalities,
        input_format,
        args.scaler_path,
        args.window,
        args.phase_windows,
        args.dequantize_waveforms,
        args.batch_size,
        args.num_workers,
        args.pin_memory,
        args.persistent_workers,
        args.prefetch_factor,
        phase_stats_path=phase_stats_path,
        slow_channels=slow_channels,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = args.output_dir / args.checkpoint_name
    best_checkpoint_path = args.output_dir / "best_checkpoint.pt"
    scheduler = _build_scheduler(optimizer, args.scheduler)
    if args.resume_from is not None:
        trainer.load_checkpoint(args.resume_from)
    else:
        _remove_stale_best_checkpoint(best_checkpoint_path)
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
        grad_clip_norm=args.grad_clip_norm,
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
            args.window,
            args.phase_windows,
            args.dequantize_waveforms,
            args.batch_size,
            args.num_workers,
            args.pin_memory,
            args.persistent_workers,
            args.prefetch_factor,
            phase_stats_path=phase_stats_path,
            slow_channels=slow_channels,
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
        "resume_from": str(args.resume_from) if args.resume_from is not None else None,
        "model_config": model_config,
        "input_format": input_format,
        "modalities": modalities,
        "slow_channels": list(slow_channels) if slow_channels is not None else None,
        "window": window_to_payload(resolve_window_config(args.window)),
        "phase_windows": _phase_windows_payload(args.phase_windows),
        "phase_stats_path": str(phase_stats_path) if phase_stats_path is not None else None,
        "dequantize_waveforms": args.dequantize_waveforms,
        "augment": _augment_payload(augment_config, args.augment_seed),
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
        "drop_last": args.drop_last,
        "amp": args.amp,
        "performance": args.performance,
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
        json.dumps(_run_config_payload(args, model_config, input_format, modalities, target_transform, slow_channels), indent=2),
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
    if config.get("resume_from") is not None:
        config["resume_from"] = Path(config["resume_from"])
    if isinstance(config.get("window"), str):
        try:
            config["window"] = json.loads(config["window"])
        except json.JSONDecodeError as exc:
            raise ValueError(f"window must be a JSON object string: {exc}") from exc
    if isinstance(config.get("phase_windows"), str):
        try:
            config["phase_windows"] = json.loads(config["phase_windows"])
        except json.JSONDecodeError as exc:
            raise ValueError(f"phase_windows must be a JSON array string: {exc}") from exc
    if isinstance(config.get("augment"), str):
        try:
            config["augment"] = json.loads(config["augment"])
        except json.JSONDecodeError as exc:
            raise ValueError(f"augment must be a JSON object string: {exc}") from exc
    if config.get("augment") is not None and not isinstance(config["augment"], dict):
        raise ValueError("augment must be a JSON object")
    config["augment_seed"] = int(config.get("augment_seed") or 0)
    if config.get("prefetch_factor") is not None:
        config["prefetch_factor"] = int(config["prefetch_factor"])
    config["drop_last"] = bool(config.get("drop_last", False))
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
    if args.resume_from is not None and not args.resume_from.is_file():
        parser.error(f"resume-from checkpoint not found: {args.resume_from}")
    try:
        phase_stats_path = _resolve_phase_stats_path(args.phase_stats_path, args.dataset_dir)
    except (FileNotFoundError, ValueError) as exc:
        parser.error(str(exc))
    if phase_stats_path is not None and args.model != "cnn1d_tcn_fusion":
        parser.error("phase-stats-path requires model='cnn1d_tcn_fusion'")
    if phase_stats_path is not None and (args.input_format or _model_input_format(args.model)) == "FEATURES":
        parser.error("phase-stats-path is only supported for sequence DL inputs")
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
    if args.grad_clip_norm < 0.0:
        parser.error(f"grad-clip-norm must be >= 0, got {args.grad_clip_norm}")
    if args.scheduler["name"] not in {"none", "reduce_on_plateau", "cosine_annealing"}:
        parser.error("scheduler.name must be one of ['none', 'reduce_on_plateau', 'cosine_annealing']")
    try:
        resolve_window_config(args.window)
        _resolve_phase_windows(args.phase_windows)
        if args.window is not None and args.phase_windows is not None:
            raise ValueError("window and phase_windows cannot be combined")
        target_transform = resolve_target_transform_spec(args.target_transform)
        validate_loss_target_transform(args.loss, None if target_transform is None else target_transform.name)
        _build_augment_config(args.augment, _parse_modalities(args.modalities))
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


def _parse_slow_channels(value: object) -> tuple[str, ...] | None:
    """解析保留的 slow 通道名（channel ablation）；None 表示保留全部。"""
    if value is None:
        return None
    if isinstance(value, (list, tuple)):
        items = tuple(str(item).strip() for item in value if str(item).strip())
    else:
        items = _parse_comma(str(value))
    return items or None


def _parse_comma(value: str) -> tuple[str, ...]:
    return tuple(s.strip() for s in value.split(",") if s.strip())


def _model_input_format(model_name: str) -> str:
    entry = MODEL_REGISTRY[model_name]
    input_format = getattr(entry, "input_format", None)
    if input_format not in {"NTC", "NCT", "FEATURES"}:
        raise ValueError(f"Model {model_name!r} does not declare input_format; pass --input-format explicitly.")
    return str(input_format)


def _infer_input_shape(sample_x: torch.Tensor | dict[str, object], input_format: str) -> tuple[int, int]:
    if isinstance(sample_x, dict):
        sample_x = sample_x["x"]
    if input_format == "FEATURES":
        if sample_x.ndim != 1:
            raise ValueError(f"Expected one feature sample shaped (F,), got {tuple(sample_x.shape)}")
        return int(sample_x.shape[0]), 1
    if sample_x.ndim == 3:
        if input_format == "NTC":
            return int(sample_x.shape[2]), int(sample_x.shape[1])
        if input_format == "NCT":
            return int(sample_x.shape[1]), int(sample_x.shape[2])
        raise ValueError(f"input_format must be NTC or NCT, got {input_format!r}")
    if sample_x.ndim != 2:
        raise ValueError(f"Expected one sample shaped (T, C), (C, T), (W, T, C), or (W, C, T), got {tuple(sample_x.shape)}")
    if input_format == "NTC":
        return int(sample_x.shape[1]), int(sample_x.shape[0])
    if input_format == "NCT":
        return int(sample_x.shape[0]), int(sample_x.shape[1])
    raise ValueError(f"input_format must be one of ['FEATURES', 'NCT', 'NTC'], got {input_format!r}")


def _resolve_phase_stats_path(value: object, dataset_dir: Path) -> Path | None:
    if value is None:
        return None
    path_text = str(value)
    if not path_text:
        raise ValueError("phase-stats-path must not be empty")
    if path_text == "auto":
        path = dataset_dir / "features" / "phase_stats.npy"
    else:
        path = Path(path_text)
    if path_text != "auto" and not path.is_absolute():
        path = dataset_dir / path
    if not path.is_file():
        raise FileNotFoundError(f"phase-stats-path not found: {path}")
    return path


def _build_dataset(
    dataset_dir: Path,
    *,
    split: str,
    modalities: tuple[str, ...],
    input_format: str,
    scaler_path: Path | None,
    window: dict[str, object] | None,
    phase_windows: list[object] | tuple[object, ...] | None,
    dequantize_waveforms: bool,
    lazy: bool,
    phase_stats_path: Path | None = None,
    augment_config: TimeSeriesAugmentConfig | None = None,
    augment_seed: int = 0,
    slow_channels: tuple[str, ...] | None = None,
):
    if input_format == "FEATURES":
        if slow_channels is not None:
            raise ValueError("slow_channels is only supported for sequence DL inputs")
        return V4FeatureMatrixDataset(
            dataset_dir,
            split=split,
            modalities=modalities,
            scaler_path=scaler_path,
            window=window,
            phase_windows=phase_windows,
        )
    return V4BenchmarkDataset(
        dataset_dir,
        split=split,
        modalities=modalities,
        input_format=input_format,
        scaler_path=scaler_path,
        window=window,
        phase_windows=phase_windows,
        dequantize_waveforms=dequantize_waveforms,
        lazy=lazy,
        phase_stats_path=phase_stats_path,
        augment_config=augment_config,
        augment_seed=augment_seed,
        slow_channels=slow_channels,
    )


_AUGMENT_KEYS = frozenset(
    (
        "jitter_std",
        "window_fraction",
        "max_shift",
        "amplitude_scale_range",
        "amplitude_apply_from_channel",
        "gaussian_noise_std",
        "apply_prob",
    )
)


def _build_augment_config(
    raw: dict[str, Any] | None,
    modalities: tuple[str, ...],
) -> TimeSeriesAugmentConfig | None:
    """从 raw dict 构造 TimeSeriesAugmentConfig；若所有字段为默认则返回 None。"""
    if not raw:
        return None
    if not isinstance(raw, dict):
        raise ValueError("augment must be a JSON object")
    unknown = set(raw) - _AUGMENT_KEYS
    if unknown:
        raise ValueError(f"Unknown augment keys: {sorted(unknown)}")

    kwargs: dict[str, Any] = {}
    if "jitter_std" in raw:
        kwargs["jitter_std"] = float(raw["jitter_std"])
    if "window_fraction" in raw:
        kwargs["window_fraction"] = float(raw["window_fraction"])
    if "max_shift" in raw:
        kwargs["max_shift"] = int(raw["max_shift"])
    if "amplitude_scale_range" in raw and raw["amplitude_scale_range"] is not None:
        rng_val = raw["amplitude_scale_range"]
        if not isinstance(rng_val, (list, tuple)) or len(rng_val) != 2:
            raise ValueError("amplitude_scale_range must be a length-2 list [lo, hi]")
        kwargs["amplitude_scale_range"] = (float(rng_val[0]), float(rng_val[1]))
    if "amplitude_apply_from_channel" in raw:
        kwargs["amplitude_apply_from_channel"] = int(raw["amplitude_apply_from_channel"])
    elif "amplitude_scale_range" in kwargs:
        # 用户未显式指定 apply_from：根据模态推断（含 slow 则跳过前 8 列，否则 0）。
        kwargs["amplitude_apply_from_channel"] = 8 if "slow" in modalities else 0
    if "gaussian_noise_std" in raw:
        kwargs["gaussian_noise_std"] = float(raw["gaussian_noise_std"])
    if "apply_prob" in raw:
        kwargs["apply_prob"] = float(raw["apply_prob"])

    config = TimeSeriesAugmentConfig(**kwargs)
    # 全部为默认值时视为关闭。
    default = TimeSeriesAugmentConfig()
    if config == default:
        return None
    return config


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


def _load_composition_scheme(dataset_dir: Path) -> tuple[str, tuple[str, ...]]:
    """Read composition_scheme and label_names from benchmark manifest/metadata.

    Falls back to hydrogen_ng defaults for legacy datasets that pre-date the
    syngas adaptation (manifest.composition_scheme absent).
    """
    manifest_path = dataset_dir / "manifest.json"
    composition_scheme = "hydrogen_ng"
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        composition_scheme = str(manifest.get("composition_scheme", "hydrogen_ng"))
    label_names_path = dataset_dir / "metadata" / "label_names.npy"
    if label_names_path.is_file():
        component_names = tuple(_load_str_array(label_names_path))
    else:
        # Fallback：legacy 数据集没有 label_names.npy 时，用 hg 默认
        from sim.core.schema import COMPONENT_FIELDS

        component_names = tuple(COMPONENT_FIELDS)
    return composition_scheme, component_names


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


def _remove_stale_best_checkpoint(path: Path) -> None:
    if path.is_file():
        path.unlink()


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


_COMPILE_MODES = frozenset(
    ("default", "reduce-overhead", "max-autotune", "max-autotune-no-cudagraphs")
)
_PERFORMANCE_KEYS = frozenset(DEFAULT_DL_CONFIG["performance"])


def _performance_config(value: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError("performance must be a JSON object")
    unknown = set(value) - _PERFORMANCE_KEYS
    if unknown:
        raise ValueError(f"Unknown performance keys: {sorted(unknown)}")
    config = dict(DEFAULT_DL_CONFIG["performance"])
    config.update(value)
    compile_mode = str(config["compile_mode"])
    if compile_mode not in _COMPILE_MODES:
        raise ValueError(f"performance.compile_mode must be one of {sorted(_COMPILE_MODES)}, got {compile_mode!r}")
    return {
        "cudnn_benchmark": bool(config["cudnn_benchmark"]),
        "tf32": bool(config["tf32"]),
        "compile": bool(config["compile"]),
        "compile_mode": compile_mode,
    }


def _apply_performance_settings(performance: dict[str, Any], device: torch.device) -> None:
    # benchmark/TF32 是全局 CUDA backend 开关，CPU 设备下静默跳过。
    # 复现性说明：seed 扩展（torch/cuda/np/random）只保证 CPU 路径和算子调度种子一致，
    # 不强制 bit-exact。cudnn_benchmark=True 时 cuDNN 会按输入 shape 选最快内核，
    # 不同进程/不同硬件下选出的内核可能不同，结果会有微小漂移；TF32 同样会损失低位精度。
    # 多 seed 验证关注的是均值与方差而非 bit-exact，因此当前不强制 cudnn.deterministic=True。
    # 若后续需要严格复现，再加 deterministic 开关并要求 cudnn_benchmark=False。
    if device.type != "cuda":
        return
    torch.set_float32_matmul_precision("high" if performance["tf32"] else "highest")
    torch.backends.cuda.matmul.allow_tf32 = bool(performance["tf32"])
    torch.backends.cudnn.allow_tf32 = bool(performance["tf32"])
    torch.backends.cudnn.benchmark = bool(performance["cudnn_benchmark"])


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
    if name == "cosine_annealing":
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=int(config.get("T_max", 50)),
            eta_min=float(config.get("eta_min", 0.0)),
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
    drop_last: bool = False,
) -> DataLoader:
    generator = torch.Generator()
    generator.manual_seed(seed)
    kwargs: dict[str, Any] = {
        "batch_size": batch_size,
        "shuffle": shuffle,
        "num_workers": num_workers,
        "generator": generator if shuffle else None,
        "pin_memory": pin_memory,
        "drop_last": drop_last,
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
    window: dict[str, object] | None,
    phase_windows: list[object] | tuple[object, ...] | None,
    dequantize_waveforms: bool,
    batch_size: int,
    num_workers: int,
    pin_memory: bool,
    persistent_workers: bool,
    prefetch_factor: int | None,
    phase_stats_path: Path | None = None,
    slow_channels: tuple[str, ...] | None = None,
) -> DataLoader | None:
    dataset = _build_dataset(
        dataset_dir,
        split=split,
        modalities=modalities,
        input_format=input_format,
        scaler_path=scaler_path,
        window=window,
        phase_windows=phase_windows,
        dequantize_waveforms=dequantize_waveforms,
        lazy=True,
        phase_stats_path=phase_stats_path,
        slow_channels=slow_channels,
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
    slow_channels: tuple[str, ...] | None = None,
) -> dict[str, Any]:
    return {
        "dataset_dir": str(args.dataset_dir),
        "output_dir": str(args.output_dir),
        "model_config": model_config,
        "input_format": input_format,
        "modalities": modalities,
        "slow_channels": list(slow_channels) if slow_channels is not None else None,
        "scaler_path": str(args.scaler_path) if args.scaler_path is not None else None,
        "resume_from": str(args.resume_from) if args.resume_from is not None else None,
        "window": window_to_payload(resolve_window_config(args.window)),
        "phase_windows": _phase_windows_payload(args.phase_windows),
        "phase_stats_path": args.phase_stats_path,
        "dequantize_waveforms": args.dequantize_waveforms,
        "augment": args.augment,
        "augment_seed": int(args.augment_seed),
        "target_transform": args.target_transform,
        "resolved_target_transform": asdict(target_transform) if target_transform is not None else None,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "pin_memory": args.pin_memory,
        "persistent_workers": args.persistent_workers,
        "prefetch_factor": args.prefetch_factor,
        "drop_last": args.drop_last,
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
        "performance": args.performance,
        "progress": args.progress,
        "grad_clip_norm": args.grad_clip_norm,
        "eval_splits": _parse_comma(args.eval_splits),
    }


def _resolve_phase_windows(value: object) -> tuple[object | None, ...] | None:
    if value is None:
        return None
    if not isinstance(value, list):
        raise ValueError("phase_windows must be a JSON array")
    if not value:
        raise ValueError("phase_windows must not be empty")
    return tuple(resolve_window_config(window) for window in value)


def _phase_windows_payload(value: object) -> list[dict[str, object] | None] | None:
    windows = _resolve_phase_windows(value)
    if windows is None:
        return None
    return [window_to_payload(resolve_window_config(window)) for window in windows]


def _augment_payload(
    config: TimeSeriesAugmentConfig | None,
    augment_seed: int,
) -> dict[str, Any] | None:
    if config is None:
        return None
    payload = asdict(config)
    if payload.get("amplitude_scale_range") is not None:
        lo, hi = payload["amplitude_scale_range"]
        payload["amplitude_scale_range"] = [lo, hi]
    payload["augment_seed"] = int(augment_seed)
    return payload


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
