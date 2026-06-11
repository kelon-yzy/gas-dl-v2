from __future__ import annotations

import argparse
import csv
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any, Sequence

from common.metrics import conditional_metrics_to_payload
from common.windows import WINDOW_KIND_EARLY, WINDOW_KIND_PHASE, resolve_window_config, window_label, window_to_payload
from dl.cli import run as run_dl_cli
from ml.evaluation_protocol import BaselineProtocolResult, run_baseline_protocol
from ml.features import MLFeatureConfig
from ml.training import MLTrainingResult, train_regressor_on_dataset
from pipeline.experiment_config import ExperimentConfig, load_experiment_config


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run a configured v4 formal experiment suite.")
    parser.add_argument("--config", type=Path, required=True, help="Experiment JSON config.")
    parser.add_argument("--dataset-dir", type=Path, default=None, help="Override config dataset_dir.")
    parser.add_argument("--output-root", type=Path, default=None, help="Override config output_root.")
    parser.add_argument("--device", type=str, default=None, help="Override config device.")
    parser.add_argument("--dry-run", action="store_true", default=False, help="Validate and print planned runs only.")
    return parser


def run(config: ExperimentConfig, *, dry_run: bool = False) -> dict[str, Any]:
    plan = {
        "experiment_name": config.experiment_name,
        "dataset_dir": str(config.dataset_dir),
        "output_root": str(config.output_root),
        "ml_runs": [run_config["name"] for run_config in config.ml_runs],
        "dl_runs": [run_config["name"] for run_config in config.dl_runs],
        "ml_run_details": [_planned_run_detail("ml", run_config) for run_config in config.ml_runs],
        "dl_run_details": [
            _planned_run_detail("dl", run_config, default_loss=config.training["loss"])
            for run_config in config.dl_runs
        ],
    }
    if dry_run:
        return {"dry_run": True, "plan": plan}
    if not config.dataset_dir.is_dir() or not (config.dataset_dir / "labels" / "y.npy").is_file():
        raise FileNotFoundError(f"dataset_dir must be a v4 benchmark root: {config.dataset_dir}")

    run_root = config.output_root / "runs" / config.experiment_name
    summary_dir = config.output_root / "summary"
    report_dir = config.output_root / "reports"
    run_root.mkdir(parents=True, exist_ok=True)
    summary_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, object]] = []
    for run_config in config.ml_runs:
        rows.extend(_run_with_progress("ml", config, run_config, run_root / str(run_config["name"])))
    for run_config in config.dl_runs:
        rows.extend(_run_with_progress("dl", config, run_config, run_root / str(run_config["name"])))

    summary_path = summary_dir / f"{config.experiment_name}_summary.csv"
    report_path = report_dir / f"{config.experiment_name}.md"
    _write_summary(summary_path, rows)
    _write_report(report_path, config, rows)
    return {
        "experiment_name": config.experiment_name,
        "summary_path": str(summary_path),
        "report_path": str(report_path),
        "rows": rows,
    }


def _run_with_progress(
    kind: str,
    config: ExperimentConfig,
    run_config: dict[str, Any],
    output_dir: Path,
) -> list[dict[str, object]]:
    run_name = str(run_config["name"])
    model_name = str(_model_name(run_config["model"]))
    print(f"[run start] kind={kind} name={run_name} model={model_name} output_dir={output_dir}", flush=True)
    try:
        if kind == "ml":
            rows = _run_ml(config, run_config, output_dir)
        elif kind == "dl":
            rows = _run_dl(config, run_config, output_dir)
        else:
            raise ValueError(f"Unknown run kind: {kind!r}")
    except Exception as exc:
        print(f"[run fail] kind={kind} name={run_name} model={model_name} error={exc}", flush=True)
        raise
    print(f"[run done] kind={kind} name={run_name} model={model_name}", flush=True)
    return rows


def _run_ml(config: ExperimentConfig, run_config: dict[str, Any], output_dir: Path) -> list[dict[str, object]]:
    output_dir.mkdir(parents=True, exist_ok=True)
    window = resolve_window_config(run_config.get("window"))
    feature_config = MLFeatureConfig(
        modalities=tuple(run_config["modalities"]),
        sequence_statistics=tuple(run_config.get("sequence_statistics", ("mean", "std", "min", "max", "last", "delta", "slope"))),
        waveform_frame_features=tuple(run_config.get("waveform_frame_features", ("mean", "std", "mean_abs", "max_abs", "energy", "peak_index"))),
        slow_scaler_path=run_config.get("scaler_path"),
        phase_filter=str(window.value) if window is not None and window.kind == WINDOW_KIND_PHASE else None,
        early_fraction=float(window.value) if window is not None and window.kind == WINDOW_KIND_EARLY else None,
    )
    model_config = run_config["model"]
    if run_config.get("protocol", False):
        if window is not None:
            raise ValueError(f"ML run {run_config['name']!r} cannot combine protocol=true with run-level window")
        result = run_baseline_protocol(
            config.dataset_dir,
            model_config=model_config,
            feature_config=feature_config,
            eval_splits=("train", *config.eval_splits),
            target_transform=run_config.get("target_transform"),
        )
        payload = _protocol_payload(result)
        rows = _ml_rows(str(run_config["name"]), str(_model_name(model_config)), result.full, window=window)
        resolved_target_transform = asdict(result.full.target_transform) if result.full.target_transform is not None else None
    else:
        result = train_regressor_on_dataset(
            config.dataset_dir,
            model_config=model_config,
            feature_config=feature_config,
            eval_splits=("train", *config.eval_splits),
            target_transform=run_config.get("target_transform"),
        )
        payload = _training_payload(result)
        rows = _ml_rows(str(run_config["name"]), str(_model_name(model_config)), result, window=window)
        resolved_target_transform = asdict(result.target_transform) if result.target_transform is not None else None
    (output_dir / "metrics.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    (output_dir / "run_config.json").write_text(
        json.dumps(_run_config_payload(run_config, resolved_target_transform), indent=2),
        encoding="utf-8",
    )
    return rows


def _run_dl(config: ExperimentConfig, run_config: dict[str, Any], output_dir: Path) -> list[dict[str, object]]:
    import argparse as _argparse

    training = config.training
    args = _argparse.Namespace(
        config=None,
        dataset_dir=config.dataset_dir,
        output_dir=output_dir,
        model=run_config["model"],
        model_kwargs=run_config.get("model_kwargs", {}),
        modalities=",".join(run_config["modalities"]),
        input_format=run_config.get("input_format"),
        scaler_path=run_config.get("scaler_path"),
        resume_from=run_config.get("resume_from"),
        window=run_config.get("window"),
        target_transform=run_config.get("target_transform"),
        epochs=training["epochs"],
        batch_size=training["batch_size"],
        num_workers=training["num_workers"],
        pin_memory=training.get("pin_memory"),
        persistent_workers=training.get("persistent_workers"),
        prefetch_factor=training.get("prefetch_factor"),
        seed=config.seed,
        device=config.device,
        loss=run_config.get("loss", training["loss"]),
        optimizer=training["optimizer"],
        lr=training["lr"],
        weight_decay=training["weight_decay"],
        eval_splits=",".join(config.eval_splits),
        checkpoint_name="checkpoint.pt",
        early_stopping=training["early_stopping"],
        scheduler=training["scheduler"],
        amp=training.get("amp"),
        progress=training.get("progress"),
        json=False,
    )
    payload = run_dl_cli(args)
    return _dl_rows(str(run_config["name"]), str(run_config["model"]), payload)


def _model_name(model_config: str | dict[str, Any]) -> str:
    if isinstance(model_config, str):
        return model_config
    return str(model_config["name"])


def _planned_run_detail(kind: str, run_config: dict[str, Any], *, default_loss: object | None = None) -> dict[str, Any]:
    detail = {
        "kind": kind,
        "name": run_config["name"],
        "model": _model_name(run_config["model"]),
        "modalities": run_config["modalities"],
        "target_transform": run_config.get("target_transform"),
        "protocol": bool(run_config.get("protocol", False)),
        "window": window_to_payload(resolve_window_config(run_config.get("window"))),
    }
    if default_loss is not None:
        detail["loss"] = run_config.get("loss", default_loss)
    return detail


def _run_config_payload(run_config: dict[str, Any], resolved_target_transform: dict[str, object] | None) -> dict[str, Any]:
    payload = dict(run_config)
    payload["resolved_target_transform"] = resolved_target_transform
    return payload


def _training_payload(result: MLTrainingResult) -> dict[str, Any]:
    return {
        "feature_config": {
            "modalities": result.feature_config.modalities,
            "sequence_statistics": result.feature_config.sequence_statistics,
            "waveform_frame_features": result.feature_config.waveform_frame_features,
            "phase_filter": result.feature_config.phase_filter,
            "early_fraction": result.feature_config.early_fraction,
        },
        "feature_names": result.feature_names,
        "label_names": result.label_names,
        "train_split": result.train_split,
        "target_transform": asdict(result.target_transform) if result.target_transform is not None else None,
        "target_transform_audits": (
            {split_name: asdict(audit) for split_name, audit in result.target_transform_audits.items()}
            if result.target_transform_audits is not None
            else None
        ),
        "evaluations": {
            split_name: {
                "metrics": asdict(split_eval.metrics),
                "component_metrics": {key: asdict(value) for key, value in split_eval.component_metrics.items()},
                "compositional_metrics": (
                    asdict(split_eval.compositional_metrics)
                    if split_eval.compositional_metrics is not None
                    else None
                ),
                "conditional_metrics": conditional_metrics_to_payload(split_eval.conditional_metrics),
            }
            for split_name, split_eval in result.evaluations.items()
        },
    }


def _protocol_payload(result: BaselineProtocolResult) -> dict[str, Any]:
    return {
        "full": _training_payload(result.full),
        "per_phase": {phase: _training_payload(value) for phase, value in result.per_phase.items()},
        "early": {str(fraction): _training_payload(value) for fraction, value in result.early.items()},
    }


def _ml_rows(
    run_name: str,
    model_name: str,
    result: MLTrainingResult,
    *,
    window: object,
) -> list[dict[str, object]]:
    rows = []
    window_text = window_label(resolve_window_config(window))
    for split_name, split_eval in result.evaluations.items():
        rows.append(
            {
                "kind": "ml",
                "run_name": run_name,
                "model": model_name,
                "window": window_text,
                "split": split_name,
                "loss": "",
                "mae": split_eval.metrics.mae,
                "rmse": split_eval.metrics.rmse,
                "r2": split_eval.metrics.r2,
                "x_n2_r2": split_eval.component_metrics["x_N2"].r2,
                "aitchison_mean": (
                    "" if split_eval.compositional_metrics is None else split_eval.compositional_metrics.aitchison_mean
                ),
                "sum_abs_error": "",
                "checkpoint_path": "",
            }
        )
    return rows


def _dl_rows(run_name: str, model_name: str, payload: dict[str, Any]) -> list[dict[str, object]]:
    rows = []
    window_text = window_label(resolve_window_config(payload.get("window")))
    for split_name, split_eval in payload["evaluations"].items():
        metrics = split_eval["metrics"]
        rows.append(
            {
                "kind": "dl",
                "run_name": run_name,
                "model": model_name,
                "window": window_text,
                "split": split_name,
                "loss": split_eval["loss"],
                "mae": metrics["mae"],
                "rmse": metrics["rmse"],
                "r2": metrics["r2"],
                "x_n2_r2": split_eval["component_metrics"]["x_N2"]["r2"],
                "aitchison_mean": (
                    "" if split_eval["compositional_metrics"] is None else split_eval["compositional_metrics"]["aitchison_mean"]
                ),
                "sum_abs_error": split_eval["sum_abs_error"],
                "checkpoint_path": payload["checkpoint_path"],
            }
        )
    return rows


def _write_summary(path: Path, rows: list[dict[str, object]]) -> None:
    fieldnames = (
        "kind",
        "run_name",
        "model",
        "window",
        "split",
        "loss",
        "mae",
        "rmse",
        "r2",
        "x_n2_r2",
        "aitchison_mean",
        "sum_abs_error",
        "checkpoint_path",
    )
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _write_report(path: Path, config: ExperimentConfig, rows: list[dict[str, object]]) -> None:
    lines = [
        f"# Experiment Report: {config.experiment_name}",
        "",
        f"- dataset: `{config.dataset_dir}`",
        f"- device: `{config.device}`",
        "",
        "| kind | run | model | window | split | loss | MAE | RMSE | R2 | x_N2 R2 | Aitchison mean | sum abs error |",
        "|---|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        loss = "" if row["loss"] == "" else f"{float(row['loss']):.6f}"
        aitchison = "" if row["aitchison_mean"] == "" else f"{float(row['aitchison_mean']):.6f}"
        sum_abs_error = "" if row["sum_abs_error"] == "" else f"{float(row['sum_abs_error']):.6f}"
        lines.append(
            f"| {row['kind']} | {row['run_name']} | {row['model']} | {row['window']} | {row['split']} | "
            f"{loss} | {float(row['mae']):.6f} | {float(row['rmse']):.6f} | {float(row['r2']):.6f} | "
            f"{float(row['x_n2_r2']):.6f} | {aitchison} | {sum_abs_error} |"
        )
    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = load_experiment_config(
        args.config,
        dataset_dir=args.dataset_dir,
        output_root=args.output_root,
        device=args.device,
    )
    result = run(config, dry_run=args.dry_run)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
