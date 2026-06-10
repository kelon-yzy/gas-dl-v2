from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from common.composition import TRAIN_MIN_POSITIVE_HALF_EPSILON, resolve_target_transform_spec
from dl.models.registry import MODEL_REGISTRY
from dl.training.losses import LOSS_REGISTRY, ILR_MSE_LOSS, validate_loss_target_transform
from ml.models import REGRESSOR_REGISTRY
from sim.core.schema import SPLIT_NAMES


ALL_MODALITIES = ("slow", "ultrasonic", "fiber_mic")


DEFAULT_ML_RUNS: tuple[dict[str, Any], ...] = (
    {
        "name": "ridge_slow",
        "model": {"name": "ridge", "alpha": 1.0},
        "modalities": ["slow"],
        "protocol": True,
    },
    {
        "name": "ridge_all_modalities",
        "model": {"name": "ridge", "alpha": 1.0},
        "modalities": list(ALL_MODALITIES),
        "protocol": True,
    },
    {
        "name": "ridge_alr_ch4_all_modalities",
        "model": {"name": "ridge", "alpha": 1.0},
        "modalities": list(ALL_MODALITIES),
        "target_transform": {"name": "alr_ch4", "epsilon": TRAIN_MIN_POSITIVE_HALF_EPSILON},
        "protocol": True,
    },
    {
        "name": "ridge_ilr_n2_first_all_modalities",
        "model": {"name": "ridge", "alpha": 1.0},
        "modalities": list(ALL_MODALITIES),
        "target_transform": {"name": "ilr_n2_first", "epsilon": TRAIN_MIN_POSITIVE_HALF_EPSILON},
        "protocol": True,
    },
    {
        "name": "dynamic_stacking_svr_all_modalities",
        "model": {"name": "dynamic_stacking_svr", "n_jobs": 4},
        "modalities": list(ALL_MODALITIES),
        "protocol": False,
    },
)

DEFAULT_DL_RUNS: tuple[dict[str, Any], ...] = (
    {"name": "cnn1d", "model": "cnn1d", "modalities": list(ALL_MODALITIES), "model_kwargs": {}},
    {"name": "tcn", "model": "tcn", "modalities": list(ALL_MODALITIES), "model_kwargs": {}},
    {"name": "lstm", "model": "lstm", "modalities": list(ALL_MODALITIES), "model_kwargs": {}},
    {"name": "transformer", "model": "transformer", "modalities": list(ALL_MODALITIES), "model_kwargs": {}},
    {"name": "patchtst", "model": "patchtst", "modalities": list(ALL_MODALITIES), "model_kwargs": {}},
    {
        "name": "cnn1d_tcn_fusion",
        "model": "cnn1d_tcn_fusion",
        "modalities": list(ALL_MODALITIES),
        "model_kwargs": {},
    },
    {
        "name": "cnn1d_tcn_fusion_ilr",
        "model": "cnn1d_tcn_fusion",
        "modalities": list(ALL_MODALITIES),
        "target_transform": {"name": "ilr_n2_first", "epsilon": TRAIN_MIN_POSITIVE_HALF_EPSILON},
        "loss": ILR_MSE_LOSS,
        "model_kwargs": {},
    },
)


@dataclass(frozen=True, slots=True)
class ExperimentConfig:
    experiment_name: str
    dataset_dir: Path
    output_root: Path
    seed: int
    device: str
    eval_splits: tuple[str, ...]
    training: dict[str, Any]
    ml_runs: tuple[dict[str, Any], ...]
    dl_runs: tuple[dict[str, Any], ...]


def load_experiment_config(
    path: Path,
    *,
    dataset_dir: Path | None = None,
    output_root: Path | None = None,
    device: str | None = None,
) -> ExperimentConfig:
    payload = _read_json(path)
    _validate_top_level(payload)
    if dataset_dir is not None:
        payload["dataset_dir"] = str(dataset_dir)
    if output_root is not None:
        payload["output_root"] = str(output_root)
    if device is not None:
        payload["device"] = device

    eval_splits = _string_tuple(payload["eval_splits"], field="eval_splits")
    unknown_splits = set(eval_splits) - set(SPLIT_NAMES)
    if unknown_splits:
        raise ValueError(f"Unknown eval_splits: {sorted(unknown_splits)}")
    training = _validate_training(payload["training"])
    ml_runs = tuple(payload["ml_runs"] or DEFAULT_ML_RUNS)
    dl_runs = tuple(payload["dl_runs"] or DEFAULT_DL_RUNS)
    _validate_ml_runs(ml_runs)
    _validate_dl_runs(dl_runs, default_loss=str(training["loss"]))
    return ExperimentConfig(
        experiment_name=str(payload["experiment_name"]),
        dataset_dir=Path(payload["dataset_dir"]),
        output_root=Path(payload["output_root"]),
        seed=int(payload["seed"]),
        device=str(payload["device"]),
        eval_splits=eval_splits,
        training=training,
        ml_runs=ml_runs,
        dl_runs=dl_runs,
    )


def _read_json(path: Path) -> dict[str, Any]:
    import json

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"experiment config must be valid JSON: {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError("experiment config must be a JSON object")
    return payload


def _validate_top_level(payload: dict[str, Any]) -> None:
    required = {
        "experiment_name",
        "dataset_dir",
        "output_root",
        "seed",
        "device",
        "eval_splits",
        "training",
        "ml_runs",
        "dl_runs",
    }
    missing = required - set(payload)
    if missing:
        raise ValueError(f"experiment config missing required keys: {sorted(missing)}")
    unknown = set(payload) - required
    if unknown:
        raise ValueError(f"experiment config has unknown keys: {sorted(unknown)}")


def _validate_training(value: object) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError("training must be a JSON object")
    required = {
        "epochs",
        "batch_size",
        "num_workers",
        "optimizer",
        "lr",
        "weight_decay",
        "loss",
        "early_stopping",
        "scheduler",
    }
    missing = required - set(value)
    if missing:
        raise ValueError(f"training missing required keys: {sorted(missing)}")
    scheduler = value["scheduler"]
    if not isinstance(scheduler, dict):
        raise ValueError("training.scheduler must be a JSON object")
    if scheduler.get("name") not in {"none", "reduce_on_plateau"}:
        raise ValueError("training.scheduler.name must be one of ['none', 'reduce_on_plateau']")
    if str(value["loss"]) not in LOSS_REGISTRY:
        raise ValueError(f"Unknown training.loss: {value['loss']!r}. Available: {sorted(LOSS_REGISTRY)}")
    return dict(value)


def _validate_ml_runs(runs: tuple[dict[str, Any], ...]) -> None:
    for run in runs:
        _validate_run_dict(run, kind="ml")
        model = run.get("model")
        if isinstance(model, str):
            model_name = model
        elif isinstance(model, dict):
            model_name = str(model.get("name"))
        else:
            raise ValueError(f"ml run {run.get('name')!r} model must be string or object")
        if model_name not in REGRESSOR_REGISTRY:
            raise ValueError(f"Unknown ML model {model_name!r} in run {run.get('name')!r}")
        _validate_target_transform(run, kind="ML")


def _validate_dl_runs(runs: tuple[dict[str, Any], ...], *, default_loss: str) -> None:
    for run in runs:
        _validate_run_dict(run, kind="dl")
        model_name = str(run["model"])
        if model_name not in MODEL_REGISTRY:
            raise ValueError(f"Unknown DL model {model_name!r} in run {run.get('name')!r}")
        target_transform = _validate_target_transform(run, kind="DL")
        loss_name = str(run.get("loss", default_loss))
        try:
            validate_loss_target_transform(loss_name, None if target_transform is None else target_transform.name)
        except ValueError as exc:
            raise ValueError(f"Invalid DL loss in run {run.get('name')!r}: {exc}") from exc


def _validate_run_dict(run: dict[str, Any], *, kind: str) -> None:
    if not isinstance(run, dict):
        raise ValueError(f"{kind} run must be a JSON object")
    required = {"name", "model", "modalities"}
    missing = required - set(run)
    if missing:
        raise ValueError(f"{kind} run missing required keys: {sorted(missing)}")
    _string_tuple(run["modalities"], field=f"{kind}.{run['name']}.modalities")


def _validate_target_transform(run: dict[str, Any], *, kind: str):
    try:
        return resolve_target_transform_spec(run.get("target_transform"))
    except ValueError as exc:
        raise ValueError(f"Invalid {kind} target_transform in run {run.get('name')!r}: {exc}") from exc


def _string_tuple(value: object, *, field: str) -> tuple[str, ...]:
    if not isinstance(value, list) or not value:
        raise ValueError(f"{field} must be a non-empty JSON array")
    if any(not isinstance(item, str) or not item for item in value):
        raise ValueError(f"{field} must contain non-empty strings")
    return tuple(value)
