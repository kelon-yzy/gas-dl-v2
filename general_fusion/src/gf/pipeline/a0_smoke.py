from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

import torch
from torch import nn

from gf.dl.adapters import ArHeCO2Adapter, XyleneENoseAdapter
from gf.dl.contracts import UnifiedSample, collate_samples
from gf.dl.fusion_core import FusionCore
from gf.dl.preprocessing import TrainGroupStandardScaler
from gf.dl.sensor_encoders import MaskedStatSensorEncoder
from gf.dl.splits import validate_group_splits
from gf.dl.task_heads import RegressionHead


DEFAULT_EXPERIMENT_CONFIG = Path("configs/experiment/a0_shared_core_smoke.json")


@dataclass(frozen=True)
class DatasetSmokeSummary:
    dataset_id: str
    sample_count: int
    sensor_count: int
    output_shape: tuple[int, int]
    group_ids: tuple[str, ...]
    fitted_scaler_groups: tuple[str, ...]
    output_checksum: float


@dataclass(frozen=True)
class A0SmokeSummary:
    core_class: str
    datasets: tuple[DatasetSmokeSummary, ...]
    all_gradients_finite: bool


def run_a0_smoke(
    *,
    project_root: Path | None = None,
    experiment_config: Path = DEFAULT_EXPERIMENT_CONFIG,
) -> A0SmokeSummary:
    root = (project_root or _default_project_root()).resolve()
    experiment_path = _resolve_project_path(root, experiment_config)
    experiment = _load_json_object(experiment_path)
    seed = _required_int(experiment, "seed")
    torch.manual_seed(seed)

    data_config_paths = experiment.get("data_configs")
    if not isinstance(data_config_paths, dict) or set(data_config_paths) != {"ar_he_co2", "xylene_e_nose"}:
        raise ValueError("data_configs must map exactly ar_he_co2 and xylene_e_nose")
    ar_config = _load_json_object(_resolve_project_path(root, Path(str(data_config_paths["ar_he_co2"]))))
    xylene_config = _load_json_object(_resolve_project_path(root, Path(str(data_config_paths["xylene_e_nose"]))))
    model_path = _resolve_project_path(root, Path(_required_string(experiment, "model_config")))
    model_config = _load_json_object(model_path)

    samples_by_dataset = {
        "ar_he_co2": ArHeCO2Adapter.from_config(ar_config).load_samples(),
        "xylene_e_nose": XyleneENoseAdapter.from_config(xylene_config, project_root=root).load_samples(),
    }
    scaled_samples: dict[str, list[UnifiedSample]] = {}
    scaler_groups: dict[str, tuple[str, ...]] = {}
    for dataset_id, samples in samples_by_dataset.items():
        splits = _splits_from_samples(samples)
        validated = validate_group_splits(splits, known_group_ids=(sample.group_id for sample in samples))
        scaler = TrainGroupStandardScaler()
        scaler.fit(samples, set(validated["train"]))
        scaled_samples[dataset_id] = [scaler.transform(sample) for sample in samples]
        scaler_groups[dataset_id] = tuple(sorted(scaler.fitted_group_ids))

    embedding_dim = _required_int(model_config, "embedding_dim")
    hidden_dim = _required_int(model_config, "hidden_dim")
    sensor_ids = _required_string_list(model_config, "sensor_ids")
    sensor_types = _required_string_list(model_config, "sensor_types")
    task_output_dims = model_config.get("task_output_dims")
    if not isinstance(task_output_dims, dict) or set(task_output_dims) != set(samples_by_dataset):
        raise ValueError("task_output_dims must match configured datasets")

    encoder = MaskedStatSensorEncoder(
        embedding_dim=embedding_dim,
        sensor_ids=sensor_ids,
        sensor_types=sensor_types,
    )
    core = FusionCore(embedding_dim=embedding_dim, hidden_dim=hidden_dim)
    heads = nn.ModuleDict(
        {
            dataset_id: RegressionHead(hidden_dim, int(task_output_dims[dataset_id]))
            for dataset_id in sorted(task_output_dims)
        }
    )

    summaries: list[DatasetSmokeSummary] = []
    losses: list[torch.Tensor] = []
    for dataset_id in sorted(scaled_samples):
        samples = scaled_samples[dataset_id]
        batch = collate_samples(samples)
        sensor_embeddings, reliability = encoder(batch)
        fused = core(sensor_embeddings, batch.sensor_mask, reliability)
        output = heads[dataset_id](fused)
        if not torch.isfinite(output).all():
            raise RuntimeError(f"non-finite output for dataset {dataset_id}")
        losses.append(output.square().mean())
        summaries.append(
            DatasetSmokeSummary(
                dataset_id=dataset_id,
                sample_count=len(samples),
                sensor_count=max(len(sample.sensor_id) for sample in samples),
                output_shape=tuple(output.shape),
                group_ids=tuple(sample.group_id for sample in samples),
                fitted_scaler_groups=scaler_groups[dataset_id],
                output_checksum=float(output.detach().sum()),
            )
        )

    sum(losses).backward()
    parameters = list(encoder.parameters()) + list(core.parameters()) + list(heads.parameters())
    gradients_finite = all(
        parameter.grad is not None and torch.isfinite(parameter.grad).all().item() for parameter in parameters
    )
    if not gradients_finite:
        raise RuntimeError("A0 smoke did not produce finite gradients for all trainable parameters")
    return A0SmokeSummary(
        core_class=type(core).__name__,
        datasets=tuple(summaries),
        all_gradients_finite=True,
    )


def _splits_from_samples(samples: list[UnifiedSample]) -> dict[str, list[str]]:
    splits = {"train": [], "val": [], "test": []}
    for sample in samples:
        split = sample.metadata.get("split")
        if split not in splits:
            raise ValueError(f"sample {sample.group_id!r} has invalid split metadata {split!r}")
        splits[str(split)].append(sample.group_id)
    return splits


def _load_json_object(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return value


def _resolve_project_path(project_root: Path, path: Path) -> Path:
    resolved = path.resolve() if path.is_absolute() else (project_root / path).resolve()
    if not resolved.is_relative_to(project_root):
        raise ValueError(f"configured path escapes project root: {path}")
    if not resolved.is_file():
        raise FileNotFoundError(resolved)
    return resolved


def _required_string(config: dict[str, Any], key: str) -> str:
    value = config.get(key)
    if not isinstance(value, str) or not value:
        raise ValueError(f"{key} must be a non-empty string")
    return value


def _required_int(config: dict[str, Any], key: str) -> int:
    value = config.get(key)
    if not isinstance(value, int) or isinstance(value, bool):
        raise ValueError(f"{key} must be an integer")
    return value


def _required_string_list(config: dict[str, Any], key: str) -> list[str]:
    value = config.get(key)
    if not isinstance(value, list) or not value or any(not isinstance(item, str) or not item for item in value):
        raise ValueError(f"{key} must be a non-empty list of strings")
    if len(set(value)) != len(value):
        raise ValueError(f"{key} must not contain duplicates")
    return value


def _default_project_root() -> Path:
    return Path(__file__).resolve().parents[3]


def main() -> None:
    summary = run_a0_smoke()
    print(json.dumps(_summary_to_dict(summary), ensure_ascii=False, indent=2))


def _summary_to_dict(summary: A0SmokeSummary) -> dict[str, Any]:
    return {
        "core_class": summary.core_class,
        "all_gradients_finite": summary.all_gradients_finite,
        "datasets": [
            {
                "dataset_id": dataset.dataset_id,
                "sample_count": dataset.sample_count,
                "sensor_count": dataset.sensor_count,
                "output_shape": list(dataset.output_shape),
                "group_ids": list(dataset.group_ids),
                "fitted_scaler_groups": list(dataset.fitted_scaler_groups),
                "output_checksum": dataset.output_checksum,
            }
            for dataset in summary.datasets
        ],
    }


if __name__ == "__main__":
    main()
