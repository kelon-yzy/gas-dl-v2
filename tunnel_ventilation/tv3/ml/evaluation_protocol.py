from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from tv3.ml.features import MLFeatureConfig
from tv3.ml.training import MLTrainingResult, train_regressor_on_dataset


DEFAULT_PHASE_WINDOWS = ("baseline", "exposure", "steady", "recovery")
DEFAULT_EARLY_FRACTIONS = (0.25, 0.5, 0.75, 1.0)


@dataclass(frozen=True, slots=True)
class BaselineProtocolResult:
    full: MLTrainingResult
    per_phase: dict[str, MLTrainingResult]
    early: dict[float, MLTrainingResult]


def run_baseline_protocol(
    dataset_dir: Path | str,
    *,
    model_config: str | dict[str, Any] | None = None,
    feature_config: MLFeatureConfig | None = None,
    phases: tuple[str, ...] = DEFAULT_PHASE_WINDOWS,
    early_fractions: tuple[float, ...] = DEFAULT_EARLY_FRACTIONS,
    train_split: str = "train",
    eval_splits: tuple[str, ...] = ("train", "val", "test", "extrapolation"),
    target_transform: str | dict[str, Any] | None = None,
) -> BaselineProtocolResult:
    base_config = feature_config or MLFeatureConfig()
    full = train_regressor_on_dataset(
        dataset_dir,
        model_config=model_config,
        feature_config=base_config,
        train_split=train_split,
        eval_splits=eval_splits,
        target_transform=target_transform,
    )
    per_phase = {
        phase: train_regressor_on_dataset(
            dataset_dir,
            model_config=model_config,
            feature_config=replace(base_config, phase_filter=phase),
            train_split=train_split,
            eval_splits=eval_splits,
            target_transform=target_transform,
        )
        for phase in phases
    }
    early = {
        fraction: train_regressor_on_dataset(
            dataset_dir,
            model_config=model_config,
            feature_config=replace(base_config, early_fraction=fraction),
            train_split=train_split,
            eval_splits=eval_splits,
            target_transform=target_transform,
        )
        for fraction in early_fractions
    }
    return BaselineProtocolResult(full=full, per_phase=per_phase, early=early)
