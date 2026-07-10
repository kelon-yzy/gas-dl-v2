from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

from tv3.ml.extratrees_head import ExtraTreesHeadConfig, _ExtraTreesRaw3Regressor
from tv3.ml.features import MLFeatureMatrix
from tv3.ml.rocket_features import (
    RocketFeatureConfig,
    build_tv3_physics_feature_cache,
    default_cache_dir,
    load_cached_split_feature_matrix,
    validate_d0_observed_feature_config,
)
from tv3.ml.rocket_training import RocketTrainingResult
from tv3.ml.training import SplitEvaluation, evaluate_regressor


def train_tv3_extratrees_regressor(
    dataset_dir: Path | str,
    *,
    feature_config: RocketFeatureConfig,
    cache_dir: Path | str | None = None,
    train_split: str = "train",
    eval_splits: tuple[str, ...] = ("val", "test", "extrapolation"),
    extratrees_config: ExtraTreesHeadConfig | None = None,
) -> RocketTrainingResult:
    """Train the R7 raw3 ExtraTrees probe on the observed physics feature contract."""

    validate_d0_observed_feature_config(feature_config)
    dataset_path = Path(dataset_dir)
    cache_path = Path(cache_dir) if cache_dir is not None else default_cache_dir(dataset_path, feature_config.feature_builder)
    feature_cache = build_tv3_physics_feature_cache(dataset_path, cache_dir=cache_path, config=feature_config)
    train_matrix = load_cached_split_feature_matrix(dataset_path, cache_path, split=train_split)
    model = _ExtraTreesRaw3Regressor(config=extratrees_config)
    model.fit(train_matrix.x, train_matrix.y, feature_names=train_matrix.feature_names)

    evaluations: dict[str, SplitEvaluation] = {}
    for split_name in (train_split, *eval_splits):
        matrix = train_matrix if split_name == train_split else load_cached_split_feature_matrix(dataset_path, cache_path, split=split_name)
        _validate_feature_contract(matrix, train_matrix)
        evaluations[split_name] = evaluate_regressor(
            model,
            matrix,
            split=split_name,
            composition_scheme="tunnel_ventilation",
        )

    return RocketTrainingResult(
        head="extratrees",
        dataset_dir=dataset_path,
        cache_dir=cache_path,
        feature_cache=feature_cache,
        feature_names=train_matrix.feature_names,
        label_names=train_matrix.label_names,
        train_split=train_split,
        evaluations=evaluations,
        diagnostics={
            "model_config": asdict(model.config),
            "feature_importances": {
                feature_name: float(importance)
                for feature_name, importance in zip(
                    train_matrix.feature_names,
                    model.model.feature_importances_,
                    strict=True,
                )
            },
        },
    )


def _validate_feature_contract(matrix: MLFeatureMatrix, reference: MLFeatureMatrix) -> None:
    if matrix.feature_names != reference.feature_names:
        raise ValueError("cached rocket feature names must match across splits")
    if matrix.label_names != reference.label_names:
        raise ValueError("cached rocket label names must match across splits")
