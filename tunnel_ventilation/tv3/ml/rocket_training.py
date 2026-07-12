from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.preprocessing import StandardScaler

from tv3.common.metrics import conditional_metrics_to_payload
from tv3.ml.features import MLFeatureMatrix
from tv3.ml.minirocket_features import (
    MiniRocketFeatureConfig,
    build_minirocket_feature_cache,
)
from tv3.ml.models import RidgeRegressor
from tv3.ml.rocket_features import (
    RAW_DSP_FEATURE_BUILDER,
    RAW_DSP_FRAME_CACHE_ROOT,
    RocketFeatureCache,
    RocketFeatureConfig,
    build_tv3_physics_feature_cache,
    default_cache_dir,
    load_cached_split_feature_matrix,
)
from tv3.ml.raw_dsp_features import RAW_DSP_FRAME_SCHEMA_VERSION
from tv3.ml.mlp_head import MlpHeadConfig, _ScaledMLPRegressor
from tv3.ml.ridge_head import ScaledRidgeCVRegressor
from tv3.ml.ridge_residual_head import (
    DEFAULT_OOF_FOLDS,
    DEFAULT_OOF_SEED,
    OofRidgeResidualMlpRegressor,
)
from tv3.ml.training import SplitEvaluation, evaluate_regressor


DEFAULT_RIDGE_ALPHAS = (1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1, 3e-1, 1.0, 3.0, 10.0, 30.0, 100.0)
RAW_DSP_FIDELITY_SCHEMA_VERSION = "tv3-d2b-frame-fidelity-1"
B7_HEAD = "oof_ridge_residual_mlp"
B6_MULTISEED_EXPECTED_MEANS = {
    "test": 0.5356,
    "extrapolation": 0.4835,
}
B6_MULTISEED_EXPECTED_SEEDS = (42, 123, 456)
FORMAL_RAW_DSP_MANIFEST_CONTRACT = {
    "schema_version": RAW_DSP_FRAME_SCHEMA_VERSION,
    "complete_dataset": True,
    "template_mode": "train_baseline_median",
    "template_source_split": "train",
    "diagnostic_only": False,
}


@dataclass(frozen=True, slots=True)
class RocketTrainingResult:
    head: str
    dataset_dir: Path
    cache_dir: Path
    feature_cache: RocketFeatureCache
    feature_names: tuple[str, ...]
    label_names: tuple[str, ...]
    train_split: str
    evaluations: dict[str, SplitEvaluation]
    diagnostics: dict[str, Any]
    raw_dsp_provenance: dict[str, Any] | None = None
    raw_dsp_fidelity: dict[str, Any] | None = None
    raw_dsp_reference: dict[str, Any] | None = None
    b6_reference: dict[str, Any] | None = None


class _ScaledClosedFormRidgeRegressor:
    def __init__(self, *, alpha: float):
        self.scaler = StandardScaler()
        self.model = RidgeRegressor(alpha=alpha, standardize=False)

    def fit(
        self,
        x: np.ndarray,
        y: np.ndarray,
        *,
        feature_names: tuple[str, ...] | None = None,
    ) -> _ScaledClosedFormRidgeRegressor:
        x_scaled = self.scaler.fit_transform(np.asarray(x, dtype=np.float64))
        self.model.fit(x_scaled, np.asarray(y, dtype=np.float64), feature_names=feature_names)
        return self

    def predict(self, x: np.ndarray) -> np.ndarray:
        x_scaled = self.scaler.transform(np.asarray(x, dtype=np.float64))
        return self.model.predict(x_scaled).astype(np.float32, copy=False)


class _TabPFNMultiRegressor:
    """TabPFN 多输出回归头。原生单输出，按标签列拆分 per-target 回归器。"""

    def __init__(self, *, device: str = "auto", n_estimators: int = 8, random_state: int = 0):
        from tabpfn import TabPFNRegressor

        self._make = lambda: TabPFNRegressor(
            device=device,
            n_estimators=n_estimators,
            random_state=random_state,
        )
        self._models: list = []

    def fit(self, x: np.ndarray, y: np.ndarray, *, feature_names=None) -> "_TabPFNMultiRegressor":
        y = np.asarray(y, dtype=np.float64)
        if y.ndim == 1:
            y = y[:, None]
        x_arr = np.asarray(x, dtype=np.float64)
        self._models = [self._make() for _ in range(y.shape[1])]
        for col, model in enumerate(self._models):
            model.fit(x_arr, y[:, col])
        return self

    def predict(self, x: np.ndarray) -> np.ndarray:
        x_arr = np.asarray(x, dtype=np.float64)
        return np.column_stack([m.predict(x_arr) for m in self._models]).astype(np.float32, copy=False)


def train_tv3_rocket_regressor(
    dataset_dir: Path | str,
    *,
    feature_config: RocketFeatureConfig | MiniRocketFeatureConfig | None = None,
    cache_dir: Path | str | None = None,
    head: str = "ridgecv",
    train_split: str = "train",
    eval_splits: tuple[str, ...] = ("val", "test", "extrapolation"),
    ridge_alphas: tuple[float, ...] = DEFAULT_RIDGE_ALPHAS,
    closed_form_alpha: float = 1.0,
    device: str = "auto",
    mlp_config: MlpHeadConfig | None = None,
    oof_folds: int = DEFAULT_OOF_FOLDS,
    oof_seed: int = DEFAULT_OOF_SEED,
    raw_dsp_fidelity_metrics_path: Path | str | None = None,
    raw_dsp_reference_metrics_path: Path | str | None = None,
    b6_multiseed_report_path: Path | str | None = None,
) -> RocketTrainingResult:
    dataset_dir = Path(dataset_dir)
    if feature_config is None:
        feature_config = RocketFeatureConfig()
    raw_dsp_provenance: dict[str, Any] | None = None
    raw_dsp_fidelity: dict[str, Any] | None = None
    raw_dsp_reference: dict[str, Any] | None = None
    b6_reference: dict[str, Any] | None = None
    if feature_config.feature_builder != RAW_DSP_FEATURE_BUILDER and (
        raw_dsp_fidelity_metrics_path is not None
        or raw_dsp_reference_metrics_path is not None
        or b6_multiseed_report_path is not None
    ):
        raise ValueError("RawDSP evidence paths require the RawDSP feature builder")
    if feature_config.feature_builder == RAW_DSP_FEATURE_BUILDER:
        if (raw_dsp_fidelity_metrics_path is None) != (raw_dsp_reference_metrics_path is None):
            raise ValueError(
                "RawDSP fidelity and reference metrics paths must be provided together for a compared run"
            )
        if head == B7_HEAD and (
            raw_dsp_fidelity_metrics_path is None
            or raw_dsp_reference_metrics_path is None
            or b6_multiseed_report_path is None
        ):
            raise ValueError(
                "B7 requires RawDSP fidelity, B1 reference metrics, and B6 multiseed report paths"
            )
        raw_dsp_provenance = load_raw_dsp_provenance(dataset_dir)
        if raw_dsp_fidelity_metrics_path is not None:
            raw_dsp_fidelity = load_raw_dsp_fidelity(
                raw_dsp_fidelity_metrics_path,
                dataset_dir=dataset_dir,
                provenance=raw_dsp_provenance,
            )
            raw_dsp_reference = load_raw_dsp_reference_metrics(
                raw_dsp_reference_metrics_path,
                dataset_dir=dataset_dir,
                feature_config=feature_config,
                provenance=raw_dsp_provenance,
            )
        if b6_multiseed_report_path is not None:
            if head != B7_HEAD:
                raise ValueError("b6_multiseed_report_path is only valid for the B7 residual head")
            b6_reference = load_b6_multiseed_report(b6_multiseed_report_path)
    elif head == B7_HEAD:
        raise ValueError("B7 oof_ridge_residual_mlp requires the RawDSP feature builder")
    feature_builder = feature_config.feature_builder
    cache_path = Path(cache_dir) if cache_dir is not None else default_cache_dir(dataset_dir, feature_builder)
    if isinstance(feature_config, MiniRocketFeatureConfig):
        feature_cache = build_minirocket_feature_cache(dataset_dir, cache_dir=cache_path, config=feature_config)
    else:
        feature_cache = build_tv3_physics_feature_cache(dataset_dir, cache_dir=cache_path, config=feature_config)
    train_matrix = load_cached_split_feature_matrix(dataset_dir, cache_path, split=train_split)
    # RocketFeatureCache 与 MiniRocketFeatureCache 字段兼容;统一进 RocketFeatureCache 供 payload 用
    if not isinstance(feature_cache, RocketFeatureCache):
        feature_cache = RocketFeatureCache(
            dataset_dir=feature_cache.dataset_dir,
            cache_dir=feature_cache.cache_dir,
            feature_config=feature_config,
            feature_names=feature_cache.feature_names,
            label_names=feature_cache.label_names,
            split_sequence_counts=feature_cache.split_sequence_counts,
        )
    resolved_mlp_config = mlp_config or MlpHeadConfig(device=device)
    model = _build_head(
        head,
        ridge_alphas=ridge_alphas,
        closed_form_alpha=closed_form_alpha,
        device=device,
        mlp_config=resolved_mlp_config,
        oof_folds=oof_folds,
        oof_seed=oof_seed,
    )
    fit_kwargs: dict[str, Any] = {"feature_names": train_matrix.feature_names}
    if head in {"mlp", B7_HEAD}:
        val_matrix = load_cached_split_feature_matrix(dataset_dir, cache_path, split="val")
        _validate_feature_contract(val_matrix, train_matrix)
        fit_kwargs["x_val"] = val_matrix.x
        fit_kwargs["y_val"] = val_matrix.y
        fit_kwargs["label_names"] = train_matrix.label_names
    model.fit(train_matrix.x, train_matrix.y, **fit_kwargs)

    evaluations: dict[str, SplitEvaluation] = {}
    for split_name in (train_split, *eval_splits):
        matrix = train_matrix if split_name == train_split else load_cached_split_feature_matrix(dataset_dir, cache_path, split=split_name)
        _validate_feature_contract(matrix, train_matrix)
        evaluations[split_name] = evaluate_regressor(
            model,
            matrix,
            split=split_name,
            composition_scheme="tunnel_ventilation",
        )

    diagnostics = _model_diagnostics(model, head=head, feature_names=train_matrix.feature_names, label_names=train_matrix.label_names)
    if head == B7_HEAD:
        assert b6_reference is not None
        assert raw_dsp_reference is not None
        diagnostics = {
            **diagnostics,
            "feature_builder": feature_builder,
            "feature_count": len(train_matrix.feature_names),
            "b1_reference": {
                "metrics_path": raw_dsp_reference["metrics_path"],
                "metrics_sha256": raw_dsp_reference["metrics_sha256"],
                "o2_r2": dict(raw_dsp_reference["o2_r2"]),
            },
            "b6_reference": {
                "report_path": b6_reference["report_path"],
                "report_sha256": b6_reference["report_sha256"],
                "o2_r2_means": dict(b6_reference["o2_r2_means"]),
                "verdict": b6_reference["verdict"],
            },
        }
    return RocketTrainingResult(
        head=head,
        dataset_dir=dataset_dir,
        cache_dir=cache_path,
        feature_cache=feature_cache,
        feature_names=train_matrix.feature_names,
        label_names=train_matrix.label_names,
        train_split=train_split,
        evaluations=evaluations,
        diagnostics=diagnostics,
        raw_dsp_provenance=raw_dsp_provenance,
        raw_dsp_fidelity=raw_dsp_fidelity,
        raw_dsp_reference=raw_dsp_reference,
        b6_reference=b6_reference,
    )


def rocket_training_payload(result: RocketTrainingResult) -> dict[str, Any]:
    feature_builder = result.feature_cache.feature_config.feature_builder
    payload: dict[str, Any] = {
        "dataset_dir": str(result.dataset_dir),
        "cache_dir": str(result.cache_dir),
        "head": result.head,
        "train_split": result.train_split,
        "feature_builder": feature_builder,
        "feature_config": asdict(result.feature_cache.feature_config),
        "feature_names": list(result.feature_names),
        "feature_count": len(result.feature_names),
        "label_names": list(result.label_names),
        "diagnostics": result.diagnostics,
        "evaluations": {},
    }
    for split_name, split_eval in result.evaluations.items():
        payload["evaluations"][split_name] = {
            "metrics": asdict(split_eval.metrics),
            "component_metrics": {name: asdict(metric) for name, metric in split_eval.component_metrics.items()},
            "conditional_metrics": conditional_metrics_to_payload(split_eval.conditional_metrics),
            "sum_abs_error": split_eval.sum_abs_error,
            "sequence_count": len(split_eval.sequence_ids),
        }
    if feature_builder == RAW_DSP_FEATURE_BUILDER:
        assert result.raw_dsp_provenance is not None
        payload["raw_dsp_provenance"] = result.raw_dsp_provenance
        if result.raw_dsp_fidelity is not None:
            payload["raw_dsp_fidelity"] = result.raw_dsp_fidelity
        if result.raw_dsp_reference is not None:
            payload["o2_audit"] = _build_o2_audit(result.evaluations, result.raw_dsp_reference)
        if result.b6_reference is not None:
            payload["b6_reference"] = result.b6_reference
            payload["o2_audit"] = {
                **payload.get("o2_audit", {}),
                "delta_vs_b6_o2_r2_means": {
                    split_name: _component_r2(result.evaluations, split_name, "x_O2") - mean_value
                    for split_name, mean_value in result.b6_reference["o2_r2_means"].items()
                    if split_name in result.evaluations
                },
            }
    return payload


def load_b6_multiseed_report(report_path: Path | str) -> dict[str, Any]:
    """Validate the frozen B6 multiseed replication report used as B7 paired baseline."""
    report_path = Path(report_path)
    payload = _read_json_object(report_path, description="B6 multiseed report")
    groups = payload.get("groups")
    if not isinstance(groups, dict) or "b6" not in groups:
        raise ValueError("B6 multiseed report is missing groups.b6")
    b6 = groups["b6"]
    if not isinstance(b6, dict):
        raise ValueError("B6 multiseed report groups.b6 must be an object")
    if b6.get("verdict") != "stable_pass":
        raise ValueError(f"B6 multiseed report verdict must be 'stable_pass', got {b6.get('verdict')!r}")
    completed_seeds = tuple(int(seed) for seed in b6.get("completed_seeds", ()))
    if completed_seeds != B6_MULTISEED_EXPECTED_SEEDS:
        raise ValueError(
            f"B6 multiseed report completed_seeds must be {B6_MULTISEED_EXPECTED_SEEDS}, got {completed_seeds}"
        )
    stats = b6.get("o2_r2_stats")
    if not isinstance(stats, dict):
        raise ValueError("B6 multiseed report is missing o2_r2_stats")
    o2_r2_means: dict[str, float] = {}
    for split_name, expected_mean in B6_MULTISEED_EXPECTED_MEANS.items():
        split_stats = stats.get(split_name)
        if not isinstance(split_stats, dict) or "mean" not in split_stats:
            raise ValueError(f"B6 multiseed report is missing o2_r2_stats.{split_name}.mean")
        mean_value = float(split_stats["mean"])
        if abs(mean_value - expected_mean) > 1e-4:
            raise ValueError(
                f"B6 multiseed report {split_name} mean {mean_value} does not match frozen {expected_mean}"
            )
        o2_r2_means[split_name] = mean_value
    val_stats = stats.get("val")
    if isinstance(val_stats, dict) and "mean" in val_stats:
        o2_r2_means["val"] = float(val_stats["mean"])
    return {
        "report_path": str(report_path),
        "report_sha256": _file_sha256(report_path),
        "verdict": b6["verdict"],
        "completed_seeds": list(completed_seeds),
        "o2_r2_means": o2_r2_means,
        "pass_count": b6.get("pass_count"),
    }


def load_raw_dsp_provenance(dataset_dir: Path | str) -> dict[str, Any]:
    """Load and validate formal RawDSP frame-cache tracing fields for metrics.json."""
    dataset_dir = Path(dataset_dir)
    cache_dir = dataset_dir / RAW_DSP_FRAME_CACHE_ROOT
    manifest_path = cache_dir / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"missing RawDSP frame cache manifest: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(manifest, dict):
        raise ValueError(f"RawDSP cache manifest must be a JSON object: {manifest_path}")
    for key, expected in FORMAL_RAW_DSP_MANIFEST_CONTRACT.items():
        if manifest.get(key) != expected:
            raise ValueError(
                f"RawDSP cache manifest field {key!r} must be {expected!r}, got {manifest.get(key)!r}"
            )
    build_signature = manifest.get("build_signature")
    template_digest = manifest.get("template_digest")
    if not isinstance(build_signature, str) or not build_signature:
        raise ValueError("RawDSP cache manifest is missing build_signature")
    if not isinstance(template_digest, str) or not template_digest:
        raise ValueError("RawDSP cache manifest is missing template_digest")
    return {
        "cache_dir": str(cache_dir),
        "manifest_path": str(manifest_path),
        "build_signature": build_signature,
        "template_digest": template_digest,
        "template_mode": manifest["template_mode"],
        "template_source_split": manifest["template_source_split"],
        "diagnostic_only": manifest["diagnostic_only"],
        "schema_version": manifest["schema_version"],
        "complete_dataset": manifest["complete_dataset"],
    }


def load_raw_dsp_fidelity(
    metrics_path: Path | str,
    *,
    dataset_dir: Path,
    provenance: dict[str, Any],
) -> dict[str, Any]:
    metrics_path = Path(metrics_path)
    payload = _read_json_object(metrics_path, description="RawDSP fidelity metrics")
    if payload.get("schema_version") != RAW_DSP_FIDELITY_SCHEMA_VERSION:
        raise ValueError("RawDSP fidelity metrics schema_version is not supported")
    if payload.get("status") != "passed":
        raise ValueError("RawDSP fidelity metrics status must be 'passed' before B6 training")
    source = payload.get("source")
    if not isinstance(source, dict):
        raise ValueError("RawDSP fidelity metrics are missing source tracing")
    _validate_dataset_path(source.get("dataset_dir"), dataset_dir, description="RawDSP fidelity metrics")
    _validate_provenance_match(source, provenance, description="RawDSP fidelity metrics")
    return {
        "metrics_path": str(metrics_path),
        "metrics_sha256": _file_sha256(metrics_path),
        "status": payload["status"],
        "schema_version": payload["schema_version"],
        "cache_build_signature": source["cache_build_signature"],
        "template_digest": source["template_digest"],
    }


def load_raw_dsp_reference_metrics(
    metrics_path: Path | str,
    *,
    dataset_dir: Path,
    feature_config: RocketFeatureConfig,
    provenance: dict[str, Any],
) -> dict[str, Any]:
    metrics_path = Path(metrics_path)
    payload = _read_json_object(metrics_path, description="RawDSP reference metrics")
    if payload.get("feature_builder") != RAW_DSP_FEATURE_BUILDER:
        raise ValueError("RawDSP reference metrics use a different feature builder")
    _validate_dataset_path(payload.get("dataset_dir"), dataset_dir, description="RawDSP reference metrics")
    if _normalize_json(payload.get("feature_config")) != _normalize_json(asdict(feature_config)):
        raise ValueError("RawDSP reference metrics feature_config does not match the current run")
    reference_provenance = payload.get("raw_dsp_provenance")
    if not isinstance(reference_provenance, dict):
        raise ValueError("RawDSP reference metrics are missing raw_dsp_provenance")
    _validate_provenance_match(reference_provenance, provenance, description="RawDSP reference metrics")
    evaluations = payload.get("evaluations")
    if not isinstance(evaluations, dict):
        raise ValueError("RawDSP reference metrics are missing evaluations")
    o2_r2 = {
        split_name: _metrics_component_r2(evaluations, split_name, "x_O2")
        for split_name in ("val", "test", "extrapolation")
    }
    return {
        "metrics_path": str(metrics_path),
        "metrics_sha256": _file_sha256(metrics_path),
        "o2_r2": o2_r2,
        "cache_build_signature": reference_provenance["build_signature"],
        "template_digest": reference_provenance["template_digest"],
    }


def _component_r2(evaluations: dict[str, SplitEvaluation], split_name: str, component: str) -> float:
    split_eval = evaluations[split_name]
    return float(split_eval.component_metrics[component].r2)


def _build_o2_audit(
    evaluations: dict[str, SplitEvaluation],
    reference_metrics: dict[str, Any],
) -> dict[str, Any]:
    if "train" not in evaluations or "val" not in evaluations:
        raise ValueError("RawDSP O2 audit requires train and val evaluations")
    train_o2 = _component_r2(evaluations, "train", "x_O2")
    val_o2 = _component_r2(evaluations, "val", "x_O2")
    deltas_vs_b1: dict[str, float] = {}
    for split_name, baseline in reference_metrics["o2_r2"].items():
        if split_name not in evaluations:
            continue
        deltas_vs_b1[split_name] = _component_r2(evaluations, split_name, "x_O2") - baseline
    return {
        "train_val_o2_r2_gap": train_o2 - val_o2,
        "reference_o2_r2": dict(reference_metrics["o2_r2"]),
        "reference_metrics_path": reference_metrics["metrics_path"],
        "reference_metrics_sha256": reference_metrics["metrics_sha256"],
        "delta_vs_b1_o2_r2": deltas_vs_b1,
    }


def _read_json_object(path: Path, *, description: str) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"missing {description}: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{description} must contain a JSON object: {path}")
    return payload


def _validate_dataset_path(value: Any, dataset_dir: Path, *, description: str) -> None:
    if not isinstance(value, str) or Path(value).resolve() != dataset_dir.resolve():
        raise ValueError(f"{description} dataset_dir does not match the current run")


def _validate_provenance_match(
    source: dict[str, Any],
    provenance: dict[str, Any],
    *,
    description: str,
) -> None:
    expected = {
        "cache_build_signature": provenance["build_signature"],
        "template_digest": provenance["template_digest"],
    }
    aliases = {"cache_build_signature": "build_signature", "template_digest": "template_digest"}
    for key, expected_value in expected.items():
        source_key = key if key in source else aliases[key]
        if source.get(source_key) != expected_value:
            raise ValueError(f"{description} {key} does not match the current RawDSP cache")


def _metrics_component_r2(evaluations: dict[str, Any], split_name: str, component: str) -> float:
    try:
        value = evaluations[split_name]["component_metrics"][component]["r2"]
    except (KeyError, TypeError) as exc:
        raise ValueError(
            f"RawDSP reference metrics are missing {split_name}.{component}.r2"
        ) from exc
    return float(value)


def _normalize_json(value: Any) -> Any:
    return json.loads(json.dumps(value, sort_keys=True))


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_rocket_training_payload(result: RocketTrainingResult, output_path: Path | str) -> dict[str, Any]:
    payload = rocket_training_payload(result)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return payload


def _build_head(
    head: str,
    *,
    ridge_alphas: tuple[float, ...],
    closed_form_alpha: float,
    device: str = "auto",
    mlp_config: MlpHeadConfig | None = None,
    oof_folds: int = DEFAULT_OOF_FOLDS,
    oof_seed: int = DEFAULT_OOF_SEED,
) -> Any:
    if head == "ridgecv":
        return ScaledRidgeCVRegressor(alphas=ridge_alphas)
    if head == "ridge_closed_form":
        return _ScaledClosedFormRidgeRegressor(alpha=closed_form_alpha)
    if head == "tabpfn":
        return _TabPFNMultiRegressor(device=device)
    if head == "mlp":
        return _ScaledMLPRegressor(config=mlp_config or MlpHeadConfig(device=device))
    if head == B7_HEAD:
        return OofRidgeResidualMlpRegressor(
            ridge_alphas=ridge_alphas,
            mlp_config=mlp_config or MlpHeadConfig(device=device),
            oof_folds=oof_folds,
            oof_seed=oof_seed,
        )
    raise ValueError(
        f"unsupported rocket head {head!r}. "
        "available=('ridgecv', 'ridge_closed_form', 'tabpfn', 'mlp', 'oof_ridge_residual_mlp')"
    )


def _validate_feature_contract(matrix: MLFeatureMatrix, reference: MLFeatureMatrix) -> None:
    if matrix.feature_names != reference.feature_names:
        raise ValueError("cached rocket feature names must match across splits")
    if matrix.label_names != reference.label_names:
        raise ValueError("cached rocket label names must match across splits")


def _model_diagnostics(
    model: Any,
    *,
    head: str,
    feature_names: tuple[str, ...],
    label_names: tuple[str, ...],
) -> dict[str, Any]:
    if head == "ridgecv":
        coef = np.asarray(model.model.coef_, dtype=np.float64)
        selected_alpha = float(model.model.alpha_)
    elif head == "ridge_closed_form":
        assert model.model.coef_ is not None
        coef = np.asarray(model.model.coef_.T, dtype=np.float64)
        selected_alpha = float(model.model.alpha)
    elif head == "tabpfn":
        return {"note": "TabPFN has no linear coefficients; diagnostics unavailable"}
    elif head == "mlp":
        return {
            "model_config": asdict(model.config),
            "hidden_dims": list(model.hidden_dims),
            "parameter_count": model.parameter_count,
            "best_epoch": model.best_epoch,
            "best_val_o2_r2": model.best_val_o2_r2,
            "standardize_targets": model.config.standardize_targets,
        }
    elif head == B7_HEAD:
        return dict(model.diagnostics)
    else:
        raise ValueError(f"unsupported diagnostics head {head!r}")
    if coef.ndim == 1:
        coef = coef.reshape(1, -1)
    return {
        "selected_alpha": selected_alpha,
        "coef_norms": {
            label_names[index]: float(np.linalg.norm(coef[index]))
            for index in range(coef.shape[0])
        },
        "top_feature_groups": {
            label_names[index]: _top_feature_groups(coef[index], feature_names)
            for index in range(coef.shape[0])
        },
    }


def _top_feature_groups(coefficients: np.ndarray, feature_names: tuple[str, ...], limit: int = 5) -> list[dict[str, Any]]:
    group_scores: dict[str, float] = {}
    for coefficient, feature_name in zip(coefficients, feature_names, strict=True):
        group_name = _feature_group_name(feature_name)
        group_scores[group_name] = group_scores.get(group_name, 0.0) + float(abs(coefficient))
    ordered = sorted(group_scores.items(), key=lambda item: item[1], reverse=True)
    return [{"group": group, "abs_coef_sum": score} for group, score in ordered[:limit]]


def _feature_group_name(feature_name: str) -> str:
    if "|" not in feature_name:
        return feature_name
    _window, remainder = feature_name.split("|", 1)
    parts = remainder.split(":")
    if len(parts) <= 2:
        return remainder
    return ":".join(parts[:-1])
