"""Deployable e1d_sb inference helpers (no LS): fit / predict / artifact export.

Uses ``e1d_sb_cal_plus_corr_psr_snr_v1`` only. Does not open E2. Does not replace B7.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from tv3.ml.e1d_sb_features import E1DSB_FEATURE_BUILDER, E1DSB_SPEC_NAME
from tv3.ml.features import MLFeatureMatrix
from tv3.ml.ridge_head import ScaledRidgeCVRegressor


@dataclass(frozen=True, slots=True)
class E1dSBInferenceArtifact:
    feature_builder: str
    spec_name: str
    feature_names: tuple[str, ...]
    label_names: tuple[str, ...]
    selected_alpha: float
    scaler_mean: np.ndarray
    scaler_scale: np.ndarray
    ridge_coef: np.ndarray
    ridge_intercept: np.ndarray
    ls_promoted: bool = False
    default_head_remains: str = "B7"
    e2_allowed: bool = False

    def to_payload(self) -> dict[str, Any]:
        return {
            "feature_builder": self.feature_builder,
            "spec_name": self.spec_name,
            "feature_names": list(self.feature_names),
            "label_names": list(self.label_names),
            "selected_alpha": self.selected_alpha,
            "scaler_mean": self.scaler_mean.astype(np.float64).tolist(),
            "scaler_scale": self.scaler_scale.astype(np.float64).tolist(),
            "ridge_coef": self.ridge_coef.astype(np.float64).tolist(),
            "ridge_intercept": self.ridge_intercept.astype(np.float64).tolist(),
            "ls_promoted": self.ls_promoted,
            "default_head_remains": self.default_head_remains,
            "e2_allowed": self.e2_allowed,
        }


def fit_e1d_sb_inference(
    train: MLFeatureMatrix,
    *,
    alphas: Sequence[float],
) -> tuple[ScaledRidgeCVRegressor, E1dSBInferenceArtifact]:
    """Fit train-only StandardScaler + RidgeCV and package a deployable artifact."""
    if train.x.ndim != 2 or train.y.ndim != 2:
        raise ValueError("train matrix must provide 2D x and y")
    if not any("ultrasonic_snr_db" in name for name in train.feature_names):
        raise ValueError("e1d_sb inference requires ultrasonic_snr_db features")
    if any("snr_weighted_ls" in name for name in train.feature_names):
        raise ValueError("deployable e1d_sb inference must not include LS ablation features")

    probe = ScaledRidgeCVRegressor(alphas=tuple(float(a) for a in alphas)).fit(
        train.x, train.y
    )
    artifact = E1dSBInferenceArtifact(
        feature_builder=E1DSB_FEATURE_BUILDER,
        spec_name=E1DSB_SPEC_NAME,
        feature_names=tuple(train.feature_names),
        label_names=tuple(train.label_names),
        selected_alpha=float(probe.selected_alpha),
        scaler_mean=np.asarray(probe.scaler.mean_, dtype=np.float64),
        scaler_scale=np.asarray(probe.scaler.scale_, dtype=np.float64),
        ridge_coef=np.asarray(probe.model.coef_, dtype=np.float64),
        ridge_intercept=np.asarray(probe.model.intercept_, dtype=np.float64),
    )
    return probe, artifact


def predict_with_artifact(artifact: E1dSBInferenceArtifact, x: np.ndarray) -> np.ndarray:
    """Predict raw3 from features using a frozen artifact (no re-fit)."""
    values = np.asarray(x, dtype=np.float64)
    if values.ndim != 2:
        raise ValueError(f"x must be 2D, got {values.shape}")
    if values.shape[1] != len(artifact.feature_names):
        raise ValueError(
            f"feature width {values.shape[1]} != artifact {len(artifact.feature_names)}"
        )
    scale = np.asarray(artifact.scaler_scale, dtype=np.float64)
    if np.any(scale == 0.0):
        raise ValueError("artifact scaler_scale contains zeros")
    x_scaled = (values - np.asarray(artifact.scaler_mean, dtype=np.float64)) / scale
    coef = np.asarray(artifact.ridge_coef, dtype=np.float64)
    intercept = np.asarray(artifact.ridge_intercept, dtype=np.float64)
    # sklearn multi-output: coef shape (n_targets, n_features)
    if coef.ndim == 1:
        pred = x_scaled @ coef + float(intercept)
        return pred.astype(np.float32).reshape(-1, 1)
    pred = x_scaled @ coef.T + intercept
    return pred.astype(np.float32, copy=False)


def write_inference_artifact(path: Path | str, artifact: E1dSBInferenceArtifact) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(artifact.to_payload(), indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return path


def load_inference_artifact(path: Path | str) -> E1dSBInferenceArtifact:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError(f"artifact JSON root must be an object: {path}")
    if payload.get("feature_builder") != E1DSB_FEATURE_BUILDER:
        raise ValueError(
            f"artifact feature_builder must be {E1DSB_FEATURE_BUILDER!r}, "
            f"got {payload.get('feature_builder')!r}"
        )
    if payload.get("ls_promoted") is True:
        raise ValueError("refusing to load LS-promoted artifact into deployable e1d_sb path")
    return E1dSBInferenceArtifact(
        feature_builder=str(payload["feature_builder"]),
        spec_name=str(payload["spec_name"]),
        feature_names=tuple(str(name) for name in payload["feature_names"]),
        label_names=tuple(str(name) for name in payload["label_names"]),
        selected_alpha=float(payload["selected_alpha"]),
        scaler_mean=np.asarray(payload["scaler_mean"], dtype=np.float64),
        scaler_scale=np.asarray(payload["scaler_scale"], dtype=np.float64),
        ridge_coef=np.asarray(payload["ridge_coef"], dtype=np.float64),
        ridge_intercept=np.asarray(payload["ridge_intercept"], dtype=np.float64),
        ls_promoted=bool(payload.get("ls_promoted", False)),
        default_head_remains=str(payload.get("default_head_remains", "B7")),
        e2_allowed=bool(payload.get("e2_allowed", False)),
    )


def artifact_summary(artifact: E1dSBInferenceArtifact) -> dict[str, Any]:
    payload = asdict(artifact)
    payload["feature_count"] = len(artifact.feature_names)
    payload["scaler_mean"] = f"shape={artifact.scaler_mean.shape}"
    payload["scaler_scale"] = f"shape={artifact.scaler_scale.shape}"
    payload["ridge_coef"] = f"shape={artifact.ridge_coef.shape}"
    payload["ridge_intercept"] = f"shape={artifact.ridge_intercept.shape}"
    return payload
