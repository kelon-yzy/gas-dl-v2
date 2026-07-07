from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

from sg.sim.core.schema import COMPONENT_FIELDS


R2_ZERO_VARIANCE_EPSILON = 1e-12
"""Total sum-of-squares below this treats the target as constant for R2."""


@dataclass(frozen=True, slots=True)
class RegressionMetrics:
    """四组分浓度回归的整体指标（pooled MAE / RMSE / R2）。

    ml（NumPy）与 dl（PyTorch）共用此数据结构与零方差阈值，计算层各自实现，
    避免指标 schema 双份维护（见 KARPATHY_REVIEW 2.2）。新增指标字段时只改这里。
    """

    mae: float
    rmse: float
    r2: float


@dataclass(frozen=True, slots=True)
class CompositionalMetrics:
    """Composition-aware metrics computed in log-ratio geometry."""

    aitchison_mean: float
    aitchison_rmse: float


def conditional_component_metrics(
    y_pred: np.ndarray,
    y_true: np.ndarray,
    component_names: tuple[str, ...] = COMPONENT_FIELDS,
    *,
    bin_count: int = 4,
    bin_components: tuple[str, ...] | None = None,
) -> dict[str, dict[str, object]]:
    """Compute conditional regression metrics binned by selected components.

    Parameters
    ----------
    bin_components:
        Components to bin by. Defaults to ("x_CO", "x_CH4") for syngas
        labels, and falls back to ("x_N2", "x_CH4") only when x_CO is absent.
        The returned dict key convention strips the ``"x_"`` prefix and adds
        ``"_bins"`` (e.g. ``"x_CO"`` → ``"co_bins"``).
    """
    if bin_components is None:
        primary = "x_CO" if "x_CO" in component_names else "x_N2"
        bin_components = (primary, "x_CH4")
    result: dict[str, dict[str, object]] = {}
    for component_name in bin_components:
        if not component_name.startswith("x_"):
            raise ValueError(f"bin component name must start with 'x_', got {component_name!r}")
        group_key = f"{component_name[2:].lower()}_bins"
        result[group_key] = _binned_component_metrics(
            y_pred,
            y_true,
            component_names,
            component_name=component_name,
            bin_count=bin_count,
        )
    return result


def conditional_metrics_to_payload(value: dict[str, dict[str, object]]) -> dict[str, object]:
    """Convert conditional metrics with RegressionMetrics values to JSON-safe dictionaries."""
    payload: dict[str, object] = {}
    for group_name, group in value.items():
        bins = {}
        for bin_name, bin_entry in group["bins"].items():
            entry = dict(bin_entry)
            if "metrics" in entry:
                entry["metrics"] = asdict(entry["metrics"])
            if "component_metrics" in entry:
                entry["component_metrics"] = {
                    key: asdict(metric)
                    for key, metric in entry["component_metrics"].items()
                }
            bins[bin_name] = entry
        payload[group_name] = {
            "component": group["component"],
            "bins": bins,
        }
    return payload


def _binned_component_metrics(
    y_pred: np.ndarray,
    y_true: np.ndarray,
    component_names: tuple[str, ...],
    *,
    component_name: str,
    bin_count: int,
) -> dict[str, object]:
    pred = _as_2d_array(y_pred, name="y_pred")
    true = _as_2d_array(y_true, name="y_true")
    if pred.shape != true.shape:
        raise ValueError(f"Prediction and target shapes must match, got {pred.shape} and {true.shape}")
    if pred.shape[1] != len(component_names):
        raise ValueError(
            f"component_names length {len(component_names)} does not match prediction channels {pred.shape[1]}"
        )
    if bin_count < 1:
        raise ValueError(f"bin_count must be >= 1, got {bin_count}")

    component_index = _component_index(component_names, component_name)
    values = true[:, component_index]
    low = float(values.min())
    high = float(values.max())
    if high == low:
        edges = np.array([low, high], dtype=np.float64)
    else:
        edges = np.linspace(low, high, bin_count + 1, dtype=np.float64)

    bins: dict[str, object] = {}
    for index in range(len(edges) - 1):
        left = float(edges[index])
        right = float(edges[index + 1])
        if index == len(edges) - 2:
            mask = (values >= left) & (values <= right)
        else:
            mask = (values >= left) & (values < right)
        count = int(np.count_nonzero(mask))
        entry: dict[str, object] = {
            "range": [left, right],
            "count": count,
        }
        if count > 0:
            entry["metrics"] = _regression_metrics_np(pred[mask], true[mask])
            entry["component_metrics"] = {
                name: _regression_metrics_np(pred[mask, idx : idx + 1], true[mask, idx : idx + 1])
                for idx, name in enumerate(component_names)
            }
        bins[f"{left:.6g}_{right:.6g}"] = entry
    return {
        "component": component_name,
        "bins": bins,
    }


def _regression_metrics_np(y_pred: np.ndarray, y_true: np.ndarray) -> RegressionMetrics:
    pred = _as_2d_array(y_pred, name="y_pred")
    true = _as_2d_array(y_true, name="y_true")
    if pred.shape != true.shape:
        raise ValueError(f"Prediction and target shapes must match, got {pred.shape} and {true.shape}")
    err = pred - true
    mae = float(np.mean(np.abs(err)))
    rmse = float(np.sqrt(np.mean(err * err)))
    ss_res = float(np.sum(err * err))
    centered = true - np.mean(true, axis=0, keepdims=True)
    ss_tot = float(np.sum(centered * centered))
    if ss_tot < R2_ZERO_VARIANCE_EPSILON:
        r2 = 1.0 if ss_res < R2_ZERO_VARIANCE_EPSILON else 0.0
    else:
        r2 = 1.0 - ss_res / ss_tot
    return RegressionMetrics(mae=mae, rmse=rmse, r2=float(r2))


def _as_2d_array(values: np.ndarray, *, name: str) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64)
    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)
    if arr.ndim != 2:
        raise ValueError(f"{name} must be a 1D or 2D array, got ndim={arr.ndim}")
    if arr.size == 0:
        raise ValueError(f"{name} must not be empty")
    return arr


def _component_index(component_names: tuple[str, ...], component_name: str) -> int:
    try:
        return component_names.index(component_name)
    except ValueError as exc:
        raise ValueError(f"Unknown component {component_name!r}. Available: {component_names}") from exc
