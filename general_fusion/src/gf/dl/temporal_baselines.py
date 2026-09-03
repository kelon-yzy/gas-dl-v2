"""A2-DYN 时序基线与 pilot 线性探针。

正式基线和 pilot 探针使用不同 ID。pilot 探针只用于数据资格预检，
不能冒充正式 MLP、GBDT、TCN、GRU 或数值 oracle。
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from gf.dl.evaluation import evaluate_predictions


PILOT_PROBE_IDS = (
    "P-B-LAST-LS",
    "P-B-EWMA-LS",
    "P-B-STAT-LS",
    "P-O-KIN-LS",
)


def pilot_probe_feature_vector(
    probe_id: str,
    observed: np.ndarray,
    clean: np.ndarray,
    layers: Any,
    tau_transport: Mapping[str, float],
    *,
    ewma_alpha: float,
) -> np.ndarray:
    signal = np.asarray(observed, dtype=np.float64)
    clean_signal = np.asarray(clean, dtype=np.float64)
    if signal.ndim != 2 or signal.shape[0] == 0 or clean_signal.shape != signal.shape:
        raise ValueError("observed and clean must be aligned non-empty (time, channel) arrays")
    endpoint = signal[-1]
    if probe_id == "P-B-LAST-LS":
        return endpoint.copy()
    if probe_id == "P-B-EWMA-LS":
        alpha = float(ewma_alpha)
        if not 0.0 < alpha <= 1.0:
            raise ValueError("ewma_alpha must lie in (0,1]")
        ewma = signal[0].copy()
        previous = ewma.copy()
        for value in signal[1:]:
            previous = ewma
            ewma = alpha * value + (1.0 - alpha) * ewma
        return np.concatenate((ewma, ewma - previous, ewma - signal[0]))
    if probe_id == "P-B-STAT-LS":
        normalized_time = np.linspace(0.0, 1.0, signal.shape[0])
        slopes = (
            np.asarray(
                [np.polyfit(normalized_time, signal[:, channel], 1)[0] for channel in range(signal.shape[1])]
            )
            if signal.shape[0] > 1
            else np.zeros(signal.shape[1], dtype=np.float64)
        )
        auc = np.trapezoid(signal, normalized_time, axis=0) if signal.shape[0] > 1 else signal[0]
        quantiles = np.quantile(signal, [0.25, 0.5, 0.75], axis=0).reshape(-1)
        crossings = _observed_half_range_crossing(signal)
        return np.concatenate(
            (
                np.mean(signal, axis=0),
                np.std(signal, axis=0),
                slopes,
                auc,
                quantiles,
                crossings,
            )
        )
    if probe_id == "P-O-KIN-LS":
        local_end = np.asarray(
            [
                layers.local_composition_pct["ultrasonic_tof"][signal.shape[0] - 1, 0],
                layers.local_composition_pct["thermal_conductivity_voltage"][signal.shape[0] - 1, 2],
                layers.local_composition_pct["ndir_co2_voltage"][signal.shape[0] - 1, 2],
            ],
            dtype=np.float64,
        )
        tau = np.asarray(
            [tau_transport[sensor_id] for sensor_id in (
                "ultrasonic_tof",
                "thermal_conductivity_voltage",
                "ndir_co2_voltage",
            )],
            dtype=np.float64,
        )
        return np.concatenate(
            (
                clean_signal[-1],
                local_end,
                layers.chamber_composition_pct[signal.shape[0] - 1],
                tau,
            )
        )
    raise ValueError(f"unsupported pilot probe {probe_id!r}")


def fit_pilot_linear_probes(
    feature_store: Mapping[str, Any],
    target_store: Mapping[str, Any],
    group_store: Mapping[str, Any],
    *,
    target_ranges: Sequence[float],
) -> dict[str, Any]:
    """拟合 pilot 最小二乘探针并复用统一 group-level 指标。"""

    metrics: dict[str, Any] = {}
    for probe_id, horizons in feature_store.items():
        metrics[probe_id] = {}
        for horizon, split_features in horizons.items():
            train_x = np.asarray(split_features["train"], dtype=np.float64)
            train_y = np.asarray(target_store[horizon]["train"], dtype=np.float64)
            metrics[probe_id][horizon] = {}
            if train_x.ndim != 2 or train_y.ndim != 2 or train_x.shape[0] == 0:
                for split in ("val", "stress_val"):
                    metrics[probe_id][horizon][split] = None
                continue
            mean = train_x.mean(axis=0)
            scale = train_x.std(axis=0)
            scale[scale < 1.0e-12] = 1.0
            design = np.column_stack((np.ones(train_x.shape[0]), (train_x - mean) / scale))
            coefficients, _, rank, _ = np.linalg.lstsq(design, train_y, rcond=None)
            for split in ("val", "stress_val"):
                eval_x = np.asarray(split_features[split], dtype=np.float64)
                eval_y = np.asarray(target_store[horizon][split], dtype=np.float64)
                groups = np.asarray(group_store[horizon][split], dtype=object)
                if eval_x.ndim != 2 or eval_y.ndim != 2 or eval_x.shape[0] == 0:
                    metrics[probe_id][horizon][split] = None
                    continue
                predictions = np.column_stack(
                    (np.ones(eval_x.shape[0]), (eval_x - mean) / scale)
                ) @ coefficients
                result = evaluate_predictions(
                    eval_y,
                    predictions,
                    groups,
                    np.arange(eval_x.shape[0]),
                    target_ranges=target_ranges,
                )
                result["samples"] = int(eval_x.shape[0])
                result["design_rank"] = int(rank)
                metrics[probe_id][horizon][split] = result
    return metrics


def _observed_half_range_crossing(signal: np.ndarray) -> np.ndarray:
    result = np.ones(signal.shape[1], dtype=np.float64)
    for channel in range(signal.shape[1]):
        values = signal[:, channel]
        baseline = float(values[0])
        observed_min = float(values.min())
        observed_max = float(values.max())
        target = baseline + 0.5 * (
            observed_max - baseline
            if abs(observed_max - baseline) >= abs(observed_min - baseline)
            else observed_min - baseline
        )
        if target >= baseline:
            indices = np.flatnonzero(values >= target)
        else:
            indices = np.flatnonzero(values <= target)
        if indices.size:
            result[channel] = float(indices[0]) / max(signal.shape[0] - 1, 1)
    return result


__all__ = [
    "PILOT_PROBE_IDS",
    "fit_pilot_linear_probes",
    "pilot_probe_feature_vector",
]
