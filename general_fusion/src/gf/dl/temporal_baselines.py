"""A2-DYN 时序基线与 pilot 线性探针。

正式基线和 pilot 探针使用不同 ID。pilot 探针只用于数据资格预检，
不能冒充正式 MLP、GBDT、TCN、GRU 或数值 oracle。
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import time
from typing import Any

import numpy as np
import torch
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.multioutput import MultiOutputRegressor
from sklearn.neural_network import MLPRegressor
from torch import nn
from torch.nn import functional as F

from gf.dl.evaluation import evaluate_predictions


PILOT_PROBE_IDS = (
    "P-B-LAST-LS",
    "P-B-EWMA-LS",
    "P-B-STAT-LS",
    "P-O-KIN-LS",
)

FORMAL_BASELINE_IDS = (
    "B-LAST",
    "B-DELTA",
    "B-EWMA",
    "B-STAT",
    "B-TCN",
    "B-GRU",
    "B-STEADY",
)
FORMAL_HORIZON_IDS = ("P005", "P015", "P030", "P060", "P120", "P150", "FULL")
EWMA_ALPHA_CANDIDATES = (0.10, 0.20, 0.40)
TEMPORAL_SEQUENCE_LENGTH = 64


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


def formal_feature_vector(
    model_id: str,
    observed_prefix: np.ndarray,
    *,
    ewma_alpha: float = 0.20,
    baseline_window_samples: int = 25,
) -> np.ndarray:
    """仅从当前观测前缀构造正式基线特征。"""

    signal = np.asarray(observed_prefix, dtype=np.float64)
    if signal.ndim != 2 or signal.shape[0] == 0 or signal.shape[1] != 3:
        raise ValueError("observed_prefix must have shape (time, 3) and be non-empty")
    if not np.isfinite(signal).all():
        raise ValueError("observed_prefix must contain finite values")
    endpoint = signal[-1]
    if model_id in {"B-LAST", "B-STEADY", "O-EQ"}:
        return endpoint.copy()
    baseline_count = min(max(int(baseline_window_samples), 1), signal.shape[0])
    baseline = np.mean(signal[:baseline_count], axis=0)
    if model_id == "B-DELTA":
        return np.concatenate((baseline, endpoint, endpoint - baseline))
    if model_id == "B-EWMA":
        alpha = float(ewma_alpha)
        if not 0.0 < alpha <= 1.0:
            raise ValueError("ewma_alpha must lie in (0,1]")
        ewma = signal[0].copy()
        previous = ewma.copy()
        for value in signal[1:]:
            previous = ewma
            ewma = alpha * value + (1.0 - alpha) * ewma
        return np.concatenate((ewma, ewma - previous, ewma - signal[0]))
    if model_id == "B-STAT":
        normalized_time = np.linspace(0.0, 1.0, signal.shape[0])
        if signal.shape[0] > 1:
            slopes = np.asarray(
                [
                    np.polyfit(normalized_time, signal[:, channel], 1)[0]
                    for channel in range(signal.shape[1])
                ],
                dtype=np.float64,
            )
            auc = np.trapezoid(signal, normalized_time, axis=0)
        else:
            slopes = np.zeros(signal.shape[1], dtype=np.float64)
            auc = signal[0].copy()
        quantiles = np.quantile(signal, [0.25, 0.50, 0.75], axis=0).reshape(-1)
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
    raise ValueError(f"unsupported formal feature model {model_id!r}")


def formal_feature_matrix(
    model_id: str,
    signals: np.ndarray,
    rows: Sequence[int] | np.ndarray,
    endpoints: Sequence[int] | np.ndarray,
    *,
    ewma_alpha: float = 0.20,
    baseline_window_samples: int = 25,
) -> np.ndarray:
    """按给定数据行和 horizon 端点保持顺序构造特征矩阵。"""

    values = np.asarray(signals, dtype=np.float32)
    row_values = np.asarray(rows, dtype=np.int64)
    endpoint_values = np.asarray(endpoints, dtype=np.int64)
    if values.ndim != 4 or values.shape[1] != 3 or values.shape[3] != 1:
        raise ValueError("signals must have shape (N, 3, T, 1)")
    if row_values.ndim != 1 or endpoint_values.shape != row_values.shape:
        raise ValueError("rows and endpoints must be aligned one-dimensional arrays")
    widths = {"B-LAST": 3, "B-STEADY": 3, "O-EQ": 3, "B-DELTA": 9, "B-EWMA": 9, "B-STAT": 24}
    if model_id not in widths:
        raise ValueError(f"unsupported formal feature model {model_id!r}")
    if row_values.size == 0:
        return np.empty((0, widths[model_id]), dtype=np.float64)
    if np.any(row_values < 0) or np.any(row_values >= values.shape[0]):
        raise IndexError("rows contain an out-of-range value")
    if np.any(endpoint_values < 0) or np.any(endpoint_values >= values.shape[2]):
        raise IndexError("endpoints contain an out-of-range value")
    features = []
    for row, endpoint in zip(row_values, endpoint_values):
        prefix = values[int(row), :, : int(endpoint) + 1, 0].T
        features.append(
            formal_feature_vector(
                model_id,
                prefix,
                ewma_alpha=ewma_alpha,
                baseline_window_samples=baseline_window_samples,
            )
        )
    return np.asarray(features, dtype=np.float64)


def select_ewma_alpha(
    signals: np.ndarray,
    rows: Sequence[int] | np.ndarray,
    endpoints: Sequence[int] | np.ndarray,
    targets: np.ndarray,
    *,
    candidates: Sequence[float] = EWMA_ALPHA_CANDIDATES,
    seed: int = 17,
) -> tuple[float, dict[str, Any]]:
    """仅用 train 内部留出选择 EWMA 平滑系数。"""

    row_values = np.asarray(rows, dtype=np.int64)
    target_values = np.asarray(targets, dtype=np.float64)
    if row_values.size != target_values.shape[0] or row_values.size < 4:
        raise ValueError("EWMA selection requires at least four aligned train rows")
    split = max(2, int(round(row_values.size * 0.8)))
    split = min(split, row_values.size - 1)
    train_positions = np.arange(split, dtype=np.int64)
    holdout_positions = np.arange(split, row_values.size, dtype=np.int64)
    scores: dict[str, float] = {}
    for alpha in candidates:
        matrix = formal_feature_matrix(
            "B-EWMA", signals, row_values, endpoints, ewma_alpha=float(alpha)
        )
        model, mean, scale = _fit_scaled_ridge(
            matrix[train_positions], target_values[train_positions], alpha=1.0e-6, seed=seed
        )
        prediction = _predict_scaled_model(
            model, matrix[holdout_positions], mean=mean, scale=scale
        )
        scores[str(float(alpha))] = float(
            np.mean((prediction - target_values[holdout_positions]) ** 2)
        )
    selected = min(scores, key=scores.get)
    return float(selected), {
        "candidates": scores,
        "selected_alpha": float(selected),
        "selection_data": "train_only",
    }


def _fit_scaled_ridge(
    features: np.ndarray,
    targets: np.ndarray,
    *,
    alpha: float,
    seed: int,
) -> tuple[Any, np.ndarray, np.ndarray]:
    del seed
    values = np.asarray(features, dtype=np.float64)
    target_values = np.asarray(targets, dtype=np.float64)
    mean = values.mean(axis=0)
    scale = values.std(axis=0)
    scale[scale < 1.0e-12] = 1.0
    model = Ridge(alpha=float(alpha))
    model.fit((values - mean) / scale, target_values / 100.0)
    return model, mean, scale


def _predict_scaled_model(
    model: Any,
    features: np.ndarray,
    mean: np.ndarray | None = None,
    scale: np.ndarray | None = None,
) -> np.ndarray:
    values = np.asarray(features, dtype=np.float64)
    if values.shape[0] == 0:
        return np.empty((0, 3), dtype=np.float64)
    if mean is not None and scale is not None:
        values = (values - mean) / scale
    return np.asarray(model.predict(values), dtype=np.float64) * 100.0


def fit_formal_classical_model(
    model_id: str,
    train_features: np.ndarray,
    train_targets: np.ndarray,
    eval_features: Sequence[np.ndarray],
    *,
    seed: int,
) -> tuple[list[np.ndarray], dict[str, Any], Any]:
    """在 train 行上拟合一个注册的标量或统计基线。"""

    if model_id not in {"B-LAST", "B-DELTA", "B-EWMA", "B-STAT", "B-STEADY", "O-EQ"}:
        raise ValueError(f"{model_id} is not a classical formal baseline")
    values = np.asarray(train_features, dtype=np.float64)
    targets = np.asarray(train_targets, dtype=np.float64)
    if values.ndim != 2 or values.shape[0] == 0 or targets.shape != (values.shape[0], 3):
        raise ValueError("train features and targets must be aligned non-empty matrices")
    mean = values.mean(axis=0)
    scale = values.std(axis=0)
    scale[scale < 1.0e-12] = 1.0
    scaled = (values - mean) / scale
    started = time.perf_counter()
    selection: dict[str, Any] = {}
    if model_id in {"B-LAST", "B-STEADY", "O-EQ"}:
        model = MLPRegressor(
            hidden_layer_sizes=(24,),
            activation="tanh",
            solver="adam",
            alpha=1.0e-6,
            max_iter=160,
            learning_rate_init=1.0e-2,
            batch_size=256,
            tol=1.0e-4,
            random_state=int(seed),
        )
        estimator_name = "small_mlp"
    elif model_id == "B-STAT":
        split = max(2, min(values.shape[0] - 1, int(round(values.shape[0] * 0.8))))
        candidates: list[tuple[str, Any]] = [
            ("ridge", Ridge(alpha=1.0e-6)),
            (
                "gbdt",
                MultiOutputRegressor(
                    GradientBoostingRegressor(
                        n_estimators=60,
                        max_depth=2,
                        learning_rate=0.05,
                        random_state=int(seed),
                    )
                ),
            ),
        ]
        candidate_scores: dict[str, float] = {}
        for name, candidate in candidates:
            candidate.fit(scaled[:split], targets[:split] / 100.0)
            candidate_scores[name] = float(
                np.mean(
                    (_predict_scaled_model(candidate, scaled[split:]) - targets[split:]) ** 2
                )
            )
        estimator_name = min(candidate_scores, key=candidate_scores.get)
        selection = {"candidates": candidate_scores, "selection_data": "train_only"}
        model = dict(candidates)[estimator_name]
    else:
        model = Ridge(alpha=1.0e-6)
        estimator_name = "ridge"
    model.fit(scaled, targets / 100.0)
    predictions = [
        _predict_scaled_model(model, np.asarray(features, dtype=np.float64), mean, scale)
        for features in eval_features
    ]
    finite = bool(all(np.isfinite(prediction).all() for prediction in predictions))
    diagnostics = {
        "status": "PASS" if finite else "FAIL",
        "estimator": estimator_name,
        "feature_count": int(values.shape[1]),
        "training_time_s": float(time.perf_counter() - started),
        "finite_predictions": finite,
        **selection,
    }
    return predictions, diagnostics, (model, mean, scale)


class _CausalConv1d(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3) -> None:
        super().__init__()
        self.kernel_size = int(kernel_size)
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size=self.kernel_size)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return self.conv(F.pad(values, (self.kernel_size - 1, 0)))


class CausalTCN(nn.Module):
    """轻量左填充 causal TCN，head 只读取当前 prefix 的最后状态。"""

    def __init__(self, hidden_channels: int = 16) -> None:
        super().__init__()
        self.conv1 = _CausalConv1d(3, hidden_channels)
        self.conv2 = _CausalConv1d(hidden_channels, hidden_channels)
        self.head = nn.Linear(hidden_channels, 3)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        encoded = torch.relu(self.conv1(values))
        encoded = torch.relu(self.conv2(encoded))
        return self.head(encoded[:, :, -1])


class CausalGRU(nn.Module):
    """单层 GRU；每次 forward 都从零状态开始，状态不跨 observation 泄漏。"""

    def __init__(self, hidden_size: int = 16) -> None:
        super().__init__()
        self.gru = nn.GRU(input_size=3, hidden_size=hidden_size, num_layers=1, batch_first=True)
        self.head = nn.Linear(hidden_size, 3)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        output, _ = self.gru(values)
        return self.head(output[:, -1, :])


def causal_sequence_matrix(
    signals: np.ndarray,
    rows: Sequence[int] | np.ndarray,
    endpoints: Sequence[int] | np.ndarray,
    *,
    sequence_length: int = TEMPORAL_SEQUENCE_LENGTH,
) -> np.ndarray:
    """将每条已观测前缀重采样到固定长度，不做未来填充。"""

    values = np.asarray(signals, dtype=np.float32)
    row_values = np.asarray(rows, dtype=np.int64)
    endpoint_values = np.asarray(endpoints, dtype=np.int64)
    if sequence_length < 2 or row_values.ndim != 1 or endpoint_values.shape != row_values.shape:
        raise ValueError("sequence_length must be >=2 and rows/endpoints must be aligned")
    if np.any(row_values < 0) or np.any(row_values >= values.shape[0]):
        raise IndexError("rows contain an out-of-range value")
    if np.any(endpoint_values < 0) or np.any(endpoint_values >= values.shape[2]):
        raise IndexError("endpoints contain an out-of-range value")
    result = np.empty((row_values.size, int(sequence_length), 3), dtype=np.float32)
    for position, (row, endpoint) in enumerate(zip(row_values, endpoint_values)):
        prefix = np.asarray(values[int(row), :, : int(endpoint) + 1, 0].T, dtype=np.float64)
        source = np.linspace(0.0, float(prefix.shape[0] - 1), int(sequence_length))
        for channel in range(3):
            result[position, :, channel] = np.interp(
                source, np.arange(prefix.shape[0], dtype=np.float64), prefix[:, channel]
            ).astype(np.float32)
    return result


def fit_causal_neural_model(
    model_id: str,
    train_sequences: np.ndarray,
    train_targets: np.ndarray,
    eval_sequences: Sequence[np.ndarray],
    *,
    seed: int,
    epochs: int = 12,
    learning_rate: float = 1.0e-2,
) -> tuple[list[np.ndarray], dict[str, Any], nn.Module]:
    """拟合注册的 TCN 或 GRU，输入始终为因果 prefix。"""

    if model_id not in {"B-TCN", "B-GRU"}:
        raise ValueError(f"{model_id} is not a causal neural baseline")
    sequences = np.asarray(train_sequences, dtype=np.float32)
    targets = np.asarray(train_targets, dtype=np.float32)
    if sequences.ndim != 3 or sequences.shape[2] != 3 or sequences.shape[0] == 0:
        raise ValueError("train_sequences must have shape (N, L, 3) and be non-empty")
    if targets.shape != (sequences.shape[0], 3):
        raise ValueError("train_targets must align with train_sequences")
    channel_mean = sequences.mean(axis=(0, 1), keepdims=True)
    channel_scale = sequences.std(axis=(0, 1), keepdims=True)
    channel_scale[channel_scale < 1.0e-7] = 1.0
    normalized = (sequences - channel_mean) / channel_scale
    torch.manual_seed(int(seed))
    model: nn.Module = CausalTCN() if model_id == "B-TCN" else CausalGRU()
    model._channel_mean = channel_mean
    model._channel_scale = channel_scale
    model._model_id = model_id
    optimizer = torch.optim.Adam(model.parameters(), lr=float(learning_rate))
    train_x = torch.from_numpy(normalized.astype(np.float32))
    train_y = torch.from_numpy(targets / 100.0)
    model.train()
    started = time.perf_counter()
    final_loss = float("nan")
    for _ in range(max(int(epochs), 1)):
        optimizer.zero_grad(set_to_none=True)
        input_tensor = train_x.transpose(1, 2) if model_id == "B-TCN" else train_x
        prediction = model(input_tensor)
        loss = F.mse_loss(prediction, train_y)
        if not torch.isfinite(loss):
            raise ValueError(f"{model_id} training produced a non-finite loss")
        loss.backward()
        optimizer.step()
        final_loss = float(loss.detach().cpu())
    training_time_s = time.perf_counter() - started
    model.eval()
    predictions: list[np.ndarray] = []
    with torch.no_grad():
        for values in eval_sequences:
            eval_array = np.asarray(values, dtype=np.float32)
            if eval_array.shape[0] == 0:
                predictions.append(np.empty((0, 3), dtype=np.float64))
                continue
            normalized_eval = (eval_array - channel_mean) / channel_scale
            tensor = torch.from_numpy(normalized_eval.astype(np.float32))
            tensor = tensor.transpose(1, 2) if model_id == "B-TCN" else tensor
            predictions.append(model(tensor).cpu().numpy().astype(np.float64) * 100.0)
    finite = bool(all(np.isfinite(prediction).all() for prediction in predictions))
    diagnostics = {
        "status": "PASS" if finite else "FAIL",
        "architecture": "lightweight_causal_tcn" if model_id == "B-TCN" else "single_layer_gru",
        "sequence_length": int(sequences.shape[1]),
        "epochs": int(max(int(epochs), 1)),
        "final_train_mse_normalized": final_loss,
        "training_time_s": float(training_time_s),
        "finite_predictions": finite,
    }
    return predictions, diagnostics, model


__all__ = [
    "CausalGRU",
    "CausalTCN",
    "EWMA_ALPHA_CANDIDATES",
    "FORMAL_BASELINE_IDS",
    "FORMAL_HORIZON_IDS",
    "PILOT_PROBE_IDS",
    "TEMPORAL_SEQUENCE_LENGTH",
    "causal_sequence_matrix",
    "fit_causal_neural_model",
    "fit_formal_classical_model",
    "fit_pilot_linear_probes",
    "formal_feature_matrix",
    "formal_feature_vector",
    "pilot_probe_feature_vector",
    "select_ewma_alpha",
]
