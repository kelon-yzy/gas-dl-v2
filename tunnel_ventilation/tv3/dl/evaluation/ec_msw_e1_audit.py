from __future__ import annotations

from dataclasses import asdict
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

from tv3.dl.data.dataset import V4BenchmarkDataset
from tv3.dl.data.waveform_preprocess import WaveformDevicePreprocessor
from tv3.dl.models.ec_msw_e1 import ECMSWE1Regressor
from tv3.dl.models.registry import build_model
from tv3.ml.metrics import component_regression_metrics, regression_metrics
from tv3.ml.ridge_head import ScaledRidgeCVRegressor
from tv3.sim.core.tunnel_ventilation_schema import COMPONENT_FIELDS
from tv3.sim.packaging.io import write_csv, write_json


PROJECT_ROOT = Path(__file__).resolve().parents[3]
EVAL_SPLITS = ("val", "test", "extrapolation")
FRAME_GATE_FIELDS = (
    "peak_mae_samples_max",
    "peak_p95_abs_error_samples_max",
    "peak_bias_abs_samples_max",
)
PARITY_GATE_FIELDS = ("o2_r2_drop_max", "co2_n2_r2_drop_max")


def run_ec_msw_e1_audit(
    config_path: Path | str,
    *,
    project_root: Path = PROJECT_ROOT,
) -> Path:
    config_path = Path(config_path)
    if not config_path.is_absolute():
        config_path = project_root / config_path
    config_path = config_path.resolve()
    config = _read_json(config_path)
    _validate_config(config)

    dataset_dir = _resolve(project_root, config["dataset_dir"])
    training_run_dir = _resolve(project_root, config["training_run_dir"])
    output_dir = _resolve(project_root, config["output_dir"])
    reference_path = _resolve(project_root, config["b1_reference_metrics"])
    checkpoint_path = _resolve(
        project_root,
        config.get("checkpoint_path", training_run_dir / "checkpoint.pt"),
    )
    run_config_path = training_run_dir / "run_config.json"
    if output_dir.exists():
        raise FileExistsError(f"EC-MSW E1 audit output already exists: {output_dir}")
    for path in (dataset_dir, checkpoint_path, run_config_path, reference_path):
        if not path.exists():
            raise FileNotFoundError(f"EC-MSW E1 audit input not found: {path}")

    run_config = _read_json(run_config_path)
    model = _load_model(run_config, checkpoint_path, device=str(config["device"]))
    reference = _read_json(reference_path)
    peak_targets = _load_peak_targets(dataset_dir)
    input_preprocess = _build_input_preprocess(run_config)
    datasets = {
        split: _build_dataset(dataset_dir, split, run_config, project_root=project_root)
        for split in ("train", *EVAL_SPLITS)
    }

    train_frames = len(datasets["train"]) * _timesteps(
        datasets["train"], input_preprocess=input_preprocess
    )
    sampled_frame_indices = _sample_frame_indices(
        train_frames,
        maximum=int(config["max_train_probe_frames"]),
        seed=int(config["probe_sample_seed"]),
    )
    train_sequence_x, train_y, train_frame_x, train_peak_y = _extract_train_features(
        model,
        datasets["train"],
        peak_targets,
        sampled_frame_indices,
        batch_size=int(config["batch_size"]),
        num_workers=int(config["num_workers"]),
        device=torch.device(str(config["device"])),
        input_preprocess=input_preprocess,
    )

    alphas = tuple(float(value) for value in config["ridge_alphas"])
    peak_probe = ScaledRidgeCVRegressor(alphas=alphas).fit(train_frame_x, train_peak_y)
    composition_probe = ScaledRidgeCVRegressor(alphas=alphas).fit(train_sequence_x, train_y)

    train_peak_error = peak_probe.predict(train_frame_x) - train_peak_y
    frame_payload: dict[str, Any] = {
        "probe": {
            "train_frame_count_total": train_frames,
            "train_frame_count_used": int(len(sampled_frame_indices)),
            "sample_seed": int(config["probe_sample_seed"]),
            "selected_alpha": peak_probe.selected_alpha,
            "target_role": "offline_audit_only_not_model_input_or_training_loss",
        },
        "gates": config["frame_fidelity_gates"],
        "train_probe_metrics": _peak_error_metrics(train_peak_error),
        "splits": {},
    }
    parity_payload: dict[str, Any] = {
        "head": "train_only_scaled_ridgecv_on_frozen_sequence_embedding",
        "selected_alpha": composition_probe.selected_alpha,
        "gates": config["parity_gates"],
        "reference_metrics_path": str(reference_path),
        "splits": {},
    }
    narrow_rows: list[dict[str, object]] = []

    for split in EVAL_SPLITS:
        sequence_x, y_true, peak_error = _extract_eval_features(
            model,
            datasets[split],
            peak_targets,
            peak_probe,
            batch_size=int(config["batch_size"]),
            num_workers=int(config["num_workers"]),
            device=torch.device(str(config["device"])),
            input_preprocess=input_preprocess,
        )
        frame_metrics = _peak_error_metrics(peak_error)
        frame_payload["splits"][split] = {
            **frame_metrics,
            "gate": _frame_split_gate(frame_metrics, config["frame_fidelity_gates"]),
        }

        y_pred = composition_probe.predict(sequence_x)
        split_metrics = _composition_metrics(y_pred, y_true)
        reference_metrics = _reference_component_metrics(reference, split)
        split_gate = _parity_split_gate(
            split_metrics["component_metrics"],
            reference_metrics,
            config["parity_gates"],
        )
        parity_payload["splits"][split] = {
            **split_metrics,
            "reference_component_metrics": reference_metrics,
            "gate": split_gate,
        }
        narrow_rows.extend(
            _narrow_window_rows(
                split,
                y_pred,
                y_true,
                config["narrow_o2_windows"],
            )
        )

    frame_payload["passed"] = all(
        entry["gate"]["passed"] for entry in frame_payload["splits"].values()
    )
    parity_payload["passed"] = all(
        entry["gate"]["passed"] for entry in parity_payload["splits"].values()
    )
    verdict = _verdict(frame_payload["passed"], parity_payload["passed"])
    verdict["e2_allowed"] = bool(verdict["status"] == "e1_pass")

    output_dir.mkdir(parents=True)
    write_json(output_dir / "frame_fidelity.json", frame_payload)
    write_json(output_dir / "b1_parity.json", parity_payload)
    write_csv(
        output_dir / "narrow_o2_windows.csv",
        (
            "split",
            "window_id",
            "low_percent",
            "high_percent",
            "count",
            "mae_percent",
            "rmse_percent",
            "p90_abs_error_percent",
            "bias_percent",
            "local_slope",
        ),
        narrow_rows,
    )
    write_json(output_dir / "verdict.json", verdict)
    write_json(
        output_dir / "manifest.json",
        {
            "schema_version": "tv3-ec-msw-e1-audit-1",
            "config_path": str(config_path),
            "config_sha256": _sha256(config_path),
            "dataset_dir": str(dataset_dir),
            "training_run_dir": str(training_run_dir),
            "checkpoint_path": str(checkpoint_path),
            "checkpoint_sha256": _sha256(checkpoint_path),
            "run_config_sha256": _sha256(run_config_path),
            "b1_reference_metrics_sha256": _sha256(reference_path),
            "component_names": list(COMPONENT_FIELDS),
            "eval_splits": list(EVAL_SPLITS),
            "model_inputs_exclude": [
                "ultrasonic_peak_index",
                "true_tof",
                "true_sound_speed",
                "true_attenuation",
                "composition_labels",
            ],
        },
    )
    return output_dir


def _load_model(
    run_config: dict[str, Any],
    checkpoint_path: Path,
    *,
    device: str,
) -> ECMSWE1Regressor:
    model_config = run_config.get("model_config")
    if not isinstance(model_config, dict) or model_config.get("name") != "ec_msw_e1":
        raise ValueError("training run must contain model_config.name='ec_msw_e1'")
    model = build_model(model_config)
    if not isinstance(model, ECMSWE1Regressor):
        raise TypeError(f"Expected ECMSWE1Regressor, got {type(model).__name__}")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if "model_state_dict" not in checkpoint:
        raise ValueError(f"checkpoint has no model_state_dict: {checkpoint_path}")
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    return model


def _build_dataset(
    dataset_dir: Path,
    split: str,
    run_config: dict[str, Any],
    *,
    project_root: Path,
) -> V4BenchmarkDataset:
    if any(run_config.get(name) is not None for name in ("window", "phase_windows", "phase_stats_path")):
        raise ValueError("EC-MSW E1 audit requires the fixed full-sequence E1 input contract")
    slow_channels = run_config.get("slow_channels")
    waveform_preprocess = str(run_config.get("waveform_preprocess", "cpu")).lower()
    return V4BenchmarkDataset(
        dataset_dir,
        split=split,
        modalities=tuple(run_config["modalities"]),
        input_format=str(run_config["input_format"]),
        scaler_path=_optional_path(project_root, run_config.get("scaler_path")),
        dequantize_waveforms=bool(run_config["dequantize_waveforms"]),
        normalize_waveforms=bool(run_config["normalize_waveforms"]),
        waveform_stats_features=tuple(run_config["waveform_stats_features"]),
        waveform_preprocess=waveform_preprocess,
        slow_channels=None if slow_channels is None else tuple(slow_channels),
    )


def _build_input_preprocess(
    run_config: dict[str, Any],
) -> WaveformDevicePreprocessor | None:
    waveform_preprocess = str(run_config.get("waveform_preprocess", "cpu")).lower()
    if waveform_preprocess != "gpu":
        return None
    return WaveformDevicePreprocessor(
        modalities=tuple(run_config["modalities"]),
        waveform_stats_features=tuple(run_config["waveform_stats_features"]),
        normalize_waveforms=bool(run_config["normalize_waveforms"]),
        input_format=str(run_config["input_format"]),
    )


def _extract_train_features(
    model: ECMSWE1Regressor,
    dataset: V4BenchmarkDataset,
    peak_targets: np.ndarray,
    sampled_frame_indices: np.ndarray,
    *,
    batch_size: int,
    num_workers: int,
    device: torch.device,
    input_preprocess: WaveformDevicePreprocessor | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    sequence_blocks: list[np.ndarray] = []
    label_blocks: list[np.ndarray] = []
    frame_blocks: list[np.ndarray] = []
    peak_blocks: list[np.ndarray] = []
    sequence_offset = 0
    timesteps = _timesteps(dataset, input_preprocess=input_preprocess)
    preprocess = None if input_preprocess is None else input_preprocess.to(device)
    with torch.inference_mode():
        for xb, yb in _loader(dataset, batch_size, num_workers, device):
            x = _batch_x(xb, device=device, input_preprocess=preprocess)
            frame_tensor = model.encode_frames(x)
            sequences = model.encode_sequence(x, frame_embeddings=frame_tensor).cpu().numpy()
            frames = frame_tensor.cpu().numpy()
            batch_sequences = int(x.shape[0])
            frame_start = sequence_offset * timesteps
            frame_end = (sequence_offset + batch_sequences) * timesteps
            left = int(np.searchsorted(sampled_frame_indices, frame_start, side="left"))
            right = int(np.searchsorted(sampled_frame_indices, frame_end, side="left"))
            local_indices = sampled_frame_indices[left:right] - frame_start
            if local_indices.size:
                flat_frames = frames.reshape(-1, frames.shape[-1])
                source_indices = dataset.indices[sequence_offset : sequence_offset + batch_sequences]
                flat_targets = np.asarray(peak_targets[source_indices]).reshape(-1)
                frame_blocks.append(flat_frames[local_indices])
                peak_blocks.append(flat_targets[local_indices])
            sequence_blocks.append(sequences)
            label_blocks.append(yb.numpy())
            sequence_offset += batch_sequences
    return (
        np.concatenate(sequence_blocks).astype(np.float32, copy=False),
        np.concatenate(label_blocks).astype(np.float32, copy=False),
        np.concatenate(frame_blocks).astype(np.float32, copy=False),
        np.concatenate(peak_blocks).astype(np.float32, copy=False),
    )


def _extract_eval_features(
    model: ECMSWE1Regressor,
    dataset: V4BenchmarkDataset,
    peak_targets: np.ndarray,
    peak_probe: ScaledRidgeCVRegressor,
    *,
    batch_size: int,
    num_workers: int,
    device: torch.device,
    input_preprocess: WaveformDevicePreprocessor | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    sequence_blocks: list[np.ndarray] = []
    label_blocks: list[np.ndarray] = []
    peak_error_blocks: list[np.ndarray] = []
    sequence_offset = 0
    preprocess = None if input_preprocess is None else input_preprocess.to(device)
    with torch.inference_mode():
        for xb, yb in _loader(dataset, batch_size, num_workers, device):
            x = _batch_x(xb, device=device, input_preprocess=preprocess)
            frame_tensor = model.encode_frames(x)
            sequences = model.encode_sequence(x, frame_embeddings=frame_tensor).cpu().numpy()
            frames = frame_tensor.cpu().numpy()
            batch_sequences = int(x.shape[0])
            source_indices = dataset.indices[sequence_offset : sequence_offset + batch_sequences]
            peak_true = np.asarray(peak_targets[source_indices]).reshape(-1)
            peak_pred = peak_probe.predict(frames.reshape(-1, frames.shape[-1]))
            peak_error_blocks.append(np.asarray(peak_pred).reshape(-1) - peak_true)
            sequence_blocks.append(sequences)
            label_blocks.append(yb.numpy())
            sequence_offset += batch_sequences
    return (
        np.concatenate(sequence_blocks).astype(np.float32, copy=False),
        np.concatenate(label_blocks).astype(np.float32, copy=False),
        np.concatenate(peak_error_blocks).astype(np.float32, copy=False),
    )


def _loader(
    dataset: V4BenchmarkDataset,
    batch_size: int,
    num_workers: int,
    device: torch.device,
) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=num_workers > 0,
    )


def _batch_x(
    batch: torch.Tensor | dict[str, torch.Tensor],
    *,
    device: torch.device,
    input_preprocess: WaveformDevicePreprocessor | None = None,
) -> torch.Tensor:
    if isinstance(batch, dict) and "x" not in batch:
        if input_preprocess is None:
            raise ValueError("raw waveform batch requires waveform_preprocess='gpu' assembler")
        moved = {
            key: value.to(device)
            for key, value in batch.items()
            if key != "aux_targets" and isinstance(value, torch.Tensor)
        }
        return input_preprocess(moved)
    tensor = batch["x"] if isinstance(batch, dict) else batch
    return tensor.to(device)


def _peak_error_metrics(errors: np.ndarray) -> dict[str, float | int]:
    values = np.asarray(errors, dtype=np.float64).reshape(-1)
    if values.size == 0 or not np.isfinite(values).all():
        raise ValueError("peak errors must be non-empty and finite")
    return {
        "frame_count": int(values.size),
        "peak_mae_samples": float(np.mean(np.abs(values))),
        "peak_rmse_samples": float(np.sqrt(np.mean(values * values))),
        "peak_p95_abs_error_samples": float(np.percentile(np.abs(values), 95.0)),
        "peak_bias_samples": float(np.mean(values)),
    }


def _composition_metrics(y_pred: np.ndarray, y_true: np.ndarray) -> dict[str, Any]:
    components = component_regression_metrics(y_pred, y_true, COMPONENT_FIELDS)
    component_payload = {}
    for index, name in enumerate(COMPONENT_FIELDS):
        errors = np.asarray(y_pred[:, index] - y_true[:, index], dtype=np.float64)
        component_payload[name] = {
            **asdict(components[name]),
            "bias": float(np.mean(errors)),
            "p90_abs_error": float(np.percentile(np.abs(errors), 90.0)),
        }
    return {
        "sequence_count": int(len(y_true)),
        "metrics": asdict(regression_metrics(y_pred, y_true)),
        "component_metrics": component_payload,
        "sum_abs_error": float(np.mean(np.abs(np.sum(y_pred, axis=1) - 100.0))),
    }


def _narrow_window_rows(
    split: str,
    y_pred: np.ndarray,
    y_true: np.ndarray,
    windows: list[dict[str, Any]],
) -> list[dict[str, object]]:
    rows = []
    o2_true = np.asarray(y_true[:, 1], dtype=np.float64)
    o2_pred = np.asarray(y_pred[:, 1], dtype=np.float64)
    for index, window in enumerate(windows):
        low = float(window["low_percent"])
        high = float(window["high_percent"])
        mask = (o2_true >= low) & (o2_true <= high if index == len(windows) - 1 else o2_true < high)
        true_values = o2_true[mask]
        pred_values = o2_pred[mask]
        errors = pred_values - true_values
        count = int(np.count_nonzero(mask))
        row: dict[str, object] = {
            "split": split,
            "window_id": str(window["id"]),
            "low_percent": low,
            "high_percent": high,
            "count": count,
            "mae_percent": None,
            "rmse_percent": None,
            "p90_abs_error_percent": None,
            "bias_percent": None,
            "local_slope": None,
        }
        if count > 0:
            row.update(
                {
                    "mae_percent": float(np.mean(np.abs(errors))),
                    "rmse_percent": float(np.sqrt(np.mean(errors * errors))),
                    "p90_abs_error_percent": float(np.percentile(np.abs(errors), 90.0)),
                    "bias_percent": float(np.mean(errors)),
                }
            )
        if count >= 2 and float(np.ptp(true_values)) > 1e-12:
            row["local_slope"] = float(np.polyfit(true_values, pred_values, 1)[0])
        rows.append(row)
    return rows


def _frame_split_gate(
    metrics: dict[str, float | int],
    gates: dict[str, float],
) -> dict[str, Any]:
    checks = {
        "peak_mae_samples": float(metrics["peak_mae_samples"]) <= float(gates["peak_mae_samples_max"]),
        "peak_p95_abs_error_samples": float(metrics["peak_p95_abs_error_samples"])
        <= float(gates["peak_p95_abs_error_samples_max"]),
        "peak_bias_abs_samples": abs(float(metrics["peak_bias_samples"]))
        <= float(gates["peak_bias_abs_samples_max"]),
    }
    return {"passed": all(checks.values()), "checks": checks}


def _parity_split_gate(
    candidate: dict[str, dict[str, float]],
    reference: dict[str, dict[str, float]],
    gates: dict[str, float],
) -> dict[str, Any]:
    deltas = {
        name: float(candidate[name]["r2"]) - float(reference[name]["r2"])
        for name in COMPONENT_FIELDS
    }
    checks = {
        "x_O2_r2_noninferiority": deltas["x_O2"] >= -float(gates["o2_r2_drop_max"]),
        "x_CO2_r2_noninferiority": deltas["x_CO2"] >= -float(gates["co2_n2_r2_drop_max"]),
        "x_N2_r2_noninferiority": deltas["x_N2"] >= -float(gates["co2_n2_r2_drop_max"]),
    }
    return {"passed": all(checks.values()), "r2_delta_vs_b1": deltas, "checks": checks}


def _reference_component_metrics(reference: dict[str, Any], split: str) -> dict[str, dict[str, float]]:
    try:
        source = reference["evaluations"][split]["component_metrics"]
        return {
            name: {
                "mae": float(source[name]["mae"]),
                "rmse": float(source[name]["rmse"]),
                "r2": float(source[name]["r2"]),
            }
            for name in COMPONENT_FIELDS
        }
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"B1 reference has invalid component metrics for split={split!r}") from exc


def _verdict(frame_passed: bool, parity_passed: bool) -> dict[str, Any]:
    if not frame_passed:
        return {
            "status": "frame_fidelity_failed",
            "reason": "At least one evaluation split failed the preregistered peak-position fidelity gate.",
        }
    if not parity_passed:
        return {
            "status": "b1_parity_failed",
            "reason": "Frame fidelity passed, but frozen sequence embeddings failed B1 non-inferiority.",
        }
    return {
        "status": "e1_pass",
        "reason": "Frame fidelity and B1 parity both passed; E2a may be considered as a separate experiment.",
    }


def _sample_frame_indices(total: int, *, maximum: int, seed: int) -> np.ndarray:
    if total < 1 or maximum < 1:
        raise ValueError("total and maximum frame counts must be positive")
    if total <= maximum:
        return np.arange(total, dtype=np.int64)
    rng = np.random.default_rng(seed)
    return np.sort(rng.choice(total, size=maximum, replace=False).astype(np.int64, copy=False))


def _load_peak_targets(dataset_dir: Path) -> np.ndarray:
    path = dataset_dir / "sequences" / "ultrasonic_peak_index.npy"
    if not path.is_file():
        raise FileNotFoundError(f"offline peak audit target not found: {path}")
    values = np.load(path, mmap_mode="r")
    if values.ndim != 2 or not np.isfinite(values).all():
        raise ValueError(f"ultrasonic_peak_index must be a finite 2D array, got {values.shape}")
    return values


def _timesteps(
    dataset: V4BenchmarkDataset,
    *,
    input_preprocess: WaveformDevicePreprocessor | None = None,
) -> int:
    if dataset.waveform_preprocess == "gpu" or input_preprocess is not None:
        return dataset.timesteps()
    x, _y = dataset[0]
    if isinstance(x, dict) and "x" in x:
        tensor = x["x"]
    elif isinstance(x, torch.Tensor):
        tensor = x
    else:
        raise ValueError("E1 cpu dataset item must provide assembled tensor 'x'")
    if tensor.ndim != 2:
        raise ValueError(f"E1 dataset item must be shaped (T, C), got {tuple(tensor.shape)}")
    return int(tensor.shape[0])


def _validate_config(config: dict[str, Any]) -> None:
    required = {
        "dataset_dir",
        "training_run_dir",
        "output_dir",
        "b1_reference_metrics",
        "device",
        "batch_size",
        "num_workers",
        "ridge_alphas",
        "max_train_probe_frames",
        "probe_sample_seed",
        "frame_fidelity_gates",
        "parity_gates",
        "narrow_o2_windows",
    }
    missing = required.difference(config)
    if missing:
        raise ValueError(f"EC-MSW E1 audit config missing fields: {sorted(missing)}")
    if int(config["batch_size"]) < 1 or int(config["num_workers"]) < 0:
        raise ValueError("batch_size must be positive and num_workers must be non-negative")
    if int(config["max_train_probe_frames"]) < 1:
        raise ValueError("max_train_probe_frames must be positive")
    alphas = [float(value) for value in config["ridge_alphas"]]
    if not alphas or any(not math.isfinite(value) or value <= 0.0 for value in alphas):
        raise ValueError("ridge_alphas must contain finite positive values")
    _validate_gate(config["frame_fidelity_gates"], FRAME_GATE_FIELDS, "frame_fidelity_gates")
    _validate_gate(config["parity_gates"], PARITY_GATE_FIELDS, "parity_gates")
    windows = config["narrow_o2_windows"]
    if not isinstance(windows, list) or not windows:
        raise ValueError("narrow_o2_windows must be a non-empty list")
    previous_high = None
    for window in windows:
        if set(window) != {"id", "low_percent", "high_percent"}:
            raise ValueError("each narrow O2 window must contain id, low_percent, high_percent")
        low = float(window["low_percent"])
        high = float(window["high_percent"])
        if not math.isfinite(low) or not math.isfinite(high) or abs((high - low) - 0.8) > 1e-9:
            raise ValueError("each narrow O2 window must have finite width 0.8 vol%")
        if previous_high is not None and low < previous_high:
            raise ValueError("narrow O2 windows must be sorted and non-overlapping")
        previous_high = high


def _validate_gate(values: object, fields: tuple[str, ...], name: str) -> None:
    if not isinstance(values, dict) or set(values) != set(fields):
        raise ValueError(f"{name} must contain exactly {list(fields)}")
    if any(not math.isfinite(float(values[field])) or float(values[field]) < 0.0 for field in fields):
        raise ValueError(f"{name} values must be finite and non-negative")


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return payload


def _resolve(root: Path, value: Path | str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else root / path


def _optional_path(root: Path, value: object) -> Path | None:
    return None if value is None else _resolve(root, str(value))


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()
