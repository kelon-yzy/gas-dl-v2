from __future__ import annotations

import csv
from functools import lru_cache
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset, get_worker_info

from tv3.common.waveform import waveform_array_path
from tv3.common.windows import WINDOW_KIND_EARLY, WINDOW_KIND_PHASE, WindowConfig, resolve_window_config
from tv3.dl.data.augmentation import TimeSeriesAugmentConfig, augment_sequence
from tv3.dl.data.scalers import apply_scaler, load_scaler
from tv3.dl.data.splits import load_splits, resolve_split_indices
from tv3.dl.data.waveform_preprocess import (
    WAVEFORM_MODALITY_ORDER,
    WAVEFORM_PREPROCESS_OPTIONS,
    dequantize_waveform_numpy,
    normalize_waveform_numpy,
    numpy_to_tensor,
    waveform_stats_numpy,
)

MODALITY_OPTIONS = ("slow", "ultrasonic", "fiber_mic")
"""v4 benchmark 支持的输入模态。"""

WAVEFORM_STATS_OPTIONS = ("log_std", "log_max_abs")
"""可追加到 slow 分支的逐帧波形幅度统计。"""


class V4BenchmarkDataset(Dataset):
    """v4 正式 benchmark PyTorch Dataset。

    消费 ``sim.pipeline.generate_benchmark`` 生成的完整数据集目录，
    按 split 和模态选择构造可训练的 Dataset 实例。

    Parameters
    ----------
    dataset_dir:
        benchmark 数据集根目录（包含 ``sequences/``, ``labels/``, ``splits/`` 等子目录）。
    split:
        要加载的 split 名称（``"train"``, ``"val"``, ``"test"``, ``"extrapolation"``）。
    modalities:
        要加载的模态列表，可选 ``"slow"``, ``"ultrasonic"``, ``"fiber_mic"``。
        默认只加载 ``"slow"``。
    input_format:
        输出张量格式：``"NTC"`` 或 ``"NCT"``。
    scaler_path:
        z-score scaler JSON 文件路径（可选）。若提供，在 ``__getitem__`` 时
        对 slow 通道做归一化。
    lazy:
        若为 ``True``，首次 ``__getitem__`` 时才从磁盘加载数组（默认）。
    waveform_preprocess:
        ``"cpu"``（默认）在 worker 内 dequant/normalize 后返回组装好的 ``x``；
        ``"gpu"`` 仅返回 int16 波形 + scale 等 raw 张量，由
        :class:`WaveformDevicePreprocessor` 在训练设备上组装。
    """

    def __init__(
        self,
        dataset_dir: Path | str,
        split: str,
        modalities: tuple[str, ...] = ("slow",),
        input_format: str = "NTC",
        scaler_path: Path | str | None = None,
        lazy: bool = True,
        augment_config: TimeSeriesAugmentConfig | None = None,
        augment_seed: int = 0,
        window: dict[str, object] | WindowConfig | None = None,
        phase_windows: tuple[dict[str, object] | WindowConfig | None, ...] | None = None,
        dequantize_waveforms: bool = False,
        normalize_waveforms: bool = False,
        waveform_stats_features: tuple[str, ...] = (),
        phase_stats_path: Path | str | None = None,
        slow_channels: tuple[str, ...] | None = None,
        aux_target_arrays: dict[str, str] | None = None,
        waveform_preprocess: str = "cpu",
    ):
        dataset_dir = Path(dataset_dir)
        self._dataset_dir = dataset_dir
        self._modalities = tuple(modalities)
        self._input_format = input_format.upper()
        self._lazy = lazy
        self._augment_config = augment_config
        self._augment_seed = augment_seed
        self._dequantize_waveforms = bool(dequantize_waveforms)
        self._normalize_waveforms = bool(normalize_waveforms)
        self._waveform_stats_features = tuple(waveform_stats_features)
        self._waveform_preprocess = str(waveform_preprocess).lower()
        self._augment_rng: np.random.Generator | None = None
        self._window = resolve_window_config(window)
        self._phase_windows = _resolve_phase_windows(phase_windows)
        if self._window is not None and self._phase_windows is not None:
            raise ValueError("window and phase_windows cannot be combined")

        _validate_modalities(self._modalities)
        _validate_waveform_stats_features(self._waveform_stats_features)
        if self._waveform_preprocess not in WAVEFORM_PREPROCESS_OPTIONS:
            raise ValueError(
                f"waveform_preprocess must be one of {WAVEFORM_PREPROCESS_OPTIONS}, "
                f"got {self._waveform_preprocess!r}"
            )
        self._slow_channels = tuple(slow_channels) if slow_channels is not None else None
        self._slow_channel_indices = _resolve_slow_channel_indices(
            dataset_dir, self._slow_channels, self._modalities
        )
        if self._input_format not in {"NTC", "NCT"}:
            raise ValueError(f"input_format must be 'NTC' or 'NCT', got {self._input_format!r}")

        if self._waveform_preprocess == "gpu":
            self._validate_gpu_preprocess_contract()

        splits = load_splits(dataset_dir / "splits")
        sequence_ids = _load_sequence_ids(dataset_dir)
        self.indices = resolve_split_indices(splits, sequence_ids)[split]
        self._sequence_ids = sequence_ids
        self._window_masks = _build_window_masks(dataset_dir, self._sequence_ids, self._window)
        self._phase_window_masks = (
            tuple(_build_window_masks(dataset_dir, self._sequence_ids, window) for window in self._phase_windows)
            if self._phase_windows is not None
            else None
        )

        self._scaler = None
        if scaler_path is not None:
            self._scaler = load_scaler(scaler_path)

        self._slow: np.ndarray | None = None
        self._ultrasonic: np.ndarray | None = None
        self._ultrasonic_scale: np.ndarray | None = None
        self._fiber_mic: np.ndarray | None = None
        self._fiber_mic_scale: np.ndarray | None = None
        self._labels: np.ndarray | None = None
        self._phase_stats: np.ndarray | None = None
        self._phase_stats_path = Path(phase_stats_path) if phase_stats_path is not None else None
        self._has_phase_stats = self._phase_stats_path is not None and self._phase_stats_path.is_file()

        self._aux_target_arrays: dict[str, str] = {}
        if aux_target_arrays is not None:
            if not isinstance(aux_target_arrays, dict) or not aux_target_arrays:
                raise ValueError("aux_target_arrays must be a non-empty dict mapping key -> npy filename")
            for key, filename in aux_target_arrays.items():
                if not isinstance(key, str) or not key.strip():
                    raise ValueError(f"aux_target_arrays keys must be non-empty strings, got {key!r}")
                if not isinstance(filename, str) or not filename.strip():
                    raise ValueError(f"aux_target_arrays values must be non-empty strings, got {filename!r}")
            self._aux_target_arrays = {str(k): str(v) for k, v in aux_target_arrays.items()}
        self._aux_data: dict[str, np.ndarray] = {}

        if not lazy:
            self._load_arrays()

    def __len__(self) -> int:
        return len(self.indices)

    @property
    def waveform_preprocess(self) -> str:
        return self._waveform_preprocess

    @property
    def modalities(self) -> tuple[str, ...]:
        return self._modalities

    @property
    def waveform_stats_features(self) -> tuple[str, ...]:
        return self._waveform_stats_features

    @property
    def normalize_waveforms(self) -> bool:
        return self._normalize_waveforms

    @property
    def dequantize_waveforms(self) -> bool:
        return self._dequantize_waveforms

    @property
    def input_format(self) -> str:
        return self._input_format

    def __getitem__(self, idx: int) -> tuple[torch.Tensor | dict[str, torch.Tensor], torch.Tensor]:
        if self._labels is None:
            self._load_arrays()
        src_idx = self.indices[idx]
        y = numpy_to_tensor(self._labels[src_idx], dtype=np.float32)
        if self._waveform_preprocess == "gpu":
            payload = self._build_raw_input(src_idx)
            if self._has_phase_stats:
                if self._phase_stats is None:
                    self._load_arrays()
                payload["phase_stats"] = numpy_to_tensor(self._phase_stats[src_idx], dtype=np.float32)
            if self._aux_target_arrays:
                payload["aux_targets"] = {
                    key: numpy_to_tensor(self._aux_data[key][src_idx], dtype=np.float32)
                    for key in self._aux_target_arrays
                }
            return payload, y

        xs = self._build_input(src_idx)
        result: dict[str, torch.Tensor] = {"x": xs}
        if self._has_phase_stats:
            if self._phase_stats is None:
                self._load_arrays()
            result["phase_stats"] = numpy_to_tensor(self._phase_stats[src_idx], dtype=np.float32)
        if self._aux_target_arrays:
            result["aux_targets"] = {
                key: numpy_to_tensor(self._aux_data[key][src_idx], dtype=np.float32)
                for key in self._aux_target_arrays
            }
        if len(result) > 1 or "x" not in result:
            return result, y
        return xs, y

    @property
    def has_phase_stats(self) -> bool:
        return self._has_phase_stats

    @property
    def phase_stat_dim(self) -> int:
        if not self._has_phase_stats:
            return 0
        if self._phase_stats is None:
            self._load_arrays()
        return int(self._phase_stats.shape[1])

    def assembled_channel_count(self) -> int:
        """组装后 NTC 通道数（含 waveform stats）。"""
        if self._labels is None:
            self._load_arrays()
        count = 0
        if self._slow is not None:
            if self._slow_channel_indices is not None:
                count += len(self._slow_channel_indices)
            else:
                count += int(self._slow.shape[-1])
        waveform_modalities = [
            modality for modality in WAVEFORM_MODALITY_ORDER if modality in self._modalities
        ]
        count += len(waveform_modalities) * len(self._waveform_stats_features)
        if self._ultrasonic is not None:
            count += int(self._ultrasonic.shape[-1])
        if self._fiber_mic is not None:
            count += int(self._fiber_mic.shape[-1])
        if count < 1:
            raise ValueError("assembled_channel_count requires at least one modality channel")
        return count

    def timesteps(self) -> int:
        """时间步数；gpu 路径禁止 window，直接读原始数组。"""
        if self._labels is None:
            self._load_arrays()
        if self._slow is not None:
            return int(self._slow.shape[1])
        if self._ultrasonic is not None:
            return int(self._ultrasonic.shape[1])
        if self._fiber_mic is not None:
            return int(self._fiber_mic.shape[1])
        raise ValueError("timesteps requires at least one loaded modality array")

    def __getstate__(self) -> dict[str, object]:
        state = self.__dict__.copy()
        state["_slow"] = None
        state["_ultrasonic"] = None
        state["_ultrasonic_scale"] = None
        state["_fiber_mic"] = None
        state["_fiber_mic_scale"] = None
        state["_labels"] = None
        state["_phase_stats"] = None
        state["_aux_data"] = {}
        state["_augment_rng"] = None
        return state

    def _validate_gpu_preprocess_contract(self) -> None:
        if not any(modality in self._modalities for modality in WAVEFORM_MODALITY_ORDER):
            raise ValueError("waveform_preprocess='gpu' requires ultrasonic and/or fiber_mic modalities")
        if not self._dequantize_waveforms:
            raise ValueError("waveform_preprocess='gpu' requires dequantize_waveforms=true")
        if self._window is not None:
            raise ValueError("waveform_preprocess='gpu' does not support window")
        if self._phase_windows is not None:
            raise ValueError("waveform_preprocess='gpu' does not support phase_windows")
        if self._augment_config is not None:
            raise ValueError("waveform_preprocess='gpu' does not support augment")
        if self._input_format not in {"NTC", "NCT"}:
            raise ValueError("waveform_preprocess='gpu' requires input_format NTC or NCT")

    def _load_arrays(self) -> None:
        seq_dir = self._dataset_dir / "sequences"
        labels_path = self._dataset_dir / "labels" / "y.npy"
        self._labels = np.load(labels_path).astype(np.float32)
        if "slow" in self._modalities:
            self._slow = np.load(seq_dir / "slow.npy", mmap_mode="r")
        if "ultrasonic" in self._modalities:
            self._ultrasonic = np.load(waveform_array_path(self._dataset_dir, "ultrasonic"), mmap_mode="r")
            if self._dequantize_waveforms:
                self._ultrasonic_scale = np.load(seq_dir / "ultrasonic_scale.npy", mmap_mode="r")
        if "fiber_mic" in self._modalities:
            self._fiber_mic = np.load(waveform_array_path(self._dataset_dir, "fiber_mic"), mmap_mode="r")
            if self._dequantize_waveforms:
                self._fiber_mic_scale = np.load(seq_dir / "fiber_mic_scale.npy", mmap_mode="r")
        if self._has_phase_stats and self._phase_stats is None:
            self._phase_stats = np.load(self._phase_stats_path, mmap_mode="r")
        if self._aux_target_arrays and not self._aux_data:
            self._load_aux_arrays()

    def _load_aux_arrays(self) -> None:
        seq_dir = self._dataset_dir / "sequences"
        n_sequences = len(self._sequence_ids)
        for key, filename in self._aux_target_arrays.items():
            path = seq_dir / f"{filename}.npy"
            if not path.is_file():
                raise FileNotFoundError(f"aux_target_arrays[{key!r}]: file not found: {path}")
            arr = np.load(path, mmap_mode="r")
            if arr.ndim != 2:
                raise ValueError(
                    f"aux_target_arrays[{key!r}]: expected 2D array (N, T), got shape {arr.shape}"
                )
            if arr.shape[0] != n_sequences:
                raise ValueError(
                    f"aux_target_arrays[{key!r}]: expected {n_sequences} sequences, got {arr.shape[0]}"
                )
            if not np.isfinite(arr).all():
                raise ValueError(f"aux_target_arrays[{key!r}]: contains non-finite values")
            self._aux_data[key] = arr

    def _ensure_augment_rng(self) -> np.random.Generator:
        # 多 worker DataLoader 下，每个 worker 进程按 worker_id 派生独立 RNG，
        # 避免 fork 复制同一状态导致各 worker 产生重复的增强序列。
        if self._augment_rng is None:
            worker_info = get_worker_info()
            worker_id = worker_info.id if worker_info is not None else 0
            self._augment_rng = np.random.default_rng(self._augment_seed + worker_id)
        return self._augment_rng

    def _build_raw_input(self, src_idx: int) -> dict[str, torch.Tensor]:
        payload: dict[str, torch.Tensor] = {}
        if self._slow is not None:
            sl = self._slow[src_idx]
            if self._scaler is not None:
                sl = apply_scaler(sl, self._scaler)
            if self._slow_channel_indices is not None:
                sl = sl[:, self._slow_channel_indices]
            payload["slow"] = numpy_to_tensor(sl, dtype=np.float32)
        if self._ultrasonic is not None:
            if self._ultrasonic_scale is None:
                raise ValueError("ultrasonic_scale is required when dequantize_waveforms=true")
            payload["ultrasonic"] = numpy_to_tensor(self._ultrasonic[src_idx])
            payload["ultrasonic_scale"] = numpy_to_tensor(self._ultrasonic_scale[src_idx], dtype=np.float32)
        if self._fiber_mic is not None:
            if self._fiber_mic_scale is None:
                raise ValueError("fiber_mic_scale is required when dequantize_waveforms=true")
            payload["fiber_mic"] = numpy_to_tensor(self._fiber_mic[src_idx])
            payload["fiber_mic_scale"] = numpy_to_tensor(self._fiber_mic_scale[src_idx], dtype=np.float32)
        return payload

    def _build_input(self, src_idx: int) -> torch.Tensor:
        if self._phase_window_masks is not None:
            views = [self._build_single_input(src_idx, masks) for masks in self._phase_window_masks]
            x = np.stack(views, axis=0)
            return numpy_to_tensor(x, dtype=np.float32)
        x = self._build_single_input(src_idx, self._window_masks)
        return numpy_to_tensor(x, dtype=np.float32)

    def _build_single_input(self, src_idx: int, window_masks: dict[int, np.ndarray] | None) -> np.ndarray:
        parts: list[np.ndarray] = []
        waveform_parts: list[np.ndarray] = []
        waveform_stat_parts: list[np.ndarray] = []
        if self._slow is not None:
            sl = self._slow[src_idx]
            if self._scaler is not None:
                sl = apply_scaler(sl, self._scaler)
            sl = self._apply_window(sl, src_idx, window_masks)
            if self._slow_channel_indices is not None:
                sl = sl[:, self._slow_channel_indices]
            parts.append(sl)
        if self._ultrasonic is not None:
            values, stats = self._waveform_values(
                self._ultrasonic,
                self._ultrasonic_scale,
                src_idx,
                window_masks,
                modality="ultrasonic",
            )
            waveform_parts.append(values)
            if stats is not None:
                waveform_stat_parts.append(stats)
        if self._fiber_mic is not None:
            values, stats = self._waveform_values(
                self._fiber_mic,
                self._fiber_mic_scale,
                src_idx,
                window_masks,
                modality="fiber_mic",
            )
            waveform_parts.append(values)
            if stats is not None:
                waveform_stat_parts.append(stats)

        parts.extend(waveform_stat_parts)
        parts.extend(waveform_parts)
        x = np.concatenate(parts, axis=-1) if len(parts) > 1 else parts[0]
        if self._augment_config is not None:
            x = augment_sequence(x, self._augment_config, self._ensure_augment_rng())
        if self._input_format == "NCT":
            x = np.transpose(x, (1, 0))
        return np.asarray(x, dtype=np.float32)

    def _apply_window(self, values: np.ndarray, src_idx: int, window_masks: dict[int, np.ndarray] | None) -> np.ndarray:
        if window_masks is None:
            return values
        mask = window_masks[src_idx]
        return _resample_masked_timesteps(values, mask)

    def _waveform_values(
        self,
        waveform: np.ndarray,
        scale: np.ndarray | None,
        src_idx: int,
        window_masks: dict[int, np.ndarray] | None,
        *,
        modality: str,
    ) -> tuple[np.ndarray, np.ndarray | None]:
        values = waveform[src_idx]
        if self._dequantize_waveforms:
            if scale is None:
                raise ValueError(f"{modality}_scale is required when dequantize_waveforms=true")
            values = dequantize_waveform_numpy(values, scale[src_idx])
        elif self._waveform_stats_features:
            values = values.astype(np.float32, copy=False)
        stats = (
            waveform_stats_numpy(values, self._waveform_stats_features)
            if self._waveform_stats_features
            else None
        )
        if self._normalize_waveforms:
            # per-timestep z-score：每帧波形点独立 zero-mean unit-var。
            # 对标 wav2vec 2.0 整段波形 z-score；tv3 每帧独立承载一个声学事件，按帧归一化更贴合物理结构。
            values = normalize_waveform_numpy(values)
        values = self._apply_window(values, src_idx, window_masks)
        if stats is not None:
            stats = self._apply_window(stats, src_idx, window_masks)
        return values, stats


def _validate_modalities(modalities: tuple[str, ...]) -> None:
    if not modalities:
        raise ValueError("modalities must not be empty")
    for m in modalities:
        if m not in MODALITY_OPTIONS:
            raise ValueError(f"Unknown modality: {m!r}. Available: {MODALITY_OPTIONS}")


def _validate_waveform_stats_features(features: tuple[str, ...]) -> None:
    unknown = set(features) - set(WAVEFORM_STATS_OPTIONS)
    if unknown:
        raise ValueError(f"Unknown waveform stats features: {sorted(unknown)}. Available: {WAVEFORM_STATS_OPTIONS}")


def _resolve_slow_channel_indices(
    dataset_dir: Path,
    slow_channels: tuple[str, ...] | None,
    modalities: tuple[str, ...],
) -> list[int] | None:
    """将保留的 slow 通道名解析为列索引（channel ablation 用）。

    slow_channels 为 None 时返回 None（保留全部通道，行为不变）。
    """
    if slow_channels is None:
        return None
    if "slow" not in modalities:
        raise ValueError("slow_channels requires 'slow' in modalities")
    if not slow_channels:
        raise ValueError("slow_channels must not be empty")
    names_path = dataset_dir / "metadata" / "slow_channel_names.npy"
    if not names_path.is_file():
        raise FileNotFoundError(f"slow_channel_names metadata not found: {names_path}")
    all_names = [str(name) for name in np.load(names_path, allow_pickle=True).tolist()]
    name_to_index = {name: index for index, name in enumerate(all_names)}
    indices: list[int] = []
    for channel in slow_channels:
        if channel not in name_to_index:
            raise ValueError(f"Unknown slow channel: {channel!r}. Available: {all_names}")
        indices.append(name_to_index[channel])
    return indices


def _load_sequence_ids(dataset_dir: Path) -> list[str]:
    path = dataset_dir / "metadata" / "sequence_ids.npy"
    if not path.is_file():
        raise FileNotFoundError(f"sequence_ids metadata not found: {path}")
    ids = np.load(path, allow_pickle=True)
    return [str(sid) for sid in ids.tolist()]


def _build_window_masks(
    dataset_dir: Path,
    sequence_ids: list[str],
    window: WindowConfig | None,
) -> dict[int, np.ndarray] | None:
    if window is None:
        return None
    phase_rows = _load_slow_phase_rows(dataset_dir / "sequences" / "slow_sequence_long.csv")
    masks: dict[int, np.ndarray] = {}
    for src_idx, sequence_id in enumerate(sequence_ids):
        rows = phase_rows.get(sequence_id)
        if not rows:
            raise ValueError(f"slow_sequence_long.csv has no rows for sequence_id={sequence_id!r}")
        if window.kind == WINDOW_KIND_PHASE:
            mask = np.array([row["phase_id"] == str(window.value) for row in rows], dtype=bool)
        elif window.kind == WINDOW_KIND_EARLY:
            count = max(1, int(np.ceil(len(rows) * float(window.value))))
            mask = np.zeros(len(rows), dtype=bool)
            mask[:count] = True
        else:
            raise ValueError(f"Unknown window kind: {window.kind!r}")
        if not bool(mask.any()):
            raise ValueError(f"empty timestep window for sequence_id={sequence_id!r}, window={window!r}")
        masks[src_idx] = mask
    return masks


def _resolve_phase_windows(
    windows: tuple[dict[str, object] | WindowConfig | None, ...] | list[dict[str, object] | WindowConfig | None] | None,
) -> tuple[WindowConfig | None, ...] | None:
    if windows is None:
        return None
    if not windows:
        raise ValueError("phase_windows must not be empty")
    return tuple(resolve_window_config(window) for window in windows)


@lru_cache(maxsize=8)
def _load_slow_phase_rows(path: Path) -> dict[str, tuple[dict[str, str], ...]]:
    if not path.is_file():
        raise FileNotFoundError(f"slow_sequence_long.csv not found: {path}")
    rows_by_sequence: dict[str, list[dict[str, str]]] = {}
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"sequence_id", "phase_id"}
        missing = required - set(reader.fieldnames or ())
        if missing:
            raise ValueError(f"slow_sequence_long.csv missing columns: {sorted(missing)}")
        for row in reader:
            rows_by_sequence.setdefault(str(row["sequence_id"]), []).append(row)
    return {sequence_id: tuple(rows) for sequence_id, rows in rows_by_sequence.items()}


def _resample_masked_timesteps(values: np.ndarray, mask: np.ndarray) -> np.ndarray:
    if values.ndim != 2:
        raise ValueError(f"windowed DL input expects 2D arrays, got shape {values.shape}")
    if values.shape[0] != mask.shape[0]:
        raise ValueError(f"window mask length {mask.shape[0]} does not match timesteps {values.shape[0]}")
    selected = np.asarray(values[mask], dtype=np.float32)
    if selected.shape[0] == values.shape[0]:
        return selected
    if selected.shape[0] == 1:
        return np.repeat(selected, values.shape[0], axis=0)
    target_positions = np.linspace(0.0, selected.shape[0] - 1, values.shape[0], dtype=np.float32)
    lower = np.floor(target_positions).astype(np.int64)
    upper = np.ceil(target_positions).astype(np.int64)
    weights = (target_positions - lower).reshape(-1, 1)
    return (selected[lower] * (1.0 - weights) + selected[upper] * weights).astype(np.float32)
