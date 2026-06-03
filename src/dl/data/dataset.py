from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset, get_worker_info

from dl.data.augmentation import TimeSeriesAugmentConfig, augment_sequence
from dl.data.scalers import apply_scaler, load_scaler
from dl.data.splits import load_splits, resolve_split_indices

MODALITY_OPTIONS = ("slow", "ultrasonic", "fiber_mic")
"""v4 benchmark 支持的输入模态。"""


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
    ):
        dataset_dir = Path(dataset_dir)
        self._dataset_dir = dataset_dir
        self._modalities = tuple(modalities)
        self._input_format = input_format.upper()
        self._lazy = lazy
        self._augment_config = augment_config
        self._augment_seed = augment_seed
        self._augment_rng: np.random.Generator | None = None

        _validate_modalities(self._modalities)
        if self._input_format not in {"NTC", "NCT"}:
            raise ValueError(f"input_format must be 'NTC' or 'NCT', got {self._input_format!r}")

        splits = load_splits(dataset_dir / "splits")
        sequence_ids = _load_sequence_ids(dataset_dir)
        self.indices = resolve_split_indices(splits, sequence_ids)[split]
        self._sequence_ids = sequence_ids

        self._scaler = None
        if scaler_path is not None:
            self._scaler = load_scaler(scaler_path)

        self._slow: np.ndarray | None = None
        self._ultrasonic: np.ndarray | None = None
        self._fiber_mic: np.ndarray | None = None
        self._labels: np.ndarray | None = None

        if not lazy:
            self._load_arrays()

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        if self._labels is None:
            self._load_arrays()
        src_idx = self.indices[idx]
        xs = self._build_input(src_idx)
        y = torch.from_numpy(self._labels[src_idx].copy())
        return xs, y

    def __getstate__(self) -> dict[str, object]:
        state = self.__dict__.copy()
        state["_slow"] = None
        state["_ultrasonic"] = None
        state["_fiber_mic"] = None
        state["_labels"] = None
        state["_augment_rng"] = None
        return state

    def _load_arrays(self) -> None:
        seq_dir = self._dataset_dir / "sequences"
        labels_path = self._dataset_dir / "labels" / "y.npy"
        self._labels = np.load(labels_path).astype(np.float32)
        if "slow" in self._modalities:
            self._slow = np.load(seq_dir / "slow.npy", mmap_mode="r")
        if "ultrasonic" in self._modalities:
            self._ultrasonic = np.load(seq_dir / "ultrasonic_int16.npy", mmap_mode="r")
        if "fiber_mic" in self._modalities:
            self._fiber_mic = np.load(seq_dir / "fiber_mic_int16.npy", mmap_mode="r")

    def _ensure_augment_rng(self) -> np.random.Generator:
        # 多 worker DataLoader 下，每个 worker 进程按 worker_id 派生独立 RNG，
        # 避免 fork 复制同一状态导致各 worker 产生重复的增强序列。
        if self._augment_rng is None:
            worker_info = get_worker_info()
            worker_id = worker_info.id if worker_info is not None else 0
            self._augment_rng = np.random.default_rng(self._augment_seed + worker_id)
        return self._augment_rng

    def _build_input(self, src_idx: int) -> torch.Tensor:
        parts: list[np.ndarray] = []
        if self._slow is not None:
            sl = self._slow[src_idx]
            if self._scaler is not None:
                sl = apply_scaler(sl, self._scaler)
            parts.append(sl)
        if self._ultrasonic is not None:
            parts.append(self._ultrasonic[src_idx])
        if self._fiber_mic is not None:
            parts.append(self._fiber_mic[src_idx])

        x = np.concatenate(parts, axis=-1) if len(parts) > 1 else parts[0]
        if self._augment_config is not None:
            x = augment_sequence(x, self._augment_config, self._ensure_augment_rng())
        if self._input_format == "NCT":
            x = np.transpose(x, (1, 0))
        return torch.from_numpy(np.array(x, dtype=np.float32, copy=True))


def _validate_modalities(modalities: tuple[str, ...]) -> None:
    if not modalities:
        raise ValueError("modalities must not be empty")
    for m in modalities:
        if m not in MODALITY_OPTIONS:
            raise ValueError(f"Unknown modality: {m!r}. Available: {MODALITY_OPTIONS}")


def _load_sequence_ids(dataset_dir: Path) -> list[str]:
    path = dataset_dir / "metadata" / "sequence_ids.npy"
    if not path.is_file():
        raise FileNotFoundError(f"sequence_ids metadata not found: {path}")
    ids = np.load(path, allow_pickle=True)
    return [str(sid) for sid in ids.tolist()]
