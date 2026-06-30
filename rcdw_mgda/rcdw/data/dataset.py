"""RCDW BenchmarkDataset：从磁盘 benchmark 目录读取 slow + ultrasonic 元数据。

对应方案 §8.1。

目录结构（由 ``scripts.generate_benchmark`` 落盘）::

    {data_root}/{slug}/
        manifest.json
        labels/y.npy                                # (N, 3) float32
        sequences/
            slow.npy                                # (N, T, 7) float32
            ultrasonic_tof_observed_s.npy           # (N, T) float32
            ultrasonic_sound_speed_estimated_m_per_s.npy   # (N, T) float32
            ultrasonic_peak_index.npy               # (N, T) int32
            ultrasonic_tof_quality.npy              # (N, T) float32
            ultrasonic_tof_accepted.npy             # (N, T) int8
            fiber_mic_int16.npy                     # 落盘但 **不读取**
        splits/{train,val,test}.csv
        metadata/sequence_ids.npy

构造 12 维输入张量（v1.2 §8.2）::

    (B, L, 12) = concat(slow[7], ultrasonic_metadata[5])

每条样本是滑窗 (L, 12)，标签取窗口末时刻的组分（与该 sequence 全局组分一致）。
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


# 12 维通道布局（与 rcdw.models.single_modal 索引一致）
SLOW_CHANNEL_ORDER: tuple[str, ...] = (
    "V_NDIR_CO2",
    "V_TCS",
    "T_C",
    "P_MPa",
    "H_RH",
    "L_m",
    "piston_position_m",
)
ULTRASONIC_META_FILES: tuple[tuple[str, str], ...] = (
    # (拼接维度的语义名, 磁盘文件名)
    ("ultrasonic_tof_observed_s", "ultrasonic_tof_observed_s.npy"),
    (
        "ultrasonic_sound_speed_estimated_m_per_s",
        "ultrasonic_sound_speed_estimated_m_per_s.npy",
    ),
    ("ultrasonic_peak_index", "ultrasonic_peak_index.npy"),
    ("ultrasonic_tof_quality", "ultrasonic_tof_quality.npy"),
    ("ultrasonic_tof_accepted", "ultrasonic_tof_accepted.npy"),
)


class BenchmarkDataset(Dataset):
    """从磁盘 benchmark 目录加载窗口张量与组分标签。"""

    def __init__(
        self,
        data_root: Path | str,
        split: str,
        window: int = 8,
        modalities: tuple[str, ...] = ("slow", "ultrasonic"),
    ):
        """
        Args:
            data_root: benchmark 数据集根目录（含 ``manifest.json`` 与
                ``sequences/`` 等）。
            split: ``"train"`` / ``"val"`` / ``"test"``。
            window: 滑窗长度 L（默认 8）。
            modalities: 加载模态白名单；当前实现要求至少包含 ``"slow"``
                与 ``"ultrasonic"``（fiber_mic 永远不读取）。
        """
        if split not in ("train", "val", "test"):
            raise ValueError(
                f"split must be one of (train, val, test), got {split!r}"
            )
        if "slow" not in modalities or "ultrasonic" not in modalities:
            raise ValueError(
                f"modalities must include both 'slow' and 'ultrasonic', "
                f"got {modalities}"
            )
        self.data_root = Path(data_root)
        self.split = split
        self.window = int(window)
        self.modalities = tuple(modalities)

        # ---- 解析 manifest 与 split CSV ----
        manifest_path = self.data_root / "manifest.json"
        if not manifest_path.is_file():
            raise FileNotFoundError(
                f"manifest.json not found at {manifest_path}; "
                f"先用 scripts.generate_benchmark 生成数据集。"
            )
        self.manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

        timesteps = int(self.manifest["timesteps"])
        if self.window > timesteps:
            raise ValueError(
                f"window={self.window} > timesteps={timesteps}; "
                f"窗口不能大于 sequence 长度。"
            )

        split_csv = self.data_root / "splits" / f"{split}.csv"
        if not split_csv.is_file():
            raise FileNotFoundError(f"split csv not found: {split_csv}")
        self.split_sequence_ids: list[str] = _read_split_sequence_ids(split_csv)

        # ---- 加载 labels（仅落盘文件中拿对应行）----
        sequence_ids = np.load(
            self.data_root / "metadata" / "sequence_ids.npy", allow_pickle=True
        )
        sequence_ids = [str(sid) for sid in sequence_ids.tolist()]
        id_to_index = {sid: idx for idx, sid in enumerate(sequence_ids)}
        try:
            self.split_indexes: list[int] = [
                id_to_index[sid] for sid in self.split_sequence_ids
            ]
        except KeyError as exc:
            raise ValueError(
                f"split sequence_id 不在 metadata/sequence_ids.npy 中: {exc}"
            ) from exc

        labels_all = np.load(self.data_root / "labels" / "y.npy")
        # 标签原始单位是百分比 (0-100, 由 generate_benchmark 直接落盘组分);
        # 训练侧统一用 [0, 1] 闭包,与 SingleModal / normalize_composition 对齐。
        # 详见方案 §8.6 历史 ckpt 废弃声明:新标签语义为 [0, 1] 比例。
        self.labels = (labels_all[self.split_indexes] / 100.0).astype(np.float32)
        # 验证 shape
        if self.labels.shape[1] != 3:
            raise ValueError(
                f"labels last dim must be 3, got {self.labels.shape}"
            )

        # ---- mmap 加载 slow + 5 个 ultrasonic 元数据数组 ----
        sequences_dir = self.data_root / "sequences"
        self._slow = np.load(sequences_dir / "slow.npy", mmap_mode="r")
        # 形状: (N_total, T, 7)
        if self._slow.shape[-1] != len(SLOW_CHANNEL_ORDER):
            raise ValueError(
                f"slow.npy 最后一维 {self._slow.shape[-1]} != "
                f"{len(SLOW_CHANNEL_ORDER)}"
            )
        self._us_arrays: dict[str, np.ndarray] = {}
        for sem_name, file_name in ULTRASONIC_META_FILES:
            path = sequences_dir / file_name
            if not path.is_file():
                raise FileNotFoundError(
                    f"ultrasonic metadata file missing: {path}"
                )
            self._us_arrays[sem_name] = np.load(path, mmap_mode="r")

        self.timesteps = timesteps
        self.n_windows_per_seq = timesteps - self.window + 1
        if self.n_windows_per_seq <= 0:
            raise ValueError(
                f"n_windows_per_seq <= 0 (timesteps={timesteps}, "
                f"window={self.window})"
            )
        self.total_windows = len(self.split_indexes) * self.n_windows_per_seq

    def __len__(self) -> int:
        return self.total_windows

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        if idx < 0 or idx >= self.total_windows:
            raise IndexError(idx)
        # 拆分到 (seq_offset, window_offset)
        seq_offset = idx // self.n_windows_per_seq
        window_offset = idx % self.n_windows_per_seq
        global_index = self.split_indexes[seq_offset]

        # slow 切片 (window, 7)
        slow_window = np.asarray(
            self._slow[global_index, window_offset : window_offset + self.window, :],
            dtype=np.float32,
        )
        # ultrasonic 元数据 5 个标量序列 → (window, 5)
        meta_cols = []
        for sem_name, _file in ULTRASONIC_META_FILES:
            arr = self._us_arrays[sem_name]
            col = np.asarray(
                arr[global_index, window_offset : window_offset + self.window],
                dtype=np.float32,
            )
            meta_cols.append(col)
        meta_window = np.stack(meta_cols, axis=-1)  # (window, 5)

        # 拼接成 (window, 12)
        x_window = np.concatenate([slow_window, meta_window], axis=-1)
        if x_window.shape != (self.window, 12):
            raise RuntimeError(
                f"unexpected window shape: {x_window.shape}, expected "
                f"({self.window}, 12)"
            )

        y = self.labels[seq_offset].astype(np.float32)  # (3,)
        return torch.from_numpy(x_window), torch.from_numpy(y)


def _read_split_sequence_ids(csv_path: Path) -> list[str]:
    """读取 splits/{name}.csv 的 sequence_id 列。"""
    ids: list[str] = []
    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            ids.append(row["sequence_id"])
    if not ids:
        raise ValueError(f"split csv has no rows: {csv_path}")
    return ids
