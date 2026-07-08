"""MiniRocket 固定核特征提取(R1a 标量序列 / R1b raw 波形)。

落地 [rocket_hydra_regression_implementation_plan.md §4.2](../../docs/rocket_hydra_regression_implementation_plan.md)
的 R1 定位:R1a 在 R0 已用满的超声标量序列上跑固定核卷积定上限,
R1b 在 raw 5000 点波形上跑帧内卷积 + 跨 timestep 池化验证增量。

本轮朴素实现,不做 chunk 流式读取(smoke 规模够);formal-6000 验证时再补。
核参数 seed 固定、requires_grad=False 思想用 numpy 固定 seed 实现。
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

from tv3.common.splits import load_splits, resolve_split_indices
from tv3.common.waveform import waveform_array_path
from tv3.ml.features import sequence_stat_features

logger = logging.getLogger(__name__)

# R1a 默认用 R0 top-3 物理标量序列(alpha_true > sound_speed > tof,见 §0 诊断)
DEFAULT_MINIROCKET_PHYSICS_ARRAYS = (
    "ultrasonic_alpha_true_npm",
    "ultrasonic_sound_speed_m_per_s",
    "ultrasonic_tof_s",
)
DEFAULT_MINIROCKET_SEQUENCE_STATISTICS = ("mean", "std", "min", "max", "slope")
DEFAULT_MINIROCKET_KERNEL_LENGTHS = (7, 9, 11)
DEFAULT_MINIROCKET_NUM_KERNELS = 128
DEFAULT_MINIROCKET_KERNEL_SEED = 42
DEFAULT_FEATURE_CACHE_ROOT = Path("features") / "rocket"

MINIROCKET_SCALAR_BUILDER = "minirocket_scalar_v1"
MINIROCKET_RAW_BUILDER = "minirocket_raw_v1"


@dataclass(frozen=True, slots=True)
class MiniRocketFeatureConfig:
    """MiniRocket 特征提取配置。feature_builder 决定 R1a/R1b 路径。"""

    feature_builder: str
    include_slow: bool = True
    slow_channels: tuple[str, ...] | None = None
    # R1a 用:在哪些超声标量序列上做固定核卷积
    physics_arrays: tuple[str, ...] = DEFAULT_MINIROCKET_PHYSICS_ARRAYS
    # 跨 timestep 池化算子(R1b 用;R1a 不池化,序列本身即"时序")
    sequence_statistics: tuple[str, ...] = DEFAULT_MINIROCKET_SEQUENCE_STATISTICS
    num_kernels: int = DEFAULT_MINIROCKET_NUM_KERNELS
    kernel_lengths: tuple[int, ...] = DEFAULT_MINIROCKET_KERNEL_LENGTHS
    kernel_seed: int = DEFAULT_MINIROCKET_KERNEL_SEED
    raw_zscore: bool = True  # R1b 只用,per-frame z-score 反量化后波形


@dataclass(frozen=True, slots=True)
class MiniRocketFeatureCache:
    dataset_dir: Path
    cache_dir: Path
    feature_config: MiniRocketFeatureConfig
    feature_names: tuple[str, ...]
    label_names: tuple[str, ...]
    split_sequence_counts: dict[str, int]


@dataclass(frozen=True, slots=True)
class FixedKernels:
    """固定卷积核集合。weights[i] 形状 (length_i,),biases[i] 标量。"""

    weights: tuple[np.ndarray, ...]
    biases: tuple[float, ...]
    lengths: tuple[int, ...]


def generate_fixed_kernels(
    num_kernels: int,
    kernel_lengths: tuple[int, ...],
    seed: int,
) -> FixedKernels:
    """生成 zero-mean 固定随机卷积核,seed 固定保证可复现。

    对标 MiniRocket(dempster 2021):固定核免去学习,小样本下稳定性远超可学习 Conv1d。
    每核从 kernel_lengths 随机选长度,权重 randn*0.1,bias randn*0.1。
    """
    if num_kernels < 1:
        raise ValueError("num_kernels 必须 >= 1")
    if not kernel_lengths:
        raise ValueError("kernel_lengths 不能为空")
    if any(length < 2 for length in kernel_lengths):
        raise ValueError("kernel_lengths 每项必须 >= 2")
    rng = np.random.default_rng(seed)
    weights: list[np.ndarray] = []
    biases: list[float] = []
    lengths: list[int] = []
    for _ in range(num_kernels):
        length = int(rng.choice(kernel_lengths))
        weight = (rng.standard_normal(length) * 0.1).astype(np.float32)
        bias = float(rng.standard_normal() * 0.1)
        # 零均值核,对标 MiniRocket zero-mean fixed kernels
        weight = weight - weight.mean()
        weights.append(weight)
        biases.append(bias)
        lengths.append(length)
    return FixedKernels(tuple(weights), tuple(biases), tuple(lengths))


def default_minirocket_cache_dir(dataset_dir: Path | str, feature_builder: str) -> Path:
    return Path(dataset_dir) / DEFAULT_FEATURE_CACHE_ROOT / feature_builder


def build_minirocket_feature_cache(
    dataset_dir: Path | str,
    *,
    cache_dir: Path | str | None = None,
    config: MiniRocketFeatureConfig | None = None,
) -> MiniRocketFeatureCache:
    """构建 R1a 或 R1b 特征缓存。按 config.feature_builder 分派。"""
    dataset_dir = Path(dataset_dir)
    config = config or MiniRocketFeatureConfig(feature_builder=MINIROCKET_SCALAR_BUILDER)
    _validate_config(config)
    _validate_tunnel_ventilation_dataset(dataset_dir)
    cache_dir = Path(cache_dir) if cache_dir is not None else default_minirocket_cache_dir(dataset_dir, config.feature_builder)
    cache_dir.mkdir(parents=True, exist_ok=True)

    splits = load_splits(dataset_dir / "splits")
    master_sequence_ids = _load_str_array(dataset_dir / "metadata" / "sequence_ids.npy")
    split_indices = resolve_split_indices(splits, master_sequence_ids)
    labels = np.load(dataset_dir / "labels" / "y.npy").astype(np.float32)
    label_names = tuple(_load_str_array(dataset_dir / "metadata" / "label_names.npy"))
    slow_names = tuple(_load_str_array(dataset_dir / "metadata" / "slow_channel_names.npy"))
    # slow_sequence_long.csv 存在性可选;R1a/R1b 不依赖 phase 窗口(序列池化用全窗)
    phase_lookup = _load_phase_lookup(dataset_dir / "sequences" / "slow_sequence_long.csv", master_sequence_ids)

    feature_names: tuple[str, ...] | None = None
    split_sequence_counts: dict[str, int] = {}
    for split_name, indices in split_indices.items():
        sequence_ids = tuple(master_sequence_ids[index] for index in indices)
        x, current_feature_names = _build_split_features(
            dataset_dir,
            split_indices=indices,
            sequence_ids=sequence_ids,
            slow_channel_names=slow_names,
            phase_lookup=phase_lookup,
            config=config,
        )
        if feature_names is None:
            feature_names = current_feature_names
        elif current_feature_names != feature_names:
            raise ValueError(f"feature names drifted across splits: {split_name}")
        if not np.isfinite(x).all():
            raise ValueError(f"non-finite features detected in split {split_name}")
        np.save(cache_dir / f"feature_matrix_{split_name}.npy", x.astype(np.float32, copy=False))
        split_labels = labels[indices]
        if split_labels.shape[0] != x.shape[0]:
            raise ValueError(f"label row mismatch for split {split_name}: {split_labels.shape[0]} != {x.shape[0]}")
        split_sequence_counts[split_name] = len(sequence_ids)

    assert feature_names is not None
    (cache_dir / "feature_names.json").write_text(
        json.dumps(list(feature_names), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    _write_manifest(
        dataset_dir=dataset_dir,
        cache_dir=cache_dir,
        config=config,
        feature_names=feature_names,
        split_sequence_counts=split_sequence_counts,
        label_names=label_names,
        slow_channel_names=slow_names,
    )
    _validate_cache_shapes(dataset_dir, cache_dir, feature_names)
    return MiniRocketFeatureCache(
        dataset_dir=dataset_dir,
        cache_dir=cache_dir,
        feature_config=config,
        feature_names=feature_names,
        label_names=label_names,
        split_sequence_counts=split_sequence_counts,
    )


def _build_split_features(
    dataset_dir: Path,
    *,
    split_indices: list[int],
    sequence_ids: tuple[str, ...],
    slow_channel_names: tuple[str, ...],
    phase_lookup: dict[str, tuple[str, ...]],
    config: MiniRocketFeatureConfig,
) -> tuple[np.ndarray, tuple[str, ...]]:
    blocks: list[np.ndarray] = []
    feature_names: list[str] = []

    kernels = generate_fixed_kernels(config.num_kernels, config.kernel_lengths, config.kernel_seed)

    if config.feature_builder == MINIROCKET_SCALAR_BUILDER:
        rocket_block, rocket_names = _build_scalar_block(
            dataset_dir, split_indices=split_indices, config=config, kernels=kernels,
        )
    elif config.feature_builder == MINIROCKET_RAW_BUILDER:
        rocket_block, rocket_names = _build_raw_block(
            dataset_dir, split_indices=split_indices, config=config, kernels=kernels,
        )
    else:
        raise ValueError(f"unsupported minirocket feature_builder {config.feature_builder!r}")
    blocks.append(rocket_block)
    feature_names.extend(rocket_names)

    if config.include_slow:
        slow = np.load(dataset_dir / "sequences" / "slow.npy", mmap_mode="r")[split_indices].astype(np.float32)
        channel_names = slow_channel_names
        if config.slow_channels is not None:
            slow, channel_names = _select_slow_channels(slow, slow_channel_names, config.slow_channels)
        # slow 统计用 sequence_stat_features(无 phase mask),均值/std 等跨整段时序
        slow_block, slow_names = sequence_stat_features(
            slow,
            channel_names=channel_names,
            statistics=config.sequence_statistics,
            prefix="slow",
        )
        blocks.append(slow_block)
        feature_names.extend(slow_names)

    if not blocks:
        raise ValueError("minirocket feature cache requires at least one enabled block")
    return np.concatenate(blocks, axis=1).astype(np.float32, copy=False), tuple(feature_names)


def _build_scalar_block(
    dataset_dir: Path,
    *,
    split_indices: list[int],
    config: MiniRocketFeatureConfig,
    kernels: FixedKernels,
) -> tuple[np.ndarray, tuple[str, ...]]:
    """R1a:对超声标量序列做固定核卷积,每核输出 PPV + max 两个标量。

    每条 physics_array 形状 (N, T),视为长度 T 的一维序列,核在 T 上滑动(valid 卷积)。
    R1a 不做跨 timestep 池化:序列本身即"时序",PPV/max 是每核一个标量。
    向量化:逐核跨所有序列一次 scipy.signal.correlate,避免 Python 序列循环。
    """
    feature_blocks: list[np.ndarray] = []
    feature_names: list[str] = []
    for array_name in config.physics_arrays:
        array = np.load(dataset_dir / "sequences" / f"{array_name}.npy", mmap_mode="r")[split_indices].astype(np.float32)
        n_sequences = array.shape[0]
        per_kernel_stats = np.zeros((n_sequences, config.num_kernels, 2), dtype=np.float32)
        for k_idx, (weight, bias) in enumerate(zip(kernels.weights, kernels.biases, strict=True)):
            # 跨所有序列一次互相关:sliding_window_view 沿时序滑窗,einsum 直积
            windows = sliding_window_view(array, weight.shape[0], axis=-1)  # (N, T-K+1, K)
            conv = np.einsum("ntk,k->nt", windows, weight, optimize=True) + bias
            per_kernel_stats[:, k_idx, 0] = (conv > 0).mean(axis=-1)  # PPV
            per_kernel_stats[:, k_idx, 1] = conv.max(axis=-1)  # max
        block = per_kernel_stats.reshape(n_sequences, config.num_kernels * 2)
        feature_blocks.append(block)
        for k_idx in range(config.num_kernels):
            feature_names.append(f"minirocket_scalar|{array_name}:kernel{k_idx}:ppv")
            feature_names.append(f"minirocket_scalar|{array_name}:kernel{k_idx}:max")
    return np.concatenate(feature_blocks, axis=1).astype(np.float32, copy=False), tuple(feature_names)


def _build_raw_block(
    dataset_dir: Path,
    *,
    split_indices: list[int],
    config: MiniRocketFeatureConfig,
    kernels: FixedKernels,
) -> tuple[np.ndarray, tuple[str, ...]]:
    """R1b:对 raw 5000 点波形逐帧固定核卷积,跨 timestep 池化。

    formal-6000 朴素加载会 OOM(6000×512×5000×4 ≈ 61 GB int32)。
    逐序列 mmap 读取:每次只实化单序列 (T, L) ≈ 10 MB,峰值可控。
    按核长度分组批量 einsum,即时池化,不存 (N, T, K, 2) 中间数组。
    """
    waveform_path = waveform_array_path(dataset_dir, "ultrasonic")
    scale = np.load(dataset_dir / "sequences" / "ultrasonic_scale.npy", mmap_mode="r")[split_indices].astype(np.float32)
    n_sequences = len(split_indices)
    waveform_mmap = np.load(waveform_path, mmap_mode="r")
    timesteps = int(waveform_mmap.shape[1])
    n_stats = len(config.sequence_statistics)
    out_dim = config.num_kernels * 2 * n_stats
    out = np.zeros((n_sequences, out_dim), dtype=np.float32)

    # channel_names 与 sequence_stat_features 内部拼接规则一致:外层 stat,内层 channel
    channel_names = tuple(
        f"kernel{k_idx}:{stat_kind}"
        for k_idx in range(config.num_kernels)
        for stat_kind in ("ppv", "max")
    )
    progress_every = max(1, n_sequences // 20)  # 约 5% 步进
    for i, global_idx in enumerate(split_indices):
        # 逐序列 mmap 读,实化 (T, L);scale[i] (T,) 广播到每帧
        frame_block = waveform_mmap[global_idx].astype(np.float32) * scale[i][..., np.newaxis]
        if config.raw_zscore:
            # per-frame z-score,对标 dataset.py normalize_waveforms
            mean = frame_block.mean(axis=-1, keepdims=True)
            std = np.maximum(frame_block.std(axis=-1, keepdims=True), 1e-6)
            frame_block = (frame_block - mean) / std
        # 按核长度分组批量 einsum:同长度核一次矩阵乘,控 conv 中间内存峰值
        # (逐核 Python 循环在 formal-6000 上过慢;分组后仅 3 次 einsum,覆盖全部核)
        ppv_acc = np.zeros((timesteps, config.num_kernels), dtype=np.float32)
        mx_acc = np.zeros((timesteps, config.num_kernels), dtype=np.float32)
        length_groups: dict[int, list[int]] = {}
        for k_idx, length in enumerate(kernels.lengths):
            length_groups.setdefault(int(length), []).append(k_idx)
        for length, k_idxs in length_groups.items():
            w_group = np.stack([kernels.weights[k] for k in k_idxs]).astype(np.float32)  # (G, length)
            b_group = np.asarray([kernels.biases[k] for k in k_idxs], dtype=np.float32)
            windows = sliding_window_view(frame_block, length, axis=-1)  # (T, L-length+1, length)
            conv = np.einsum("tlk,gk->tgl", windows, w_group, optimize=True) + b_group[None, :, None]
            ppv_acc[:, k_idxs] = (conv > 0).mean(axis=-1)
            mx_acc[:, k_idxs] = conv.max(axis=-1)
        kernel_frame_stats = np.zeros((timesteps, config.num_kernels * 2), dtype=np.float32)
        kernel_frame_stats[:, 0::2] = ppv_acc
        kernel_frame_stats[:, 1::2] = mx_acc
        # 跨 T 池化:复用 sequence_stat_features,(1, T, C) → (1, C*n_stats)
        pooled, _ = sequence_stat_features(
            kernel_frame_stats[np.newaxis, :, :],
            channel_names=channel_names,
            statistics=config.sequence_statistics,
            prefix="minirocket_raw",
        )
        out[i] = pooled[0]
        if (i + 1) % progress_every == 0 or (i + 1) == n_sequences:
            logger.info("minirocket_raw: %d/%d sequences done", i + 1, n_sequences)
    return out, _raw_feature_names(config)


def _raw_feature_names(config: MiniRocketFeatureConfig) -> tuple[str, ...]:
    """生成 R1b 特征名,顺序与 sequence_stat_features 输出列对齐:外层 stat,内层 kernel+stat_kind。"""
    names: list[str] = []
    for stat in config.sequence_statistics:
        for k_idx in range(config.num_kernels):
            names.append(f"minirocket_raw:kernel{k_idx}:ppv:{stat}")
            names.append(f"minirocket_raw:kernel{k_idx}:max:{stat}")
    return tuple(names)


def _validate_config(config: MiniRocketFeatureConfig) -> None:
    if config.feature_builder not in (MINIROCKET_SCALAR_BUILDER, MINIROCKET_RAW_BUILDER):
        raise ValueError(f"unsupported feature_builder {config.feature_builder!r}")
    if config.feature_builder == MINIROCKET_SCALAR_BUILDER and not config.physics_arrays:
        raise ValueError("R1a (minirocket_scalar) 需至少一个 physics_arrays")
    if config.num_kernels < 1:
        raise ValueError("num_kernels 必须 >= 1")
    if not config.kernel_lengths:
        raise ValueError("kernel_lengths 不能为空")
    if len(set(config.kernel_lengths)) != len(config.kernel_lengths):
        raise ValueError("kernel_lengths 不能有重复")
    if config.include_slow is False and config.feature_builder == MINIROCKET_RAW_BUILDER:
        # R1b 设计上仅拼 slow 归因,关 slow 无意义;允许但不鼓励
        pass


def _validate_tunnel_ventilation_dataset(dataset_dir: Path) -> None:
    manifest_path = dataset_dir / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"missing manifest: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    composition_scheme = str(manifest.get("composition_scheme", ""))
    if composition_scheme != "tunnel_ventilation":
        raise ValueError(f"minirocket cache only supports tunnel_ventilation, got {composition_scheme!r}")


def _select_slow_channels(
    slow: np.ndarray,
    channel_names: tuple[str, ...],
    keep: tuple[str, ...],
) -> tuple[np.ndarray, tuple[str, ...]]:
    name_to_index = {name: index for index, name in enumerate(channel_names)}
    indices: list[int] = []
    for channel in keep:
        if channel not in name_to_index:
            raise ValueError(f"unknown slow channel {channel!r}. available={list(channel_names)}")
        indices.append(name_to_index[channel])
    return slow[:, :, indices], tuple(channel_names[index] for index in indices)


def _load_phase_lookup(path: Path, sequence_ids: list[str]) -> dict[str, tuple[str, ...]]:
    """读取 phase_id 序列。R1a/R1b 暂不用 phase 窗口,保留供后续 phase-aware 池化。"""
    if not path.is_file():
        return {}
    rows: dict[str, list[tuple[int, str]]] = {}
    with path.open("r", encoding="utf-8") as handle:
        header = handle.readline().strip().split(",")
        idx_seq = header.index("sequence_id")
        idx_ts = header.index("timestep")
        idx_phase = header.index("phase_id")
        for line in handle:
            parts = line.rstrip("\n").split(",")
            rows.setdefault(parts[idx_seq], []).append((int(parts[idx_ts]), parts[idx_phase]))
    return {
        seq_id: tuple(phase for _ts, phase in sorted(items, key=lambda item: item[0]))
        for seq_id, items in rows.items()
    }


def _load_str_array(path: Path) -> list[str]:
    values = np.load(path, allow_pickle=True)
    return [str(value) for value in values.tolist()]


def _write_manifest(
    *,
    dataset_dir: Path,
    cache_dir: Path,
    config: MiniRocketFeatureConfig,
    feature_names: tuple[str, ...],
    split_sequence_counts: dict[str, int],
    label_names: tuple[str, ...],
    slow_channel_names: tuple[str, ...],
) -> None:
    split_summary_path = dataset_dir / "splits" / "split_summary.json"
    split_policy = None
    if split_summary_path.is_file():
        split_summary = json.loads(split_summary_path.read_text(encoding="utf-8"))
        split_policy = split_summary.get("split_policy")
    manifest = {
        "dataset_slug": dataset_dir.name,
        "schema_version": "tv3-minirocket-feature-1",
        "sequence_count": int(sum(split_sequence_counts.values())),
        "split_sequence_counts": split_sequence_counts,
        "split_policy": split_policy,
        "feature_builder": config.feature_builder,
        "kernel_seed": int(config.kernel_seed),
        "kernel_count": int(config.num_kernels),
        "kernel_lengths": list(config.kernel_lengths),
        "dilations": [],  # 第一版固定核无 dilation,留空
        "pooling_stats": list(config.sequence_statistics),
        "modalities": _modalities_payload(config),
        "slow_channels": list(config.slow_channels or slow_channel_names),
        "source_arrays": list(config.physics_arrays) if config.feature_builder == MINIROCKET_SCALAR_BUILDER else ["ultrasonic_raw"],
        "raw_zscore": bool(config.raw_zscore),
        "label_names": list(label_names),
        "feature_count": len(feature_names),
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    (cache_dir / "feature_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def _modalities_payload(config: MiniRocketFeatureConfig) -> list[str]:
    modalities: list[str] = []
    if config.feature_builder == MINIROCKET_SCALAR_BUILDER:
        modalities.append("minirocket_scalar")
    else:
        modalities.append("minirocket_raw")
    if config.include_slow:
        modalities.append("slow")
    return modalities


def _validate_cache_shapes(dataset_dir: Path, cache_dir: Path, feature_names: tuple[str, ...]) -> None:
    splits = load_splits(dataset_dir / "splits")
    feature_name_count = len(feature_names)
    for split_name, rows in splits.items():
        matrix = np.load(cache_dir / f"feature_matrix_{split_name}.npy", mmap_mode="r")
        if matrix.shape[0] != len(rows):
            raise ValueError(f"cached row count mismatch for {split_name}: {matrix.shape[0]} != {len(rows)}")
        if matrix.shape[1] != feature_name_count:
            raise ValueError(
                f"cached column count mismatch for {split_name}: {matrix.shape[1]} != {feature_name_count}"
            )
        if not np.isfinite(matrix).all():
            raise ValueError(f"cached matrix contains non-finite values for split {split_name}")