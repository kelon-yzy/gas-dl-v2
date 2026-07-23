"""F5 five-arm feature contracts on top of ``raw_dsp_bidirectional_v1`` frame cache.

Arms answer the plan questions; heads reuse frozen B1 RidgeCV / B7 residual recipes.
A2 is audit-only (oracle ``v_path``); deploy arms never include ``*_true*`` columns.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from tv3.common.splits import load_splits, resolve_split_indices
from tv3.ml.features import MLFeatureMatrix
from tv3.ml.raw_dsp_features import FORMAL_SLOW_CHANNELS
from tv3.ml.rocket_features import (
    DEFAULT_EARLY_FRACTIONS,
    DEFAULT_PHASE_WINDOWS,
    DEFAULT_ROCKET_SEQUENCE_STATISTICS,
    _load_phase_lookup,
    _load_str_array,
    _select_slow_channels,
    _windowed_sequence_features,
)
from tv3.ml.bidir_features import assert_no_oracle_inputs
from tv3.sim.core.tunnel_ventilation_bidir_schema import FORMAL_FEATURE_BUILDER

B1_SEVEN_STATS = ("mean", "std", "min", "max", "last", "delta", "slope")
B1_FULL_STATS = DEFAULT_ROCKET_SEQUENCE_STATISTICS

BIDIR_FRAME_CACHE_ROOT = Path("features") / "raw_dsp_bidir" / FORMAL_FEATURE_BUILDER

# Frame arrays written by build_tv3_bidir_features (deployable names only).
AB_COMPACT_FRAMES = (
    "ultrasonic_tof_corrected_ab_raw_dsp_s",
    "ultrasonic_peak_index_ab_raw_dsp",
    "ultrasonic_sound_speed_ab_raw_dsp_m_per_s",
    "ultrasonic_snr_db_ab",
    "ultrasonic_psr_ab",
)
BA_COMPACT_FRAMES = (
    "ultrasonic_tof_corrected_ba_raw_dsp_s",
    "ultrasonic_peak_index_ba_raw_dsp",
    "ultrasonic_sound_speed_ba_raw_dsp_m_per_s",
    "ultrasonic_snr_db_ba",
    "ultrasonic_psr_ba",
)
AB_EXTRA_FRAMES = (
    "ultrasonic_tof_observed_ab_raw_dsp_s",
    "ultrasonic_quality_ab_raw_dsp",
    "ultrasonic_accepted_ab_raw_dsp",
)
BA_EXTRA_FRAMES = (
    "ultrasonic_tof_observed_ba_raw_dsp_s",
    "ultrasonic_quality_ba_raw_dsp",
    "ultrasonic_accepted_ba_raw_dsp",
)
PAIR_FRAMES = (
    "ultrasonic_sound_speed_pair_raw_dsp_m_per_s",
    "ultrasonic_v_path_hat_raw_dsp_m_per_s",
    "ultrasonic_reciprocity_residual_s",
)
PAIR_FRAMES_NO_V = (
    "ultrasonic_sound_speed_pair_raw_dsp_m_per_s",
    "ultrasonic_reciprocity_residual_s",
)

SEQ_SCALARS_BASE = (
    "tau_ab_s",
    "tau_ba_s",
    "c_hat_seq_m_per_s",
    "reciprocity_residual_p95_s",
    "c_hat_dispersion_m_per_s",
)
SEQ_SCALARS_WITH_V = SEQ_SCALARS_BASE + ("v_hat_seq_m_per_s",)
ORACLE_V_SCALAR = "oracle_v_path_m_per_s"


@dataclass(frozen=True, slots=True)
class BidirArmSpec:
    arm_id: str
    feature_builder: str
    deployable: bool
    frame_arrays: tuple[str, ...]
    sequence_scalars: tuple[str, ...]
    physics_statistics: tuple[str, ...]
    physics_phase_windows: tuple[str, ...]
    physics_early_fractions: tuple[float, ...]
    include_slow: bool = True


def arm_specs() -> dict[str, BidirArmSpec]:
    """Frozen F5 arm contracts."""
    a1_frames = AB_COMPACT_FRAMES
    a3_frames = AB_COMPACT_FRAMES + BA_COMPACT_FRAMES + PAIR_FRAMES
    a4_frames = (
        AB_COMPACT_FRAMES
        + BA_COMPACT_FRAMES
        + AB_EXTRA_FRAMES
        + BA_EXTRA_FRAMES
        + PAIR_FRAMES
    )
    a5_frames = AB_COMPACT_FRAMES + BA_COMPACT_FRAMES + PAIR_FRAMES_NO_V
    return {
        "A1": BidirArmSpec(
            arm_id="A1",
            feature_builder="bidir_arm_a1_ab_only_v1",
            deployable=True,
            frame_arrays=a1_frames,
            sequence_scalars=("tau_ab_s",),
            physics_statistics=B1_SEVEN_STATS,
            physics_phase_windows=DEFAULT_PHASE_WINDOWS,
            physics_early_fractions=(),
        ),
        "A2": BidirArmSpec(
            arm_id="A2",
            feature_builder="bidir_arm_a2_ab_oracle_v_v1",
            deployable=False,
            frame_arrays=a1_frames,
            sequence_scalars=("tau_ab_s", ORACLE_V_SCALAR),
            physics_statistics=B1_SEVEN_STATS,
            physics_phase_windows=DEFAULT_PHASE_WINDOWS,
            physics_early_fractions=(),
        ),
        "A3": BidirArmSpec(
            arm_id="A3",
            feature_builder="bidir_arm_a3_pair_compact_v1",
            deployable=True,
            frame_arrays=a3_frames,
            sequence_scalars=SEQ_SCALARS_WITH_V,
            physics_statistics=B1_SEVEN_STATS,
            physics_phase_windows=DEFAULT_PHASE_WINDOWS,
            physics_early_fractions=(),
        ),
        "A4": BidirArmSpec(
            arm_id="A4",
            feature_builder="bidir_arm_a4_full_stats_v1",
            deployable=True,
            frame_arrays=a4_frames,
            sequence_scalars=SEQ_SCALARS_WITH_V,
            physics_statistics=B1_FULL_STATS,
            physics_phase_windows=DEFAULT_PHASE_WINDOWS,
            physics_early_fractions=DEFAULT_EARLY_FRACTIONS,
        ),
        "A5": BidirArmSpec(
            arm_id="A5",
            feature_builder="bidir_arm_a5_no_vhat_v1",
            deployable=True,
            frame_arrays=a5_frames,
            sequence_scalars=SEQ_SCALARS_BASE,
            physics_statistics=B1_SEVEN_STATS,
            physics_phase_windows=DEFAULT_PHASE_WINDOWS,
            physics_early_fractions=(),
        ),
    }


def default_frame_cache_dir(dataset_dir: Path | str) -> Path:
    return Path(dataset_dir) / BIDIR_FRAME_CACHE_ROOT


def default_arm_cache_dir(dataset_dir: Path | str, arm_id: str) -> Path:
    spec = arm_specs()[arm_id]
    return Path(dataset_dir) / "features" / "rocket" / spec.feature_builder


def load_frame_cache_arrays(
    cache_dir: Path | str,
    names: Sequence[str],
) -> dict[str, np.ndarray]:
    cache_dir = Path(cache_dir)
    out: dict[str, np.ndarray] = {}
    for name in names:
        path = cache_dir / f"{name}.npy"
        if not path.is_file():
            raise FileNotFoundError(f"missing bidir frame array: {path}")
        out[name] = np.load(path, mmap_mode="r")
    return out


def compute_shared_slow_windowed_block(
    *,
    slow: np.ndarray,
    slow_channel_names: Sequence[str],
    sequence_ids: Sequence[str],
    phase_lookup: Mapping[str, tuple[str, ...]],
) -> tuple[np.ndarray, tuple[str, ...]]:
    """Slow windowed block shared by all F5 arms within one split (35 dims)."""
    slow_arr = np.asarray(slow, dtype=np.float32)
    selected_slow, selected_names = _select_slow_channels(
        slow_arr, tuple(slow_channel_names), FORMAL_SLOW_CHANNELS
    )
    sequence_ids_t = tuple(sequence_ids)
    slow_block, slow_names = _windowed_sequence_features(
        selected_slow,
        sequence_ids=sequence_ids_t,
        channel_names=selected_names,
        phase_lookup=dict(phase_lookup),
        statistics=B1_FULL_STATS,
        source_prefix="slow",
        phase_windows=DEFAULT_PHASE_WINDOWS,
        early_fractions=DEFAULT_EARLY_FRACTIONS,
    )
    return slow_block, tuple(slow_names)


def assemble_arm_feature_matrix(
    *,
    slow: np.ndarray,
    slow_channel_names: Sequence[str],
    sequence_ids: Sequence[str],
    labels: np.ndarray,
    label_names: Sequence[str],
    phase_lookup: Mapping[str, tuple[str, ...]],
    frame_arrays: Mapping[str, np.ndarray],
    sequence_scalars: Mapping[str, np.ndarray],
    arm: BidirArmSpec,
    shared_slow_block: tuple[np.ndarray, tuple[str, ...]] | None = None,
) -> MLFeatureMatrix:
    """Assemble one arm matrix from already-extracted arrays."""
    if arm.deployable:
        assert_no_oracle_inputs(list(arm.frame_arrays) + list(arm.sequence_scalars))
    elif ORACLE_V_SCALAR not in arm.sequence_scalars:
        raise ValueError("audit arm A2 must include oracle_v_path_m_per_s")

    sequence_ids_t = tuple(sequence_ids)
    blocks: list[np.ndarray] = []
    feature_names: list[str] = []

    if arm.include_slow:
        if shared_slow_block is not None:
            slow_block, slow_names = shared_slow_block
            if slow_block.shape[0] != len(sequence_ids_t):
                raise ValueError("shared_slow_block row count mismatch")
        else:
            slow_block, slow_names = compute_shared_slow_windowed_block(
                slow=slow,
                slow_channel_names=slow_channel_names,
                sequence_ids=sequence_ids_t,
                phase_lookup=phase_lookup,
            )
        blocks.append(np.asarray(slow_block, dtype=np.float32))
        feature_names.extend(slow_names)

    for array_name in arm.frame_arrays:
        if array_name not in frame_arrays:
            raise KeyError(f"missing frame array {array_name!r} for arm {arm.arm_id}")
        values = np.asarray(frame_arrays[array_name], dtype=np.float32)
        if values.ndim != 2:
            raise ValueError(f"{array_name} must be (N, T), got {values.shape}")
        if values.shape[0] != len(sequence_ids_t):
            raise ValueError(f"{array_name} row count mismatch")
        block, names = _windowed_sequence_features(
            values[..., np.newaxis],
            sequence_ids=sequence_ids_t,
            channel_names=(array_name,),
            phase_lookup=dict(phase_lookup),
            statistics=arm.physics_statistics,
            source_prefix="physics",
            phase_windows=arm.physics_phase_windows,
            early_fractions=arm.physics_early_fractions,
        )
        blocks.append(block)
        feature_names.extend(names)

    for scalar_name in arm.sequence_scalars:
        if scalar_name not in sequence_scalars:
            raise KeyError(f"missing sequence scalar {scalar_name!r} for arm {arm.arm_id}")
        values = np.asarray(sequence_scalars[scalar_name], dtype=np.float32).reshape(-1, 1)
        if values.shape[0] != len(sequence_ids_t):
            raise ValueError(f"{scalar_name} row count mismatch")
        if not np.isfinite(values).all():
            raise ValueError(f"non-finite values in {scalar_name}")
        blocks.append(values)
        feature_names.append(f"seq|{scalar_name}")

    x = np.concatenate(blocks, axis=1).astype(np.float32, copy=False)
    if not np.isfinite(x).all():
        raise ValueError(f"non-finite features in arm {arm.arm_id}")
    return MLFeatureMatrix(
        x=x,
        y=np.asarray(labels, dtype=np.float32),
        feature_names=tuple(feature_names),
        label_names=tuple(label_names),
        sequence_ids=sequence_ids_t,
    )


def build_arm_feature_cache(
    dataset_dir: Path | str,
    arm_id: str,
    *,
    frame_cache_dir: Path | str | None = None,
    cache_dir: Path | str | None = None,
) -> dict[str, Any]:
    """Build per-split feature matrices for one arm into features/rocket/<builder>/."""
    return build_arm_feature_caches(
        dataset_dir,
        (arm_id,),
        frame_cache_dir=frame_cache_dir,
        cache_dirs={arm_id: cache_dir} if cache_dir is not None else None,
    )[arm_id]


def build_arm_feature_caches(
    dataset_dir: Path | str,
    arm_ids: Sequence[str],
    *,
    frame_cache_dir: Path | str | None = None,
    cache_dirs: Mapping[str, Path | str | None] | None = None,
) -> dict[str, dict[str, Any]]:
    """Build multiple arms, sharing the slow windowed block once per split."""
    dataset_dir = Path(dataset_dir)
    if not arm_ids:
        raise ValueError("arm_ids must be non-empty")
    specs = arm_specs()
    unknown = [arm_id for arm_id in arm_ids if arm_id not in specs]
    if unknown:
        raise KeyError(f"unknown arm_id(s): {unknown}")
    arms = [specs[arm_id] for arm_id in arm_ids]
    frame_cache_dir = Path(frame_cache_dir) if frame_cache_dir else default_frame_cache_dir(dataset_dir)
    resolved_cache_dirs: dict[str, Path] = {}
    for arm in arms:
        override = None if cache_dirs is None else cache_dirs.get(arm.arm_id)
        resolved_cache_dirs[arm.arm_id] = (
            Path(override) if override is not None else default_arm_cache_dir(dataset_dir, arm.arm_id)
        )
        resolved_cache_dirs[arm.arm_id].mkdir(parents=True, exist_ok=True)

    needed_frames = sorted({name for arm in arms for name in arm.frame_arrays})
    frame_arrays_all = load_frame_cache_arrays(frame_cache_dir, needed_frames)
    needed_scalars = sorted({name for arm in arms for name in arm.sequence_scalars})
    sequence_scalars_all: dict[str, np.ndarray] = {}
    for name in needed_scalars:
        path = frame_cache_dir / f"{name}.npy"
        if not path.is_file():
            raise FileNotFoundError(f"missing sequence scalar array: {path}")
        sequence_scalars_all[name] = np.load(path, mmap_mode="r")

    master_sequence_ids = _load_str_array(dataset_dir / "metadata" / "sequence_ids.npy")
    label_names = _load_str_array(dataset_dir / "metadata" / "label_names.npy")
    slow_names = _load_str_array(dataset_dir / "metadata" / "slow_channel_names.npy")
    labels = np.load(dataset_dir / "labels" / "y.npy").astype(np.float32)
    slow = np.load(dataset_dir / "sequences" / "slow.npy", mmap_mode="r")
    phase_lookup = _load_phase_lookup(dataset_dir / "sequences" / "slow_sequence_long.csv")
    splits = load_splits(dataset_dir / "splits")
    split_indices = resolve_split_indices(splits, master_sequence_ids)

    feature_names_by_arm: dict[str, tuple[str, ...] | None] = {arm.arm_id: None for arm in arms}
    split_counts_by_arm: dict[str, dict[str, int]] = {arm.arm_id: {} for arm in arms}

    for split_name, indices in split_indices.items():
        seq_ids = tuple(master_sequence_ids[i] for i in indices)
        slow_sub = np.asarray(slow[indices], dtype=np.float32)
        shared_slow = compute_shared_slow_windowed_block(
            slow=slow_sub,
            slow_channel_names=slow_names,
            sequence_ids=seq_ids,
            phase_lookup=phase_lookup,
        )
        for arm in arms:
            frame_sub = {
                k: np.asarray(frame_arrays_all[k][indices], dtype=np.float32) for k in arm.frame_arrays
            }
            scalar_sub = {
                k: np.asarray(sequence_scalars_all[k][indices], dtype=np.float32)
                for k in arm.sequence_scalars
            }
            matrix = assemble_arm_feature_matrix(
                slow=slow_sub,
                slow_channel_names=slow_names,
                sequence_ids=seq_ids,
                labels=labels[indices],
                label_names=label_names,
                phase_lookup=phase_lookup,
                frame_arrays=frame_sub,
                sequence_scalars=scalar_sub,
                arm=arm,
                shared_slow_block=shared_slow,
            )
            prev_names = feature_names_by_arm[arm.arm_id]
            if prev_names is None:
                feature_names_by_arm[arm.arm_id] = matrix.feature_names
            elif matrix.feature_names != prev_names:
                raise ValueError(f"feature names drifted on split {split_name} for arm {arm.arm_id}")
            cache_dir = resolved_cache_dirs[arm.arm_id]
            np.save(cache_dir / f"feature_matrix_{split_name}.npy", matrix.x)
            split_counts_by_arm[arm.arm_id][split_name] = len(seq_ids)

    manifests: dict[str, dict[str, Any]] = {}
    for arm in arms:
        feature_names = feature_names_by_arm[arm.arm_id]
        assert feature_names is not None
        cache_dir = resolved_cache_dirs[arm.arm_id]
        (cache_dir / "feature_names.json").write_text(
            json.dumps(list(feature_names), indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        manifest = {
            "schema_version": "tv3-bidir-arm-feature-1",
            "arm_id": arm.arm_id,
            "feature_builder": arm.feature_builder,
            "deployable": arm.deployable,
            "base_feature_builder": FORMAL_FEATURE_BUILDER,
            "frame_cache_dir": str(frame_cache_dir.resolve()),
            "dataset_dir": str(dataset_dir.resolve()),
            "feature_count": len(feature_names),
            "feature_names_digest": __import__("hashlib")
            .sha256("\n".join(feature_names).encode("utf-8"))
            .hexdigest(),
            "split_sequence_counts": split_counts_by_arm[arm.arm_id],
            "frame_arrays": list(arm.frame_arrays),
            "sequence_scalars": list(arm.sequence_scalars),
            "physics_statistics": list(arm.physics_statistics),
        }
        (cache_dir / "feature_manifest.json").write_text(
            json.dumps(manifest, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        manifests[arm.arm_id] = manifest
    return manifests


def load_arm_split_matrix(
    dataset_dir: Path | str,
    arm_id: str,
    *,
    split: str,
    cache_dir: Path | str | None = None,
) -> MLFeatureMatrix:
    dataset_dir = Path(dataset_dir)
    cache_dir = Path(cache_dir) if cache_dir else default_arm_cache_dir(dataset_dir, arm_id)
    master_sequence_ids = _load_str_array(dataset_dir / "metadata" / "sequence_ids.npy")
    label_names = _load_str_array(dataset_dir / "metadata" / "label_names.npy")
    splits = load_splits(dataset_dir / "splits")
    indices = resolve_split_indices(splits, master_sequence_ids)[split]
    sequence_ids = tuple(master_sequence_ids[i] for i in indices)
    x = np.load(cache_dir / f"feature_matrix_{split}.npy", mmap_mode="r")
    x = np.asarray(x, dtype=np.float32)
    y = np.load(dataset_dir / "labels" / "y.npy").astype(np.float32)[indices]
    feature_names = tuple(json.loads((cache_dir / "feature_names.json").read_text(encoding="utf-8")))
    if x.shape[0] != len(sequence_ids):
        raise ValueError(f"cached row mismatch for {arm_id}/{split}")
    return MLFeatureMatrix(
        x=x,
        y=y,
        feature_names=feature_names,
        label_names=tuple(label_names),
        sequence_ids=sequence_ids,
    )


__all__ = [
    "AB_COMPACT_FRAMES",
    "BA_COMPACT_FRAMES",
    "BIDIR_FRAME_CACHE_ROOT",
    "BidirArmSpec",
    "ORACLE_V_SCALAR",
    "PAIR_FRAMES",
    "arm_specs",
    "assemble_arm_feature_matrix",
    "build_arm_feature_cache",
    "build_arm_feature_caches",
    "compute_shared_slow_windowed_block",
    "default_arm_cache_dir",
    "default_frame_cache_dir",
    "load_arm_split_matrix",
    "load_frame_cache_arrays",
]
