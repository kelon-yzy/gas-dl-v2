"""E1d-SB deployable sequence features: cal_plus_corr_psr_snr semantics.

Builds the compact E1d parity set from RawDSP-extractable quantities only
(raw waveform + deployable slow). Does not include full B1 physics arrays,
peak_width, quality, or accepted flags.
"""
from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Literal, Mapping, Sequence

import numpy as np

from tv3.common.splits import load_splits, resolve_split_indices
from tv3.common.waveform import waveform_array_filename
from tv3.ml.features import MLFeatureMatrix
from tv3.ml.raw_dsp_features import (
    FORMAL_SLOW_CHANNELS,
    RawDSPConfig,
    RawDSPSequenceResult,
    extract_raw_dsp_sequence,
    fit_tof_vs_path_length_snr_weighted,
)
from tv3.ml.rocket_features import (
    DEFAULT_EARLY_FRACTIONS,
    DEFAULT_PHASE_WINDOWS,
    DEFAULT_ROCKET_SEQUENCE_STATISTICS,
    RAW_DSP_FRAME_CACHE_ROOT,
    _load_phase_lookup,
    _load_str_array,
    _select_slow_channels,
    _windowed_sequence_features,
)

E1DSB_FEATURE_BUILDER = "e1d_sb_cal_plus_corr_psr_snr_v1"
E1DSB_SPEC_NAME = "cal_plus_corr_psr_snr"
E1DSB_LS_FEATURE_BUILDER = "e1d_sb_cal_plus_corr_psr_snr_ls_v1"
E1DSB_LS_SPEC_NAME = "cal_plus_corr_psr_snr_ls"
E1DSB_LS_EXTRA_SCALARS = (
    "ultrasonic_tof_l_m_intercept_snr_weighted_ls_s",
    "ultrasonic_sound_speed_snr_weighted_ls_m_per_s",
)
FEATURE_SOURCE = Literal["raw_dsp_cache", "waveform"]

B1_SEVEN_STATS = ("mean", "std", "min", "max", "last", "delta", "slope")
B1_FULL_STATS = DEFAULT_ROCKET_SEQUENCE_STATISTICS

E1DSB_FRAME_ARRAYS = (
    "ultrasonic_peak_index_raw_dsp",
    "ultrasonic_tof_corrected_raw_dsp_s",
    "ultrasonic_sound_speed_raw_dsp_m_per_s",
    "ultrasonic_corr_peak",
    "ultrasonic_peak_to_sidelobe_ratio",
    "ultrasonic_snr_db",
)
E1DSB_SEQUENCE_SCALARS = (
    "ultrasonic_delay_calibration_s",
    "ultrasonic_tof_l_m_intercept_s",
    "ultrasonic_sound_speed_slope_raw_dsp_m_per_s",
)

_RESULT_FRAME_ATTRS = {
    "ultrasonic_peak_index_raw_dsp": "peak_index",
    "ultrasonic_tof_corrected_raw_dsp_s": "tof_corrected_s",
    "ultrasonic_sound_speed_raw_dsp_m_per_s": "sound_speed_m_per_s",
    "ultrasonic_corr_peak": "corr_peak",
    "ultrasonic_peak_to_sidelobe_ratio": "peak_to_sidelobe_ratio",
    "ultrasonic_snr_db": "snr_db",
}
_RESULT_SCALAR_ATTRS = {
    "ultrasonic_delay_calibration_s": "delay_calibration_s",
    "ultrasonic_tof_l_m_intercept_s": "tof_l_m_intercept_s",
    "ultrasonic_sound_speed_slope_raw_dsp_m_per_s": "sound_speed_slope_m_per_s",
}


@dataclass(frozen=True, slots=True)
class E1dSBBuilderInfo:
    feature_builder: str
    spec_name: str
    frame_arrays: tuple[str, ...]
    sequence_scalars: tuple[str, ...]
    physics_statistics: tuple[str, ...]
    physics_phase_windows: tuple[str, ...]
    physics_early_fractions: tuple[float, ...]
    slow_statistics: tuple[str, ...]
    slow_phase_windows: tuple[str, ...]
    slow_early_fractions: tuple[float, ...]


def e1d_sb_builder_info() -> E1dSBBuilderInfo:
    return E1dSBBuilderInfo(
        feature_builder=E1DSB_FEATURE_BUILDER,
        spec_name=E1DSB_SPEC_NAME,
        frame_arrays=E1DSB_FRAME_ARRAYS,
        sequence_scalars=E1DSB_SEQUENCE_SCALARS,
        physics_statistics=B1_SEVEN_STATS,
        physics_phase_windows=DEFAULT_PHASE_WINDOWS,
        physics_early_fractions=(),
        slow_statistics=B1_FULL_STATS,
        slow_phase_windows=DEFAULT_PHASE_WINDOWS,
        slow_early_fractions=DEFAULT_EARLY_FRACTIONS,
    )


def e1d_sb_ls_builder_info() -> E1dSBBuilderInfo:
    base = e1d_sb_builder_info()
    return E1dSBBuilderInfo(
        feature_builder=E1DSB_LS_FEATURE_BUILDER,
        spec_name=E1DSB_LS_SPEC_NAME,
        frame_arrays=base.frame_arrays,
        sequence_scalars=base.sequence_scalars + E1DSB_LS_EXTRA_SCALARS,
        physics_statistics=base.physics_statistics,
        physics_phase_windows=base.physics_phase_windows,
        physics_early_fractions=base.physics_early_fractions,
        slow_statistics=base.slow_statistics,
        slow_phase_windows=base.slow_phase_windows,
        slow_early_fractions=base.slow_early_fractions,
    )


def diagnostic_feature_count(feature_names: Sequence[str]) -> int:
    return sum("|slow:" not in name for name in feature_names)


def build_e1d_sb_feature_matrix(
    dataset_dir: Path | str,
    *,
    split: str,
    feature_source: FEATURE_SOURCE = "raw_dsp_cache",
) -> MLFeatureMatrix:
    """Build the compact E1d-SB matrix for one split.

    ``raw_dsp_cache`` reuses validated RawDSP frame arrays (bit-identical to E1d
    ``cal_plus_corr_psr_snr``). ``waveform`` re-runs ``extract_raw_dsp_sequence``
    with the frozen train-baseline template from the same cache.
    """
    dataset_dir = Path(dataset_dir)
    if feature_source == "raw_dsp_cache":
        return _build_from_cache(dataset_dir, split=split)
    if feature_source == "waveform":
        return _build_from_waveforms(dataset_dir, split=split)
    raise ValueError(f"unsupported feature_source: {feature_source!r}")


def frame_and_scalar_arrays_from_result(
    result: RawDSPSequenceResult,
) -> tuple[dict[str, np.ndarray], dict[str, float]]:
    """Map one RawDSP sequence result onto E1d-SB array names."""
    frames = {
        name: np.asarray(getattr(result, attr), dtype=np.float32)
        for name, attr in _RESULT_FRAME_ATTRS.items()
    }
    scalars = {
        name: float(getattr(result, attr)) for name, attr in _RESULT_SCALAR_ATTRS.items()
    }
    for name, values in frames.items():
        if values.ndim != 1:
            raise ValueError(f"{name} must be 1D, got {values.shape}")
        if not np.isfinite(values).all():
            raise ValueError(f"non-finite values in {name}")
    for name, value in scalars.items():
        if not np.isfinite(value):
            raise ValueError(f"non-finite sequence scalar {name}")
    return frames, scalars


def assemble_e1d_sb_feature_matrix(
    *,
    slow: np.ndarray,
    slow_channel_names: Sequence[str],
    sequence_ids: Sequence[str],
    labels: np.ndarray,
    label_names: Sequence[str],
    phase_lookup: Mapping[str, tuple[str, ...]],
    frame_arrays: Mapping[str, np.ndarray],
    sequence_scalars: Mapping[str, np.ndarray],
    builder_info: E1dSBBuilderInfo | None = None,
) -> MLFeatureMatrix:
    """Assemble the frozen E1d-SB window contract from already-extracted arrays."""
    info = e1d_sb_builder_info() if builder_info is None else builder_info
    sequence_ids_t = tuple(sequence_ids)
    blocks: list[np.ndarray] = []
    feature_names: list[str] = []

    slow_arr = np.asarray(slow, dtype=np.float32)
    if slow_arr.ndim != 3:
        raise ValueError(f"slow must be (N, T, C), got {slow_arr.shape}")
    selected_slow, selected_names = _select_slow_channels(
        slow_arr, tuple(slow_channel_names), FORMAL_SLOW_CHANNELS
    )
    slow_block, slow_names = _windowed_sequence_features(
        selected_slow,
        sequence_ids=sequence_ids_t,
        channel_names=selected_names,
        phase_lookup=dict(phase_lookup),
        statistics=info.slow_statistics,
        source_prefix="slow",
        phase_windows=info.slow_phase_windows,
        early_fractions=info.slow_early_fractions,
    )
    blocks.append(slow_block)
    feature_names.extend(slow_names)

    for array_name in info.frame_arrays:
        if array_name not in frame_arrays:
            raise KeyError(f"missing frame array {array_name!r}")
        values = np.asarray(frame_arrays[array_name], dtype=np.float32)
        if values.ndim != 2:
            raise ValueError(f"{array_name} must be (N, T), got {values.shape}")
        if values.shape[0] != len(sequence_ids_t):
            raise ValueError(
                f"{array_name} row count {values.shape[0]} != {len(sequence_ids_t)}"
            )
        block, names = _windowed_sequence_features(
            values[..., np.newaxis],
            sequence_ids=sequence_ids_t,
            channel_names=(array_name,),
            phase_lookup=dict(phase_lookup),
            statistics=info.physics_statistics,
            source_prefix="physics",
            phase_windows=info.physics_phase_windows,
            early_fractions=info.physics_early_fractions,
        )
        blocks.append(block)
        feature_names.extend(names)

    for array_name in info.sequence_scalars:
        if array_name not in sequence_scalars:
            raise KeyError(f"missing sequence scalar {array_name!r}")
        values = np.asarray(sequence_scalars[array_name], dtype=np.float32).reshape(-1, 1)
        if values.shape[0] != len(sequence_ids_t):
            raise ValueError(
                f"{array_name} length {values.shape[0]} != {len(sequence_ids_t)}"
            )
        if not np.isfinite(values).all():
            raise ValueError(f"non-finite sequence scalar values in {array_name}")
        blocks.append(values)
        feature_names.append(f"seq|{array_name}")

    if "ultrasonic_snr_db" not in info.frame_arrays:
        raise ValueError("E1d-SB builders must retain ultrasonic_snr_db frame features")

    x = np.concatenate(blocks, axis=1).astype(np.float32, copy=False)
    if not np.isfinite(x).all():
        raise ValueError("non-finite E1d-SB features")
    y = np.asarray(labels, dtype=np.float32)
    if y.shape[0] != x.shape[0]:
        raise ValueError(f"label row count {y.shape[0]} != feature rows {x.shape[0]}")
    return MLFeatureMatrix(
        x=x,
        y=y,
        feature_names=tuple(feature_names),
        label_names=tuple(label_names),
        sequence_ids=sequence_ids_t,
    )


def _split_context(dataset_dir: Path, *, split: str) -> dict[str, Any]:
    splits = load_splits(dataset_dir / "splits")
    master_sequence_ids = _load_str_array(dataset_dir / "metadata" / "sequence_ids.npy")
    split_indices = resolve_split_indices(splits, master_sequence_ids)[split]
    return {
        "master_sequence_ids": master_sequence_ids,
        "split_indices": split_indices,
        "sequence_ids": tuple(master_sequence_ids[index] for index in split_indices),
        "labels": np.load(dataset_dir / "labels" / "y.npy").astype(np.float32)[split_indices],
        "label_names": tuple(_load_str_array(dataset_dir / "metadata" / "label_names.npy")),
        "slow_names": tuple(_load_str_array(dataset_dir / "metadata" / "slow_channel_names.npy")),
        "phase_lookup": _load_phase_lookup(dataset_dir / "sequences" / "slow_sequence_long.csv"),
    }


def _build_from_cache(dataset_dir: Path, *, split: str) -> MLFeatureMatrix:
    ctx = _split_context(dataset_dir, split=split)
    raw_dsp_dir = dataset_dir / RAW_DSP_FRAME_CACHE_ROOT
    slow = np.load(dataset_dir / "sequences" / "slow.npy", mmap_mode="r")[ctx["split_indices"]]
    frame_arrays = {}
    for array_name in E1DSB_FRAME_ARRAYS:
        path = raw_dsp_dir / f"{array_name}.npy"
        if not path.is_file():
            raise FileNotFoundError(f"missing RawDSP frame array: {path}")
        values = np.asarray(np.load(path, mmap_mode="r")[ctx["split_indices"]], dtype=np.float32)
        frame_arrays[array_name] = values
    sequence_scalars = {}
    for array_name in E1DSB_SEQUENCE_SCALARS:
        path = raw_dsp_dir / f"{array_name}.npy"
        if not path.is_file():
            raise FileNotFoundError(f"missing RawDSP sequence scalar: {path}")
        values = np.asarray(np.load(path), dtype=np.float32)[ctx["split_indices"]]
        sequence_scalars[array_name] = values
    return assemble_e1d_sb_feature_matrix(
        slow=np.asarray(slow, dtype=np.float32),
        slow_channel_names=ctx["slow_names"],
        sequence_ids=ctx["sequence_ids"],
        labels=ctx["labels"],
        label_names=ctx["label_names"],
        phase_lookup=ctx["phase_lookup"],
        frame_arrays=frame_arrays,
        sequence_scalars=sequence_scalars,
    )


def _build_from_waveforms(dataset_dir: Path, *, split: str) -> MLFeatureMatrix:
    raw_dsp_dir = dataset_dir / RAW_DSP_FRAME_CACHE_ROOT
    manifest_path = raw_dsp_dir / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(
            f"waveform E1d-SB build requires RawDSP cache manifest for template/config: {manifest_path}"
        )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    template_path = raw_dsp_dir / "template.npy"
    if not template_path.is_file():
        raise FileNotFoundError(f"missing RawDSP template: {template_path}")

    template = np.asarray(np.load(template_path), dtype=np.float32)
    template_peak_offset = int(manifest["template_peak_offset_samples"])
    raw_dsp_payload = manifest.get("raw_dsp")
    if not isinstance(raw_dsp_payload, dict):
        raise ValueError("RawDSP manifest missing raw_dsp config object")
    config = RawDSPConfig(**{key: raw_dsp_payload[key] for key in RawDSPConfig.__dataclass_fields__})

    waveform_spec = json.loads(
        (dataset_dir / "metadata" / "waveform_spec.json").read_text(encoding="utf-8")
    )["ultrasonic"]
    waveform_dtype = str(waveform_spec["waveform_dtype"])
    daq_full_scale_v = float(waveform_spec["daq_full_scale_v"])
    waveform_path = dataset_dir / "sequences" / waveform_array_filename("ultrasonic", waveform_dtype)
    scale_path = dataset_dir / "sequences" / "ultrasonic_scale.npy"

    ctx = _split_context(dataset_dir, split=split)
    waveforms = np.load(waveform_path, mmap_mode="r")
    scales = np.load(scale_path, mmap_mode="r")
    slow_all = np.load(dataset_dir / "sequences" / "slow.npy", mmap_mode="r")

    frame_buffers = {name: [] for name in E1DSB_FRAME_ARRAYS}
    scalar_buffers = {name: [] for name in E1DSB_SEQUENCE_SCALARS}
    slow_rows: list[np.ndarray] = []

    for index in ctx["split_indices"]:
        sequence_id = ctx["master_sequence_ids"][index]
        phases = ctx["phase_lookup"].get(sequence_id)
        if phases is None:
            raise ValueError(f"phase CSV missing sequence_id={sequence_id!r}")
        result = extract_raw_dsp_sequence(
            waveforms[index],
            scales[index],
            slow_all[index],
            ctx["slow_names"],
            phases,
            template,
            daq_full_scale_v=daq_full_scale_v,
            config=config,
            template_peak_offset_samples=template_peak_offset,
        )
        frames, scalars = frame_and_scalar_arrays_from_result(result)
        for name in E1DSB_FRAME_ARRAYS:
            frame_buffers[name].append(frames[name])
        for name in E1DSB_SEQUENCE_SCALARS:
            scalar_buffers[name].append(scalars[name])
        slow_rows.append(np.asarray(slow_all[index], dtype=np.float32))

    return assemble_e1d_sb_feature_matrix(
        slow=np.stack(slow_rows, axis=0),
        slow_channel_names=ctx["slow_names"],
        sequence_ids=ctx["sequence_ids"],
        labels=ctx["labels"],
        label_names=ctx["label_names"],
        phase_lookup=ctx["phase_lookup"],
        frame_arrays={name: np.stack(rows, axis=0) for name, rows in frame_buffers.items()},
        sequence_scalars={
            name: np.asarray(values, dtype=np.float32) for name, values in scalar_buffers.items()
        },
    )


def builder_manifest_payload(*, include_snr_weighted_ls: bool = False) -> dict[str, Any]:
    info = e1d_sb_ls_builder_info() if include_snr_weighted_ls else e1d_sb_builder_info()
    notes = [
        "compact E1d set only; not full B1",
        "SNR is required; TOF-L alone is insufficient",
        "oracle TOF/true sound speed/true alpha/labels are not inputs",
    ]
    if include_snr_weighted_ls:
        notes.extend(
            [
                "additive SNR-weighted closed-form LS TOF-L scalars only",
                "frame ultrasonic_snr_db retained; LS does not replace SNR",
            ]
        )
    return {
        "feature_builder": info.feature_builder,
        "spec_name": info.spec_name,
        "frame_arrays": list(info.frame_arrays),
        "sequence_scalars": list(info.sequence_scalars),
        "physics_statistics": list(info.physics_statistics),
        "physics_phase_windows": list(info.physics_phase_windows),
        "physics_early_fractions": list(info.physics_early_fractions),
        "slow_statistics": list(info.slow_statistics),
        "slow_phase_windows": list(info.slow_phase_windows),
        "slow_early_fractions": list(info.slow_early_fractions),
        "include_snr_weighted_ls": include_snr_weighted_ls,
        "notes": notes,
    }


def build_e1d_sb_ls_feature_matrix(
    dataset_dir: Path | str,
    *,
    split: str,
    feature_source: FEATURE_SOURCE = "raw_dsp_cache",
    snr_ls_weight_mode: str = "amplitude",
) -> MLFeatureMatrix:
    """Build base E1d-SB features plus additive SNR-weighted LS TOF-L scalars."""
    dataset_dir = Path(dataset_dir)
    if feature_source == "raw_dsp_cache":
        return _build_ls_from_cache(
            dataset_dir, split=split, snr_ls_weight_mode=snr_ls_weight_mode
        )
    if feature_source == "waveform":
        return _build_ls_from_waveforms(
            dataset_dir, split=split, snr_ls_weight_mode=snr_ls_weight_mode
        )
    raise ValueError(f"unsupported feature_source: {feature_source!r}")


def _compute_snr_weighted_ls_scalars(
    *,
    tof_corrected: np.ndarray,
    path_lengths_m: np.ndarray,
    snr_db: np.ndarray,
    phase_ids: Sequence[str],
    accepted: np.ndarray | None,
    weight_mode: str,
) -> dict[str, float]:
    intercept, sound_speed = fit_tof_vs_path_length_snr_weighted(
        tof_corrected,
        path_lengths_m,
        snr_db,
        phase_ids,
        accepted,
        weight_mode=weight_mode,
    )
    return {
        "ultrasonic_tof_l_m_intercept_snr_weighted_ls_s": float(intercept),
        "ultrasonic_sound_speed_snr_weighted_ls_m_per_s": float(sound_speed),
    }


def _path_lengths_from_slow(slow_row: np.ndarray, slow_names: Sequence[str]) -> np.ndarray:
    if "L_m" not in slow_names:
        raise ValueError("slow channels missing L_m required for SNR-weighted LS")
    index = list(slow_names).index("L_m")
    values = np.asarray(slow_row[:, index], dtype=np.float64)
    if values.ndim != 1:
        raise ValueError(f"L_m must be 1D per sequence, got {values.shape}")
    return values


def _build_ls_from_cache(
    dataset_dir: Path,
    *,
    split: str,
    snr_ls_weight_mode: str,
) -> MLFeatureMatrix:
    ctx = _split_context(dataset_dir, split=split)
    raw_dsp_dir = dataset_dir / RAW_DSP_FRAME_CACHE_ROOT
    slow = np.asarray(
        np.load(dataset_dir / "sequences" / "slow.npy", mmap_mode="r")[ctx["split_indices"]],
        dtype=np.float32,
    )
    frame_arrays = {}
    for array_name in E1DSB_FRAME_ARRAYS:
        path = raw_dsp_dir / f"{array_name}.npy"
        if not path.is_file():
            raise FileNotFoundError(f"missing RawDSP frame array: {path}")
        values = np.asarray(np.load(path, mmap_mode="r")[ctx["split_indices"]], dtype=np.float32)
        frame_arrays[array_name] = values
    sequence_scalars = {}
    for array_name in E1DSB_SEQUENCE_SCALARS:
        path = raw_dsp_dir / f"{array_name}.npy"
        if not path.is_file():
            raise FileNotFoundError(f"missing RawDSP sequence scalar: {path}")
        values = np.asarray(np.load(path), dtype=np.float32)[ctx["split_indices"]]
        sequence_scalars[array_name] = values

    accepted_path = raw_dsp_dir / "ultrasonic_raw_dsp_accepted.npy"
    accepted_all = None
    if accepted_path.is_file():
        accepted_all = np.asarray(
            np.load(accepted_path, mmap_mode="r")[ctx["split_indices"]], dtype=bool
        )

    intercepts: list[float] = []
    speeds: list[float] = []
    for row_index, sequence_id in enumerate(ctx["sequence_ids"]):
        phases = ctx["phase_lookup"].get(sequence_id)
        if phases is None:
            raise ValueError(f"phase CSV missing sequence_id={sequence_id!r}")
        accepted_row = None if accepted_all is None else accepted_all[row_index]
        ls_scalars = _compute_snr_weighted_ls_scalars(
            tof_corrected=frame_arrays["ultrasonic_tof_corrected_raw_dsp_s"][row_index],
            path_lengths_m=_path_lengths_from_slow(slow[row_index], ctx["slow_names"]),
            snr_db=frame_arrays["ultrasonic_snr_db"][row_index],
            phase_ids=phases,
            accepted=accepted_row,
            weight_mode=snr_ls_weight_mode,
        )
        intercepts.append(ls_scalars["ultrasonic_tof_l_m_intercept_snr_weighted_ls_s"])
        speeds.append(ls_scalars["ultrasonic_sound_speed_snr_weighted_ls_m_per_s"])
    sequence_scalars["ultrasonic_tof_l_m_intercept_snr_weighted_ls_s"] = np.asarray(
        intercepts, dtype=np.float32
    )
    sequence_scalars["ultrasonic_sound_speed_snr_weighted_ls_m_per_s"] = np.asarray(
        speeds, dtype=np.float32
    )
    return assemble_e1d_sb_feature_matrix(
        slow=slow,
        slow_channel_names=ctx["slow_names"],
        sequence_ids=ctx["sequence_ids"],
        labels=ctx["labels"],
        label_names=ctx["label_names"],
        phase_lookup=ctx["phase_lookup"],
        frame_arrays=frame_arrays,
        sequence_scalars=sequence_scalars,
        builder_info=e1d_sb_ls_builder_info(),
    )


def _build_ls_from_waveforms(
    dataset_dir: Path,
    *,
    split: str,
    snr_ls_weight_mode: str,
) -> MLFeatureMatrix:
    raw_dsp_dir = dataset_dir / RAW_DSP_FRAME_CACHE_ROOT
    manifest_path = raw_dsp_dir / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(
            f"waveform E1d-SB LS build requires RawDSP cache manifest: {manifest_path}"
        )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    template_path = raw_dsp_dir / "template.npy"
    if not template_path.is_file():
        raise FileNotFoundError(f"missing RawDSP template: {template_path}")

    template = np.asarray(np.load(template_path), dtype=np.float32)
    template_peak_offset = int(manifest["template_peak_offset_samples"])
    raw_dsp_payload = manifest.get("raw_dsp")
    if not isinstance(raw_dsp_payload, dict):
        raise ValueError("RawDSP manifest missing raw_dsp config object")
    config = RawDSPConfig(**{key: raw_dsp_payload[key] for key in RawDSPConfig.__dataclass_fields__})

    waveform_spec = json.loads(
        (dataset_dir / "metadata" / "waveform_spec.json").read_text(encoding="utf-8")
    )["ultrasonic"]
    waveform_dtype = str(waveform_spec["waveform_dtype"])
    daq_full_scale_v = float(waveform_spec["daq_full_scale_v"])
    waveform_path = dataset_dir / "sequences" / waveform_array_filename("ultrasonic", waveform_dtype)
    scale_path = dataset_dir / "sequences" / "ultrasonic_scale.npy"

    ctx = _split_context(dataset_dir, split=split)
    waveforms = np.load(waveform_path, mmap_mode="r")
    scales = np.load(scale_path, mmap_mode="r")
    slow_all = np.load(dataset_dir / "sequences" / "slow.npy", mmap_mode="r")

    frame_buffers = {name: [] for name in E1DSB_FRAME_ARRAYS}
    scalar_buffers = {name: [] for name in E1DSB_SEQUENCE_SCALARS + E1DSB_LS_EXTRA_SCALARS}
    slow_rows: list[np.ndarray] = []

    for index in ctx["split_indices"]:
        sequence_id = ctx["master_sequence_ids"][index]
        phases = ctx["phase_lookup"].get(sequence_id)
        if phases is None:
            raise ValueError(f"phase CSV missing sequence_id={sequence_id!r}")
        result = extract_raw_dsp_sequence(
            waveforms[index],
            scales[index],
            slow_all[index],
            ctx["slow_names"],
            phases,
            template,
            daq_full_scale_v=daq_full_scale_v,
            config=config,
            template_peak_offset_samples=template_peak_offset,
        )
        frames, scalars = frame_and_scalar_arrays_from_result(result)
        ls_scalars = _compute_snr_weighted_ls_scalars(
            tof_corrected=frames["ultrasonic_tof_corrected_raw_dsp_s"],
            path_lengths_m=_path_lengths_from_slow(
                np.asarray(slow_all[index], dtype=np.float32), ctx["slow_names"]
            ),
            snr_db=frames["ultrasonic_snr_db"],
            phase_ids=phases,
            accepted=np.asarray(result.accepted, dtype=bool),
            weight_mode=snr_ls_weight_mode,
        )
        for name in E1DSB_FRAME_ARRAYS:
            frame_buffers[name].append(frames[name])
        for name in E1DSB_SEQUENCE_SCALARS:
            scalar_buffers[name].append(scalars[name])
        for name, value in ls_scalars.items():
            scalar_buffers[name].append(value)
        slow_rows.append(np.asarray(slow_all[index], dtype=np.float32))

    return assemble_e1d_sb_feature_matrix(
        slow=np.stack(slow_rows, axis=0),
        slow_channel_names=ctx["slow_names"],
        sequence_ids=ctx["sequence_ids"],
        labels=ctx["labels"],
        label_names=ctx["label_names"],
        phase_lookup=ctx["phase_lookup"],
        frame_arrays={name: np.stack(rows, axis=0) for name, rows in frame_buffers.items()},
        sequence_scalars={
            name: np.asarray(values, dtype=np.float32) for name, values in scalar_buffers.items()
        },
        builder_info=e1d_sb_ls_builder_info(),
    )
