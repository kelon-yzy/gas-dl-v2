"""Build train-calibrated ``raw_dsp_bidirectional_v1`` frame cache for F5 arms."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from tv3.common.splits import load_splits, resolve_split_indices
from tv3.common.waveform import waveform_array_filename
from tv3.ml.bidir_arm_features import BIDIR_FRAME_CACHE_ROOT, ORACLE_V_SCALAR, default_frame_cache_dir
from tv3.ml.bidir_features import (
    BIDIR_RAW_DSP_SCHEMA_VERSION,
    FEATURE_BUILDER,
    assert_no_oracle_inputs,
    build_direction_template,
    calibrate_session_delay_shared_s,
    extract_bidir_sequence,
    freeze_session_delay_calibration,
    session_calibration_as_dict,
    templates_digest_payload,
)
from tv3.ml.raw_dsp_features import RawDSPConfig, dequantize_waveforms
from tv3.sim.core.tunnel_ventilation_bidir_schema import COMPOSITION_SCHEME, SCHEMA_VERSION

logger = logging.getLogger(__name__)

FRAME_ARRAY_SPECS: dict[str, tuple[str, np.dtype[Any]]] = {
    "ultrasonic_tof_observed_ab_raw_dsp_s": ("tof_observed_ab_s", np.dtype(np.float32)),
    "ultrasonic_tof_observed_ba_raw_dsp_s": ("tof_observed_ba_s", np.dtype(np.float32)),
    "ultrasonic_tof_corrected_ab_raw_dsp_s": ("tof_corrected_ab_s", np.dtype(np.float32)),
    "ultrasonic_tof_corrected_ba_raw_dsp_s": ("tof_corrected_ba_s", np.dtype(np.float32)),
    "ultrasonic_peak_index_ab_raw_dsp": ("peak_index_ab", np.dtype(np.float32)),
    "ultrasonic_peak_index_ba_raw_dsp": ("peak_index_ba", np.dtype(np.float32)),
    "ultrasonic_snr_db_ab": ("snr_db_ab", np.dtype(np.float32)),
    "ultrasonic_snr_db_ba": ("snr_db_ba", np.dtype(np.float32)),
    "ultrasonic_psr_ab": ("peak_to_sidelobe_ratio_ab", np.dtype(np.float32)),
    "ultrasonic_psr_ba": ("peak_to_sidelobe_ratio_ba", np.dtype(np.float32)),
    "ultrasonic_quality_ab_raw_dsp": ("quality_ab", np.dtype(np.float32)),
    "ultrasonic_quality_ba_raw_dsp": ("quality_ba", np.dtype(np.float32)),
    "ultrasonic_accepted_ab_raw_dsp": ("accepted_ab", np.dtype(np.bool_)),
    "ultrasonic_accepted_ba_raw_dsp": ("accepted_ba", np.dtype(np.bool_)),
    "ultrasonic_sound_speed_pair_raw_dsp_m_per_s": ("sound_speed_m_per_s", np.dtype(np.float32)),
    "ultrasonic_v_path_hat_raw_dsp_m_per_s": ("v_path_m_per_s", np.dtype(np.float32)),
    "ultrasonic_reciprocity_residual_s": ("reciprocity_residual_s", np.dtype(np.float32)),
}

SEQ_SCALAR_SPECS: dict[str, str] = {
    "tau_ab_s": "tau_ab_s",
    "tau_ba_s": "tau_ba_s",
    "c_hat_seq_m_per_s": "sound_speed_seq_m_per_s",
    "v_hat_seq_m_per_s": "v_path_seq_m_per_s",
    "reciprocity_residual_p95_s": "reciprocity_residual_p95_s",
    "c_hat_dispersion_m_per_s": "sound_speed_dispersion_m_per_s",
}


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return payload


def _load_str_array(path: Path) -> tuple[str, ...]:
    values = np.load(path, allow_pickle=False)
    return tuple(str(item) for item in values.tolist())


def _load_phase_lookup(path: Path) -> dict[str, tuple[str, ...]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    lookup: dict[str, list[str]] = {}
    for row in rows:
        lookup.setdefault(row["sequence_id"], []).append(row["phase_id"])
    return {sid: tuple(phases) for sid, phases in lookup.items()}


def _select_baseline_frames(
    *,
    waveform: np.ndarray,
    scale: np.ndarray,
    slow: np.ndarray,
    path_index: int,
    train_indices: Sequence[int],
    sequence_ids: Sequence[str],
    phase_lookup: dict[str, tuple[str, ...]],
    max_frames: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    frames: list[np.ndarray] = []
    scales: list[float] = []
    path_lengths: list[float] = []
    for sequence_index in train_indices:
        sequence_id = sequence_ids[sequence_index]
        for timestep, phase in enumerate(phase_lookup[sequence_id]):
            if phase != "baseline":
                continue
            frames.append(np.asarray(waveform[sequence_index, timestep]))
            scales.append(float(scale[sequence_index, timestep]))
            path_lengths.append(float(slow[sequence_index, timestep, path_index]))
            if len(frames) >= max_frames:
                break
        if len(frames) >= max_frames:
            break
    if not frames:
        raise ValueError("train split contains no baseline frames for template construction")
    return (
        np.stack(frames),
        np.asarray(scales, dtype=np.float32),
        np.asarray(path_lengths, dtype=np.float64),
    )


def _direction_tof_for_calibration(
    *,
    waveform: np.ndarray,
    scale: np.ndarray,
    slow: np.ndarray,
    path_index: int,
    sequence_index: int,
    template: np.ndarray,
    template_peak_offset: int,
    daq_full_scale_v: float,
    config: RawDSPConfig,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    from tv3.ml.raw_dsp_features import extract_raw_dsp_frame

    waves = dequantize_waveforms(waveform[sequence_index], scale[sequence_index])
    path_lengths = np.asarray(slow[sequence_index, :, path_index], dtype=np.float64)
    peaks: list[float] = []
    accepted: list[bool] = []
    for wave, path_length_m in zip(waves, path_lengths, strict=True):
        frame = extract_raw_dsp_frame(
            wave,
            template,
            path_length_m=float(path_length_m),
            daq_full_scale_v=daq_full_scale_v,
            config=config,
            template_peak_offset_samples=template_peak_offset,
        )
        peaks.append(float(frame.peak_index))
        accepted.append(bool(frame.accepted))
    tof = np.asarray(peaks, dtype=np.float64) / float(config.sample_rate_hz)
    return tof, path_lengths, np.asarray(accepted, dtype=bool)


def build_tv3_bidir_feature_cache(
    dataset_dir: Path | str,
    *,
    cache_dir: Path | str | None = None,
    template_source_split: str = "train",
    template_max_frames: int = 512,
    template_pre_samples: int = 25,
    template_post_samples: int = 33,
    template_min_snr_db: float = 20.0,
    template_reference_peak_polarity: int = -1,
    raw_dsp_overrides: dict[str, Any] | None = None,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Extract bidirectional RawDSP frame + sequence scalars (train-calibrated)."""
    dataset_dir = Path(dataset_dir)
    cache_dir = Path(cache_dir) if cache_dir is not None else default_frame_cache_dir(dataset_dir)
    if cache_dir.exists() and any(cache_dir.iterdir()) and not overwrite:
        raise FileExistsError(
            f"bidir feature cache already exists: {cache_dir} (pass overwrite=True to rebuild)"
        )
    cache_dir.mkdir(parents=True, exist_ok=True)

    manifest = _read_json(dataset_dir / "manifest.json")
    if manifest.get("composition_scheme") != COMPOSITION_SCHEME:
        raise ValueError(
            f"expected composition_scheme={COMPOSITION_SCHEME!r}, "
            f"got {manifest.get('composition_scheme')!r}"
        )
    if manifest.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(
            f"expected schema_version={SCHEMA_VERSION!r}, got {manifest.get('schema_version')!r}"
        )

    waveform_spec = _read_json(dataset_dir / "metadata" / "waveform_spec.json")["ultrasonic"]
    sample_rate_hz = float(waveform_spec["sample_rate_hz"])
    daq_full_scale_v = float(waveform_spec["daq_full_scale_v"])
    carrier_hz = float(waveform_spec["center_frequency_hz"])
    overrides = dict(raw_dsp_overrides or {})
    if {"sample_rate_hz", "carrier_frequency_hz"} & set(overrides):
        raise ValueError("raw_dsp sample_rate_hz/carrier_frequency_hz are fixed by waveform metadata")
    config = RawDSPConfig(
        sample_rate_hz=sample_rate_hz,
        carrier_frequency_hz=carrier_hz,
        **overrides,
    )

    sequence_ids = _load_str_array(dataset_dir / "metadata" / "sequence_ids.npy")
    slow_names = _load_str_array(dataset_dir / "metadata" / "slow_channel_names.npy")
    path_index = list(slow_names).index("L_m")
    phase_lookup = _load_phase_lookup(dataset_dir / "sequences" / "slow_sequence_long.csv")
    splits = load_splits(dataset_dir / "splits")
    split_indices = resolve_split_indices(splits, list(sequence_ids))
    train_indices = list(split_indices[template_source_split])

    dtype = str(waveform_spec.get("waveform_dtype", "int16"))
    wave_ab = np.load(
        dataset_dir / "sequences" / waveform_array_filename("ultrasonic_ab", dtype),
        mmap_mode="r",
    )
    wave_ba = np.load(
        dataset_dir / "sequences" / waveform_array_filename("ultrasonic_ba", dtype),
        mmap_mode="r",
    )
    scale_ab = np.load(dataset_dir / "sequences" / "ultrasonic_ab_scale.npy", mmap_mode="r")
    scale_ba = np.load(dataset_dir / "sequences" / "ultrasonic_ba_scale.npy", mmap_mode="r")
    slow = np.load(dataset_dir / "sequences" / "slow.npy", mmap_mode="r")
    n_seq, n_t = int(wave_ab.shape[0]), int(wave_ab.shape[1])

    assert_no_oracle_inputs(["ultrasonic_ab", "ultrasonic_ba", "slow"])

    ab_frames, ab_scales, ab_paths = _select_baseline_frames(
        waveform=wave_ab,
        scale=scale_ab,
        slow=slow,
        path_index=path_index,
        train_indices=train_indices,
        sequence_ids=sequence_ids,
        phase_lookup=phase_lookup,
        max_frames=template_max_frames,
    )
    ba_frames, ba_scales, ba_paths = _select_baseline_frames(
        waveform=wave_ba,
        scale=scale_ba,
        slow=slow,
        path_index=path_index,
        train_indices=train_indices,
        sequence_ids=sequence_ids,
        phase_lookup=phase_lookup,
        max_frames=template_max_frames,
    )
    template_ab = build_direction_template(
        ab_frames,
        ab_scales,
        ab_paths,
        config=config,
        daq_full_scale_v=daq_full_scale_v,
        template_pre_samples=template_pre_samples,
        template_post_samples=template_post_samples,
        min_template_snr_db=template_min_snr_db,
        reference_peak_polarity=template_reference_peak_polarity,
    )
    template_ba = build_direction_template(
        ba_frames,
        ba_scales,
        ba_paths,
        config=config,
        daq_full_scale_v=daq_full_scale_v,
        template_pre_samples=template_pre_samples,
        template_post_samples=template_post_samples,
        min_template_snr_db=template_min_snr_db,
        reference_peak_polarity=template_reference_peak_polarity,
    )
    # Peak offset equals template pre-samples (same freeze as F3).
    peak_off_ab = int(template_pre_samples)
    peak_off_ba = int(template_pre_samples)

    # Session τ̂: train-only mid-pair shared intercept (F3 freeze; asymmetry≈0).
    cal_tofs_ab: list[np.ndarray] = []
    cal_tofs_ba: list[np.ndarray] = []
    cal_paths: list[np.ndarray] = []
    cal_phases: list[tuple[str, ...]] = []
    cal_acc_ab: list[np.ndarray] = []
    cal_acc_ba: list[np.ndarray] = []
    for sequence_index in train_indices:
        phases = phase_lookup[sequence_ids[sequence_index]]
        tof_ab, paths, acc_ab = _direction_tof_for_calibration(
            waveform=wave_ab,
            scale=scale_ab,
            slow=slow,
            path_index=path_index,
            sequence_index=sequence_index,
            template=template_ab,
            template_peak_offset=peak_off_ab,
            daq_full_scale_v=daq_full_scale_v,
            config=config,
        )
        tof_ba, _, acc_ba = _direction_tof_for_calibration(
            waveform=wave_ba,
            scale=scale_ba,
            slow=slow,
            path_index=path_index,
            sequence_index=sequence_index,
            template=template_ba,
            template_peak_offset=peak_off_ba,
            daq_full_scale_v=daq_full_scale_v,
            config=config,
        )
        cal_tofs_ab.append(tof_ab)
        cal_tofs_ba.append(tof_ba)
        cal_paths.append(paths)
        cal_phases.append(phases)
        cal_acc_ab.append(acc_ab)
        cal_acc_ba.append(acc_ba)

    tau_shared, c_eff_shared, n_shared = calibrate_session_delay_shared_s(
        cal_tofs_ab,
        cal_tofs_ba,
        cal_paths,
        cal_phases,
        cal_acc_ab,
        cal_acc_ba,
    )
    calibration = freeze_session_delay_calibration(
        tau_ab_s=tau_shared,
        tau_ba_s=tau_shared,
        c_eff_ab_m_per_s=c_eff_shared,
        c_eff_ba_m_per_s=c_eff_shared,
        n_sequences_ab=n_shared,
        n_sequences_ba=n_shared,
        method="train_steady_shared_midpair_theilsen_v1",
    )

    # Allocate outputs.
    frame_store: dict[str, np.ndarray] = {
        name: np.empty((n_seq, n_t), dtype=dtype)
        for name, (_attr, dtype) in FRAME_ARRAY_SPECS.items()
    }
    sound_speed_ab = np.empty((n_seq, n_t), dtype=np.float32)
    sound_speed_ba = np.empty((n_seq, n_t), dtype=np.float32)
    seq_store: dict[str, np.ndarray] = {
        name: np.empty((n_seq,), dtype=np.float32) for name in SEQ_SCALAR_SPECS
    }

    for seq_idx in range(n_seq):
        result = extract_bidir_sequence(
            wave_ab[seq_idx],
            scale_ab[seq_idx],
            wave_ba[seq_idx],
            scale_ba[seq_idx],
            slow[seq_idx],
            slow_names,
            template_ab=template_ab,
            template_ba=template_ba,
            calibration=calibration,
            daq_full_scale_v=daq_full_scale_v,
            config=config,
            template_peak_offset_ab=peak_off_ab,
            template_peak_offset_ba=peak_off_ba,
        )
        for name, (attr, _dtype) in FRAME_ARRAY_SPECS.items():
            frame_store[name][seq_idx] = np.asarray(getattr(result, attr), dtype=_dtype)
        # Unidirectional (flow-polluted) sound speeds for A1/A2.
        path_lengths = np.asarray(slow[seq_idx, :, path_index], dtype=np.float64)
        t_ab = np.asarray(result.tof_corrected_ab_s, dtype=np.float64)
        t_ba = np.asarray(result.tof_corrected_ba_s, dtype=np.float64)
        with np.errstate(divide="ignore", invalid="ignore"):
            c_ab = np.where(t_ab > 0.0, path_lengths / t_ab, np.nan)
            c_ba = np.where(t_ba > 0.0, path_lengths / t_ba, np.nan)
        sound_speed_ab[seq_idx] = c_ab.astype(np.float32)
        sound_speed_ba[seq_idx] = c_ba.astype(np.float32)
        for name, attr in SEQ_SCALAR_SPECS.items():
            seq_store[name][seq_idx] = float(getattr(result, attr))
        if (seq_idx + 1) % 50 == 0 or seq_idx + 1 == n_seq:
            logger.info("bidir feature extract %s/%s", seq_idx + 1, n_seq)

    for name, array in frame_store.items():
        np.save(cache_dir / f"{name}.npy", array)
    np.save(cache_dir / "ultrasonic_sound_speed_ab_raw_dsp_m_per_s.npy", sound_speed_ab)
    np.save(cache_dir / "ultrasonic_sound_speed_ba_raw_dsp_m_per_s.npy", sound_speed_ba)
    for name, array in seq_store.items():
        np.save(cache_dir / f"{name}.npy", array)

    # Audit-only oracle v (A2). Stored beside cache; never listed in deployable arm names.
    conditions_path = dataset_dir / "condition_grid_sequence.csv"
    with conditions_path.open(encoding="utf-8", newline="") as handle:
        cond_rows = list(csv.DictReader(handle))
    cond_by_sid = {row["sequence_id"]: float(row["v_path_m_per_s"]) for row in cond_rows}
    oracle_v = np.asarray([cond_by_sid[sid] for sid in sequence_ids], dtype=np.float32)
    np.save(cache_dir / f"{ORACLE_V_SCALAR}.npy", oracle_v)

    np.save(cache_dir / "template_ab.npy", template_ab.astype(np.float32, copy=False))
    np.save(cache_dir / "template_ba.npy", template_ba.astype(np.float32, copy=False))
    (cache_dir / "session_delay_calibration.json").write_text(
        json.dumps(session_calibration_as_dict(calibration), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    digests = templates_digest_payload(template_ab, template_ba)
    build_signature = hashlib.sha256(
        json.dumps(
            {
                "feature_builder": FEATURE_BUILDER,
                "schema": BIDIR_RAW_DSP_SCHEMA_VERSION,
                "template_source_split": template_source_split,
                "calibration_digest": calibration.digest,
                **digests,
            },
            sort_keys=True,
        ).encode("utf-8")
    ).hexdigest()
    payload = {
        "schema_version": BIDIR_RAW_DSP_SCHEMA_VERSION,
        "feature_builder": FEATURE_BUILDER,
        "dataset_dir": str(dataset_dir.resolve()),
        "cache_dir": str(cache_dir.resolve()),
        "cache_relpath": str(BIDIR_FRAME_CACHE_ROOT).replace("\\", "/"),
        "template_source_split": template_source_split,
        "template_digests": digests,
        "template_ab_digest": digests["template_ab_digest"],
        "template_ba_digest": digests["template_ba_digest"],
        "session_delay_calibration": session_calibration_as_dict(calibration),
        "build_signature": build_signature,
        "sequence_count": n_seq,
        "timesteps": n_t,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "oracle_arrays_for_audit_only": [ORACLE_V_SCALAR],
        "deploy_arrays": sorted(
            list(FRAME_ARRAY_SPECS)
            + [
                "ultrasonic_sound_speed_ab_raw_dsp_m_per_s",
                "ultrasonic_sound_speed_ba_raw_dsp_m_per_s",
            ]
            + list(SEQ_SCALAR_SPECS)
        ),
    }
    (cache_dir / "manifest.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--template-source-split", default="train")
    parser.add_argument("--template-max-frames", type=int, default=512)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    args = build_parser().parse_args(argv)
    payload = build_tv3_bidir_feature_cache(
        args.dataset_dir,
        cache_dir=args.cache_dir,
        template_source_split=args.template_source_split,
        template_max_frames=args.template_max_frames,
        overwrite=args.overwrite,
    )
    print(json.dumps({"cache_dir": payload["cache_dir"], "build_signature": payload["build_signature"]}, indent=2))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
