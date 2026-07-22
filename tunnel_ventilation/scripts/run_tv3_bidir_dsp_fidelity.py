#!/usr/bin/env python3
"""F3 audit: bidirectional RawDSP fidelity + session delay + reciprocal-sum recovery."""
from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import numpy as np
from scipy.stats import theilslopes

_TV3_ROOT = Path(__file__).resolve().parents[1]
if str(_TV3_ROOT) not in sys.path:
    sys.path.insert(0, str(_TV3_ROOT))

from tv3.common.splits import load_splits, resolve_split_indices  # noqa: E402
from tv3.common.waveform import waveform_array_filename  # noqa: E402
from tv3.ml.bidir_features import (  # noqa: E402
    BIDIR_RAW_DSP_SCHEMA_VERSION,
    FEATURE_BUILDER,
    assert_no_oracle_inputs,
    build_direction_template,
    calibrate_session_delay_s,
    calibrate_session_delay_shared_s,
    extract_bidir_sequence,
    freeze_session_delay_calibration,
    session_calibration_as_dict,
    templates_digest_payload,
    true_fixed_delay_s,
)
from tv3.ml.raw_dsp_features import RawDSPConfig, dequantize_waveforms, template_digest  # noqa: E402
from tv3.sim.core.tunnel_ventilation_bidir_schema import (  # noqa: E402
    COMPOSITION_SCHEME,
    SCHEMA_VERSION,
)
from tv3.sim.generation.tunnel_ventilation.bidir_registry import (  # noqa: E402
    default_config_dir,
)
from tv3.sim.generation.tunnel_ventilation.conditions import (  # noqa: E402
    COMPOSITION_DOMAIN_NARROW,
    COMPOSITION_DOMAIN_WIDE,
)


AUDIT_SCHEMA_VERSION = "tv3-bidir-dsp-fidelity-1"


@dataclass(frozen=True, slots=True)
class F3Thresholds:
    peak_p95_abs_samples: float = 0.25
    tau_abs_error_s: float = 0.10e-6
    sound_speed_bias_m_per_s: float = 0.05
    v_path_bias_m_per_s: float = 0.05
    reciprocity_residual_p95_s: float = 0.10e-6


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--config",
        type=Path,
        default=_TV3_ROOT / "configs" / "tv3_bidir_dsp_fidelity.json",
    )
    p.add_argument("--dataset-dir", type=Path, default=None)
    p.add_argument("--output-dir", type=Path, default=None)
    p.add_argument("--allow-overwrite", action="store_true")
    return p.parse_args(argv)


def _load_config(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("config must be a JSON object")
    return payload


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


def _prepare_output_dir(path: Path, *, allow_overwrite: bool) -> None:
    if path.exists():
        if not allow_overwrite:
            existing = sorted(p.name for p in path.iterdir())
            if existing:
                raise FileExistsError(
                    f"output dir already has files and overwrite is disabled: {path} ({existing})"
                )
    path.mkdir(parents=True, exist_ok=True)


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


def _snr_weighted_mean_pair(
    c_frames: np.ndarray,
    v_frames: np.ndarray,
    weights: np.ndarray,
    accepted: np.ndarray,
) -> tuple[float, float]:
    mask = accepted & np.isfinite(c_frames) & np.isfinite(v_frames) & np.isfinite(weights)
    if not np.any(mask):
        return float("nan"), float("nan")
    w = np.clip(weights[mask], 1e-12, None)
    w = w / float(np.sum(w))
    return float(np.sum(w * c_frames[mask])), float(np.sum(w * v_frames[mask]))


def _direction_tof_for_calibration(
    *,
    waveform: np.ndarray,
    scale: np.ndarray,
    slow: np.ndarray,
    path_index: int,
    sequence_index: int,
    phase_ids: Sequence[str],
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


def _peak_metrics(err: np.ndarray) -> dict[str, float]:
    finite = err[np.isfinite(err)]
    if finite.size == 0:
        return {
            "n_frames": 0.0,
            "mae_samples": float("nan"),
            "p95_abs_samples": float("nan"),
            "bias_samples": float("nan"),
            "max_abs_samples": float("nan"),
        }
    return {
        "n_frames": float(finite.size),
        "mae_samples": float(np.mean(np.abs(finite))),
        "p95_abs_samples": float(np.percentile(np.abs(finite), 95)),
        "bias_samples": float(np.mean(finite)),
        "max_abs_samples": float(np.max(np.abs(finite))),
    }


def _wide_stress_peak_gate_passed(
    *,
    max_ab: float,
    max_ba: float,
    n_stress: int,
    limit: float,
) -> bool:
    """Wide stress peak gate: non-finite errors fail (no vacuous pass)."""
    if n_stress <= 0:
        return True
    return bool(
        np.isfinite(max_ab)
        and np.isfinite(max_ba)
        and float(max_ab) <= limit
        and float(max_ba) <= limit
    )


def _wide_stress_peak_report(
    *,
    peak_err_ab: np.ndarray,
    peak_err_ba: np.ndarray,
    labels: np.ndarray,
    slow: np.ndarray,
    path_index: int,
    co2_min_vol_pct: float,
    l_m_min: float,
) -> dict[str, Any]:
    """Peak errors on high-CO2 + long-L frames (wide-domain physical risk set)."""
    n_seq, n_t = peak_err_ab.shape
    co2 = np.asarray(labels[:, 0], dtype=np.float64)  # x_CO2
    mask = np.zeros((n_seq, n_t), dtype=bool)
    for si in range(n_seq):
        if co2[si] < co2_min_vol_pct:
            continue
        l_row = np.asarray(slow[si, :, path_index], dtype=np.float64)
        mask[si] = l_row >= l_m_min
    return {
        "co2_min_vol_pct": co2_min_vol_pct,
        "l_m_min": l_m_min,
        "n_sequences_high_co2": int(np.sum(co2 >= co2_min_vol_pct)),
        "n_frames": int(np.sum(mask)),
        "ab": _peak_metrics(np.where(mask, peak_err_ab, np.nan)),
        "ba": _peak_metrics(np.where(mask, peak_err_ba, np.nan)),
    }


def run_f3_dsp_fidelity(
    *,
    dataset_dir: Path,
    output_dir: Path,
    config: dict[str, Any],
    allow_overwrite: bool = False,
) -> dict[str, Any]:
    thresholds = F3Thresholds(**config.get("thresholds", {}))
    template_cfg = config.get("template", {})
    raw_dsp_overrides = dict(config.get("raw_dsp", {}))
    composition_domain = str(config.get("composition_domain", COMPOSITION_DOMAIN_NARROW))

    dataset_dir = Path(dataset_dir)
    output_dir = Path(output_dir)
    _prepare_output_dir(output_dir, allow_overwrite=allow_overwrite)

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
    man_domain = (manifest.get("sim_revision") or {}).get(
        "composition_domain", COMPOSITION_DOMAIN_NARROW
    )
    if man_domain != composition_domain:
        raise ValueError(
            f"config composition_domain={composition_domain!r} != "
            f"manifest sim_revision.composition_domain={man_domain!r}"
        )

    waveform_spec = _read_json(dataset_dir / "metadata" / "waveform_spec.json")["ultrasonic"]
    sample_rate_hz = float(waveform_spec["sample_rate_hz"])
    daq_full_scale_v = float(waveform_spec["daq_full_scale_v"])
    carrier_hz = float(waveform_spec["center_frequency_hz"])
    system_delay_s = float(waveform_spec["system_delay_s"])
    cable_delay_s = float(waveform_spec["cable_delay_s"])

    if {"sample_rate_hz", "carrier_frequency_hz"} & set(raw_dsp_overrides):
        raise ValueError("raw_dsp sample_rate_hz/carrier_frequency_hz are fixed by waveform metadata")
    config_dsp = RawDSPConfig(
        sample_rate_hz=sample_rate_hz,
        carrier_frequency_hz=carrier_hz,
        **raw_dsp_overrides,
    )

    sequence_ids = _load_str_array(dataset_dir / "metadata" / "sequence_ids.npy")
    slow_names = _load_str_array(dataset_dir / "metadata" / "slow_channel_names.npy")
    path_index = list(slow_names).index("L_m")
    phase_lookup = _load_phase_lookup(dataset_dir / "sequences" / "slow_sequence_long.csv")
    splits = load_splits(dataset_dir / "splits")
    split_indices = resolve_split_indices(splits, list(sequence_ids))
    train_indices = list(split_indices["train"])

    ab_path = dataset_dir / "sequences" / waveform_array_filename("ultrasonic_ab", "int16")
    ba_path = dataset_dir / "sequences" / waveform_array_filename("ultrasonic_ba", "int16")
    wave_ab = np.load(ab_path, mmap_mode="r")
    wave_ba = np.load(ba_path, mmap_mode="r")
    scale_ab = np.load(dataset_dir / "sequences" / "ultrasonic_ab_scale.npy", mmap_mode="r")
    scale_ba = np.load(dataset_dir / "sequences" / "ultrasonic_ba_scale.npy", mmap_mode="r")
    slow = np.load(dataset_dir / "sequences" / "slow.npy", mmap_mode="r")
    tof_obs_ab = np.load(dataset_dir / "sequences" / "ultrasonic_tof_observed_ab_s.npy")
    tof_obs_ba = np.load(dataset_dir / "sequences" / "ultrasonic_tof_observed_ba_s.npy")
    c_true = np.load(dataset_dir / "sequences" / "ultrasonic_sound_speed_m_per_s.npy")
    v_true = np.load(dataset_dir / "sequences" / "ultrasonic_v_path_true_m_per_s.npy")
    labels = np.load(dataset_dir / "labels" / "y.npy")

    # Deploy path must not accept oracle names as feature inputs.
    assert_no_oracle_inputs(
        [
            "ultrasonic_ab",
            "ultrasonic_ba",
            "slow",
            "ultrasonic_tof_observed_ab_s",
            "ultrasonic_tof_observed_ba_s",
        ]
    )

    max_frames = int(template_cfg.get("max_frames", 512))
    pre = int(template_cfg.get("pre_samples", 25))
    post = int(template_cfg.get("post_samples", 33))
    min_snr = float(template_cfg.get("min_snr_db", 20.0))
    polarity = int(template_cfg.get("reference_peak_polarity", -1))

    ab_frames, ab_scales, ab_paths = _select_baseline_frames(
        waveform=wave_ab,
        scale=scale_ab,
        slow=slow,
        path_index=path_index,
        train_indices=train_indices,
        sequence_ids=sequence_ids,
        phase_lookup=phase_lookup,
        max_frames=max_frames,
    )
    ba_frames, ba_scales, ba_paths = _select_baseline_frames(
        waveform=wave_ba,
        scale=scale_ba,
        slow=slow,
        path_index=path_index,
        train_indices=train_indices,
        sequence_ids=sequence_ids,
        phase_lookup=phase_lookup,
        max_frames=max_frames,
    )
    template_ab = build_direction_template(
        ab_frames,
        ab_scales,
        ab_paths,
        config=config_dsp,
        daq_full_scale_v=daq_full_scale_v,
        template_pre_samples=pre,
        template_post_samples=post,
        min_template_snr_db=min_snr,
        reference_peak_polarity=polarity,
    )
    template_ba = build_direction_template(
        ba_frames,
        ba_scales,
        ba_paths,
        config=config_dsp,
        daq_full_scale_v=daq_full_scale_v,
        template_pre_samples=pre,
        template_post_samples=post,
        min_template_snr_db=min_snr,
        reference_peak_polarity=polarity,
    )
    template_peak_offset_ab = pre
    template_peak_offset_ba = pre

    # Session delay: train-only per-sequence steady Theil–Sen → median τ̂
    tof_ab_seq: list[np.ndarray] = []
    path_ab_seq: list[np.ndarray] = []
    phase_ab_seq: list[tuple[str, ...]] = []
    acc_ab_seq: list[np.ndarray] = []
    tof_ba_seq: list[np.ndarray] = []
    path_ba_seq: list[np.ndarray] = []
    phase_ba_seq: list[tuple[str, ...]] = []
    acc_ba_seq: list[np.ndarray] = []
    for sequence_index in train_indices:
        sid = sequence_ids[sequence_index]
        phases = phase_lookup[sid]
        tof_ab, path_ab, acc_ab = _direction_tof_for_calibration(
            waveform=wave_ab,
            scale=scale_ab,
            slow=slow,
            path_index=path_index,
            sequence_index=sequence_index,
            phase_ids=phases,
            template=template_ab,
            template_peak_offset=template_peak_offset_ab,
            daq_full_scale_v=daq_full_scale_v,
            config=config_dsp,
        )
        tof_ba, path_ba, acc_ba = _direction_tof_for_calibration(
            waveform=wave_ba,
            scale=scale_ba,
            slow=slow,
            path_index=path_index,
            sequence_index=sequence_index,
            phase_ids=phases,
            template=template_ba,
            template_peak_offset=template_peak_offset_ba,
            daq_full_scale_v=daq_full_scale_v,
            config=config_dsp,
        )
        tof_ab_seq.append(tof_ab)
        path_ab_seq.append(path_ab)
        phase_ab_seq.append(phases)
        acc_ab_seq.append(acc_ab)
        tof_ba_seq.append(tof_ba)
        path_ba_seq.append(path_ba)
        phase_ba_seq.append(phases)
        acc_ba_seq.append(acc_ba)

    cond_path = dataset_dir / "condition_grid_sequence.csv"
    with cond_path.open("r", encoding="utf-8", newline="") as handle:
        cond_rows = list(csv.DictReader(handle))
    asymmetry_by_sid = {
        row["sequence_id"]: float(row.get("delay_asymmetry_s", 0.0)) for row in cond_rows
    }

    tau_ab, c_eff_ab, n_ab = calibrate_session_delay_s(
        tof_ab_seq, path_ab_seq, phase_ab_seq, acc_ab_seq
    )
    tau_ba, c_eff_ba, n_ba = calibrate_session_delay_s(
        tof_ba_seq, path_ba_seq, phase_ba_seq, acc_ba_seq
    )
    # Default contract: delay_asymmetry_s=0 → shared τ (avoids fake v̂ from τ̂_ab−τ̂_ba noise).
    train_asym_values = [
        float(asymmetry_by_sid.get(sequence_ids[i], 0.0)) for i in train_indices
    ]
    max_train_asym = max(abs(v) for v in train_asym_values) if train_asym_values else 0.0
    delay_method = "train_steady_theilsen_median_intercept_v1"
    if max_train_asym <= 1e-15:
        tau_shared, c_eff_shared, n_shared = calibrate_session_delay_shared_s(
            tof_ab_seq,
            tof_ba_seq,
            path_ab_seq,
            phase_ab_seq,
            acc_ab_seq,
            acc_ba_seq,
        )
        tau_ab = tau_ba = tau_shared
        c_eff_ab = c_eff_ba = c_eff_shared
        n_ab = n_ba = n_shared
        delay_method = "train_steady_shared_midpair_theilsen_v1"
    calibration = freeze_session_delay_calibration(
        tau_ab_s=tau_ab,
        tau_ba_s=tau_ba,
        c_eff_ab_m_per_s=c_eff_ab,
        c_eff_ba_m_per_s=c_eff_ba,
        n_sequences_ab=n_ab,
        n_sequences_ba=n_ba,
        method=delay_method,
    )

    n_seq, n_t = wave_ab.shape[0], wave_ab.shape[1]
    peak_err_ab = np.full((n_seq, n_t), np.nan, dtype=np.float64)
    peak_err_ba = np.full((n_seq, n_t), np.nan, dtype=np.float64)
    c_hat_seq = np.full(n_seq, np.nan, dtype=np.float64)
    v_hat_seq = np.full(n_seq, np.nan, dtype=np.float64)
    c_true_seq = np.full(n_seq, np.nan, dtype=np.float64)
    v_true_seq = np.full(n_seq, np.nan, dtype=np.float64)
    rec_p95_seq = np.full(n_seq, np.nan, dtype=np.float64)
    seq_mean_reciprocity: list[float] = []

    for sequence_index, sequence_id in enumerate(sequence_ids):
        result = extract_bidir_sequence(
            wave_ab[sequence_index],
            scale_ab[sequence_index],
            wave_ba[sequence_index],
            scale_ba[sequence_index],
            slow[sequence_index],
            slow_names,
            template_ab=template_ab,
            template_ba=template_ba,
            calibration=calibration,
            daq_full_scale_v=daq_full_scale_v,
            config=config_dsp,
            template_peak_offset_ab=template_peak_offset_ab,
            template_peak_offset_ba=template_peak_offset_ba,
        )
        ref_peak_ab = np.asarray(tof_obs_ab[sequence_index], dtype=np.float64) * sample_rate_hz
        ref_peak_ba = np.asarray(tof_obs_ba[sequence_index], dtype=np.float64) * sample_rate_hz
        peak_err_ab[sequence_index] = result.peak_index_ab.astype(np.float64) - ref_peak_ab
        peak_err_ba[sequence_index] = result.peak_index_ba.astype(np.float64) - ref_peak_ba

        # Physics gate uses steady multi-L frames only. Full-sequence ĉ_seq mixes
        # baseline/exposure/recovery compositions and is not the F3 recovery target.
        phase_ids = phase_lookup[sequence_id]
        steady_accepted = np.asarray(
            [
                phase == "steady" and bool(acc)
                for phase, acc in zip(phase_ids, result.accepted_pair, strict=True)
            ],
            dtype=bool,
        )
        c_frames = result.sound_speed_m_per_s.astype(np.float64)
        v_frames = result.v_path_m_per_s.astype(np.float64)
        snr_w = np.asarray(
            [
                0.5
                * (
                    10.0 ** (0.05 * float(snr_ab))
                    + 10.0 ** (0.05 * float(snr_ba))
                )
                for snr_ab, snr_ba in zip(result.snr_db_ab, result.snr_db_ba, strict=True)
            ],
            dtype=np.float64,
        )
        c_hat_seq[sequence_index], v_hat_seq[sequence_index] = _snr_weighted_mean_pair(
            c_frames, v_frames, snr_w, steady_accepted
        )
        if int(np.sum(steady_accepted)) > 0:
            c_true_seq[sequence_index] = float(np.median(c_true[sequence_index][steady_accepted]))
            v_true_seq[sequence_index] = float(np.median(v_true[sequence_index][steady_accepted]))
        rec_p95_seq[sequence_index] = result.reciprocity_residual_p95_s
        finite_rec = result.reciprocity_residual_s[
            steady_accepted & np.isfinite(result.reciprocity_residual_s)
        ]
        # Plan residual: multi-L fit intercept of delay-corrected mid-pair TOF (should be ~0).
        if int(np.sum(steady_accepted)) >= 2:
            path_lengths = np.asarray(slow[sequence_index, :, path_index], dtype=np.float64)
            t_mid = 0.5 * (
                result.tof_corrected_ab_s.astype(np.float64)
                + result.tof_corrected_ba_s.astype(np.float64)
            )
            if np.unique(path_lengths[steady_accepted]).size >= 2:
                _slope, intercept, _lo, _hi = theilslopes(
                    t_mid[steady_accepted], path_lengths[steady_accepted]
                )
                seq_mean_reciprocity.append(float(intercept))
            elif finite_rec.size:
                seq_mean_reciprocity.append(float(np.mean(finite_rec)))
        elif finite_rec.size:
            seq_mean_reciprocity.append(float(np.mean(finite_rec)))

    def _peak_metrics_local(err: np.ndarray) -> dict[str, float]:
        return _peak_metrics(err)

    peak_ab_metrics = _peak_metrics_local(peak_err_ab)
    peak_ba_metrics = _peak_metrics_local(peak_err_ba)

    # τ̂ vs true fixed delay (asymmetry from first train row is representative; smoke default 0)
    train_asym = float(asymmetry_by_sid.get(sequence_ids[train_indices[0]], 0.0))
    tau_true_ab = true_fixed_delay_s(
        system_delay_s=system_delay_s,
        cable_delay_s=cable_delay_s,
        delay_asymmetry_s=train_asym,
        direction="ab",
    )
    tau_true_ba = true_fixed_delay_s(
        system_delay_s=system_delay_s,
        cable_delay_s=cable_delay_s,
        delay_asymmetry_s=train_asym,
        direction="ba",
    )
    tau_err_ab = abs(calibration.tau_ab_s - tau_true_ab)
    tau_err_ba = abs(calibration.tau_ba_s - tau_true_ba)

    c_mask = np.isfinite(c_hat_seq) & np.isfinite(c_true_seq)
    v_mask = np.isfinite(v_hat_seq) & np.isfinite(v_true_seq)
    c_bias = float(np.mean(c_hat_seq[c_mask] - c_true_seq[c_mask])) if np.any(c_mask) else float("nan")
    v_bias = float(np.mean(v_hat_seq[v_mask] - v_true_seq[v_mask])) if np.any(v_mask) else float("nan")
    # Gate uses P95 of |sequence-mean reciprocity residual| (frame jitter averages inside a sequence).
    if seq_mean_reciprocity:
        reciprocity_p95 = float(
            np.percentile(np.abs(np.asarray(seq_mean_reciprocity, dtype=np.float64)), 95)
        )
    else:
        reciprocity_p95 = float("nan")

    gates = {
        "peak_p95_ab": {
            "value": peak_ab_metrics["p95_abs_samples"],
            "limit": thresholds.peak_p95_abs_samples,
            "passed": peak_ab_metrics["p95_abs_samples"] <= thresholds.peak_p95_abs_samples,
        },
        "peak_p95_ba": {
            "value": peak_ba_metrics["p95_abs_samples"],
            "limit": thresholds.peak_p95_abs_samples,
            "passed": peak_ba_metrics["p95_abs_samples"] <= thresholds.peak_p95_abs_samples,
        },
        "tau_abs_error_ab": {
            "value": tau_err_ab,
            "limit": thresholds.tau_abs_error_s,
            "passed": tau_err_ab <= thresholds.tau_abs_error_s,
        },
        "tau_abs_error_ba": {
            "value": tau_err_ba,
            "limit": thresholds.tau_abs_error_s,
            "passed": tau_err_ba <= thresholds.tau_abs_error_s,
        },
        "sound_speed_bias": {
            "value": abs(c_bias),
            "signed_bias": c_bias,
            "limit": thresholds.sound_speed_bias_m_per_s,
            "passed": abs(c_bias) <= thresholds.sound_speed_bias_m_per_s,
        },
        "v_path_bias": {
            "value": abs(v_bias),
            "signed_bias": v_bias,
            "limit": thresholds.v_path_bias_m_per_s,
            "passed": abs(v_bias) <= thresholds.v_path_bias_m_per_s,
        },
        "reciprocity_residual_p95": {
            "value": reciprocity_p95,
            "limit": thresholds.reciprocity_residual_p95_s,
            "passed": reciprocity_p95 <= thresholds.reciprocity_residual_p95_s,
        },
    }

    wide_stress_report: dict[str, Any] | None = None
    if composition_domain == COMPOSITION_DOMAIN_WIDE:
        stress_cfg = dict(config.get("wide_stress") or {})
        co2_min = float(stress_cfg.get("co2_min_vol_pct", 8.0))
        l_min = float(stress_cfg.get("l_m_min", 0.28))
        require_frames = bool(stress_cfg.get("require_frames", True))
        wide_stress_report = _wide_stress_peak_report(
            peak_err_ab=peak_err_ab,
            peak_err_ba=peak_err_ba,
            labels=labels,
            slow=slow,
            path_index=path_index,
            co2_min_vol_pct=co2_min,
            l_m_min=l_min,
        )
        n_stress = int(wide_stress_report["n_frames"])
        stress_ok = n_stress > 0 or not require_frames
        max_ab = wide_stress_report["ab"]["max_abs_samples"]
        max_ba = wide_stress_report["ba"]["max_abs_samples"]
        # Non-finite peak errors are failures (NaN/Inf must not vacuously pass).
        peak_ok = _wide_stress_peak_gate_passed(
            max_ab=float(max_ab) if max_ab is not None else float("nan"),
            max_ba=float(max_ba) if max_ba is not None else float("nan"),
            n_stress=n_stress,
            limit=float(thresholds.peak_p95_abs_samples),
        )
        gates["wide_stress_frames_present"] = {
            "value": float(n_stress),
            "limit": 1.0 if require_frames else 0.0,
            "passed": stress_ok,
            "co2_min_vol_pct": co2_min,
            "l_m_min": l_min,
        }
        gates["wide_stress_peak_max_ab"] = {
            "value": max_ab,
            "limit": thresholds.peak_p95_abs_samples,
            "passed": peak_ok if n_stress > 0 else True,
        }
        gates["wide_stress_peak_max_ba"] = {
            "value": max_ba,
            "limit": thresholds.peak_p95_abs_samples,
            "passed": peak_ok if n_stress > 0 else True,
        }

    passed = all(item["passed"] for item in gates.values())
    if composition_domain == COMPOSITION_DOMAIN_WIDE:
        verdict = "f3_wide_dsp_passed" if passed else "estimator_failed"
        next_stage = "F4_wide_identifiability_v2"
    else:
        verdict = "f3_dsp_passed" if passed else "estimator_failed"
        next_stage = "F4_identifiability_v2"

    result = {
        "schema_version": AUDIT_SCHEMA_VERSION,
        "feature_builder": FEATURE_BUILDER,
        "bidir_raw_dsp_schema": BIDIR_RAW_DSP_SCHEMA_VERSION,
        "composition_domain": composition_domain,
        "status": "passed" if passed else "failed",
        "verdict": verdict,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "dataset_dir": str(dataset_dir),
        "thresholds": asdict(thresholds),
        "gates": gates,
        "peak_metrics": {"ab": peak_ab_metrics, "ba": peak_ba_metrics},
        "wide_stress": wide_stress_report,
        "delay_calibration": session_calibration_as_dict(calibration),
        "tau_true_s": {"ab": tau_true_ab, "ba": tau_true_ba},
        "tau_abs_error_s": {"ab": tau_err_ab, "ba": tau_err_ba},
        "physics_recovery": {
            "sound_speed_bias_m_per_s": c_bias,
            "v_path_bias_m_per_s": v_bias,
            "reciprocity_residual_p95_s": reciprocity_p95,
            "aggregation": "snr_weighted_steady_accepted_vs_median_oracle_steady",
            "reciprocity_aggregation": "p95_abs_steady_midpair_tof_l_intercept",
            "n_sequences_c": int(np.sum(c_mask)),
            "n_sequences_v": int(np.sum(v_mask)),
            "n_reciprocity_sequences": len(seq_mean_reciprocity),
        },
        "templates": {
            **templates_digest_payload(template_ab, template_ba),
            "template_source_split": "train",
            "template_pre_samples": pre,
            "template_post_samples": post,
            "n_baseline_frames_ab": int(ab_frames.shape[0]),
            "n_baseline_frames_ba": int(ba_frames.shape[0]),
        },
        "allowed_next_stage_on_pass": next_stage,
    }

    (output_dir / "metrics.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    np.save(output_dir / "template_ab.npy", template_ab)
    np.save(output_dir / "template_ba.npy", template_ba)
    (output_dir / "delay_calibration.json").write_text(
        json.dumps(session_calibration_as_dict(calibration), indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    f3_verdict = {
        "stage": "F3_wide" if composition_domain == COMPOSITION_DOMAIN_WIDE else "F3",
        "verdict": verdict,
        "passed": passed,
        "composition_domain": composition_domain,
        "created_at": result["created_at"],
        "metrics_path": str(output_dir / "metrics.json"),
        "feature_builder": FEATURE_BUILDER,
        "template_ab_digest": template_digest(template_ab),
        "template_ba_digest": template_digest(template_ba),
        "delay_calibration_digest": calibration.digest,
        "gates": gates,
        "wide_stress": wide_stress_report,
        "allowed_next_stage_on_pass": next_stage,
    }
    (output_dir / "f3_verdict.json").write_text(
        json.dumps(f3_verdict, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return result


def _update_stage_status_f3_wide(*, output_dir: Path, result: dict[str, Any]) -> None:
    """Write f3_wide only — never rewrite narrow f3 / allowed_next_stage."""
    if result.get("composition_domain") != COMPOSITION_DOMAIN_WIDE or result.get("status") != "passed":
        return
    stage_path = default_config_dir() / "stage_status.json"
    if not stage_path.is_file():
        return
    stage = json.loads(stage_path.read_text(encoding="utf-8"))
    stage["f3_wide"] = {
        "verdict": result["verdict"],
        "passed_at": datetime.now(timezone.utc).date().isoformat(),
        "dataset": "data/tv3-bidir-f3-wide",
        "feature_builder": FEATURE_BUILDER,
        "verdict_path": "outputs/tv3_bidir/dsp_fidelity_wide/f3_verdict.json",
        "metrics_path": "outputs/tv3_bidir/dsp_fidelity_wide/metrics.json",
        "allowed_next_stage": result["allowed_next_stage_on_pass"],
        "physics_aggregation": "snr_weighted_steady_accepted_vs_median_oracle_steady",
        "wide_stress_n_frames": (result.get("wide_stress") or {}).get("n_frames"),
    }
    stage_path.write_text(json.dumps(stage, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    cfg = _load_config(args.config)
    dataset_dir = Path(args.dataset_dir or cfg.get("dataset_dir", "data/tv3-bidir-smoke"))
    if not dataset_dir.is_absolute():
        dataset_dir = _TV3_ROOT / dataset_dir
    output_dir = Path(args.output_dir or cfg.get("output_dir", "outputs/tv3_bidir/dsp_fidelity"))
    if not output_dir.is_absolute():
        output_dir = _TV3_ROOT / output_dir
    result = run_f3_dsp_fidelity(
        dataset_dir=dataset_dir,
        output_dir=output_dir,
        config=cfg,
        allow_overwrite=bool(args.allow_overwrite or cfg.get("allow_overwrite", False)),
    )
    _update_stage_status_f3_wide(output_dir=output_dir, result=result)
    print(
        json.dumps(
            {
                "status": result["status"],
                "verdict": result["verdict"],
                "composition_domain": result.get("composition_domain"),
                "gates": {k: v["passed"] for k, v in result["gates"].items()},
                "wide_stress": result.get("wide_stress"),
                "output_dir": str(output_dir),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0 if result["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
