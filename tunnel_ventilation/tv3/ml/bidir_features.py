"""Deployable bidirectional RawDSP estimator: ``raw_dsp_bidirectional_v1``.

Four layers (F3 freeze):
1. Per-direction matched-filter frames (reuse D2b ``extract_raw_dsp_frame``).
2. Train-only session delay calibration ``t_dir(L)=τ_dir+L/c_eff`` (steady Theil–Sen
   per train sequence, then median intercept; digest-frozen).
3. Frame-pair physics via reciprocal-sum on delay-corrected transit times.
4. Compact sequence features (E1d-SB discipline × 2 directions + pair block).

Oracle arrays must never enter the deploy path; callers that need audit metrics
compare estimator outputs to oracle externally.
"""
from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass
from typing import Sequence

import numpy as np

from tv3.ml.raw_dsp_features import (
    FORMAL_SLOW_CHANNELS,
    RawDSPConfig,
    RawDSPFrameResult,
    build_baseline_median_template,
    dequantize_waveforms,
    extract_raw_dsp_frame,
    fit_tof_vs_path_length,
    template_digest,
    validate_raw_dsp_config,
)
from tv3.sim.core.tunnel_ventilation_bidir_schema import FORMAL_FEATURE_BUILDER
from tv3.sim.generation.tunnel_ventilation.flow_physics import (
    reciprocal_sum_path_velocity_m_per_s,
    reciprocal_sum_sound_speed_m_per_s,
)

FEATURE_BUILDER = FORMAL_FEATURE_BUILDER
BIDIR_RAW_DSP_SCHEMA_VERSION = "tv3-bidir-raw-dsp-1"
DIRECTIONS = ("ab", "ba")


@dataclass(frozen=True, slots=True)
class BidirSessionDelayCalibration:
    """Train-only frozen per-direction delay intercepts."""

    tau_ab_s: float
    tau_ba_s: float
    c_eff_ab_m_per_s: float
    c_eff_ba_m_per_s: float
    n_sequences_ab: int
    n_sequences_ba: int
    method: str
    digest: str

    def tau_s(self, direction: str) -> float:
        if direction == "ab":
            return self.tau_ab_s
        if direction == "ba":
            return self.tau_ba_s
        raise ValueError(f"direction must be 'ab' or 'ba', got {direction!r}")


@dataclass(frozen=True, slots=True)
class BidirFramePairResult:
    peak_index_ab: float
    peak_index_ba: float
    tof_observed_ab_s: float
    tof_observed_ba_s: float
    tof_corrected_ab_s: float
    tof_corrected_ba_s: float
    sound_speed_m_per_s: float
    v_path_m_per_s: float
    reciprocity_residual_s: float
    snr_db_ab: float
    snr_db_ba: float
    peak_to_sidelobe_ratio_ab: float
    peak_to_sidelobe_ratio_ba: float
    quality_ab: float
    quality_ba: float
    accepted_ab: bool
    accepted_ba: bool
    accepted_pair: bool


@dataclass(frozen=True, slots=True)
class BidirSequenceResult:
    peak_index_ab: np.ndarray
    peak_index_ba: np.ndarray
    tof_observed_ab_s: np.ndarray
    tof_observed_ba_s: np.ndarray
    tof_corrected_ab_s: np.ndarray
    tof_corrected_ba_s: np.ndarray
    sound_speed_m_per_s: np.ndarray
    v_path_m_per_s: np.ndarray
    reciprocity_residual_s: np.ndarray
    snr_db_ab: np.ndarray
    snr_db_ba: np.ndarray
    peak_to_sidelobe_ratio_ab: np.ndarray
    peak_to_sidelobe_ratio_ba: np.ndarray
    quality_ab: np.ndarray
    quality_ba: np.ndarray
    accepted_ab: np.ndarray
    accepted_ba: np.ndarray
    accepted_pair: np.ndarray
    tau_ab_s: float
    tau_ba_s: float
    sound_speed_seq_m_per_s: float
    v_path_seq_m_per_s: float
    reciprocity_residual_p95_s: float
    sound_speed_dispersion_m_per_s: float


def assert_no_oracle_inputs(names: Sequence[str]) -> None:
    """Reject oracle / true arrays if a caller accidentally feeds them as features."""
    forbidden = {
        "ultrasonic_tof_true_ab_s",
        "ultrasonic_tof_true_ba_s",
        "ultrasonic_v_path_true_m_per_s",
        "ultrasonic_sound_speed_m_per_s",
        "ultrasonic_alpha_true_npm",
    }
    bad = [name for name in names if name in forbidden or name.endswith("_true") or "_true_" in name]
    # Allow estimated / raw_dsp names that happen to contain substrings.
    bad = [
        name
        for name in bad
        if name in forbidden
        or (
            ("_true" in name or name.endswith("_true"))
            and "estimated" not in name
            and "raw_dsp" not in name
            and "hat" not in name
        )
    ]
    if bad:
        raise ValueError(
            f"oracle array(s) forbidden in deployable {FEATURE_BUILDER} inputs: {bad}"
        )


def build_direction_template(
    waveform_int_frames: np.ndarray,
    scales: np.ndarray,
    path_lengths_m: np.ndarray,
    *,
    config: RawDSPConfig,
    daq_full_scale_v: float,
    template_pre_samples: int,
    template_post_samples: int,
    min_template_snr_db: float,
    reference_peak_polarity: int = -1,
) -> np.ndarray:
    return build_baseline_median_template(
        waveform_int_frames,
        scales,
        path_lengths_m,
        config=config,
        daq_full_scale_v=daq_full_scale_v,
        template_pre_samples=template_pre_samples,
        template_post_samples=template_post_samples,
        min_template_snr_db=min_template_snr_db,
        reference_peak_polarity=reference_peak_polarity,
    )


def calibrate_session_delay_s(
    tof_observed_by_sequence: Sequence[np.ndarray],
    path_lengths_by_sequence: Sequence[np.ndarray],
    phase_ids_by_sequence: Sequence[Sequence[str]],
    accepted_by_sequence: Sequence[np.ndarray],
    *,
    min_sequences: int = 1,
    refine_iterations: int = 3,
) -> tuple[float, float, int]:
    """Train-only session τ̂ from multi-L steady frames (one direction).

    1. Per-sequence Theil–Sen ``t(L)=τ+L/c_eff`` (handles sequence-constant flow).
    2. Pooled refinement: shared τ, per-sequence ``c_eff``, median over all steady frames.

    Returns ``(tau_s, median_c_eff_m_per_s, n_sequences_used)``.
    Prefer :func:`calibrate_session_delay_shared_s` when ``delay_asymmetry_s≈0``.
    """
    return _calibrate_session_delay_from_tof_series(
        tof_observed_by_sequence,
        path_lengths_by_sequence,
        phase_ids_by_sequence,
        accepted_by_sequence,
        min_sequences=min_sequences,
        refine_iterations=refine_iterations,
    )


def calibrate_session_delay_shared_s(
    tof_ab_by_sequence: Sequence[np.ndarray],
    tof_ba_by_sequence: Sequence[np.ndarray],
    path_lengths_by_sequence: Sequence[np.ndarray],
    phase_ids_by_sequence: Sequence[Sequence[str]],
    accepted_ab_by_sequence: Sequence[np.ndarray],
    accepted_ba_by_sequence: Sequence[np.ndarray],
    *,
    min_sequences: int = 1,
    refine_iterations: int = 3,
) -> tuple[float, float, int]:
    """Shared τ̂ from mid-pair transit times when delay asymmetry is zero.

    Uses ``t_mid=(t_ab+t_ba)/2 = τ + L·c/(c²−v²)`` so AB/BA jitter averages and
    a single intercept is identified. Required for v̂ not to absorb τ̂_ab−τ̂_ba noise.
    """
    if len(tof_ab_by_sequence) != len(tof_ba_by_sequence):
        raise ValueError("AB/BA sequence counts must match for shared delay calibration")
    mid_tofs: list[np.ndarray] = []
    accepted_mid: list[np.ndarray] = []
    for tof_ab, tof_ba, acc_ab, acc_ba in zip(
        tof_ab_by_sequence,
        tof_ba_by_sequence,
        accepted_ab_by_sequence,
        accepted_ba_by_sequence,
        strict=True,
    ):
        ab = np.asarray(tof_ab, dtype=np.float64)
        ba = np.asarray(tof_ba, dtype=np.float64)
        if ab.shape != ba.shape:
            raise ValueError(f"AB/BA TOF shape mismatch: {ab.shape} vs {ba.shape}")
        mid_tofs.append(0.5 * (ab + ba))
        accepted_mid.append(np.asarray(acc_ab, dtype=bool) & np.asarray(acc_ba, dtype=bool))
    return _calibrate_session_delay_from_tof_series(
        mid_tofs,
        path_lengths_by_sequence,
        phase_ids_by_sequence,
        accepted_mid,
        min_sequences=min_sequences,
        refine_iterations=refine_iterations,
    )


def _calibrate_session_delay_from_tof_series(
    tof_observed_by_sequence: Sequence[np.ndarray],
    path_lengths_by_sequence: Sequence[np.ndarray],
    phase_ids_by_sequence: Sequence[Sequence[str]],
    accepted_by_sequence: Sequence[np.ndarray],
    *,
    min_sequences: int,
    refine_iterations: int,
) -> tuple[float, float, int]:
    if not (
        len(tof_observed_by_sequence)
        == len(path_lengths_by_sequence)
        == len(phase_ids_by_sequence)
        == len(accepted_by_sequence)
    ):
        raise ValueError("session delay calibration inputs must have matching sequence counts")

    usable: list[tuple[np.ndarray, np.ndarray]] = []
    taus: list[float] = []
    c_effs: list[float] = []
    for tof, path_lengths, phase_ids, accepted in zip(
        tof_observed_by_sequence,
        path_lengths_by_sequence,
        phase_ids_by_sequence,
        accepted_by_sequence,
        strict=True,
    ):
        tof_arr = np.asarray(tof, dtype=np.float64)
        path_arr = np.asarray(path_lengths, dtype=np.float64)
        accepted_mask = np.asarray(accepted, dtype=bool)
        steady = np.asarray([phase == "steady" for phase in phase_ids], dtype=bool) & accepted_mask
        try:
            intercept_s, c_eff = fit_tof_vs_path_length(tof_arr, path_arr, phase_ids, accepted_mask)
        except ValueError:
            continue
        if not math.isfinite(intercept_s) or intercept_s <= 0.0:
            continue
        if int(np.sum(steady)) < 2:
            continue
        usable.append((tof_arr[steady], path_arr[steady]))
        taus.append(float(intercept_s))
        c_effs.append(float(c_eff))

    if len(taus) < min_sequences:
        raise ValueError(
            f"session delay calibration requires at least {min_sequences} usable train sequences, "
            f"got {len(taus)}"
        )

    tau = float(np.median(taus))
    for _ in range(max(0, int(refine_iterations))):
        refined_c: list[float] = []
        tau_candidates: list[float] = []
        for tof_steady, path_steady in usable:
            t_prop = tof_steady - tau
            valid = t_prop > 0.0
            if int(np.sum(valid)) < 2 or np.unique(path_steady[valid]).size < 2:
                continue
            slowness = float(np.median(t_prop[valid] / path_steady[valid]))
            if not math.isfinite(slowness) or slowness <= 0.0:
                continue
            c_eff = 1.0 / slowness
            refined_c.append(c_eff)
            tau_candidates.extend((tof_steady[valid] - path_steady[valid] / c_eff).tolist())
        if not tau_candidates or not refined_c:
            break
        tau = float(np.median(tau_candidates))
        c_effs = refined_c

    if not math.isfinite(tau) or tau <= 0.0:
        raise ValueError(f"session delay calibration produced invalid tau={tau}")
    return tau, float(np.median(c_effs)), len(usable)


def freeze_session_delay_calibration(
    *,
    tau_ab_s: float,
    tau_ba_s: float,
    c_eff_ab_m_per_s: float,
    c_eff_ba_m_per_s: float,
    n_sequences_ab: int,
    n_sequences_ba: int,
    method: str = "train_steady_theilsen_median_intercept_v1",
) -> BidirSessionDelayCalibration:
    payload = {
        "method": method,
        "tau_ab_s": float(tau_ab_s),
        "tau_ba_s": float(tau_ba_s),
        "c_eff_ab_m_per_s": float(c_eff_ab_m_per_s),
        "c_eff_ba_m_per_s": float(c_eff_ba_m_per_s),
        "n_sequences_ab": int(n_sequences_ab),
        "n_sequences_ba": int(n_sequences_ba),
    }
    digest = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return BidirSessionDelayCalibration(digest=digest, **payload)


def reciprocity_residual_s(
    path_length_m: float,
    sound_speed_frame_m_per_s: float,
    sound_speed_ref_m_per_s: float,
) -> float:
    """Time-domain residual: ``L*(1/ĉ_frame − 1/ĉ_ref)`` (frame vs sequence aggregate)."""
    if path_length_m <= 0.0:
        raise ValueError("path_length_m must be > 0")
    if sound_speed_frame_m_per_s <= 0.0 or sound_speed_ref_m_per_s <= 0.0:
        raise ValueError("sound speeds must be > 0")
    return float(path_length_m) * (
        1.0 / float(sound_speed_frame_m_per_s) - 1.0 / float(sound_speed_ref_m_per_s)
    )


def extract_bidir_frame_pair(
    waveform_ab_v: np.ndarray,
    waveform_ba_v: np.ndarray,
    *,
    path_length_m: float,
    template_ab: np.ndarray,
    template_ba: np.ndarray,
    tau_ab_s: float,
    tau_ba_s: float,
    daq_full_scale_v: float,
    config: RawDSPConfig,
    sound_speed_ref_m_per_s: float | None = None,
    template_peak_offset_ab: int | None = None,
    template_peak_offset_ba: int | None = None,
) -> BidirFramePairResult:
    validate_raw_dsp_config(config)
    if tau_ab_s <= 0.0 or tau_ba_s <= 0.0:
        raise ValueError("session delay estimates must be > 0")

    ab = extract_raw_dsp_frame(
        waveform_ab_v,
        template_ab,
        path_length_m=path_length_m,
        daq_full_scale_v=daq_full_scale_v,
        config=config,
        template_peak_offset_samples=template_peak_offset_ab,
    )
    ba = extract_raw_dsp_frame(
        waveform_ba_v,
        template_ba,
        path_length_m=path_length_m,
        daq_full_scale_v=daq_full_scale_v,
        config=config,
        template_peak_offset_samples=template_peak_offset_ba,
    )
    return _pair_from_direction_frames(
        ab,
        ba,
        path_length_m=path_length_m,
        sample_rate_hz=config.sample_rate_hz,
        tau_ab_s=tau_ab_s,
        tau_ba_s=tau_ba_s,
        sound_speed_ref_m_per_s=sound_speed_ref_m_per_s,
    )


def extract_bidir_sequence(
    waveform_ab_int: np.ndarray,
    scale_ab: np.ndarray,
    waveform_ba_int: np.ndarray,
    scale_ba: np.ndarray,
    slow: np.ndarray,
    slow_channel_names: Sequence[str],
    *,
    template_ab: np.ndarray,
    template_ba: np.ndarray,
    calibration: BidirSessionDelayCalibration,
    daq_full_scale_v: float,
    config: RawDSPConfig,
    template_peak_offset_ab: int | None = None,
    template_peak_offset_ba: int | None = None,
) -> BidirSequenceResult:
    """Extract one sequence with frozen session τ̂ (not per-sequence fresh-air τ)."""
    validate_raw_dsp_config(config)
    waves_ab = dequantize_waveforms(waveform_ab_int, scale_ab)
    waves_ba = dequantize_waveforms(waveform_ba_int, scale_ba)
    slow_values = np.asarray(slow, dtype=np.float64)
    if waves_ab.ndim != 2 or waves_ba.ndim != 2:
        raise ValueError("bidirectional sequence waveforms must be 2D")
    if waves_ab.shape != waves_ba.shape:
        raise ValueError(f"AB/BA shape mismatch: {waves_ab.shape} vs {waves_ba.shape}")
    if slow_values.ndim != 2 or slow_values.shape[0] != waves_ab.shape[0]:
        raise ValueError(f"slow sequence shape mismatch: {slow_values.shape}")

    channel_indices = {name: idx for idx, name in enumerate(slow_channel_names)}
    missing = [name for name in ("L_m",) if name not in channel_indices]
    if missing:
        raise ValueError(f"missing slow channels required by {FEATURE_BUILDER}: {missing}")
    path_lengths = slow_values[:, channel_indices["L_m"]]

    # First pass: physics with temporary reciprocity ref = medium-agnostic pair ĉ
    provisional: list[BidirFramePairResult] = []
    for waveform_ab, waveform_ba, path_length_m in zip(waves_ab, waves_ba, path_lengths, strict=True):
        provisional.append(
            extract_bidir_frame_pair(
                waveform_ab,
                waveform_ba,
                path_length_m=float(path_length_m),
                template_ab=template_ab,
                template_ba=template_ba,
                tau_ab_s=calibration.tau_ab_s,
                tau_ba_s=calibration.tau_ba_s,
                daq_full_scale_v=daq_full_scale_v,
                config=config,
                sound_speed_ref_m_per_s=None,
                template_peak_offset_ab=template_peak_offset_ab,
                template_peak_offset_ba=template_peak_offset_ba,
            )
        )

    accepted_pair = np.asarray([pair.accepted_pair for pair in provisional], dtype=bool)
    c_frames = np.asarray([pair.sound_speed_m_per_s for pair in provisional], dtype=np.float64)
    v_frames = np.asarray([pair.v_path_m_per_s for pair in provisional], dtype=np.float64)
    snr_w = np.asarray(
        [
            0.5
            * (
                10.0 ** (0.05 * float(pair.snr_db_ab))
                + 10.0 ** (0.05 * float(pair.snr_db_ba))
            )
            for pair in provisional
        ],
        dtype=np.float64,
    )
    c_seq, v_seq = _snr_weighted_mean_pair(c_frames, v_frames, snr_w, accepted_pair)

    reciprocity = np.asarray(
        [
            reciprocity_residual_s(float(path_length), float(pair.sound_speed_m_per_s), c_seq)
            if pair.accepted_pair and math.isfinite(pair.sound_speed_m_per_s)
            else float("nan")
            for pair, path_length in zip(provisional, path_lengths, strict=True)
        ],
        dtype=np.float64,
    )
    final_pairs = [
        BidirFramePairResult(
            peak_index_ab=pair.peak_index_ab,
            peak_index_ba=pair.peak_index_ba,
            tof_observed_ab_s=pair.tof_observed_ab_s,
            tof_observed_ba_s=pair.tof_observed_ba_s,
            tof_corrected_ab_s=pair.tof_corrected_ab_s,
            tof_corrected_ba_s=pair.tof_corrected_ba_s,
            sound_speed_m_per_s=pair.sound_speed_m_per_s,
            v_path_m_per_s=pair.v_path_m_per_s,
            reciprocity_residual_s=float(reciprocity[idx])
            if math.isfinite(reciprocity[idx])
            else float("nan"),
            snr_db_ab=pair.snr_db_ab,
            snr_db_ba=pair.snr_db_ba,
            peak_to_sidelobe_ratio_ab=pair.peak_to_sidelobe_ratio_ab,
            peak_to_sidelobe_ratio_ba=pair.peak_to_sidelobe_ratio_ba,
            quality_ab=pair.quality_ab,
            quality_ba=pair.quality_ba,
            accepted_ab=pair.accepted_ab,
            accepted_ba=pair.accepted_ba,
            accepted_pair=pair.accepted_pair,
        )
        for idx, pair in enumerate(provisional)
    ]

    finite_rec = reciprocity[np.isfinite(reciprocity)]
    if finite_rec.size:
        rec_p95 = float(np.percentile(np.abs(finite_rec), 95))
    else:
        rec_p95 = float("nan")
    if int(np.sum(accepted_pair)) >= 2:
        dispersion = float(np.std(c_frames[accepted_pair], ddof=0))
    elif int(np.sum(accepted_pair)) == 1:
        dispersion = 0.0
    else:
        dispersion = float("nan")

    return BidirSequenceResult(
        peak_index_ab=np.asarray([p.peak_index_ab for p in final_pairs], dtype=np.float32),
        peak_index_ba=np.asarray([p.peak_index_ba for p in final_pairs], dtype=np.float32),
        tof_observed_ab_s=np.asarray([p.tof_observed_ab_s for p in final_pairs], dtype=np.float32),
        tof_observed_ba_s=np.asarray([p.tof_observed_ba_s for p in final_pairs], dtype=np.float32),
        tof_corrected_ab_s=np.asarray([p.tof_corrected_ab_s for p in final_pairs], dtype=np.float32),
        tof_corrected_ba_s=np.asarray([p.tof_corrected_ba_s for p in final_pairs], dtype=np.float32),
        sound_speed_m_per_s=np.asarray([p.sound_speed_m_per_s for p in final_pairs], dtype=np.float32),
        v_path_m_per_s=np.asarray([p.v_path_m_per_s for p in final_pairs], dtype=np.float32),
        reciprocity_residual_s=np.asarray(
            [p.reciprocity_residual_s for p in final_pairs], dtype=np.float32
        ),
        snr_db_ab=np.asarray([p.snr_db_ab for p in final_pairs], dtype=np.float32),
        snr_db_ba=np.asarray([p.snr_db_ba for p in final_pairs], dtype=np.float32),
        peak_to_sidelobe_ratio_ab=np.asarray(
            [p.peak_to_sidelobe_ratio_ab for p in final_pairs], dtype=np.float32
        ),
        peak_to_sidelobe_ratio_ba=np.asarray(
            [p.peak_to_sidelobe_ratio_ba for p in final_pairs], dtype=np.float32
        ),
        quality_ab=np.asarray([p.quality_ab for p in final_pairs], dtype=np.float32),
        quality_ba=np.asarray([p.quality_ba for p in final_pairs], dtype=np.float32),
        accepted_ab=np.asarray([p.accepted_ab for p in final_pairs], dtype=bool),
        accepted_ba=np.asarray([p.accepted_ba for p in final_pairs], dtype=bool),
        accepted_pair=accepted_pair,
        tau_ab_s=float(calibration.tau_ab_s),
        tau_ba_s=float(calibration.tau_ba_s),
        sound_speed_seq_m_per_s=float(c_seq),
        v_path_seq_m_per_s=float(v_seq),
        reciprocity_residual_p95_s=rec_p95,
        sound_speed_dispersion_m_per_s=dispersion,
    )


def compact_sequence_feature_vector(result: BidirSequenceResult) -> dict[str, float]:
    """E1d-SB-style compact scalars for one sequence (deployable; no oracle)."""
    def _phase_stats(values: np.ndarray, accepted: np.ndarray) -> dict[str, float]:
        mask = accepted & np.isfinite(values)
        if not np.any(mask):
            return {"mean": float("nan"), "std": float("nan"), "last": float("nan")}
        selected = values[mask]
        return {
            "mean": float(np.mean(selected)),
            "std": float(np.std(selected, ddof=0)),
            "last": float(selected[-1]),
        }

    features: dict[str, float] = {
        "tau_ab_s": float(result.tau_ab_s),
        "tau_ba_s": float(result.tau_ba_s),
        "c_hat_seq_m_per_s": float(result.sound_speed_seq_m_per_s),
        "v_hat_seq_m_per_s": float(result.v_path_seq_m_per_s),
        "reciprocity_residual_p95_s": float(result.reciprocity_residual_p95_s),
        "c_hat_dispersion_m_per_s": float(result.sound_speed_dispersion_m_per_s),
    }
    for direction in DIRECTIONS:
        tof_corr = getattr(result, f"tof_corrected_{direction}_s")
        snr = getattr(result, f"snr_db_{direction}")
        psr = getattr(result, f"peak_to_sidelobe_ratio_{direction}")
        accepted = getattr(result, f"accepted_{direction}")
        for prefix, arr in (
            (f"tof_corrected_{direction}_s", tof_corr),
            (f"snr_db_{direction}", snr),
            (f"psr_{direction}", psr),
        ):
            stats = _phase_stats(np.asarray(arr, dtype=np.float64), np.asarray(accepted, dtype=bool))
            for key, value in stats.items():
                features[f"{prefix}_{key}"] = value
    return features


def session_calibration_as_dict(calibration: BidirSessionDelayCalibration) -> dict[str, object]:
    return asdict(calibration)


def templates_digest_payload(template_ab: np.ndarray, template_ba: np.ndarray) -> dict[str, str]:
    return {
        "template_ab_digest": template_digest(template_ab),
        "template_ba_digest": template_digest(template_ba),
    }


def _pair_from_direction_frames(
    ab: RawDSPFrameResult,
    ba: RawDSPFrameResult,
    *,
    path_length_m: float,
    sample_rate_hz: float,
    tau_ab_s: float,
    tau_ba_s: float,
    sound_speed_ref_m_per_s: float | None,
) -> BidirFramePairResult:
    tof_ab = float(ab.peak_index) / float(sample_rate_hz)
    tof_ba = float(ba.peak_index) / float(sample_rate_hz)
    t_ab = tof_ab - float(tau_ab_s)
    t_ba = tof_ba - float(tau_ba_s)
    accepted_pair = bool(ab.accepted) and bool(ba.accepted) and t_ab > 0.0 and t_ba > 0.0
    if accepted_pair:
        c_hat = reciprocal_sum_sound_speed_m_per_s(path_length_m, t_ab, t_ba)
        v_hat = reciprocal_sum_path_velocity_m_per_s(path_length_m, t_ab, t_ba)
        ref = float(sound_speed_ref_m_per_s) if sound_speed_ref_m_per_s is not None else c_hat
        residual = reciprocity_residual_s(path_length_m, c_hat, ref)
    else:
        c_hat = float("nan")
        v_hat = float("nan")
        residual = float("nan")

    return BidirFramePairResult(
        peak_index_ab=float(ab.peak_index),
        peak_index_ba=float(ba.peak_index),
        tof_observed_ab_s=tof_ab,
        tof_observed_ba_s=tof_ba,
        tof_corrected_ab_s=t_ab,
        tof_corrected_ba_s=t_ba,
        sound_speed_m_per_s=float(c_hat),
        v_path_m_per_s=float(v_hat),
        reciprocity_residual_s=float(residual),
        snr_db_ab=float(ab.snr_db),
        snr_db_ba=float(ba.snr_db),
        peak_to_sidelobe_ratio_ab=float(ab.peak_to_sidelobe_ratio),
        peak_to_sidelobe_ratio_ba=float(ba.peak_to_sidelobe_ratio),
        quality_ab=float(ab.quality),
        quality_ba=float(ba.quality),
        accepted_ab=bool(ab.accepted),
        accepted_ba=bool(ba.accepted),
        accepted_pair=accepted_pair,
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


def true_fixed_delay_s(
    *,
    system_delay_s: float,
    cable_delay_s: float,
    delay_asymmetry_s: float = 0.0,
    direction: str,
) -> float:
    """Simulator fixed delay for one direction (excludes trigger jitter)."""
    half = 0.5 * float(delay_asymmetry_s)
    if direction == "ab":
        return float(system_delay_s) + float(cable_delay_s) + half
    if direction == "ba":
        return float(system_delay_s) + float(cable_delay_s) - half
    raise ValueError(f"direction must be 'ab' or 'ba', got {direction!r}")


__all__ = [
    "BIDIR_RAW_DSP_SCHEMA_VERSION",
    "BidirFramePairResult",
    "BidirSequenceResult",
    "BidirSessionDelayCalibration",
    "DIRECTIONS",
    "FEATURE_BUILDER",
    "FORMAL_SLOW_CHANNELS",
    "assert_no_oracle_inputs",
    "build_direction_template",
    "calibrate_session_delay_s",
    "calibrate_session_delay_shared_s",
    "compact_sequence_feature_vector",
    "extract_bidir_frame_pair",
    "extract_bidir_sequence",
    "freeze_session_delay_calibration",
    "reciprocity_residual_s",
    "session_calibration_as_dict",
    "templates_digest_payload",
    "true_fixed_delay_s",
]
