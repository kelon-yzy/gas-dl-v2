from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import pytest

from tv3.common.splits import load_splits, resolve_split_indices
from tv3.ml.raw_dsp_features import (
    RawDSPConfig,
    build_baseline_median_template,
    calibrate_sequence_delay_s,
    dequantize_waveforms,
    exact_simulator_template,
    extract_raw_dsp_frame,
    fit_tof_vs_path_length,
    fit_tof_vs_path_length_snr_weighted,
    parabolic_peak_offset,
    physical_peak_window_samples,
)
from tv3.sim.generation.tunnel_ventilation.acoustic_physics import (
    hidden_attenuation_v2,
    hidden_sound_speed_v2,
)
from tv3.sim.generation.waveforms import WaveformSpec, simulate_waveform_measurement


def _simulate_frame(
    *,
    path_length_m: float = 0.237,
    noise_std_v: float = 0.0,
    waveform_dtype: str = "int16",
    per_timestep_scale: bool = True,
    seed: int = 17,
) -> tuple[WaveformSpec, dict[str, object]]:
    spec = WaveformSpec(
        noise_std_v=noise_std_v,
        trigger_jitter_std_s=0.0,
        waveform_dtype=waveform_dtype,
        per_timestep_scale=per_timestep_scale,
    )
    measurement = simulate_waveform_measurement(
        x_h2=0.0,
        x_ch4=0.0,
        x_co2=0.04,
        x_n2=79.06,
        t_c=27.0,
        p_mpa=0.101325,
        h_rh=45.0,
        l_m=path_length_m,
        seed=seed,
        spec=spec,
        sound_speed_fn=hidden_sound_speed_v2,
        attenuation_fn=hidden_attenuation_v2,
        extra_gas_kwargs={"x_o2": 20.90},
    )
    return spec, measurement


def _extract_simulated_frame(
    spec: WaveformSpec,
    measurement: dict[str, object],
    *,
    path_length_m: float,
):
    config = RawDSPConfig(sample_rate_hz=float(spec.sample_rate_hz))
    waveform = dequantize_waveforms(
        np.asarray(measurement["waveform_int"]),
        float(measurement["scale_factor"]),
    )
    return extract_raw_dsp_frame(
        waveform,
        exact_simulator_template(spec.to_dict()),
        path_length_m=path_length_m,
        daq_full_scale_v=spec.daq_full_scale_v,
        config=config,
    )


def test_exact_template_recovers_fractional_simulator_peak_without_noise():
    path_length_m = 0.237
    spec, measurement = _simulate_frame(path_length_m=path_length_m)

    result = _extract_simulated_frame(spec, measurement, path_length_m=path_length_m)
    expected_peak = float(measurement["tof_observed_s"]) * spec.sample_rate_hz

    assert result.peak_index == pytest.approx(expected_peak, abs=0.05)
    assert result.tof_observed_s == pytest.approx(float(measurement["tof_observed_s"]), abs=0.05e-6)
    assert result.corr_peak > 0.94
    assert result.accepted


def test_noise_amplitude_and_integer_storage_do_not_change_peak_coordinate():
    path_length_m = 0.251
    spec16, measurement16 = _simulate_frame(
        path_length_m=path_length_m,
        noise_std_v=1.0e-3,
        waveform_dtype="int16",
        per_timestep_scale=True,
        seed=29,
    )
    spec32, measurement32 = _simulate_frame(
        path_length_m=path_length_m,
        noise_std_v=1.0e-3,
        waveform_dtype="int32",
        per_timestep_scale=False,
        seed=29,
    )

    result16 = _extract_simulated_frame(spec16, measurement16, path_length_m=path_length_m)
    result32 = _extract_simulated_frame(spec32, measurement32, path_length_m=path_length_m)
    waveform16 = dequantize_waveforms(
        np.asarray(measurement16["waveform_int"]),
        float(measurement16["scale_factor"]),
    )
    scaled_result = extract_raw_dsp_frame(
        waveform16 * 0.35,
        exact_simulator_template(spec16.to_dict()),
        path_length_m=path_length_m,
        daq_full_scale_v=spec16.daq_full_scale_v,
        config=RawDSPConfig(sample_rate_hz=float(spec16.sample_rate_hz)),
    )

    assert result16.peak_index == pytest.approx(result32.peak_index, abs=0.02)
    assert scaled_result.peak_index == pytest.approx(result16.peak_index, abs=0.01)
    assert scaled_result.peak_amplitude_v == pytest.approx(result16.peak_amplitude_v * 0.35, rel=1e-5)


def test_physical_peak_window_uses_only_path_length_and_configured_bounds():
    config = RawDSPConfig(
        sample_rate_hz=1_000_000.0,
        sound_speed_min_m_per_s=250.0,
        sound_speed_max_m_per_s=400.0,
        delay_min_s=40.0e-6,
        delay_max_s=130.0e-6,
    )

    lower, upper = physical_peak_window_samples(0.20, waveform_samples=5000, config=config)

    assert lower == 540
    assert upper == 930


@pytest.mark.parametrize("offset", [-0.4, -0.2, 0.0, 0.2, 0.4])
def test_parabolic_interpolation_recovers_known_subsample_offsets(offset: float):
    left = -((-1.0 - offset) ** 2)
    center = -(offset**2)
    right = -((1.0 - offset) ** 2)

    assert parabolic_peak_offset(left, center, right) == pytest.approx(offset, abs=1e-12)


def test_timing_normalization_does_not_pollute_amplitude_template_path():
    path_lengths = np.full(8, 0.237, dtype=np.float64)
    frames: list[np.ndarray] = []
    scales: list[float] = []
    for seed in range(8):
        _spec, measurement = _simulate_frame(
            path_length_m=0.237,
            noise_std_v=5.0e-4,
            seed=seed,
        )
        frames.append(np.asarray(measurement["waveform_int"]))
        scales.append(float(measurement["scale_factor"]))
    config = RawDSPConfig(sample_rate_hz=1_000_000.0)

    template = build_baseline_median_template(
        np.stack(frames),
        np.asarray(scales, dtype=np.float32),
        path_lengths,
        config=config,
        daq_full_scale_v=2.5,
        template_pre_samples=25,
        template_post_samples=33,
        min_template_snr_db=20.0,
    )

    assert template.shape == (59,)
    assert np.max(np.abs(template)) == pytest.approx(1.0)
    assert not np.allclose(template, dequantize_waveforms(frames[0], scales[0])[774:833])


def test_baseline_delay_calibration_and_tof_path_fit_are_label_free():
    phase_ids = ("baseline",) * 4 + ("steady",) * 4
    temperature = np.full(8, 25.0)
    pressure = np.full(8, 0.101325)
    humidity = np.full(8, 50.0)
    path_lengths = np.asarray([0.20, 0.20, 0.20, 0.20, 0.18, 0.20, 0.22, 0.25])
    fresh_speed = hidden_sound_speed_v2(0.0, 0.0, 0.04, 79.06, t_c=25.0, x_o2=20.90)
    delay_s = 82.0e-6
    tof = path_lengths / fresh_speed + delay_s
    accepted = np.ones(8, dtype=bool)

    calibrated = calibrate_sequence_delay_s(
        tof,
        path_lengths,
        temperature,
        pressure,
        humidity,
        phase_ids,
        accepted,
        min_frames=4,
    )
    intercept, slope_speed = fit_tof_vs_path_length(tof, path_lengths, phase_ids, accepted)

    assert calibrated == pytest.approx(delay_s, abs=1e-12)
    assert intercept == pytest.approx(delay_s, abs=1e-12)
    assert slope_speed == pytest.approx(fresh_speed, rel=1e-12)

    snr = np.linspace(15.0, 35.0, path_lengths.size)
    ls_intercept, ls_speed = fit_tof_vs_path_length_snr_weighted(
        tof, path_lengths, snr, phase_ids, accepted, weight_mode="amplitude"
    )
    assert ls_intercept == pytest.approx(delay_s, abs=1e-12)
    assert ls_speed == pytest.approx(fresh_speed, rel=1e-12)


def test_local_512_frame_exact_template_fidelity_gate():
    project_root = Path(__file__).resolve().parents[1]
    dataset_dir = project_root / "data" / "tv3-formal"
    waveform_path = dataset_dir / "sequences" / "ultrasonic_int32.npy"
    if not waveform_path.is_file():
        pytest.skip("optional legacy tv3-formal waveform dataset is not available")
    waveform_spec = json.loads(
        (dataset_dir / "metadata" / "waveform_spec.json").read_text(encoding="utf-8")
    )["ultrasonic"]
    config = RawDSPConfig(sample_rate_hz=float(waveform_spec["sample_rate_hz"]))
    template = exact_simulator_template(waveform_spec)
    waveforms = np.load(waveform_path, mmap_mode="r")
    scales = np.load(dataset_dir / "sequences" / "ultrasonic_scale.npy", mmap_mode="r")
    slow = np.load(dataset_dir / "sequences" / "slow.npy", mmap_mode="r")
    targets = np.load(dataset_dir / "sequences" / "ultrasonic_tof_observed_s.npy", mmap_mode="r")
    slow_names = np.load(dataset_dir / "metadata" / "slow_channel_names.npy", allow_pickle=True).tolist()
    path_index = slow_names.index("L_m")
    errors: list[float] = []

    for sequence_index in range(8):
        for timestep in range(64):
            waveform = dequantize_waveforms(
                waveforms[sequence_index, timestep],
                scales[sequence_index, timestep],
            )
            result = extract_raw_dsp_frame(
                waveform,
                template,
                path_length_m=float(slow[sequence_index, timestep, path_index]),
                daq_full_scale_v=float(waveform_spec["daq_full_scale_v"]),
                config=config,
            )
            target_peak = float(targets[sequence_index, timestep]) * config.sample_rate_hz
            errors.append(result.peak_index - target_peak)

    error = np.asarray(errors)
    assert np.mean(np.abs(error)) <= 0.05
    assert np.percentile(np.abs(error), 95) <= 0.10
    assert abs(np.mean(error)) <= 0.02


def test_local_512_frame_train_baseline_template_fidelity_gate():
    project_root = Path(__file__).resolve().parents[1]
    dataset_dir = project_root / "data" / "tv3-formal"
    waveform_path = dataset_dir / "sequences" / "ultrasonic_int32.npy"
    if not waveform_path.is_file():
        pytest.skip("optional legacy tv3-formal waveform dataset is not available")
    waveform_spec = json.loads(
        (dataset_dir / "metadata" / "waveform_spec.json").read_text(encoding="utf-8")
    )["ultrasonic"]
    config = RawDSPConfig(sample_rate_hz=float(waveform_spec["sample_rate_hz"]))
    sequence_ids = np.load(
        dataset_dir / "metadata" / "sequence_ids.npy", allow_pickle=True
    ).tolist()
    split_indices = resolve_split_indices(
        load_splits(dataset_dir / "splits"),
        [str(sequence_id) for sequence_id in sequence_ids],
    )
    phase_lookup: dict[str, list[str]] = {}
    with (dataset_dir / "sequences" / "slow_sequence_long.csv").open(
        "r", encoding="utf-8", newline=""
    ) as handle:
        for row in csv.DictReader(handle):
            phase_lookup.setdefault(row["sequence_id"], []).append(row["phase_id"])
    waveforms = np.load(waveform_path, mmap_mode="r")
    scales = np.load(dataset_dir / "sequences" / "ultrasonic_scale.npy", mmap_mode="r")
    slow = np.load(dataset_dir / "sequences" / "slow.npy", mmap_mode="r")
    targets = np.load(dataset_dir / "sequences" / "ultrasonic_tof_observed_s.npy", mmap_mode="r")
    slow_names = np.load(dataset_dir / "metadata" / "slow_channel_names.npy", allow_pickle=True).tolist()
    path_index = slow_names.index("L_m")
    template_frames: list[np.ndarray] = []
    template_scales: list[float] = []
    template_lengths: list[float] = []
    for sequence_index in split_indices["train"]:
        sequence_id = str(sequence_ids[sequence_index])
        for timestep, phase in enumerate(phase_lookup[sequence_id]):
            if phase != "baseline":
                continue
            template_frames.append(np.asarray(waveforms[sequence_index, timestep]))
            template_scales.append(float(scales[sequence_index, timestep]))
            template_lengths.append(float(slow[sequence_index, timestep, path_index]))
            if len(template_frames) >= 512:
                break
        if len(template_frames) >= 512:
            break
    template = build_baseline_median_template(
        np.stack(template_frames),
        np.asarray(template_scales, dtype=np.float32),
        np.asarray(template_lengths, dtype=np.float64),
        config=config,
        daq_full_scale_v=float(waveform_spec["daq_full_scale_v"]),
        template_pre_samples=25,
        template_post_samples=33,
        min_template_snr_db=20.0,
    )
    errors: list[float] = []
    for sequence_index in split_indices["val"][:8]:
        for timestep in range(64):
            result = extract_raw_dsp_frame(
                dequantize_waveforms(
                    waveforms[sequence_index, timestep],
                    scales[sequence_index, timestep],
                ),
                template,
                path_length_m=float(slow[sequence_index, timestep, path_index]),
                daq_full_scale_v=float(waveform_spec["daq_full_scale_v"]),
                config=config,
                template_peak_offset_samples=25,
            )
            target_peak = float(targets[sequence_index, timestep]) * config.sample_rate_hz
            errors.append(result.peak_index - target_peak)

    error = np.asarray(errors)
    assert np.mean(np.abs(error)) <= 0.15, {
        "template_abs_peak": int(np.argmax(np.abs(template))),
        "template_center": float(template[25]),
        "bias": float(np.mean(error)),
    }
    assert np.percentile(np.abs(error), 95) <= 0.25
    assert abs(np.mean(error)) <= 0.05
