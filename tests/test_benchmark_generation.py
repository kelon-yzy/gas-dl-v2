import csv
import json

import numpy as np

from sim.generation.benchmark import BenchmarkGenerationSpec, generate_benchmark_dataset, resolve_time_axis_preset
from sim.generation.slow import _multi_tau_channel_step


def _read_csv(path):
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def test_generate_benchmark_dataset_writes_v4_assets(tmp_path):
    summary = generate_benchmark_dataset(
        tmp_path,
        BenchmarkGenerationSpec(dataset_slug="wv4-smoke", sequence_count=16, seed=7, optical_absorption_backend="empirical_v1"),
    )

    dataset_dir = tmp_path / "wv4-smoke"
    condition_rows = _read_csv(dataset_dir / "condition_grid_sequence.csv")
    index_rows = _read_csv(dataset_dir / "sequence_index.csv")
    label_rows = _read_csv(dataset_dir / "sequence_labels.csv")
    train_rows = _read_csv(dataset_dir / "splits" / "train.csv")
    val_rows = _read_csv(dataset_dir / "splits" / "val.csv")
    test_rows = _read_csv(dataset_dir / "splits" / "test.csv")
    extrapolation_rows = _read_csv(dataset_dir / "splits" / "extrapolation.csv")
    manifest = json.loads((dataset_dir / "manifest.json").read_text(encoding="utf-8"))

    assert summary["dataset_slug"] == "wv4-smoke"
    assert summary["sequence_count"] == 16
    assert len(condition_rows) == 16
    assert len(index_rows) == 16
    assert len(label_rows) == 16
    assert manifest["schema_version"] == "v4-benchmark-1"
    assert manifest["path_lms"] == [0.2, 0.25, 0.3, 0.35, 0.4]
    assert manifest["stage_profile"] == "standard_exposure"
    assert manifest["stage_jitter"] == 0.0
    assert manifest["phase_schedule"]["name"] == "standard_exposure"
    assert manifest["optical_absorption_backend"] == "empirical_v1"
    assert manifest["ultrasonic_model"] == "tof_observed_transducer_proxy_v1"
    assert manifest["ultrasonic_system_delay_model"] == "fixed_delay_plus_trigger_jitter_v1"
    assert manifest["fiber_mic_model"] == "fiber_interferometric_proxy_v1"
    assert manifest["fiber_optical_demodulation_model"] == "linear_phase_demodulation_proxy_v1"

    forbidden_fields = {"base_condition_id", "noise_seed_index", "noise_seed"}
    assert forbidden_fields.isdisjoint(condition_rows[0])
    assert condition_rows[0]["mixture_id"].startswith("M")
    assert condition_rows[0]["sequence_id"].startswith("Q")
    assert condition_rows[0]["mixture_id"] != condition_rows[0]["sequence_id"]

    split_sequence_ids = {
        row["sequence_id"]
        for rows in (train_rows, val_rows, test_rows, extrapolation_rows)
        for row in rows
    }
    assert split_sequence_ids == {row["sequence_id"] for row in condition_rows}


def test_generated_condition_rows_have_component_sum_100(tmp_path):
    generate_benchmark_dataset(
        tmp_path,
        BenchmarkGenerationSpec(dataset_slug="wv4-components", sequence_count=8, seed=11, optical_absorption_backend="empirical_v1"),
    )

    rows = _read_csv(tmp_path / "wv4-components" / "condition_grid_sequence.csv")

    for row in rows:
        total = sum(float(row[name]) for name in ("x_H2", "x_CH4", "x_CO2", "x_N2"))
        assert abs(total - 100.0) < 1e-5


def test_generate_benchmark_dataset_writes_npz_storage_arrays_and_metadata(tmp_path):
    generate_benchmark_dataset(
        tmp_path,
        BenchmarkGenerationSpec(
            dataset_slug="wv4-npz",
            sequence_count=5,
            seed=19,
            timesteps=8,
            dt_s=0.25,
            storage="npz",
            multi_path_phase="off",
            optical_absorption_backend="empirical_v1",
        ),
    )

    dataset_dir = tmp_path / "wv4-npz"
    condition_rows = _read_csv(dataset_dir / "condition_grid_sequence.csv")
    slow_rows = _read_csv(dataset_dir / "sequences" / "slow_sequence_long.csv")
    manifest = json.loads((dataset_dir / "manifest.json").read_text(encoding="utf-8"))
    waveform_spec = json.loads((dataset_dir / "metadata" / "waveform_spec.json").read_text(encoding="utf-8"))
    validation = json.loads((dataset_dir / "quality" / "validation_summary.json").read_text(encoding="utf-8"))

    y = np.load(dataset_dir / "labels" / "y.npy")
    sequence_ids = np.load(dataset_dir / "metadata" / "sequence_ids.npy", allow_pickle=True)
    slow_channel_names = np.load(dataset_dir / "metadata" / "slow_channel_names.npy", allow_pickle=True)
    label_names = np.load(dataset_dir / "metadata" / "label_names.npy", allow_pickle=True)
    slow = np.load(dataset_dir / "sequences" / "slow.npy")
    ultrasonic = np.load(dataset_dir / "sequences" / "ultrasonic_int16.npy")
    ultrasonic_tof_s = np.load(dataset_dir / "sequences" / "ultrasonic_tof_s.npy")
    ultrasonic_tof_observed_s = np.load(dataset_dir / "sequences" / "ultrasonic_tof_observed_s.npy")
    ultrasonic_peak_index = np.load(dataset_dir / "sequences" / "ultrasonic_peak_index.npy")
    ultrasonic_sound_speed = np.load(dataset_dir / "sequences" / "ultrasonic_sound_speed_m_per_s.npy")
    ultrasonic_sound_speed_estimated = np.load(dataset_dir / "sequences" / "ultrasonic_sound_speed_estimated_m_per_s.npy")
    ultrasonic_alpha = np.load(dataset_dir / "sequences" / "ultrasonic_alpha_true_npm.npy")
    ultrasonic_tof_quality = np.load(dataset_dir / "sequences" / "ultrasonic_tof_quality.npy")
    ultrasonic_tof_accepted = np.load(dataset_dir / "sequences" / "ultrasonic_tof_accepted.npy")
    fiber_mic = np.load(dataset_dir / "sequences" / "fiber_mic_int16.npy")
    bundle = np.load(dataset_dir / "sequences" / "waveform_sequence.npz")

    assert manifest["storage"] == "npz"
    assert manifest["shapes"]["slow"] == [5, 8, len(slow_channel_names)]
    assert waveform_spec["optical_absorption_backend"] == "empirical_v1"
    assert waveform_spec["stage_profile"] == "standard_exposure"
    assert waveform_spec["stage_jitter"] == 0.0
    assert waveform_spec["phase_schedule"]["segments"][0]["name"] == "baseline"
    assert waveform_spec["ultrasonic"]["model_name"] == "tof_observed_transducer_proxy_v1"
    assert waveform_spec["ultrasonic"]["system_delay_model"] == "fixed_delay_plus_trigger_jitter_v1"
    assert waveform_spec["ultrasonic"]["system_delay_s"] > 0.0
    assert waveform_spec["ultrasonic"]["trigger_jitter_std_s"] > 0.0
    assert waveform_spec["ultrasonic"]["transducer_response_model"] == "second_order_resonant_bandpass_proxy_v1"
    assert waveform_spec["fiber_mic"]["model_name"] == "fiber_interferometric_proxy_v1"
    assert waveform_spec["fiber_mic"]["acoustic_field_model"] == "probe_pressure_with_optional_reflections_v1"
    assert waveform_spec["fiber_mic"]["fiber_optical_demodulation_model"] == "linear_phase_demodulation_proxy_v1"
    assert waveform_spec["fiber_mic"]["probe"]["pressure_sensitivity_rad_per_pa"] > 0.0
    assert waveform_spec["fiber_mic_model"] == "fiber_interferometric_proxy_v1"
    assert waveform_spec["fiber_optical_demodulation_model"] == "linear_phase_demodulation_proxy_v1"
    assert waveform_spec["acoustic_attenuation_model"] == "semi_empirical_multigas_relaxation_proxy_v2"
    assert validation["status"] == "pass"
    assert y.shape == (5, 4)
    assert slow.shape == (5, 8, len(slow_channel_names))
    assert ultrasonic.shape[:2] == (5, 8)
    assert ultrasonic_tof_s.shape == (5, 8)
    assert ultrasonic_tof_observed_s.shape == (5, 8)
    assert ultrasonic_peak_index.shape == (5, 8)
    assert ultrasonic_sound_speed.shape == (5, 8)
    assert ultrasonic_sound_speed_estimated.shape == (5, 8)
    assert ultrasonic_alpha.shape == (5, 8)
    assert ultrasonic_tof_quality.shape == (5, 8)
    assert ultrasonic_tof_accepted.shape == (5, 8)
    assert ultrasonic_peak_index.dtype == np.int32
    assert ultrasonic_tof_accepted.dtype == np.int8
    assert np.isfinite(ultrasonic_tof_s).all()
    assert np.isfinite(ultrasonic_tof_observed_s).all()
    assert np.isfinite(ultrasonic_sound_speed).all()
    assert np.isfinite(ultrasonic_sound_speed_estimated).all()
    assert np.isfinite(ultrasonic_alpha).all()
    assert np.isfinite(ultrasonic_tof_quality).all()
    assert np.all(ultrasonic_tof_observed_s > ultrasonic_tof_s)
    assert np.all((ultrasonic_tof_quality >= 0.0) & (ultrasonic_tof_quality <= 1.0))
    assert fiber_mic.shape[:2] == (5, 8)
    assert bundle["slow"].shape == slow.shape
    assert bundle["ultrasonic_tof_s"].shape == ultrasonic_tof_s.shape
    assert bundle["ultrasonic_tof_observed_s"].shape == ultrasonic_tof_observed_s.shape
    assert bundle["ultrasonic_peak_index"].shape == ultrasonic_peak_index.shape
    assert list(sequence_ids.astype(str)) == [row["sequence_id"] for row in condition_rows]
    assert list(label_names.astype(str)) == ["x_H2", "x_CH4", "x_CO2", "x_N2"]
    assert len(slow_rows) == 5 * 8


def test_generate_benchmark_dataset_uses_configured_path_lms(tmp_path):
    generate_benchmark_dataset(
        tmp_path,
        BenchmarkGenerationSpec(
            dataset_slug="wv4-path-lms",
            sequence_count=4,
            seed=31,
            timesteps=8,
            storage="npz",
            multi_path_phase="steady",
            path_lms=(0.25, 0.35),
            optical_absorption_backend="empirical_v1",
        ),
    )

    dataset_dir = tmp_path / "wv4-path-lms"
    slow_rows = _read_csv(dataset_dir / "sequences" / "slow_sequence_long.csv")
    manifest = json.loads((dataset_dir / "manifest.json").read_text(encoding="utf-8"))
    waveform_spec = json.loads((dataset_dir / "metadata" / "waveform_spec.json").read_text(encoding="utf-8"))

    first_sequence_rows = [row for row in slow_rows if row["sequence_id"] == "Q000001"]

    assert manifest["path_lms"] == [0.25, 0.35]
    assert waveform_spec["path_lms"] == [0.25, 0.35]
    assert first_sequence_rows[4]["phase_id"] == "steady"
    assert first_sequence_rows[5]["phase_id"] == "steady"
    assert first_sequence_rows[4]["L_m"] == "0.25000"
    assert first_sequence_rows[5]["L_m"] == "0.35000"


def test_generate_benchmark_dataset_uses_nonstandard_stage_profile_and_jitter(tmp_path):
    generate_benchmark_dataset(
        tmp_path,
        BenchmarkGenerationSpec(
            dataset_slug="wv4-variable-onset",
            sequence_count=4,
            seed=41,
            timesteps=32,
            storage="npz",
            multi_path_phase="off",
            stage_profile="variable_onset",
            stage_jitter=0.15,
            optical_absorption_backend="empirical_v1",
        ),
    )

    dataset_dir = tmp_path / "wv4-variable-onset"
    slow_rows = _read_csv(dataset_dir / "sequences" / "slow_sequence_long.csv")
    index_rows = _read_csv(dataset_dir / "sequence_index.csv")
    manifest = json.loads((dataset_dir / "manifest.json").read_text(encoding="utf-8"))
    waveform_spec = json.loads((dataset_dir / "metadata" / "waveform_spec.json").read_text(encoding="utf-8"))

    first_sequence_rows = [row for row in slow_rows if row["sequence_id"] == "Q000001"]

    assert {row["stage_profile"] for row in index_rows} == {"variable_onset"}
    assert manifest["stage_profile"] == "variable_onset"
    assert manifest["stage_jitter"] == 0.15
    assert waveform_spec["stage_profile"] == "variable_onset"
    assert waveform_spec["stage_jitter"] == 0.15
    assert waveform_spec["phase_schedule"]["segments"][0]["duration_frac"] == 0.35
    assert first_sequence_rows[0]["phase_id"] == "baseline"
    assert first_sequence_rows[8]["phase_id"] == "baseline"
    assert first_sequence_rows[12]["phase_id"] == "exposure"


def test_generate_benchmark_dataset_supports_multi_pulse_profile(tmp_path):
    generate_benchmark_dataset(
        tmp_path,
        BenchmarkGenerationSpec(
            dataset_slug="wv4-multi-pulse",
            sequence_count=4,
            seed=43,
            timesteps=24,
            storage="npz",
            multi_path_phase="off",
            stage_profile="multi_pulse",
            optical_absorption_backend="empirical_v1",
        ),
    )

    rows = _read_csv(tmp_path / "wv4-multi-pulse" / "sequences" / "slow_sequence_long.csv")
    first_sequence_phases = [row["phase_id"] for row in rows if row["sequence_id"] == "Q000001"]

    assert first_sequence_phases.count("baseline") == 6
    assert first_sequence_phases.count("exposure") == 6
    assert first_sequence_phases.count("steady") == 6
    assert first_sequence_phases.count("recovery") == 6


def test_generate_benchmark_dataset_writes_memmap_storage_arrays_without_npz(tmp_path):
    generate_benchmark_dataset(
        tmp_path,
        BenchmarkGenerationSpec(
            dataset_slug="wv4-memmap",
            sequence_count=4,
            seed=23,
            timesteps=8,
            storage="memmap",
            optical_absorption_backend="empirical_v1",
        ),
    )

    dataset_dir = tmp_path / "wv4-memmap"

    assert (dataset_dir / "sequences" / "slow.npy").is_file()
    assert (dataset_dir / "sequences" / "ultrasonic_int16.npy").is_file()
    assert (dataset_dir / "sequences" / "ultrasonic_tof_s.npy").is_file()
    assert (dataset_dir / "sequences" / "ultrasonic_tof_observed_s.npy").is_file()
    assert (dataset_dir / "sequences" / "ultrasonic_peak_index.npy").is_file()
    assert (dataset_dir / "sequences" / "ultrasonic_sound_speed_m_per_s.npy").is_file()
    assert (dataset_dir / "sequences" / "ultrasonic_sound_speed_estimated_m_per_s.npy").is_file()
    assert (dataset_dir / "sequences" / "ultrasonic_alpha_true_npm.npy").is_file()
    assert (dataset_dir / "sequences" / "ultrasonic_tof_quality.npy").is_file()
    assert (dataset_dir / "sequences" / "ultrasonic_tof_accepted.npy").is_file()
    assert (dataset_dir / "sequences" / "fiber_mic_int16.npy").is_file()
    assert not (dataset_dir / "sequences" / "waveform_sequence.npz").exists()


def test_generate_benchmark_dataset_rejects_invalid_storage(tmp_path):
    try:
        generate_benchmark_dataset(
            tmp_path,
            BenchmarkGenerationSpec(dataset_slug="bad-storage", sequence_count=4, seed=1, storage="csv"),
        )
    except ValueError as exc:
        assert "storage must be one of" in str(exc)
    else:
        raise AssertionError("invalid storage was accepted")


def test_time_axis_presets_define_long_sequence_tiers():
    assert resolve_time_axis_preset("short").timesteps == 128
    assert resolve_time_axis_preset("standard").timesteps == 512
    assert resolve_time_axis_preset("long").timesteps == 1024
    assert resolve_time_axis_preset("xlong").timesteps == 2048


def test_multi_tau_channel_step_moves_toward_target_with_recovery_floor():
    params = {
        "tau_rise_system_s": 10.0,
        "tau_decay_system_s": 10.0,
        "fast_tau_fraction": 0.3,
        "slow_tau_multiplier": 3.0,
        "fast_response_weight": 0.7,
        "recovery_floor_fraction": 0.05,
    }

    rising = _multi_tau_channel_step(previous=1.0, target=2.0, params=params)
    decaying = _multi_tau_channel_step(previous=2.0, target=1.0, params=params)

    assert 1.0 < rising < 2.0
    assert 1.0 < decaying < 2.0


def test_multi_tau_channel_step_recovery_floor_slows_decay():
    base = {
        "tau_rise_system_s": 10.0,
        "tau_decay_system_s": 10.0,
        "fast_tau_fraction": 0.3,
        "slow_tau_multiplier": 3.0,
        "fast_response_weight": 0.7,
    }
    sticky = _multi_tau_channel_step(previous=2.0, target=1.0, params={**base, "recovery_floor_fraction": 0.3})
    free = _multi_tau_channel_step(previous=2.0, target=1.0, params={**base, "recovery_floor_fraction": 0.0})

    # recovery_floor 把衰减目标拉向当前值，单步衰减更慢（结果更接近起点 2.0）。
    assert sticky > free
    assert free > 1.0
