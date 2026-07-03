import csv
import json

import numpy as np
import pytest

from pipeline.bundle_waveform_sequence import bundle_waveform_sequence
from sim.generation import benchmark as benchmark_module
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
    assert manifest["path_lms"] == [0.18, 0.2, 0.22, 0.25, 0.28]
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


def test_split_summary_contains_component_stats(tmp_path):
    summary = generate_benchmark_dataset(
        tmp_path,
        BenchmarkGenerationSpec(dataset_slug="wv4-split-stats", sequence_count=16, seed=7, optical_absorption_backend="empirical_v1"),
    )
    split_summary = json.loads((tmp_path / "wv4-split-stats" / "splits" / "split_summary.json").read_text(encoding="utf-8"))

    # 每个 split 应有 components
    for split_name in ("train", "val", "test", "extrapolation"):
        split_entry = split_summary["splits"][split_name]
        assert "components" in split_entry, f"{split_name} missing components"
        comps = split_entry["components"]
        for field in ("x_H2", "x_CH4", "x_CO2", "x_N2"):
            assert field in comps, f"{split_name}/{field} missing"
            stats = comps[field]
            for key in ("mean", "std", "min", "max", "p25", "p50", "p75"):
                assert key in stats, f"{split_name}/{field}/{key} missing"
                assert isinstance(stats[key], (int, float)), f"{split_name}/{field}/{key} not numeric"

    train_h2 = split_summary["splits"]["train"]["components"]["x_H2"]
    # H2 双峰映射范围 [0, 30]，验证边界和有限性
    assert train_h2["min"] >= 0.0
    assert train_h2["max"] <= 30.0
    assert train_h2["mean"] >= train_h2["min"]
    assert train_h2["mean"] <= train_h2["max"]
    assert train_h2["std"] > 0.0  # H2 为双峰分布，必有方差
    assert train_h2["p25"] <= train_h2["p50"] <= train_h2["p75"]


def test_split_summary_distribution_checks_pass(tmp_path):
    generate_benchmark_dataset(
        tmp_path,
        BenchmarkGenerationSpec(dataset_slug="wv4-dist-check", sequence_count=200, seed=13, optical_absorption_backend="empirical_v1"),
    )
    split_summary = json.loads((tmp_path / "wv4-dist-check" / "splits" / "split_summary.json").read_text(encoding="utf-8"))

    assert "distribution_checks" in split_summary, "distribution_checks key missing"
    checks = split_summary["distribution_checks"]

    # 应有 val_vs_train / test_vs_train / extrapolation_vs_train
    for check_key in ("val_vs_train", "test_vs_train", "extrapolation_vs_train"):
        assert check_key in checks, f"{check_key} missing"
        entry = checks[check_key]
        assert "status" in entry, f"{check_key} missing status"
        assert "ks_tests" in entry, f"{check_key} missing ks_tests"
        # 随机分箱不应产生显著分布偏移，预期 pass
        assert entry["status"] == "pass", f"{check_key} status={entry['status']} — unexpected distribution shift"

        ks = entry["ks_tests"]
        for field in ("x_H2", "x_CH4", "x_CO2", "x_N2"):
            assert field in ks, f"{check_key}/{field} missing"
            assert "statistic" in ks[field], f"{check_key}/{field} missing statistic"
            assert "p_value" in ks[field], f"{check_key}/{field} missing p_value"
            assert ks[field]["p_value"] >= 0.05, f"{check_key}/{field} p={ks[field]['p_value']} < 0.05"


def test_split_summary_backward_compatible_without_conditions(tmp_path):
    """调用 _split_summary(split_rows) 不带 conditions 应返回最小摘要。"""
    split_rows = {
        "train": [{"sequence_id": "Q1", "mixture_id": "M1"}],
        "val": [{"sequence_id": "Q2", "mixture_id": "M2"}],
        "test": [{"sequence_id": "Q3", "mixture_id": "M3"}],
        "extrapolation": [{"sequence_id": "Q4", "mixture_id": "M4"}],
    }
    result = benchmark_module._split_summary(split_rows)

    assert "split_policy" in result
    assert "group_field" in result
    assert "splits" in result
    assert "components" not in result["splits"]["train"]
    assert "distribution_checks" not in result


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
    ultrasonic = np.load(dataset_dir / "sequences" / "ultrasonic_int32.npy")
    ultrasonic_tof_s = np.load(dataset_dir / "sequences" / "ultrasonic_tof_s.npy")
    ultrasonic_tof_observed_s = np.load(dataset_dir / "sequences" / "ultrasonic_tof_observed_s.npy")
    ultrasonic_peak_index = np.load(dataset_dir / "sequences" / "ultrasonic_peak_index.npy")
    ultrasonic_sound_speed = np.load(dataset_dir / "sequences" / "ultrasonic_sound_speed_m_per_s.npy")
    ultrasonic_sound_speed_estimated = np.load(dataset_dir / "sequences" / "ultrasonic_sound_speed_estimated_m_per_s.npy")
    ultrasonic_alpha = np.load(dataset_dir / "sequences" / "ultrasonic_alpha_true_npm.npy")
    ultrasonic_tof_quality = np.load(dataset_dir / "sequences" / "ultrasonic_tof_quality.npy")
    ultrasonic_tof_accepted = np.load(dataset_dir / "sequences" / "ultrasonic_tof_accepted.npy")
    fiber_mic = np.load(dataset_dir / "sequences" / "fiber_mic_int32.npy")
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
    # sim_revision.l_m_range 应从 spec.path_lms 派生，而非硬编码（防止与 path_lms 矛盾）
    assert manifest["sim_revision"]["l_m_range"] == [0.25, 0.35]
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
    assert (dataset_dir / "sequences" / "ultrasonic_int32.npy").is_file()
    assert (dataset_dir / "sequences" / "ultrasonic_tof_s.npy").is_file()
    assert (dataset_dir / "sequences" / "ultrasonic_tof_observed_s.npy").is_file()
    assert (dataset_dir / "sequences" / "ultrasonic_peak_index.npy").is_file()
    assert (dataset_dir / "sequences" / "ultrasonic_sound_speed_m_per_s.npy").is_file()
    assert (dataset_dir / "sequences" / "ultrasonic_sound_speed_estimated_m_per_s.npy").is_file()
    assert (dataset_dir / "sequences" / "ultrasonic_alpha_true_npm.npy").is_file()
    assert (dataset_dir / "sequences" / "ultrasonic_tof_quality.npy").is_file()
    assert (dataset_dir / "sequences" / "ultrasonic_tof_accepted.npy").is_file()
    assert (dataset_dir / "sequences" / "fiber_mic_int32.npy").is_file()
    assert not (dataset_dir / "sequences" / "waveform_sequence.npz").exists()


def test_publish_staging_restores_existing_output_when_move_fails(tmp_path, monkeypatch):
    output_dir = tmp_path / "wv4-existing"
    staging_dir = tmp_path / "wv4-existing.tmp-new"
    output_dir.mkdir()
    staging_dir.mkdir()
    (output_dir / "manifest.json").write_text("old", encoding="utf-8")
    (staging_dir / "manifest.json").write_text("new", encoding="utf-8")

    def fail_move(src, dst):
        raise RuntimeError("publish failed")

    monkeypatch.setattr(benchmark_module.shutil, "move", fail_move)

    with pytest.raises(RuntimeError, match="publish failed"):
        benchmark_module._publish_staging_dir(staging_dir, output_dir)

    assert (output_dir / "manifest.json").read_text(encoding="utf-8") == "old"
    assert (staging_dir / "manifest.json").read_text(encoding="utf-8") == "new"
    assert not list(tmp_path.glob("wv4-existing.bak-*"))


def test_publish_staging_replaces_existing_output_after_success(tmp_path):
    output_dir = tmp_path / "wv4-existing"
    staging_dir = tmp_path / "wv4-existing.tmp-new"
    output_dir.mkdir()
    staging_dir.mkdir()
    (output_dir / "manifest.json").write_text("old", encoding="utf-8")
    (staging_dir / "manifest.json").write_text("new", encoding="utf-8")

    benchmark_module._publish_staging_dir(staging_dir, output_dir)

    assert (output_dir / "manifest.json").read_text(encoding="utf-8") == "new"
    assert not staging_dir.exists()
    assert not list(tmp_path.glob("wv4-existing.bak-*"))


def test_parallel_generation_preserves_schema_shape_split_manifest_and_quality(tmp_path):
    serial_summary = generate_benchmark_dataset(
        tmp_path,
        BenchmarkGenerationSpec(
            dataset_slug="wv4-serial-contract",
            sequence_count=6,
            seed=53,
            timesteps=8,
            storage="memmap",
            optical_absorption_backend="empirical_v1",
            workers=1,
        ),
    )
    parallel_summary = generate_benchmark_dataset(
        tmp_path,
        BenchmarkGenerationSpec(
            dataset_slug="wv4-parallel-contract",
            sequence_count=6,
            seed=53,
            timesteps=8,
            storage="memmap",
            optical_absorption_backend="empirical_v1",
            workers=2,
            chunk_size=2,
        ),
    )

    serial_dir = tmp_path / "wv4-serial-contract"
    parallel_dir = tmp_path / "wv4-parallel-contract"
    serial_manifest = json.loads((serial_dir / "manifest.json").read_text(encoding="utf-8"))
    parallel_manifest = json.loads((parallel_dir / "manifest.json").read_text(encoding="utf-8"))
    serial_quality = json.loads((serial_dir / "quality" / "validation_summary.json").read_text(encoding="utf-8"))
    parallel_quality = json.loads((parallel_dir / "quality" / "validation_summary.json").read_text(encoding="utf-8"))

    assert serial_summary["sequence_count"] == parallel_summary["sequence_count"] == 6
    assert serial_manifest["shapes"] == parallel_manifest["shapes"]
    assert serial_manifest["storage"] == parallel_manifest["storage"] == "memmap"
    assert serial_manifest["slow_channels"] == parallel_manifest["slow_channels"]
    assert serial_manifest["labels"] == parallel_manifest["labels"]
    assert serial_quality["status"] == parallel_quality["status"] == "pass"
    assert _read_csv(serial_dir / "condition_grid_sequence.csv") == _read_csv(parallel_dir / "condition_grid_sequence.csv")
    assert _read_csv(serial_dir / "splits" / "train.csv") == _read_csv(parallel_dir / "splits" / "train.csv")
    assert _read_csv(serial_dir / "splits" / "val.csv") == _read_csv(parallel_dir / "splits" / "val.csv")
    assert _read_csv(serial_dir / "splits" / "test.csv") == _read_csv(parallel_dir / "splits" / "test.csv")
    assert _read_csv(serial_dir / "splits" / "extrapolation.csv") == _read_csv(parallel_dir / "splits" / "extrapolation.csv")
    assert np.load(serial_dir / "metadata" / "sequence_ids.npy", allow_pickle=True).tolist() == np.load(
        parallel_dir / "metadata" / "sequence_ids.npy",
        allow_pickle=True,
    ).tolist()


def test_parallel_generation_is_stable_across_chunk_sizes(tmp_path):
    for slug, chunk_size in (("wv4-chunk-1", 1), ("wv4-chunk-3", 3)):
        generate_benchmark_dataset(
            tmp_path,
            BenchmarkGenerationSpec(
                dataset_slug=slug,
                sequence_count=6,
                seed=59,
                timesteps=8,
                storage="memmap",
                multi_path_phase="off",
                optical_absorption_backend="empirical_v1",
                workers=2,
                chunk_size=chunk_size,
            ),
        )

    chunk_1_dir = tmp_path / "wv4-chunk-1"
    chunk_3_dir = tmp_path / "wv4-chunk-3"

    assert _read_csv(chunk_1_dir / "condition_grid_sequence.csv") == _read_csv(chunk_3_dir / "condition_grid_sequence.csv")
    assert _read_csv(chunk_1_dir / "sequences" / "slow_sequence_long.csv") == _read_csv(chunk_3_dir / "sequences" / "slow_sequence_long.csv")
    np.testing.assert_allclose(np.load(chunk_1_dir / "sequences" / "slow.npy"), np.load(chunk_3_dir / "sequences" / "slow.npy"))
    np.testing.assert_array_equal(
        np.load(chunk_1_dir / "sequences" / "ultrasonic_int32.npy"),
        np.load(chunk_3_dir / "sequences" / "ultrasonic_int32.npy"),
    )
    np.testing.assert_array_equal(
        np.load(chunk_1_dir / "sequences" / "fiber_mic_int32.npy"),
        np.load(chunk_3_dir / "sequences" / "fiber_mic_int32.npy"),
    )
    np.testing.assert_allclose(np.load(chunk_1_dir / "labels" / "y.npy"), np.load(chunk_3_dir / "labels" / "y.npy"))


def test_bundle_waveform_sequence_builds_npz_after_memmap_generation(tmp_path):
    generate_benchmark_dataset(
        tmp_path,
        BenchmarkGenerationSpec(
            dataset_slug="wv4-memmap-bundle",
            sequence_count=4,
            seed=61,
            timesteps=8,
            storage="memmap",
            optical_absorption_backend="empirical_v1",
            workers=2,
            chunk_size=2,
        ),
    )

    dataset_dir = tmp_path / "wv4-memmap-bundle"
    assert not (dataset_dir / "sequences" / "waveform_sequence.npz").exists()

    summary = bundle_waveform_sequence(dataset_dir)
    bundle = np.load(dataset_dir / "sequences" / "waveform_sequence.npz")

    assert summary["dataset_dir"] == str(dataset_dir)
    assert bundle["slow"].shape == np.load(dataset_dir / "sequences" / "slow.npy").shape
    assert bundle["ultrasonic"].shape == np.load(dataset_dir / "sequences" / "ultrasonic_int32.npy").shape
    assert bundle["y"].shape == np.load(dataset_dir / "labels" / "y.npy").shape


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
