from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from tv3.pipeline.build_tv3_raw_dsp_features import (
    FRAME_OUTPUTS,
    SEQUENCE_OUTPUTS,
    build_tv3_raw_dsp_feature_cache,
    main,
    preflight_tv3_raw_dsp_dataset,
)
from tv3.pipeline.run_tv3_rocket_baseline import main as run_rocket_main


def _make_tv3_dataset(tmp_path: Path, slug: str = "tv3-raw-dsp-smoke", sequences: int = 16) -> Path:
    from tv3.sim.generation.tunnel_ventilation import (
        TunnelVentilationBenchmarkGenerationSpec,
        generate_tunnel_ventilation_benchmark_dataset,
    )

    generate_tunnel_ventilation_benchmark_dataset(
        tmp_path,
        TunnelVentilationBenchmarkGenerationSpec(
            dataset_slug=slug,
            sequence_count=sequences,
            seed=20260710,
            timesteps=16,
            storage="npz",
            optical_absorption_backend="empirical_v1",
            workers=1,
        ),
    )
    return tmp_path / slug


@pytest.fixture(scope="module")
def raw_dsp_dataset(tmp_path_factory: pytest.TempPathFactory) -> Path:
    return _make_tv3_dataset(tmp_path_factory.mktemp("raw_dsp_dataset"))


def test_preflight_freezes_waveform_scale_phase_split_and_named_slow_contract(raw_dsp_dataset: Path):
    preflight = preflight_tv3_raw_dsp_dataset(raw_dsp_dataset)

    assert preflight.waveform_path.is_file()
    assert preflight.waveform_scale_path.is_file()
    assert preflight.phase_csv_path.is_file()
    assert preflight.waveform_dtype == "int16"
    assert preflight.slow_channel_names == (
        "V_NDIR_CO2",
        "V_TCS",
        "T_C",
        "P_MPa",
        "H_RH",
        "L_m",
        "piston_position_m",
    )
    assert preflight.extra_slow_channels == ()
    assert sum(len(indices) for indices in preflight.split_indices.values()) == preflight.sequence_count


def test_exact_template_cache_writes_only_derived_feature_directory_and_auditable_manifest(
    raw_dsp_dataset: Path,
    tmp_path: Path,
):
    preflight = preflight_tv3_raw_dsp_dataset(raw_dsp_dataset)
    cache_dir = tmp_path / "raw_dsp_frame_v1"

    cache = build_tv3_raw_dsp_feature_cache(
        preflight,
        cache_dir=cache_dir,
        template_mode="exact_simulator_debug",
        workers=1,
    )
    manifest = json.loads((cache.cache_dir / "manifest.json").read_text(encoding="utf-8"))

    assert cache.cache_dir == cache_dir
    assert manifest["schema_version"] == "tv3-raw-dsp-frame-1"
    assert manifest["diagnostic_only"] is True
    assert manifest["template_mode"] == "exact_simulator_debug"
    assert manifest["template_peak_offset_samples"] == 25
    assert manifest["template_reference_peak_polarity"] == -1
    assert manifest["peak_interpolation_method"] == "three_point_parabolic"
    assert manifest["delay_calibration_method"] == "per_sequence_baseline_fresh_air_median"
    assert set(manifest["input_files"]) == {
        "manifest",
        "waveform_spec",
        "sequence_ids",
        "slow_channel_names",
        "waveform",
        "waveform_scale",
        "slow",
        "phase_csv",
        "split_train",
        "split_val",
        "split_test",
        "split_extrapolation",
    }
    assert not any("tof_observed" in item["path"] for item in manifest["input_files"].values())
    assert not (raw_dsp_dataset / "sequences" / "ultrasonic_peak_index_raw_dsp.npy").exists()
    for filename in (*FRAME_OUTPUTS, *SEQUENCE_OUTPUTS):
        assert (cache.cache_dir / filename).is_file()


def test_train_baseline_template_is_train_split_only_and_cache_reuse_is_explicit(
    raw_dsp_dataset: Path,
    tmp_path: Path,
):
    preflight = preflight_tv3_raw_dsp_dataset(raw_dsp_dataset)
    cache_dir = tmp_path / "raw_dsp_train_template"
    kwargs = {
        "cache_dir": cache_dir,
        "template_mode": "train_baseline_median",
        "template_source_split": "train",
        "template_max_frames": 32,
        "workers": 1,
    }

    first = build_tv3_raw_dsp_feature_cache(preflight, **kwargs)
    second = build_tv3_raw_dsp_feature_cache(preflight, **kwargs)
    manifest = json.loads((cache_dir / "manifest.json").read_text(encoding="utf-8"))

    assert first.reused is False
    assert second.reused is True
    assert manifest["template_source_split"] == "train"
    assert manifest["template_source_frame_count"] == 32
    assert manifest["template_peak_offset_samples"] == 25
    assert manifest["template_reference_peak_polarity"] == -1
    assert manifest["diagnostic_only"] is False
    assert len(manifest["template_digest"]) == 64


def test_matching_filter_cpu_multiprocess_is_numerically_identical(raw_dsp_dataset: Path, tmp_path: Path):
    preflight = preflight_tv3_raw_dsp_dataset(raw_dsp_dataset)
    single = build_tv3_raw_dsp_feature_cache(
        preflight,
        cache_dir=tmp_path / "single",
        template_mode="exact_simulator_debug",
        workers=1,
    )
    multi = build_tv3_raw_dsp_feature_cache(
        preflight,
        cache_dir=tmp_path / "multi",
        template_mode="exact_simulator_debug",
        workers=2,
    )

    for filename in (*FRAME_OUTPUTS, *SEQUENCE_OUTPUTS):
        single_values = np.load(single.cache_dir / filename)
        multi_values = np.load(multi.cache_dir / filename)
        np.testing.assert_array_equal(single_values, multi_values)


def test_existing_cache_manifest_mismatch_fails_instead_of_silent_reuse(
    raw_dsp_dataset: Path,
    tmp_path: Path,
):
    preflight = preflight_tv3_raw_dsp_dataset(raw_dsp_dataset)
    cache_dir = tmp_path / "mismatch"
    build_tv3_raw_dsp_feature_cache(
        preflight,
        cache_dir=cache_dir,
        template_mode="exact_simulator_debug",
        workers=1,
    )

    with pytest.raises(ValueError, match="manifest mismatch"):
        build_tv3_raw_dsp_feature_cache(
            preflight,
            cache_dir=cache_dir,
            template_mode="exact_simulator_debug",
            workers=1,
            raw_dsp_overrides={"min_corr_peak": 0.6},
        )


def test_cli_config_builds_cache_and_reports_result(raw_dsp_dataset: Path, tmp_path: Path, capsys):
    cache_dir = tmp_path / "cli_cache"
    config_path = tmp_path / "raw_dsp_config.json"
    config_path.write_text(
        json.dumps(
            {
                "dataset_dir": str(raw_dsp_dataset),
                "cache_dir": str(cache_dir),
                "template_mode": "exact_simulator_debug",
                "workers": 1,
            }
        ),
        encoding="utf-8",
    )

    exit_code = main(["--config", str(config_path)])
    payload = json.loads(capsys.readouterr().out)

    assert exit_code == 0
    assert Path(payload["cache_dir"]) == cache_dir
    assert payload["template_mode"] == "exact_simulator_debug"
    assert payload["reused"] is False


def test_preflight_rejects_missing_waveform_scale(tmp_path: Path):
    dataset_dir = _make_tv3_dataset(tmp_path, slug="missing-scale", sequences=8)
    (dataset_dir / "sequences" / "ultrasonic_scale.npy").unlink()

    with pytest.raises(FileNotFoundError, match="ultrasonic_scale"):
        preflight_tv3_raw_dsp_dataset(dataset_dir)


def test_raw_dsp_ridge_smoke_runs_from_derived_cache(raw_dsp_dataset: Path, tmp_path: Path, capsys):
    preflight = preflight_tv3_raw_dsp_dataset(raw_dsp_dataset)
    build_tv3_raw_dsp_feature_cache(
        preflight,
        template_mode="train_baseline_median",
        template_max_frames=32,
        workers=1,
    )
    project_root = Path(__file__).resolve().parents[1]
    config = json.loads(
        (project_root / "configs" / "tv3_d2b_raw_dsp_ridge.json").read_text(encoding="utf-8")
    )
    output_dir = tmp_path / "ridge_output"
    config["dataset_dir"] = str(raw_dsp_dataset)
    config["output_dir"] = str(output_dir)
    config_path = tmp_path / "ridge_config.json"
    config_path.write_text(json.dumps(config), encoding="utf-8")

    exit_code = run_rocket_main(["--config", str(config_path)])
    payload = json.loads(capsys.readouterr().out)

    assert exit_code == 0
    assert payload["feature_builder"] == "d0_raw_dsp_physics_stats_v1"
    assert payload["head"] == "ridgecv"
    assert set(payload["evaluations"]) == {"train", "val", "test", "extrapolation"}
    assert (output_dir / "metrics.json").is_file()
