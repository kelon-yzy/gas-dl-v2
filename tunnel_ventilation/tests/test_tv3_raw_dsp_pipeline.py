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
from tv3.ml.rocket_training import RAW_DSP_FIDELITY_SCHEMA_VERSION, load_raw_dsp_fidelity


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


def _write_passed_fidelity_metrics(dataset_dir: Path, cache_dir: Path, output_dir: Path) -> Path:
    manifest = json.loads((cache_dir / "manifest.json").read_text(encoding="utf-8"))
    output_dir.mkdir(parents=True)
    metrics_path = output_dir / "metrics.json"
    metrics_path.write_text(
        json.dumps(
            {
                "schema_version": RAW_DSP_FIDELITY_SCHEMA_VERSION,
                "status": "passed",
                "source": {
                    "dataset_dir": str(dataset_dir),
                    "cache_build_signature": manifest["build_signature"],
                    "template_digest": manifest["template_digest"],
                },
            }
        ),
        encoding="utf-8",
    )
    return metrics_path


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


def test_b6_mlp_config_matches_b1_feature_contract_plus_r5t_mlp_fields():
    project_root = Path(__file__).resolve().parents[1]
    ridge = json.loads((project_root / "configs" / "tv3_d2b_raw_dsp_ridge.json").read_text(encoding="utf-8"))
    mlp = json.loads(
        (project_root / "configs" / "tv3_d2b_raw_dsp_mlp_target_scaled.json").read_text(encoding="utf-8")
    )
    feature_keys = (
        "feature_set",
        "feature_builder",
        "include_slow",
        "slow_channels",
        "physics_arrays",
        "sequence_statistics",
        "phase_windows",
        "early_fractions",
        "eval_splits",
    )
    for key in feature_keys:
        assert mlp[key] == ridge[key], key
    assert mlp["dataset_dir"] == ridge["dataset_dir"]
    assert mlp["head"] == "mlp"
    assert mlp["output_dir"] == "outputs/tv3_d2b/raw_dsp_mlp_target_scaled"
    assert mlp["mlp_hidden_dims"] == [256, 128]
    assert mlp["mlp_dropout"] == 0.1
    assert mlp["mlp_weight_decay"] == 0.0001
    assert mlp["mlp_lr"] == 0.001
    assert mlp["mlp_batch_size"] == 256
    assert mlp["mlp_max_epochs"] == 200
    assert mlp["mlp_patience"] == 20
    assert mlp["mlp_loss_weights"] == [1.0, 2.0, 1.0]
    assert mlp["mlp_standardize_targets"] is True
    assert mlp["seed"] == 20260704
    assert mlp["device"] == "cuda"
    assert "ridge_alphas" not in mlp


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
    cache_manifest = json.loads(
        (raw_dsp_dataset / "features" / "raw_dsp" / "raw_dsp_frame_v1" / "manifest.json").read_text(
            encoding="utf-8"
        )
    )

    assert exit_code == 0
    assert payload["feature_builder"] == "d0_raw_dsp_physics_stats_v1"
    assert payload["head"] == "ridgecv"
    assert set(payload["evaluations"]) == {"train", "val", "test", "extrapolation"}
    assert (output_dir / "metrics.json").is_file()
    provenance = payload["raw_dsp_provenance"]
    assert provenance["build_signature"] == cache_manifest["build_signature"]
    assert provenance["template_digest"] == cache_manifest["template_digest"]
    assert provenance["template_mode"] == "train_baseline_median"
    assert provenance["template_source_split"] == "train"
    assert provenance["diagnostic_only"] is False
    assert "o2_audit" not in payload


def test_raw_dsp_mlp_smoke_writes_provenance_and_target_scaled_diagnostics(
    raw_dsp_dataset: Path,
    tmp_path: Path,
    capsys,
):
    preflight = preflight_tv3_raw_dsp_dataset(raw_dsp_dataset)
    build_tv3_raw_dsp_feature_cache(
        preflight,
        template_mode="train_baseline_median",
        template_max_frames=32,
        workers=1,
    )
    project_root = Path(__file__).resolve().parents[1]
    b1_config = json.loads(
        (project_root / "configs" / "tv3_d2b_raw_dsp_ridge.json").read_text(encoding="utf-8")
    )
    b1_output_dir = tmp_path / "b1_output"
    b1_config["dataset_dir"] = str(raw_dsp_dataset)
    b1_config["output_dir"] = str(b1_output_dir)
    b1_config_path = tmp_path / "b1_config.json"
    b1_config_path.write_text(json.dumps(b1_config), encoding="utf-8")
    assert run_rocket_main(["--config", str(b1_config_path)]) == 0
    capsys.readouterr()

    config = json.loads(
        (project_root / "configs" / "tv3_d2b_raw_dsp_mlp_target_scaled.json").read_text(encoding="utf-8")
    )
    output_dir = tmp_path / "mlp_output"
    config["dataset_dir"] = str(raw_dsp_dataset)
    config["output_dir"] = str(output_dir)
    config["device"] = "cpu"
    config["mlp_hidden_dims"] = [32, 16]
    config["mlp_batch_size"] = 8
    config["mlp_max_epochs"] = 3
    config["mlp_patience"] = 2
    config["raw_dsp_fidelity_metrics_path"] = str(
        _write_passed_fidelity_metrics(
            raw_dsp_dataset,
            raw_dsp_dataset / "features" / "raw_dsp" / "raw_dsp_frame_v1",
            tmp_path / "fidelity",
        )
    )
    config["raw_dsp_reference_metrics_path"] = str(b1_output_dir / "metrics.json")
    config_path = tmp_path / "mlp_config.json"
    config_path.write_text(json.dumps(config), encoding="utf-8")

    exit_code = run_rocket_main(["--config", str(config_path)])
    payload = json.loads(capsys.readouterr().out)

    assert exit_code == 0
    assert payload["head"] == "mlp"
    assert payload["feature_builder"] == "d0_raw_dsp_physics_stats_v1"
    assert payload["diagnostics"]["model_config"]["standardize_targets"] is True
    assert payload["diagnostics"]["parameter_count"] > 0
    assert payload["raw_dsp_provenance"]["template_mode"] == "train_baseline_median"
    assert payload["raw_dsp_provenance"]["diagnostic_only"] is False
    assert payload["raw_dsp_fidelity"]["status"] == "passed"
    assert "delta_vs_b1_o2_r2" in payload["o2_audit"]
    assert (output_dir / "metrics.json").is_file()


def _write_b6_multiseed_report(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "seeds": [42, 123, 456],
                "groups": {
                    "b6": {
                        "verdict": "stable_pass",
                        "pass_count": 3,
                        "completed_seeds": [42, 123, 456],
                        "o2_r2_stats": {
                            "val": {"mean": 0.5581, "std": 0.0096},
                            "test": {"mean": 0.5356, "std": 0.0170},
                            "extrapolation": {"mean": 0.4835, "std": 0.0036},
                        },
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    return path


def test_raw_dsp_b7_smoke_writes_oof_diagnostics_and_b6_reference(
    raw_dsp_dataset: Path,
    tmp_path: Path,
    capsys,
):
    preflight = preflight_tv3_raw_dsp_dataset(raw_dsp_dataset)
    build_tv3_raw_dsp_feature_cache(
        preflight,
        template_mode="train_baseline_median",
        template_max_frames=32,
        workers=1,
    )
    project_root = Path(__file__).resolve().parents[1]
    b1_config = json.loads(
        (project_root / "configs" / "tv3_d2b_raw_dsp_ridge.json").read_text(encoding="utf-8")
    )
    b1_output_dir = tmp_path / "b1_output"
    b1_config["dataset_dir"] = str(raw_dsp_dataset)
    b1_config["output_dir"] = str(b1_output_dir)
    b1_config_path = tmp_path / "b1_config.json"
    b1_config_path.write_text(json.dumps(b1_config), encoding="utf-8")
    assert run_rocket_main(["--config", str(b1_config_path)]) == 0
    capsys.readouterr()

    config = json.loads(
        (project_root / "configs" / "tv3_d2b_oof_ridge_residual_mlp.json").read_text(encoding="utf-8")
    )
    output_dir = tmp_path / "b7_output"
    config["dataset_dir"] = str(raw_dsp_dataset)
    config["output_dir"] = str(output_dir)
    config["device"] = "cpu"
    config["mlp_hidden_dims"] = [16, 16]
    config["mlp_batch_size"] = 8
    config["mlp_max_epochs"] = 3
    config["mlp_patience"] = 2
    config["oof_folds"] = 3
    config["raw_dsp_fidelity_metrics_path"] = str(
        _write_passed_fidelity_metrics(
            raw_dsp_dataset,
            raw_dsp_dataset / "features" / "raw_dsp" / "raw_dsp_frame_v1",
            tmp_path / "fidelity_b7",
        )
    )
    config["raw_dsp_reference_metrics_path"] = str(b1_output_dir / "metrics.json")
    config["b6_multiseed_report_path"] = str(
        _write_b6_multiseed_report(tmp_path / "b6_multiseed" / "replication_report.json")
    )
    config_path = tmp_path / "b7_config.json"
    config_path.write_text(json.dumps(config), encoding="utf-8")

    exit_code = run_rocket_main(["--config", str(config_path)])
    payload = json.loads(capsys.readouterr().out)

    assert exit_code == 0
    assert payload["head"] == "oof_ridge_residual_mlp"
    assert payload["feature_builder"] == "d0_raw_dsp_physics_stats_v1"
    assert payload["diagnostics"]["oof"]["fold_count"] == 3
    assert payload["diagnostics"]["oof"]["fold_seed"] == 20260711
    assert payload["diagnostics"]["oof"]["coverage_complete"] is True
    assert payload["diagnostics"]["ridge"]["full_selected_alpha"] > 0.0
    assert payload["diagnostics"]["residual_mlp"]["parameter_count"] > 0
    assert payload["diagnostics"]["residual_mlp"]["zero_init_output"] is True
    assert payload["diagnostics"]["b1_reference"]["metrics_sha256"]
    assert payload["diagnostics"]["b6_reference"]["report_sha256"]
    assert payload["raw_dsp_fidelity"]["status"] == "passed"
    assert "delta_vs_b1_o2_r2" in payload["o2_audit"]
    assert "delta_vs_b6_o2_r2_means" in payload["o2_audit"]
    assert payload["b6_reference"]["verdict"] == "stable_pass"
    assert (output_dir / "metrics.json").is_file()


def test_raw_dsp_b7_rejects_missing_b6_multiseed_report(
    raw_dsp_dataset: Path,
    tmp_path: Path,
):
    preflight = preflight_tv3_raw_dsp_dataset(raw_dsp_dataset)
    build_tv3_raw_dsp_feature_cache(
        preflight,
        template_mode="train_baseline_median",
        template_max_frames=32,
        workers=1,
    )
    project_root = Path(__file__).resolve().parents[1]
    b1_config = json.loads(
        (project_root / "configs" / "tv3_d2b_raw_dsp_ridge.json").read_text(encoding="utf-8")
    )
    b1_output_dir = tmp_path / "b1_output"
    b1_config["dataset_dir"] = str(raw_dsp_dataset)
    b1_config["output_dir"] = str(b1_output_dir)
    b1_config_path = tmp_path / "b1_config.json"
    b1_config_path.write_text(json.dumps(b1_config), encoding="utf-8")
    assert run_rocket_main(["--config", str(b1_config_path)]) == 0

    config = json.loads(
        (project_root / "configs" / "tv3_d2b_oof_ridge_residual_mlp.json").read_text(encoding="utf-8")
    )
    config["dataset_dir"] = str(raw_dsp_dataset)
    config["output_dir"] = str(tmp_path / "b7_missing_b6")
    config["device"] = "cpu"
    config["mlp_hidden_dims"] = [8]
    config["mlp_batch_size"] = 8
    config["mlp_max_epochs"] = 1
    config["mlp_patience"] = 1
    config["oof_folds"] = 2
    config["raw_dsp_fidelity_metrics_path"] = str(
        _write_passed_fidelity_metrics(
            raw_dsp_dataset,
            raw_dsp_dataset / "features" / "raw_dsp" / "raw_dsp_frame_v1",
            tmp_path / "fidelity_missing",
        )
    )
    config["raw_dsp_reference_metrics_path"] = str(b1_output_dir / "metrics.json")
    config.pop("b6_multiseed_report_path", None)
    config_path = tmp_path / "b7_missing.json"
    config_path.write_text(json.dumps(config), encoding="utf-8")

    with pytest.raises(ValueError, match="B6 multiseed report"):
        run_rocket_main(["--config", str(config_path)])


def test_raw_dsp_provenance_rejects_diagnostic_exact_template(tmp_path: Path):
    from tv3.ml.rocket_training import load_raw_dsp_provenance

    dataset_dir = tmp_path / "fake_dataset"
    cache_dir = dataset_dir / "features" / "raw_dsp" / "raw_dsp_frame_v1"
    cache_dir.mkdir(parents=True)
    (cache_dir / "manifest.json").write_text(
        json.dumps(
            {
                "schema_version": "tv3-raw-dsp-frame-1",
                "complete_dataset": True,
                "template_mode": "train_baseline_median",
                "template_source_split": "train",
                "diagnostic_only": True,
                "build_signature": "a" * 64,
                "template_digest": "b" * 64,
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="diagnostic_only"):
        load_raw_dsp_provenance(dataset_dir)


def test_raw_dsp_fidelity_rejects_mismatched_cache_signature(tmp_path: Path):
    metrics_path = tmp_path / "fidelity.json"
    metrics_path.write_text(
        json.dumps(
            {
                "schema_version": RAW_DSP_FIDELITY_SCHEMA_VERSION,
                "status": "passed",
                "source": {
                    "dataset_dir": str(tmp_path),
                    "cache_build_signature": "wrong",
                    "template_digest": "wrong",
                },
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="cache_build_signature"):
        load_raw_dsp_fidelity(
            metrics_path,
            dataset_dir=tmp_path,
            provenance={"build_signature": "expected", "template_digest": "expected"},
        )


def test_runner_refuses_to_overwrite_existing_metrics(tmp_path: Path):
    config_path = tmp_path / "config.json"
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    (output_dir / "metrics.json").write_text("{}", encoding="utf-8")
    config_path.write_text(
        json.dumps(
            {
                "dataset_dir": str(tmp_path / "unused_dataset"),
                "output_dir": str(output_dir),
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        run_rocket_main(["--config", str(config_path)])
