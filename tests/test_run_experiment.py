from __future__ import annotations

import json
from pathlib import Path

import pytest

from common.composition import TRAIN_MIN_POSITIVE_HALF_EPSILON
from dl.training.losses import (
    FREE_COMPONENT_MSE_LOSS,
    WEIGHTED_COMPONENT_MSE_LOSS,
    WEIGHTED_FREE_COMPONENT_MSE_LOSS,
)
from pipeline.experiment_config import ALL_MODALITIES, load_experiment_config
from pipeline.run_experiment import run
from sim.generation.benchmark import BenchmarkGenerationSpec, generate_benchmark_dataset


def _make_smoke_dataset(tmp_path: Path, slug: str = "experiment-smoke", sequences: int = 8) -> Path:
    generate_benchmark_dataset(
        tmp_path,
        BenchmarkGenerationSpec(
            dataset_slug=slug,
            sequence_count=sequences,
            seed=31,
            timesteps=8,
            storage="npz",
            optical_absorption_backend="empirical_v1",
        ),
    )
    return tmp_path / slug


def _base_config(dataset_dir: Path, output_root: Path) -> dict[str, object]:
    return {
        "experiment_name": "smoke_suite",
        "dataset_dir": str(dataset_dir),
        "output_root": str(output_root),
        "seed": 123,
        "device": "cpu",
        "eval_splits": ["val"],
        "training": {
            "epochs": 1,
            "batch_size": 4,
            "num_workers": 0,
            "optimizer": "adamw",
            "lr": 0.001,
            "weight_decay": 0.0,
            "loss": "mse",
            "early_stopping": {"enabled": False, "monitor": "val_loss", "patience": 2, "min_delta": 0.0, "mode": "min"},
            "scheduler": {"name": "none"},
        },
        "ml_runs": [
            {
                "name": "mean_slow",
                "model": "mean",
                "modalities": ["slow"],
                "protocol": False,
                "sequence_statistics": ["mean"],
                "target_transform": {"name": "alr_ch4", "epsilon": TRAIN_MIN_POSITIVE_HALF_EPSILON},
            }
        ],
        "dl_runs": [
            {
                "name": "cnn1d_smoke",
                "model": "cnn1d",
                "modalities": ["slow"],
                "model_kwargs": {"hidden_channels": [4], "kernel_size": 3, "dropout": 0.0},
            }
        ],
    }


def _write_config(path: Path, payload: dict[str, object]) -> Path:
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_load_experiment_config_applies_overrides(tmp_path: Path):
    dataset_dir = _make_smoke_dataset(tmp_path)
    config_path = _write_config(tmp_path / "config.json", _base_config(dataset_dir, tmp_path / "outputs"))
    override_dataset = tmp_path / "override-dataset"

    config = load_experiment_config(
        config_path,
        dataset_dir=override_dataset,
        output_root=tmp_path / "override-outputs",
        device="cpu",
    )

    assert config.dataset_dir == override_dataset
    assert config.output_root == tmp_path / "override-outputs"
    assert config.device == "cpu"


def test_load_experiment_config_rejects_unknown_model(tmp_path: Path):
    dataset_dir = _make_smoke_dataset(tmp_path)
    payload = _base_config(dataset_dir, tmp_path / "outputs")
    payload["dl_runs"] = [{"name": "bad", "model": "missing", "modalities": ["slow"]}]
    config_path = _write_config(tmp_path / "bad_config.json", payload)

    with pytest.raises(ValueError, match="Unknown DL model"):
        load_experiment_config(config_path)


def test_default_dl_runs_use_all_modalities(tmp_path: Path):
    payload = _base_config(tmp_path / "dataset", tmp_path / "outputs")
    payload["ml_runs"] = None
    payload["dl_runs"] = None
    config_path = _write_config(tmp_path / "defaults_config.json", payload)

    config = load_experiment_config(config_path)

    assert [run["name"] for run in config.dl_runs] == [
        "cnn1d",
        "tcn",
        "lstm",
        "transformer",
        "patchtst",
        "cnn1d_tcn_fusion",
        "cnn1d_tcn_fusion_ilr",
        "cnn1d_tcn_fusion_phase_exposure",
        "cnn1d_tcn_fusion_phase_recovery",
        "cnn1d_tcn_fusion_early_050",
        "cnn1d_tcn_fusion_early_075",
    ]
    assert len(config.dl_runs) == 11
    for run in config.dl_runs:
        assert tuple(run["modalities"]) == ALL_MODALITIES
    dynamic_run = [run for run in config.ml_runs if run["name"] == "dynamic_stacking_svr_all_modalities"][0]
    ml_phase_run = [run for run in config.ml_runs if run["name"] == "ridge_all_modalities_phase_exposure"][0]
    alr_run = [run for run in config.ml_runs if run["name"] == "ridge_alr_ch4_all_modalities"][0]
    ilr_run = [run for run in config.ml_runs if run["name"] == "ridge_ilr_n2_first_all_modalities"][0]
    fusion_ilr_run = [run for run in config.dl_runs if run["name"] == "cnn1d_tcn_fusion_ilr"][0]
    dl_phase_run = [run for run in config.dl_runs if run["name"] == "cnn1d_tcn_fusion_phase_exposure"][0]
    assert dynamic_run["model"]["n_jobs"] == 4
    assert ml_phase_run["window"] == {"kind": "phase", "value": "exposure"}
    assert dl_phase_run["window"] == {"kind": "phase", "value": "exposure"}
    assert "target_transform" not in ml_phase_run
    assert "target_transform" not in dl_phase_run
    assert alr_run["target_transform"]["epsilon"] == TRAIN_MIN_POSITIVE_HALF_EPSILON
    assert ilr_run["target_transform"]["epsilon"] == TRAIN_MIN_POSITIVE_HALF_EPSILON
    assert fusion_ilr_run["target_transform"]["epsilon"] == TRAIN_MIN_POSITIVE_HALF_EPSILON
    assert fusion_ilr_run["loss"] == "ilr_mse"


def test_phase_window_tcn_improvement_config_plans_gas_head_runs():
    config = load_experiment_config(
        Path("configs/experiment/phase_window_tcn_improvement/phase_window_tcn_improvement.json")
    )

    result = run(config, dry_run=True)

    assert config.training["lr"] == 0.0001
    assert config.training["batch_size"] == 16
    assert config.training["performance"] == {
        "cudnn_benchmark": True,
        "tf32": True,
        "compile": False,
        "compile_mode": "default",
    }
    assert [run_config["name"] for run_config in config.dl_runs] == [
        "phase_window_tcn_gas_4mse",
        "phase_window_tcn_gas_free",
    ]
    first, second = config.dl_runs
    assert first["model_kwargs"]["output_mode"] == "gas_head"
    assert second["model_kwargs"]["output_mode"] == "gas_head"
    assert second["loss"] == FREE_COMPONENT_MSE_LOSS
    assert result["plan"]["dl_runs"] == ["phase_window_tcn_gas_4mse", "phase_window_tcn_gas_free"]
    assert result["plan"]["dl_run_details"][0]["loss"] == "mse"
    assert result["plan"]["dl_run_details"][1]["loss"] == FREE_COMPONENT_MSE_LOSS


def test_phase_window_tcn_ablation_configs_diagnostic_batch_then_structure_followup():
    config = load_experiment_config(
        Path("configs/experiment/phase_window_tcn_ablation/phase_window_tcn_ablation.json")
    )
    structure = load_experiment_config(
        Path("configs/experiment/phase_window_tcn_ablation/phase_window_tcn_ablation_structure.json")
    )
    followup = load_experiment_config(
        Path("configs/experiment/phase_window_tcn_ablation/phase_window_tcn_ablation_followup.json")
    )

    result = run(config, dry_run=True)
    structure_result = run(structure, dry_run=True)
    followup_result = run(followup, dry_run=True)

    assert config.seed == structure.seed == followup.seed == 20260615
    assert [run_config["name"] for run_config in config.dl_runs] == [
        "phase_window_tcn_gas_free",
        "phase_window_tcn_gas_varweight",
        "phase_window_tcn_gas_free_varweight",
        "phase_window_tcn_handcraft_mlp",
    ]
    assert [run_config["name"] for run_config in structure.dl_runs] == [
        "phase_window_tcn_gas_free_split",
        "phase_window_tcn_gas_free_deep",
    ]
    assert [run_config["name"] for run_config in followup.dl_runs] == [
        "phase_window_tcn_gas_free_split_deep"
    ]
    baseline, varweight, free_varweight, handcraft = config.dl_runs
    assert baseline["loss"] == FREE_COMPONENT_MSE_LOSS
    assert baseline["model_kwargs"]["output_mode"] == "gas_head"
    assert "share_window_encoder" not in baseline["model_kwargs"]
    assert varweight["loss"] == {"name": WEIGHTED_COMPONENT_MSE_LOSS, "weighting": "inverse_train_var"}
    assert varweight["model_kwargs"]["output_mode"] == "gas_head"
    assert free_varweight["loss"] == {"name": WEIGHTED_FREE_COMPONENT_MSE_LOSS, "weighting": "inverse_train_var"}
    assert handcraft["model"] == "handcraft_mlp"
    assert handcraft["loss"] == {"name": WEIGHTED_COMPONENT_MSE_LOSS, "weighting": "inverse_train_var"}
    split, deep = structure.dl_runs
    assert split["model_kwargs"]["share_window_encoder"] is False
    assert split["model_kwargs"]["tcn_channels"] == [64, 64, 64]
    assert "share_window_encoder" not in deep["model_kwargs"]
    assert deep["model_kwargs"]["tcn_channels"] == [64, 64, 64, 64, 64]
    assert followup.dl_runs[0]["model_kwargs"]["share_window_encoder"] is False
    assert followup.dl_runs[0]["model_kwargs"]["tcn_channels"] == [64, 64, 64, 64, 64]
    assert result["plan"]["dl_runs"] == [
        "phase_window_tcn_gas_free",
        "phase_window_tcn_gas_varweight",
        "phase_window_tcn_gas_free_varweight",
        "phase_window_tcn_handcraft_mlp",
    ]
    assert structure_result["plan"]["dl_runs"] == [
        "phase_window_tcn_gas_free_split",
        "phase_window_tcn_gas_free_deep",
    ]
    assert followup_result["plan"]["dl_runs"] == ["phase_window_tcn_gas_free_split_deep"]


def test_empty_run_lists_disable_that_family(tmp_path: Path):
    payload = _base_config(tmp_path / "dataset", tmp_path / "outputs")
    payload["ml_runs"] = []
    payload["dl_runs"] = []
    config_path = _write_config(tmp_path / "empty_runs_config.json", payload)

    config = load_experiment_config(config_path)

    assert config.ml_runs == ()
    assert config.dl_runs == ()


def test_experiment_config_accepts_window_object(tmp_path: Path):
    payload = _base_config(tmp_path / "dataset", tmp_path / "outputs")
    payload["ml_runs"][0]["window"] = {"kind": "early", "value": 0.5}
    payload["dl_runs"][0]["window"] = {"kind": "phase", "value": "recovery"}
    config_path = _write_config(tmp_path / "window_config.json", payload)

    config = load_experiment_config(config_path)

    assert config.ml_runs[0]["window"] == {"kind": "early", "value": 0.5}
    assert config.dl_runs[0]["window"] == {"kind": "phase", "value": "recovery"}


def test_experiment_config_accepts_ml_windows(tmp_path: Path):
    payload = _base_config(tmp_path / "dataset", tmp_path / "outputs")
    payload["ml_runs"][0].pop("protocol")
    payload["ml_runs"][0]["windows"] = [None, {"kind": "phase", "value": "exposure"}]
    config_path = _write_config(tmp_path / "multiwindow_config.json", payload)

    config = load_experiment_config(config_path)

    assert config.ml_runs[0]["windows"] == [None, {"kind": "phase", "value": "exposure"}]


def test_experiment_config_accepts_dl_phase_windows(tmp_path: Path):
    payload = _base_config(tmp_path / "dataset", tmp_path / "outputs")
    payload["dl_runs"][0]["model"] = "phase_window_tcn"
    payload["dl_runs"][0]["phase_windows"] = [None, {"kind": "phase", "value": "exposure"}]
    config_path = _write_config(tmp_path / "phase_windows_config.json", payload)

    config = load_experiment_config(config_path)
    dry_run = run(config, dry_run=True)

    detail = dry_run["plan"]["dl_run_details"][0]
    assert config.dl_runs[0]["phase_windows"] == [None, {"kind": "phase", "value": "exposure"}]
    assert detail["phase_windows"] == [None, {"kind": "phase", "value": "exposure"}]


def test_experiment_config_rejects_windows_with_protocol(tmp_path: Path):
    payload = _base_config(tmp_path / "dataset", tmp_path / "outputs")
    payload["ml_runs"][0]["windows"] = [None, {"kind": "phase", "value": "exposure"}]
    config_path = _write_config(tmp_path / "bad_multiwindow_config.json", payload)

    with pytest.raises(ValueError, match="cannot combine windows"):
        load_experiment_config(config_path)


def test_experiment_config_rejects_phase_windows_for_ml_or_with_window(tmp_path: Path):
    payload = _base_config(tmp_path / "dataset", tmp_path / "outputs")
    payload["ml_runs"][0]["phase_windows"] = [None, {"kind": "phase", "value": "exposure"}]
    config_path = _write_config(tmp_path / "bad_ml_phase_windows_config.json", payload)

    with pytest.raises(ValueError, match="DL-only"):
        load_experiment_config(config_path)

    payload = _base_config(tmp_path / "dataset", tmp_path / "outputs")
    payload["dl_runs"][0]["phase_windows"] = [None, {"kind": "phase", "value": "exposure"}]
    payload["dl_runs"][0]["window"] = {"kind": "phase", "value": "recovery"}
    config_path = _write_config(tmp_path / "bad_dl_phase_windows_config.json", payload)

    with pytest.raises(ValueError, match="cannot combine phase_windows"):
        load_experiment_config(config_path)


def test_experiment_config_rejects_invalid_window(tmp_path: Path):
    payload = _base_config(tmp_path / "dataset", tmp_path / "outputs")
    payload["ml_runs"][0]["window"] = {"kind": "early", "value": 0.0}
    config_path = _write_config(tmp_path / "bad_window_config.json", payload)

    with pytest.raises(ValueError, match="Invalid ml window"):
        load_experiment_config(config_path)


def test_experiment_config_accepts_target_transform_object(tmp_path: Path):
    payload = _base_config(tmp_path / "dataset", tmp_path / "outputs")
    payload["ml_runs"] = [
        {
            "name": "ridge_alr_custom_eps",
            "model": {"name": "ridge", "alpha": 1.0},
            "modalities": ["slow"],
            "target_transform": {"name": "alr_ch4", "epsilon": 0.0002},
        }
    ]
    payload["dl_runs"] = [
        {
            "name": "fusion_ilr_custom_eps",
            "model": "cnn1d_tcn_fusion",
            "modalities": list(ALL_MODALITIES),
            "target_transform": {"name": "ilr_n2_first", "epsilon": TRAIN_MIN_POSITIVE_HALF_EPSILON},
        }
    ]
    config_path = _write_config(tmp_path / "target_transform_config.json", payload)

    config = load_experiment_config(config_path)

    assert config.ml_runs[0]["target_transform"]["epsilon"] == 0.0002
    assert config.dl_runs[0]["target_transform"]["epsilon"] == TRAIN_MIN_POSITIVE_HALF_EPSILON


def test_experiment_config_rejects_ilr_loss_without_target_transform(tmp_path: Path):
    payload = _base_config(tmp_path / "dataset", tmp_path / "outputs")
    payload["dl_runs"] = [
        {
            "name": "bad_ilr_loss",
            "model": "cnn1d",
            "modalities": ["slow"],
            "loss": "ilr_mse",
        }
    ]
    config_path = _write_config(tmp_path / "bad_loss_config.json", payload)

    with pytest.raises(ValueError, match="ilr_mse requires target_transform"):
        load_experiment_config(config_path)


def test_experiment_config_accepts_configured_training_loss(tmp_path: Path):
    payload = _base_config(tmp_path / "dataset", tmp_path / "outputs")
    payload["training"]["loss"] = {"name": "smooth_l1", "beta": 0.5}
    config_path = _write_config(tmp_path / "configured_training_loss.json", payload)

    config = load_experiment_config(config_path)
    dry_run = run(config, dry_run=True)

    assert config.training["loss"] == {"name": "smooth_l1", "beta": 0.5}
    assert dry_run["plan"]["dl_run_details"][0]["loss"] == {"name": "smooth_l1", "beta": 0.5}


def test_experiment_config_rejects_free_component_loss_without_gas_head(tmp_path: Path):
    payload = _base_config(tmp_path / "dataset", tmp_path / "outputs")
    payload["dl_runs"] = [
        {
            "name": "bad_free_loss",
            "model": "phase_window_tcn",
            "modalities": ["slow"],
            "loss": FREE_COMPONENT_MSE_LOSS,
            "model_kwargs": {"window_count": 1, "output_mode": "raw4"},
        }
    ]
    config_path = _write_config(tmp_path / "bad_free_loss_config.json", payload)

    with pytest.raises(ValueError, match="output_mode='gas_head'"):
        load_experiment_config(config_path)


def test_run_experiment_dry_run_does_not_write_outputs(tmp_path: Path):
    dataset_dir = _make_smoke_dataset(tmp_path)
    output_root = tmp_path / "outputs"
    config_path = _write_config(tmp_path / "config.json", _base_config(dataset_dir, output_root))
    config = load_experiment_config(config_path)

    result = run(config, dry_run=True)

    assert result["dry_run"] is True
    ml_detail = result["plan"]["ml_run_details"][0]
    assert ml_detail["name"] == "mean_slow"
    assert ml_detail["target_transform"]["epsilon"] == TRAIN_MIN_POSITIVE_HALF_EPSILON
    assert result["plan"]["dl_run_details"][0]["target_transform"] is None
    assert result["plan"]["dl_run_details"][0]["loss"] == "mse"
    assert result["plan"]["ml_run_details"][0]["window"] is None
    assert not output_root.exists()


def test_run_experiment_writes_multiwindow_ml_summary(tmp_path: Path):
    dataset_dir = _make_smoke_dataset(tmp_path)
    output_root = tmp_path / "outputs"
    payload = _base_config(dataset_dir, output_root)
    payload["ml_runs"] = [
        {
            "name": "ridge_multiwindow",
            "model": {"name": "ridge", "alpha": 1.0},
            "modalities": ["slow"],
            "sequence_statistics": ["mean"],
            "windows": [None, {"kind": "phase", "value": "exposure"}, {"kind": "phase", "value": "recovery"}],
        }
    ]
    payload["dl_runs"] = []
    config_path = _write_config(tmp_path / "multiwindow_config.json", payload)
    config = load_experiment_config(config_path)

    dry_run = run(config, dry_run=True)
    result = run(config)

    detail = dry_run["plan"]["ml_run_details"][0]
    metrics = json.loads(
        (output_root / "runs" / "smoke_suite" / "ridge_multiwindow" / "metrics.json").read_text(encoding="utf-8")
    )
    summary = Path(result["summary_path"]).read_text(encoding="utf-8")
    assert detail["windows"] == [None, {"kind": "phase", "value": "exposure"}, {"kind": "phase", "value": "recovery"}]
    assert metrics["feature_config"]["feature_windows"] == detail["windows"]
    assert metrics["feature_names"][0].startswith("full|")
    assert "ph_exposure|" in "\n".join(metrics["feature_names"])
    assert "multi:full+exp+rec" in summary
    assert not (output_root / "runs" / "smoke_suite" / "cnn1d_smoke").exists()


def test_run_experiment_writes_phase_window_dl_summary(tmp_path: Path):
    dataset_dir = _make_smoke_dataset(tmp_path)
    output_root = tmp_path / "outputs"
    payload = _base_config(dataset_dir, output_root)
    payload["ml_runs"] = []
    payload["dl_runs"] = [
        {
            "name": "phase_window_tcn_smoke",
            "model": "phase_window_tcn",
            "modalities": ["slow", "ultrasonic", "fiber_mic"],
            "loss": FREE_COMPONENT_MSE_LOSS,
            "phase_windows": [None, {"kind": "phase", "value": "exposure"}, {"kind": "phase", "value": "recovery"}],
            "model_kwargs": {
                "window_count": 3,
                "waveform_embedding_dim": 4,
                "acoustic_channels": [2, 4],
                "slow_hidden_dim": 4,
                "slow_embedding_dim": 4,
                "tcn_channels": [4],
                "shared_hidden_dims": [8, 4],
                "output_mode": "gas_head",
            },
        }
    ]
    config_path = _write_config(tmp_path / "phase_window_dl_config.json", payload)
    config = load_experiment_config(config_path)

    result = run(config)

    run_dir = output_root / "runs" / "smoke_suite" / "phase_window_tcn_smoke"
    run_config = json.loads((run_dir / "run_config.json").read_text(encoding="utf-8"))
    metrics = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
    summary = Path(result["summary_path"]).read_text(encoding="utf-8")
    assert run_config["phase_windows"] == [
        None,
        {"kind": "phase", "value": "exposure"},
        {"kind": "phase", "value": "recovery"},
    ]
    assert run_config["loss"] == FREE_COMPONENT_MSE_LOSS
    assert run_config["model_config"]["output_mode"] == "gas_head"
    assert metrics["loss"] == FREE_COMPONENT_MSE_LOSS
    assert metrics["evaluations"]["val"]["sum_abs_error"] < 1e-4
    assert "phase_window_tcn_smoke" in summary
    assert "multi:full+exp+rec" in summary


def test_run_experiment_writes_handcraft_mlp_summary(tmp_path: Path):
    dataset_dir = _make_smoke_dataset(tmp_path)
    output_root = tmp_path / "outputs"
    payload = _base_config(dataset_dir, output_root)
    payload["ml_runs"] = []
    payload["dl_runs"] = [
        {
            "name": "handcraft_mlp_smoke",
            "model": "handcraft_mlp",
            "modalities": ["slow"],
            "loss": {"name": WEIGHTED_COMPONENT_MSE_LOSS, "weighting": "inverse_train_var"},
            "phase_windows": [None, {"kind": "phase", "value": "exposure"}, {"kind": "phase", "value": "recovery"}],
            "model_kwargs": {"hidden_dims": [8, 4], "dropout": 0.0},
        }
    ]
    config_path = _write_config(tmp_path / "handcraft_mlp_config.json", payload)
    config = load_experiment_config(config_path)

    result = run(config)

    run_dir = output_root / "runs" / "smoke_suite" / "handcraft_mlp_smoke"
    run_config = json.loads((run_dir / "run_config.json").read_text(encoding="utf-8"))
    metrics = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
    summary = Path(result["summary_path"]).read_text(encoding="utf-8")
    assert run_config["input_format"] == "FEATURES"
    assert run_config["model_config"]["name"] == "handcraft_mlp"
    assert run_config["loss"] == {"name": WEIGHTED_COMPONENT_MSE_LOSS, "weighting": "inverse_train_var"}
    assert metrics["evaluations"]["val"]["sum_abs_error"] < 1e-4
    assert "handcraft_mlp_smoke" in summary
    assert "multi:full+exp+rec" in summary


def test_run_experiment_writes_runs_summary_report_and_progress_logs(tmp_path: Path, capsys):
    dataset_dir = _make_smoke_dataset(tmp_path)
    output_root = tmp_path / "outputs"
    config_path = _write_config(tmp_path / "config.json", _base_config(dataset_dir, output_root))
    config = load_experiment_config(config_path)

    result = run(config)

    output = capsys.readouterr().out
    assert Path(result["summary_path"]).is_file()
    assert Path(result["report_path"]).is_file()
    assert (output_root / "runs" / "smoke_suite" / "mean_slow" / "metrics.json").is_file()
    ml_run_config = json.loads(
        (output_root / "runs" / "smoke_suite" / "mean_slow" / "run_config.json").read_text(encoding="utf-8")
    )
    assert (output_root / "runs" / "smoke_suite" / "cnn1d_smoke" / "metrics.json").is_file()
    assert (output_root / "runs" / "smoke_suite" / "cnn1d_smoke" / "metrics_live.jsonl").is_file()
    assert "[run start] kind=ml name=mean_slow" in output
    assert "[run done] kind=ml name=mean_slow" in output
    assert "[run start] kind=dl name=cnn1d_smoke" in output
    assert "[run done] kind=dl name=cnn1d_smoke" in output
    assert "[epoch] model=cnn1d epoch=1/1" in output
    summary = Path(result["summary_path"]).read_text(encoding="utf-8")
    report = Path(result["report_path"]).read_text(encoding="utf-8")
    assert "mean_slow" in summary
    assert "cnn1d_smoke" in summary
    assert "x_n2_r2" in summary
    assert "aitchison_mean" in summary
    assert "window" in summary
    assert "full" in summary
    assert ml_run_config["target_transform"]["epsilon"] == TRAIN_MIN_POSITIVE_HALF_EPSILON
    assert isinstance(ml_run_config["resolved_target_transform"]["epsilon"], float)
    assert "# Experiment Report: smoke_suite" in report
    assert "x_N2 R2" in report
    assert "Aitchison mean" in report
    assert "| kind | run | model | window | split |" in report


def test_run_experiment_stops_on_failed_run(tmp_path: Path):
    dataset_dir = _make_smoke_dataset(tmp_path)
    output_root = tmp_path / "outputs"
    payload = _base_config(dataset_dir, output_root)
    payload["dl_runs"] = [
        {"name": "bad_fusion", "model": "cnn1d_tcn_fusion", "modalities": ["slow"], "model_kwargs": {}},
        {"name": "should_not_run", "model": "cnn1d", "modalities": ["slow"], "model_kwargs": {}},
    ]
    config_path = _write_config(tmp_path / "config.json", payload)
    config = load_experiment_config(config_path)

    with pytest.raises(ValueError, match="does not match"):
        run(config)

    assert not (output_root / "runs" / "smoke_suite" / "should_not_run").exists()
