from __future__ import annotations

import json
from pathlib import Path

import pytest

from pipeline.experiment_config import load_experiment_config
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


def test_run_experiment_dry_run_does_not_write_outputs(tmp_path: Path):
    dataset_dir = _make_smoke_dataset(tmp_path)
    output_root = tmp_path / "outputs"
    config_path = _write_config(tmp_path / "config.json", _base_config(dataset_dir, output_root))
    config = load_experiment_config(config_path)

    result = run(config, dry_run=True)

    assert result["dry_run"] is True
    assert not output_root.exists()


def test_run_experiment_writes_runs_summary_and_report(tmp_path: Path):
    dataset_dir = _make_smoke_dataset(tmp_path)
    output_root = tmp_path / "outputs"
    config_path = _write_config(tmp_path / "config.json", _base_config(dataset_dir, output_root))
    config = load_experiment_config(config_path)

    result = run(config)

    assert Path(result["summary_path"]).is_file()
    assert Path(result["report_path"]).is_file()
    assert (output_root / "runs" / "smoke_suite" / "mean_slow" / "metrics.json").is_file()
    assert (output_root / "runs" / "smoke_suite" / "cnn1d_smoke" / "metrics.json").is_file()
    summary = Path(result["summary_path"]).read_text(encoding="utf-8")
    report = Path(result["report_path"]).read_text(encoding="utf-8")
    assert "mean_slow" in summary
    assert "cnn1d_smoke" in summary
    assert "# Experiment Report: smoke_suite" in report


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
