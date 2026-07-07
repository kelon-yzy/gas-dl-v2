from __future__ import annotations

import json
from pathlib import Path

from pipeline.run_tv3_rocket_baseline import main


def _make_tv3_smoke_dataset(tmp_path: Path, slug: str = "tv3-rocket-cli-smoke", sequences: int = 16) -> Path:
    from sim.generation.tunnel_ventilation import (
        TunnelVentilationBenchmarkGenerationSpec,
        generate_tunnel_ventilation_benchmark_dataset,
    )

    generate_tunnel_ventilation_benchmark_dataset(
        tmp_path,
        TunnelVentilationBenchmarkGenerationSpec(
            dataset_slug=slug,
            sequence_count=sequences,
            seed=20260706,
            timesteps=16,
            storage="npz",
            optical_absorption_backend="empirical_v1",
            workers=1,
        ),
    )
    return tmp_path / slug


def test_run_tv3_rocket_baseline_writes_metrics_json(tmp_path: Path, capsys):
    dataset_dir = _make_tv3_smoke_dataset(tmp_path)
    output_dir = tmp_path / "outputs" / "tv3_rocket_smoke"

    exit_code = main(
        [
            "--dataset-dir",
            str(dataset_dir),
            "--feature-set",
            "physics_stats",
            "--head",
            "ridgecv",
            "--output-dir",
            str(output_dir),
        ]
    )

    payload = json.loads(capsys.readouterr().out)
    metrics_path = output_dir / "metrics.json"

    assert exit_code == 0
    assert metrics_path.is_file()
    assert payload["head"] == "ridgecv"
    assert payload["feature_builder"] == "physics_stats_v1"
    assert set(payload["evaluations"]) == {"train", "val", "test", "extrapolation"}
    assert payload["evaluations"]["val"]["sequence_count"] > 0
    assert payload["diagnostics"]["selected_alpha"] > 0.0


def test_run_tv3_rocket_baseline_reads_json_config_with_lists(tmp_path: Path, capsys):
    dataset_dir = _make_tv3_smoke_dataset(tmp_path, slug="tv3-rocket-config-smoke")
    output_dir = tmp_path / "outputs" / "tv3_rocket_config"
    config_path = tmp_path / "tv3_rocket_config.json"
    config_path.write_text(
        json.dumps(
            {
                "dataset_dir": str(dataset_dir),
                "output_dir": str(output_dir),
                "feature_set": "physics_stats",
                "head": "ridgecv",
                "physics_arrays": ["ultrasonic_tof_s", "ultrasonic_sound_speed_estimated_m_per_s"],
                "sequence_statistics": ["mean", "std", "delta"],
                "phase_windows": ["baseline", "steady"],
                "early_fractions": [0.5],
                "eval_splits": ["val", "test"],
                "ridge_alphas": [0.001, 0.01, 0.1],
            }
        ),
        encoding="utf-8",
    )

    exit_code = main(["--config", str(config_path)])
    payload = json.loads(capsys.readouterr().out)

    assert exit_code == 0
    assert payload["feature_count"] > 0
    assert set(payload["evaluations"]) == {"train", "val", "test"}
