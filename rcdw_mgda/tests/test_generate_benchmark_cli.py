"""测试 scripts.generate_benchmark 的 cache-only CLI 分支。"""
from __future__ import annotations

from pathlib import Path

import yaml

from rcdw.sim.generation.conditions import generate_condition_rows
from scripts import generate_benchmark


def _write_config(path: Path, *, backend: str = "hitran_hapi_v1") -> None:
    payload = {
        "data": {
            "dataset_root": "data/rcdw-cli-test",
            "train_modalities": ["slow", "ultrasonic"],
        },
        "generation": {
            "sequence_count": 4,
            "seed": 123,
            "timesteps": 8,
            "dt_s": 0.5,
            "storage": "memmap",
            "multi_path_phase": "steady",
            "stage_profile": "standard_exposure",
            "stage_jitter": 0.0,
            "sampling_strategy": "random",
            "path_lms": [0.2, 0.3],
            "optical_absorption_backend": backend,
            "hitran_cache_root": str(path.parent / "hitran_cache"),
        },
    }
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")


def test_precompute_only_does_not_create_benchmark_dir(
    tmp_path: Path, monkeypatch
) -> None:
    config_path = tmp_path / "config.yaml"
    output_root = tmp_path / "out"
    _write_config(config_path)

    captured: dict[str, object] = {}

    def fake_precompute_hitran_benchmark_cache(conditions, *, cache_root):
        captured["conditions"] = conditions
        captured["cache_root"] = cache_root
        return {"total": 0, "cached": 0, "filled": 0, "cache_root": str(cache_root)}

    def fail_generate_benchmark_dataset(output_root_arg, spec):
        raise AssertionError("generate_benchmark_dataset must not be called")

    monkeypatch.setattr(
        generate_benchmark,
        "precompute_hitran_benchmark_cache",
        fake_precompute_hitran_benchmark_cache,
    )
    monkeypatch.setattr(
        generate_benchmark,
        "generate_benchmark_dataset",
        fail_generate_benchmark_dataset,
    )
    monkeypatch.setattr(
        "sys.argv",
        [
            "generate_benchmark.py",
            "--config",
            str(config_path),
            "--dataset-slug",
            "rcdw-cli-test",
            "--output-root",
            str(output_root),
            "--precompute-cache-only",
        ],
    )

    generate_benchmark.main()

    assert not (output_root / "rcdw-cli-test").exists()
    assert captured["conditions"] == generate_condition_rows(
        4, seed=123, sampling_strategy="random"
    )


def test_precompute_only_empirical_backend_noop(tmp_path: Path, monkeypatch) -> None:
    config_path = tmp_path / "config.yaml"
    output_root = tmp_path / "out"
    _write_config(config_path, backend="empirical_v1")

    def fail_precompute_hitran_benchmark_cache(conditions, *, cache_root):
        raise AssertionError("precompute_hitran_benchmark_cache must not be called")

    def fail_generate_benchmark_dataset(output_root_arg, spec):
        raise AssertionError("generate_benchmark_dataset must not be called")

    monkeypatch.setattr(
        generate_benchmark,
        "precompute_hitran_benchmark_cache",
        fail_precompute_hitran_benchmark_cache,
    )
    monkeypatch.setattr(
        generate_benchmark,
        "generate_benchmark_dataset",
        fail_generate_benchmark_dataset,
    )
    monkeypatch.setattr(
        "sys.argv",
        [
            "generate_benchmark.py",
            "--config",
            str(config_path),
            "--dataset-slug",
            "rcdw-cli-test",
            "--output-root",
            str(output_root),
            "--precompute-cache-only",
        ],
    )

    generate_benchmark.main()

    assert not (output_root / "rcdw-cli-test").exists()
