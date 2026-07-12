from __future__ import annotations

import json
from pathlib import Path

import pytest

from tv3.pipeline.audit_d2b_frame_fidelity import (
    OUTPUT_FILENAMES,
    audit_d2b_frame_fidelity,
)
from tv3.pipeline.build_tv3_raw_dsp_features import (
    build_tv3_raw_dsp_feature_cache,
    preflight_tv3_raw_dsp_dataset,
)


def _make_tv3_dataset(tmp_path: Path, slug: str) -> Path:
    from tv3.sim.generation.tunnel_ventilation import (
        TunnelVentilationBenchmarkGenerationSpec,
        generate_tunnel_ventilation_benchmark_dataset,
    )

    generate_tunnel_ventilation_benchmark_dataset(
        tmp_path,
        TunnelVentilationBenchmarkGenerationSpec(
            dataset_slug=slug,
            sequence_count=16,
            seed=20260711,
            timesteps=16,
            storage="npz",
            optical_absorption_backend="empirical_v1",
            workers=1,
        ),
    )
    return tmp_path / slug


def _build_formal_cache(dataset_dir: Path, cache_dir: Path) -> None:
    build_tv3_raw_dsp_feature_cache(
        preflight_tv3_raw_dsp_dataset(dataset_dir),
        cache_dir=cache_dir,
        template_mode="train_baseline_median",
        template_max_frames=32,
        workers=1,
    )


def test_audit_writes_split_phase_quality_and_manifest_evidence(tmp_path: Path):
    dataset_dir = _make_tv3_dataset(tmp_path, "d2b-fidelity")
    cache_dir = tmp_path / "raw_dsp_cache"
    output_dir = tmp_path / "audit"
    _build_formal_cache(dataset_dir, cache_dir)

    result = audit_d2b_frame_fidelity(
        dataset_dir=dataset_dir,
        cache_dir=cache_dir,
        output_dir=output_dir,
    )

    assert result["schema_version"] == "tv3-d2b-frame-fidelity-1"
    assert result["primary_population"] == "all_frames"
    assert set(result["splits"]) == {"train", "val", "test", "extrapolation"}
    assert all((output_dir / filename).is_file() for filename in OUTPUT_FILENAMES)

    metrics = json.loads((output_dir / "metrics.json").read_text(encoding="utf-8"))
    assert metrics["source"]["template_mode"] == "train_baseline_median"
    assert metrics["splits"]["val"]["gate"]["required"] is True

    phase_rows = (output_dir / "peak_error_by_phase.csv").read_text(encoding="utf-8")
    assert "baseline" in phase_rows
    assert "steady" in phase_rows


def test_audit_rejects_debug_template_cache(tmp_path: Path):
    dataset_dir = _make_tv3_dataset(tmp_path, "d2b-debug-cache")
    cache_dir = tmp_path / "debug_cache"
    build_tv3_raw_dsp_feature_cache(
        preflight_tv3_raw_dsp_dataset(dataset_dir),
        cache_dir=cache_dir,
        template_mode="exact_simulator_debug",
        workers=1,
    )

    with pytest.raises(ValueError, match="template_mode"):
        audit_d2b_frame_fidelity(
            dataset_dir=dataset_dir,
            cache_dir=cache_dir,
            output_dir=tmp_path / "audit",
        )


def test_audit_refuses_to_overwrite_existing_evidence(tmp_path: Path):
    dataset_dir = _make_tv3_dataset(tmp_path, "d2b-overwrite")
    cache_dir = tmp_path / "raw_dsp_cache"
    output_dir = tmp_path / "audit"
    _build_formal_cache(dataset_dir, cache_dir)
    audit_d2b_frame_fidelity(dataset_dir=dataset_dir, cache_dir=cache_dir, output_dir=output_dir)

    with pytest.raises(FileExistsError, match="audit output already exists"):
        audit_d2b_frame_fidelity(dataset_dir=dataset_dir, cache_dir=cache_dir, output_dir=output_dir)
