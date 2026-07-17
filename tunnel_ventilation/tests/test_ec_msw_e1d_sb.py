from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from tv3.dl.evaluation.ec_msw_e1d_diagnosis import build_e1d_feature_matrix, default_e1d_specs
from tv3.dl.evaluation.ec_msw_e1d_sb_audit import (
    _build_verdict,
    _validate_config,
    run_ec_msw_e1d_sb_audit,
)
from tv3.ml.e1d_sb_features import (
    E1DSB_FEATURE_BUILDER,
    E1DSB_FRAME_ARRAYS,
    E1DSB_SEQUENCE_SCALARS,
    E1DSB_SPEC_NAME,
    build_e1d_sb_feature_matrix,
    diagnostic_feature_count,
    e1d_sb_builder_info,
)
from tv3.pipeline.build_tv3_raw_dsp_features import (
    build_tv3_raw_dsp_feature_cache,
    preflight_tv3_raw_dsp_dataset,
)
from tv3.sim.generation.tunnel_ventilation import (
    TunnelVentilationBenchmarkGenerationSpec,
    generate_tunnel_ventilation_benchmark_dataset,
)


def _make_dataset(tmp_path: Path, *, sequences: int = 16) -> Path:
    generate_tunnel_ventilation_benchmark_dataset(
        tmp_path,
        TunnelVentilationBenchmarkGenerationSpec(
            dataset_slug="tv3-e1d-sb-smoke",
            sequence_count=sequences,
            seed=20260717,
            timesteps=16,
            storage="npz",
            optical_absorption_backend="empirical_v1",
            workers=1,
        ),
    )
    dataset_dir = tmp_path / "tv3-e1d-sb-smoke"
    preflight = preflight_tv3_raw_dsp_dataset(dataset_dir)
    build_tv3_raw_dsp_feature_cache(
        preflight,
        cache_dir=dataset_dir / "features" / "raw_dsp" / "raw_dsp_frame_v1",
        template_mode="train_baseline_median",
        workers=1,
    )
    return dataset_dir


@pytest.fixture(scope="module")
def e1d_sb_dataset(tmp_path_factory: pytest.TempPathFactory) -> Path:
    return _make_dataset(tmp_path_factory.mktemp("e1d_sb_dataset"))


class TestE1dSBBuilder:
    def test_builder_info_matches_compact_recipe(self):
        info = e1d_sb_builder_info()
        assert info.feature_builder == E1DSB_FEATURE_BUILDER
        assert info.spec_name == E1DSB_SPEC_NAME
        assert info.frame_arrays == E1DSB_FRAME_ARRAYS
        assert info.sequence_scalars == E1DSB_SEQUENCE_SCALARS
        assert info.physics_early_fractions == ()
        assert "snr" in "".join(info.frame_arrays)

    def test_cache_matrix_bit_identical_to_e1d_spec(self, e1d_sb_dataset: Path):
        e1d_spec = next(item for item in default_e1d_specs() if item.name == E1DSB_SPEC_NAME)
        expected = build_e1d_feature_matrix(e1d_sb_dataset, split="train", spec=e1d_spec)
        actual = build_e1d_sb_feature_matrix(
            e1d_sb_dataset,
            split="train",
            feature_source="raw_dsp_cache",
        )
        assert actual.feature_names == expected.feature_names
        assert actual.x.shape == expected.x.shape
        np.testing.assert_allclose(actual.x, expected.x, rtol=0.0, atol=0.0)
        assert diagnostic_feature_count(actual.feature_names) == 213
        assert diagnostic_feature_count(actual.feature_names) <= 252

    def test_waveform_matrix_matches_cache(self, e1d_sb_dataset: Path):
        cache = build_e1d_sb_feature_matrix(
            e1d_sb_dataset,
            split="val",
            feature_source="raw_dsp_cache",
        )
        waveform = build_e1d_sb_feature_matrix(
            e1d_sb_dataset,
            split="val",
            feature_source="waveform",
        )
        assert waveform.feature_names == cache.feature_names
        np.testing.assert_allclose(waveform.x, cache.x, rtol=0.0, atol=1e-5)


class TestE1dSBVerdict:
    def test_smoke_verdict(self):
        verdict = _build_verdict(
            run_kind="smoke",
            has_reference=False,
            parity_passed=False,
            compact=True,
            diagnostic_feature_count=213,
        )
        assert verdict["status"] == "smoke_only"
        assert verdict["e2_allowed"] is False
        assert verdict["continue_e1r_attachment"] is False

    def test_parity_pass_verdict(self):
        verdict = _build_verdict(
            run_kind="formal",
            has_reference=True,
            parity_passed=True,
            compact=True,
            diagnostic_feature_count=213,
        )
        assert verdict["status"] == "parity_passed"
        assert verdict["continue_e1r_attachment"] is True
        assert verdict["e2_allowed"] is False

    def test_parity_fail_verdict(self):
        verdict = _build_verdict(
            run_kind="formal",
            has_reference=True,
            parity_passed=False,
            compact=True,
            diagnostic_feature_count=213,
        )
        assert verdict["status"] == "parity_failed"
        assert verdict["continue_e1r_attachment"] is False

    def test_formal_requires_b1_reference(self):
        with pytest.raises(ValueError, match="b1_reference_metrics"):
            _validate_config(
                {
                    "dataset_dir": "data/example",
                    "output_dir": "outputs/example",
                    "run_kind": "formal",
                    "ridge_alphas": [0.1],
                }
            )


def test_run_e1d_sb_smoke_writes_artifacts(e1d_sb_dataset: Path, tmp_path: Path):
    config_path = tmp_path / "e1d_sb_config.json"
    output_dir = tmp_path / "e1d_sb_out"
    config_path.write_text(
        json.dumps(
            {
                "dataset_dir": str(e1d_sb_dataset),
                "output_dir": str(output_dir),
                "run_kind": "smoke",
                "feature_source": "raw_dsp_cache",
                "ridge_alphas": [0.01, 0.1, 1.0, 10.0],
            }
        ),
        encoding="utf-8",
    )

    written = run_ec_msw_e1d_sb_audit(config_path, project_root=tmp_path)
    assert written == output_dir
    for name in (
        "manifest.json",
        "feature_sets.json",
        "summary.json",
        "verdict.json",
        "ablation_table.csv",
        "narrow_o2_windows.csv",
    ):
        assert (output_dir / name).is_file()

    verdict = json.loads((output_dir / "verdict.json").read_text(encoding="utf-8"))
    assert verdict["status"] == "smoke_only"
    assert verdict["e2_allowed"] is False
    assert verdict["feature_builder"] == E1DSB_FEATURE_BUILDER

    summary = json.loads((output_dir / "summary.json").read_text(encoding="utf-8"))
    assert summary["feature_builder"] == E1DSB_FEATURE_BUILDER
    assert summary["diagnostic_feature_count"] == 213
    assert summary["compact"] is True
    for split in ("val", "test", "extrapolation"):
        assert split in summary["eval"]
