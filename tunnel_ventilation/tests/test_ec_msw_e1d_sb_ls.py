from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from tv3.dl.evaluation.ec_msw_e1d_sb_ls_audit import (
    _assert_safe_output_dir,
    _build_verdict,
    _validate_config,
    run_ec_msw_e1d_sb_ls_audit,
)
from tv3.ml.e1d_sb_features import (
    E1DSB_FEATURE_BUILDER,
    E1DSB_FRAME_ARRAYS,
    E1DSB_LS_EXTRA_SCALARS,
    E1DSB_LS_FEATURE_BUILDER,
    E1DSB_LS_SPEC_NAME,
    build_e1d_sb_feature_matrix,
    build_e1d_sb_ls_feature_matrix,
    diagnostic_feature_count,
    e1d_sb_ls_builder_info,
)
from tv3.ml.raw_dsp_features import fit_tof_vs_path_length_snr_weighted
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
            dataset_slug="tv3-e1d-sb-ls-smoke",
            sequence_count=sequences,
            seed=20260717,
            timesteps=16,
            storage="npz",
            optical_absorption_backend="empirical_v1",
            workers=1,
        ),
    )
    dataset_dir = tmp_path / "tv3-e1d-sb-ls-smoke"
    preflight = preflight_tv3_raw_dsp_dataset(dataset_dir)
    build_tv3_raw_dsp_feature_cache(
        preflight,
        cache_dir=dataset_dir / "features" / "raw_dsp" / "raw_dsp_frame_v1",
        template_mode="train_baseline_median",
        workers=1,
    )
    return dataset_dir


@pytest.fixture(scope="module")
def e1d_sb_ls_dataset(tmp_path_factory: pytest.TempPathFactory) -> Path:
    return _make_dataset(tmp_path_factory.mktemp("e1d_sb_ls_dataset"))


class TestSnrWeightedLsFit:
    def test_recovers_known_sound_speed(self):
        path_lengths = np.linspace(0.20, 0.30, 8)
        true_c = 345.0
        delay = 8.0e-5
        tof = path_lengths / true_c + delay
        snr = np.full(path_lengths.shape, 30.0)
        phase_ids = ["steady"] * path_lengths.size
        intercept, speed = fit_tof_vs_path_length_snr_weighted(
            tof, path_lengths, snr, phase_ids, weight_mode="amplitude"
        )
        assert intercept == pytest.approx(delay, rel=0.0, abs=1e-12)
        assert speed == pytest.approx(true_c, rel=0.0, abs=1e-9)

    def test_rejects_bad_weight_mode(self):
        with pytest.raises(ValueError, match="weight_mode"):
            fit_tof_vs_path_length_snr_weighted(
                np.array([1.0, 2.0]),
                np.array([0.2, 0.3]),
                np.array([20.0, 20.0]),
                ["steady", "steady"],
                weight_mode="invalid",
            )


class TestE1dSBLSBuilder:
    def test_builder_info_keeps_snr_and_adds_ls(self):
        info = e1d_sb_ls_builder_info()
        assert info.feature_builder == E1DSB_LS_FEATURE_BUILDER
        assert info.spec_name == E1DSB_LS_SPEC_NAME
        assert info.frame_arrays == E1DSB_FRAME_ARRAYS
        assert "ultrasonic_snr_db" in info.frame_arrays
        for name in E1DSB_LS_EXTRA_SCALARS:
            assert name in info.sequence_scalars

    def test_ls_matrix_contains_snr_and_ls_scalars(self, e1d_sb_ls_dataset: Path):
        base = build_e1d_sb_feature_matrix(
            e1d_sb_ls_dataset, split="train", feature_source="raw_dsp_cache"
        )
        ls = build_e1d_sb_ls_feature_matrix(
            e1d_sb_ls_dataset, split="train", feature_source="raw_dsp_cache"
        )
        assert any("ultrasonic_snr_db" in name for name in ls.feature_names)
        for name in E1DSB_LS_EXTRA_SCALARS:
            assert f"seq|{name}" in ls.feature_names
        assert ls.x.shape[1] == base.x.shape[1] + 2
        assert diagnostic_feature_count(ls.feature_names) == diagnostic_feature_count(
            base.feature_names
        ) + 2
        # Base columns remain identical; LS only appends.
        np.testing.assert_allclose(ls.x[:, : base.x.shape[1]], base.x, rtol=0.0, atol=0.0)

    def test_waveform_matches_cache(self, e1d_sb_ls_dataset: Path):
        cache = build_e1d_sb_ls_feature_matrix(
            e1d_sb_ls_dataset, split="val", feature_source="raw_dsp_cache"
        )
        waveform = build_e1d_sb_ls_feature_matrix(
            e1d_sb_ls_dataset, split="val", feature_source="waveform"
        )
        assert waveform.feature_names == cache.feature_names
        np.testing.assert_allclose(waveform.x, cache.x, rtol=0.0, atol=1e-5)


class TestE1dSBLSAudit:
    def test_formal_config_requires_attachment_and_flag(self):
        _validate_config(
            {
                "dataset_dir": "data/x",
                "output_dir": "outputs/e1d_sb_ls_s1",
                "ridge_alphas": [0.1],
                "feature_builder": E1DSB_LS_FEATURE_BUILDER,
                "include_snr_weighted_ls": True,
                "run_kind": "formal",
                "b1_reference_metrics": "outputs/b1.json",
                "attachment_verdict_path": "outputs/attach/verdict.json",
                "baseline_e1d_sb_summary": "outputs/e1d_sb/summary.json",
            }
        )
        with pytest.raises(ValueError, match="baseline_e1d_sb_summary"):
            _validate_config(
                {
                    "dataset_dir": "data/x",
                    "output_dir": "outputs/e1d_sb_ls_s1",
                    "ridge_alphas": [0.1],
                    "feature_builder": E1DSB_LS_FEATURE_BUILDER,
                    "include_snr_weighted_ls": True,
                    "run_kind": "formal",
                    "b1_reference_metrics": "outputs/b1.json",
                    "attachment_verdict_path": "outputs/attach/verdict.json",
                }
            )
        with pytest.raises(ValueError, match="include_snr_weighted_ls"):
            _validate_config(
                {
                    "dataset_dir": "data/x",
                    "output_dir": "outputs/e1d_sb_ls_s1",
                    "ridge_alphas": [0.1],
                    "feature_builder": E1DSB_LS_FEATURE_BUILDER,
                    "include_snr_weighted_ls": False,
                    "run_kind": "smoke",
                }
            )
        with pytest.raises(ValueError, match="feature_builder"):
            _validate_config(
                {
                    "dataset_dir": "data/x",
                    "output_dir": "outputs/e1d_sb_ls_s1",
                    "ridge_alphas": [0.1],
                    "feature_builder": E1DSB_FEATURE_BUILDER,
                    "include_snr_weighted_ls": True,
                    "run_kind": "smoke",
                }
            )

    def test_refuses_frozen_output_names(self):
        with pytest.raises(ValueError, match="collides|refusing"):
            _assert_safe_output_dir(Path("outputs/tv3_ec_msw/e1d_sb_s20260704"))
        _assert_safe_output_dir(Path("outputs/tv3_ec_msw/e1d_sb_ls_s20260704"))

    def test_verdict_blocks_without_attachment(self):
        verdict = _build_verdict(
            run_kind="formal",
            attachment_gate={"status": "b1_parity_failed"},
            has_reference=True,
            parity_passed=True,
            compact=True,
            diagnostic_feature_count=215,
        )
        assert verdict["status"] == "attachment_gate_failed"
        assert verdict["e2_allowed"] is False
        assert verdict["snr_retained"] is True

    def test_verdict_blocks_wrong_attachment_builder(self):
        verdict = _build_verdict(
            run_kind="formal",
            attachment_gate={
                "status": "attachment_passed",
                "feature_builder": "wrong_builder",
                "e2_allowed": False,
                "frame_fidelity_passed": True,
                "sequence_parity_passed": True,
            },
            has_reference=True,
            parity_passed=True,
            compact=True,
            diagnostic_feature_count=215,
        )
        assert verdict["status"] == "attachment_gate_failed"
        assert "feature_builder" in verdict["reason"]

    def test_delta_vs_baseline_requires_complete_eval(self):
        from tv3.dl.evaluation.ec_msw_e1d_sb_ls_audit import _delta_vs_baseline

        split_payload = {
            "val": {
                "component_metrics": {
                    "x_O2": {"r2": 0.4},
                    "x_CO2": {"r2": 0.9},
                    "x_N2": {"r2": 0.8},
                }
            },
            "test": {
                "component_metrics": {
                    "x_O2": {"r2": 0.41},
                    "x_CO2": {"r2": 0.91},
                    "x_N2": {"r2": 0.81},
                }
            },
            "extrapolation": {
                "component_metrics": {
                    "x_O2": {"r2": 0.39},
                    "x_CO2": {"r2": 0.89},
                    "x_N2": {"r2": 0.79},
                }
            },
        }
        with pytest.raises(ValueError, match="missing required splits"):
            _delta_vs_baseline(
                split_payload,
                {"eval": {"val": {"x_O2_r2": 0.39, "x_CO2_r2": 0.9, "x_N2_r2": 0.8}}},
                ("val", "test", "extrapolation"),
                require=True,
            )
        delta = _delta_vs_baseline(
            split_payload,
            {
                "eval": {
                    "val": {"x_O2_r2": 0.39, "x_CO2_r2": 0.9, "x_N2_r2": 0.8},
                    "test": {"x_O2_r2": 0.40, "x_CO2_r2": 0.90, "x_N2_r2": 0.80},
                    "extrapolation": {"x_O2_r2": 0.38, "x_CO2_r2": 0.88, "x_N2_r2": 0.78},
                }
            },
            ("val", "test", "extrapolation"),
            require=True,
        )
        assert delta is not None
        assert "val" in delta and "test" in delta and "extrapolation" in delta

    def test_cache_builder_requires_accepted_array(self, e1d_sb_ls_dataset: Path, tmp_path: Path):
        import shutil

        from tv3.ml.e1d_sb_features import build_e1d_sb_ls_feature_matrix

        dataset_copy = tmp_path / "dataset_no_accepted"
        shutil.copytree(e1d_sb_ls_dataset, dataset_copy)
        accepted = (
            dataset_copy
            / "features"
            / "raw_dsp"
            / "raw_dsp_frame_v1"
            / "ultrasonic_raw_dsp_accepted.npy"
        )
        assert accepted.is_file()
        accepted.unlink()
        with pytest.raises(FileNotFoundError, match="accepted"):
            build_e1d_sb_ls_feature_matrix(
                dataset_copy, split="train", feature_source="raw_dsp_cache"
            )

    def test_smoke_audit_writes_artifacts(self, e1d_sb_ls_dataset: Path, tmp_path: Path):
        config_path = tmp_path / "e1d_sb_ls_smoke.json"
        output_dir = tmp_path / "e1d_sb_ls_smoke_out"
        config_path.write_text(
            json.dumps(
                {
                    "dataset_dir": str(e1d_sb_ls_dataset),
                    "output_dir": str(output_dir),
                    "run_kind": "smoke",
                    "feature_source": "raw_dsp_cache",
                    "feature_builder": E1DSB_LS_FEATURE_BUILDER,
                    "include_snr_weighted_ls": True,
                    "snr_ls_weight_mode": "amplitude",
                    "ridge_alphas": [0.1, 1.0],
                    "eval_splits": ["val", "test", "extrapolation"],
                }
            ),
            encoding="utf-8",
        )
        wrote = run_ec_msw_e1d_sb_ls_audit(config_path, project_root=tmp_path)
        assert wrote == output_dir
        verdict = json.loads((output_dir / "verdict.json").read_text(encoding="utf-8"))
        assert verdict["status"] == "smoke_only"
        assert verdict["e2_allowed"] is False
        assert verdict["snr_retained"] is True
        manifest = json.loads((output_dir / "manifest.json").read_text(encoding="utf-8"))
        assert manifest["feature_builder"] == E1DSB_LS_FEATURE_BUILDER
        assert "ultrasonic_snr_db" in manifest["builder"]["frame_arrays"]
        assert any("snr_weighted_ls" in name for name in manifest["builder"]["sequence_scalars"])
