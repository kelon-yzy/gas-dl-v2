from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from tv3.dl.evaluation.ec_msw_e1d_sb_deploy_probe import (
    _assert_safe_output_dir,
    _build_verdict,
    _validate_config,
    run_ec_msw_e1d_sb_deploy_probe,
)
from tv3.ml.e1d_sb_features import E1DSB_FEATURE_BUILDER, build_e1d_sb_feature_matrix
from tv3.ml.e1d_sb_inference import (
    fit_e1d_sb_inference,
    load_inference_artifact,
    predict_with_artifact,
    write_inference_artifact,
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
            dataset_slug="tv3-e1d-sb-deploy-smoke",
            sequence_count=sequences,
            seed=20260717,
            timesteps=16,
            storage="npz",
            optical_absorption_backend="empirical_v1",
            workers=1,
        ),
    )
    dataset_dir = tmp_path / "tv3-e1d-sb-deploy-smoke"
    preflight = preflight_tv3_raw_dsp_dataset(dataset_dir)
    build_tv3_raw_dsp_feature_cache(
        preflight,
        cache_dir=dataset_dir / "features" / "raw_dsp" / "raw_dsp_frame_v1",
        template_mode="train_baseline_median",
        workers=1,
    )
    return dataset_dir


@pytest.fixture(scope="module")
def deploy_probe_dataset(tmp_path_factory: pytest.TempPathFactory) -> Path:
    return _make_dataset(tmp_path_factory.mktemp("e1d_sb_deploy_dataset"))


class TestE1dSBInference:
    def test_fit_predict_roundtrip(self, deploy_probe_dataset: Path, tmp_path: Path):
        train = build_e1d_sb_feature_matrix(
            deploy_probe_dataset, split="train", feature_source="raw_dsp_cache"
        )
        probe, artifact = fit_e1d_sb_inference(train, alphas=(0.1, 1.0, 10.0))
        assert artifact.feature_builder == E1DSB_FEATURE_BUILDER
        assert artifact.ls_promoted is False
        assert artifact.e2_allowed is False
        assert artifact.default_head_remains == "B7"

        live = probe.predict(train.x)
        frozen = predict_with_artifact(artifact, train.x)
        np.testing.assert_allclose(frozen, live, rtol=0.0, atol=1e-5)

        path = write_inference_artifact(tmp_path / "artifact.json", artifact)
        loaded = load_inference_artifact(path)
        reloaded = predict_with_artifact(loaded, train.x)
        np.testing.assert_allclose(reloaded, live, rtol=0.0, atol=1e-5)

    def test_rejects_ls_features(self, deploy_probe_dataset: Path):
        from tv3.ml.e1d_sb_features import build_e1d_sb_ls_feature_matrix

        train = build_e1d_sb_ls_feature_matrix(
            deploy_probe_dataset, split="train", feature_source="raw_dsp_cache"
        )
        with pytest.raises(ValueError, match="LS"):
            fit_e1d_sb_inference(train, alphas=(1.0,))


class TestDeployProbeGates:
    def test_formal_config_requires_gates(self):
        _validate_config(
            {
                "dataset_dir": "data/x",
                "output_dir": "outputs/e1d_sb_deploy_probe_s1",
                "ridge_alphas": [0.1],
                "feature_builder": E1DSB_FEATURE_BUILDER,
                "run_kind": "formal",
                "b1_reference_metrics": "outputs/b1.json",
                "e1d_sb_verdict_path": "outputs/e1d_sb/verdict.json",
                "attachment_verdict_path": "outputs/attach/verdict.json",
                "ls_promoted": False,
                "e2_allowed": False,
            }
        )
        with pytest.raises(ValueError, match="e1d_sb_verdict_path"):
            _validate_config(
                {
                    "dataset_dir": "data/x",
                    "output_dir": "outputs/e1d_sb_deploy_probe_s1",
                    "ridge_alphas": [0.1],
                    "feature_builder": E1DSB_FEATURE_BUILDER,
                    "run_kind": "formal",
                    "b1_reference_metrics": "outputs/b1.json",
                    "attachment_verdict_path": "outputs/attach/verdict.json",
                }
            )
        with pytest.raises(ValueError, match="feature_builder"):
            _validate_config(
                {
                    "dataset_dir": "data/x",
                    "output_dir": "outputs/e1d_sb_deploy_probe_s1",
                    "ridge_alphas": [0.1],
                    "feature_builder": "e1d_sb_cal_plus_corr_psr_snr_ls_v1",
                    "run_kind": "smoke",
                }
            )

    def test_refuses_frozen_output_names(self):
        with pytest.raises(ValueError, match="collides"):
            _assert_safe_output_dir(Path("outputs/tv3_ec_msw/e1d_sb_s20260704"))
        with pytest.raises(ValueError, match="collides"):
            _assert_safe_output_dir(Path("outputs/tv3_ec_msw/e1d_sb_ls_s20260704"))
        _assert_safe_output_dir(Path("outputs/tv3_ec_msw/e1d_sb_deploy_probe_s20260704"))

    def test_verdict_blocks_without_upstream_gates(self):
        verdict = _build_verdict(
            run_kind="formal",
            e1d_sb_gate={"status": "parity_failed"},
            attachment_gate={"status": "attachment_passed"},
            has_reference=True,
            parity_passed=True,
            waveform_align_passed=True,
            compact=True,
            diagnostic_feature_count=213,
        )
        assert verdict["status"] == "gate_blocked"
        assert verdict["e2_allowed"] is False
        assert verdict["ls_promoted"] is False
        assert verdict["default_head_remains"] == "B7"

    def test_verdict_pass(self):
        verdict = _build_verdict(
            run_kind="formal",
            e1d_sb_gate={"status": "parity_passed"},
            attachment_gate={"status": "attachment_passed"},
            has_reference=True,
            parity_passed=True,
            waveform_align_passed=True,
            compact=True,
            diagnostic_feature_count=213,
        )
        assert verdict["status"] == "deploy_probe_passed"
        assert verdict["e2_allowed"] is False


class TestDeployProbeSmoke:
    def test_smoke_probe_writes_artifacts(self, deploy_probe_dataset: Path, tmp_path: Path):
        config_path = tmp_path / "deploy_probe_smoke.json"
        output_dir = tmp_path / "e1d_sb_deploy_probe_smoke_out"
        config_path.write_text(
            json.dumps(
                {
                    "dataset_dir": str(deploy_probe_dataset),
                    "output_dir": str(output_dir),
                    "run_kind": "smoke",
                    "feature_source": "waveform",
                    "feature_builder": E1DSB_FEATURE_BUILDER,
                    "ls_promoted": False,
                    "e2_allowed": False,
                    "ridge_alphas": [0.1, 1.0],
                    "eval_splits": ["val", "test", "extrapolation"],
                }
            ),
            encoding="utf-8",
        )
        wrote = run_ec_msw_e1d_sb_deploy_probe(config_path)
        assert wrote == output_dir
        assert (output_dir / "manifest.json").is_file()
        assert (output_dir / "verdict.json").is_file()
        assert (output_dir / "summary.json").is_file()
        assert (output_dir / "predictions.csv").is_file()
        assert (output_dir / "inference_artifact.json").is_file()
        verdict = json.loads((output_dir / "verdict.json").read_text(encoding="utf-8"))
        assert verdict["status"] == "smoke_only"
        assert verdict["e2_allowed"] is False
        assert verdict["ls_promoted"] is False
        assert verdict["default_head_remains"] == "B7"
        manifest = json.loads((output_dir / "manifest.json").read_text(encoding="utf-8"))
        assert manifest["feature_alignment"]["passed"] is True
        assert manifest["ls_promoted"] is False
