from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch

from tv3.dl.evaluation.ec_msw_e1r_attachment_audit import (
    _build_verdict,
    _validate_config,
    run_ec_msw_e1r_attachment_audit,
)
from tv3.dl.models.registry import build_model
from tv3.ml.e1d_sb_features import E1DSB_FEATURE_BUILDER
from tv3.ml.raw_dsp_features import template_digest
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
            dataset_slug="tv3-e1r-attach-smoke",
            sequence_count=sequences,
            seed=20260717,
            timesteps=16,
            storage="npz",
            optical_absorption_backend="empirical_v1",
            workers=1,
        ),
    )
    dataset_dir = tmp_path / "tv3-e1r-attach-smoke"
    preflight = preflight_tv3_raw_dsp_dataset(dataset_dir)
    build_tv3_raw_dsp_feature_cache(
        preflight,
        cache_dir=dataset_dir / "features" / "raw_dsp" / "raw_dsp_frame_v1",
        template_mode="train_baseline_median",
        workers=1,
    )
    return dataset_dir


@pytest.fixture(scope="module")
def attach_dataset(tmp_path_factory: pytest.TempPathFactory) -> Path:
    return _make_dataset(tmp_path_factory.mktemp("e1r_attach_dataset"))


class TestE1rAttachmentVerdict:
    def test_smoke_verdict(self):
        verdict = _build_verdict(
            run_kind="smoke",
            frame_passed=True,
            parity_passed=True,
            e1d_sb_gate=None,
            feature_source="raw_dsp_cache",
        )
        assert verdict["status"] == "smoke_only"
        assert verdict["e2_allowed"] is False

    def test_attachment_passed(self):
        verdict = _build_verdict(
            run_kind="formal",
            frame_passed=True,
            parity_passed=True,
            e1d_sb_gate={"continue_e1r_attachment": True},
            feature_source="raw_dsp_cache",
        )
        assert verdict["status"] == "attachment_passed"
        assert verdict["e2_allowed"] is False

    def test_frame_fail(self):
        verdict = _build_verdict(
            run_kind="formal",
            frame_passed=False,
            parity_passed=True,
            e1d_sb_gate=None,
            feature_source="raw_dsp_cache",
        )
        assert verdict["status"] == "frame_fidelity_failed"

    def test_formal_config_accepts_required_keys(self):
        _validate_config(
            {
                "dataset_dir": "data/x",
                "training_run_dir": "outputs/x",
                "output_dir": "outputs/y",
                "b1_reference_metrics": "outputs/z.json",
                "device": "cpu",
                "batch_size": 1,
                "num_workers": 0,
                "ridge_alphas": [0.1],
                "max_train_probe_frames": 10,
                "probe_sample_seed": 1,
                "run_kind": "formal",
                "feature_builder": E1DSB_FEATURE_BUILDER,
            }
        )

    def test_rejects_wrong_feature_builder(self):
        with pytest.raises(ValueError, match="feature_builder must be"):
            _validate_config(
                {
                    "dataset_dir": "data/x",
                    "training_run_dir": "outputs/x",
                    "output_dir": "outputs/y",
                    "b1_reference_metrics": "outputs/z.json",
                    "device": "cpu",
                    "batch_size": 1,
                    "num_workers": 0,
                    "ridge_alphas": [0.1],
                    "max_train_probe_frames": 10,
                    "probe_sample_seed": 1,
                    "feature_builder": "not_the_builder",
                }
            )


def test_run_attachment_smoke_writes_artifacts(attach_dataset: Path, tmp_path: Path):
    cache_dir = attach_dataset / "features" / "raw_dsp" / "raw_dsp_frame_v1"
    template = np.load(cache_dir / "template.npy").astype(np.float32)
    digest = template_digest(template)

    run_dir = tmp_path / "e1r_run"
    run_dir.mkdir()
    model_config = {
        "name": "ec_msw_e1",
        "in_channels": 5009,
        "out_dim": 3,
        "output_mode": "raw3",
        "slow_channels": 9,
        "ultrasonic_channels": 5000,
        "fiber_mic_channels": 0,
        "waveform_embedding_dim": 4,
        "slow_embedding_dim": 4,
        "stem_channels": 2,
        "branch_channels": 2,
        "kernel_sizes": [3],
        "dilations": [1],
        "downsample_factor": 4,
        "dropout": 0.0,
        "head_hidden_dim": 4,
        "peak_coordinate_template": template.tolist(),
        "peak_coordinate_template_digest": digest,
    }
    model = build_model(model_config)
    torch.save({"model_state_dict": model.state_dict()}, run_dir / "checkpoint.pt")
    (run_dir / "run_config.json").write_text(
        json.dumps(
            {
                "model_config": model_config,
                "modalities": ["slow", "ultrasonic"],
                "input_format": "NTC",
                "scaler_path": None,
                "slow_channels": None,
                "phase_windows": None,
                "phase_stats_path": None,
                "dequantize_waveforms": True,
                "normalize_waveforms": True,
                "waveform_stats_features": ["log_std", "log_max_abs"],
            }
        ),
        encoding="utf-8",
    )

    reference_path = tmp_path / "b1.json"
    component_metrics = {
        name: {"mae": 0.0, "rmse": 0.0, "r2": 0.0}
        for name in ("x_CO2", "x_O2", "x_N2")
    }
    reference_path.write_text(
        json.dumps(
            {
                "evaluations": {
                    split: {"component_metrics": component_metrics}
                    for split in ("val", "test", "extrapolation")
                }
            }
        ),
        encoding="utf-8",
    )

    e1d_sb_verdict = tmp_path / "e1d_sb_verdict.json"
    e1d_sb_verdict.write_text(
        json.dumps(
            {
                "status": "parity_passed",
                "continue_e1r_attachment": True,
                "feature_builder": E1DSB_FEATURE_BUILDER,
                "e2_allowed": False,
            }
        ),
        encoding="utf-8",
    )

    output_dir = tmp_path / "attach_out"
    config_path = tmp_path / "attach.json"
    config_path.write_text(
        json.dumps(
            {
                "dataset_dir": str(attach_dataset),
                "training_run_dir": str(run_dir),
                "output_dir": str(output_dir),
                "run_kind": "smoke",
                "feature_source": "raw_dsp_cache",
                "feature_builder": E1DSB_FEATURE_BUILDER,
                "e1d_sb_verdict_path": str(e1d_sb_verdict),
                "b1_reference_metrics": str(reference_path),
                "device": "cpu",
                "batch_size": 4,
                "num_workers": 0,
                "ridge_alphas": [0.01, 0.1, 1.0],
                "max_train_probe_frames": 64,
                "probe_sample_seed": 17,
                "frame_fidelity_gates": {
                    "peak_mae_samples_max": 10000.0,
                    "peak_p95_abs_error_samples_max": 10000.0,
                    "peak_bias_abs_samples_max": 10000.0,
                },
                "parity_gates": {
                    "o2_r2_drop_max": 10000.0,
                    "co2_n2_r2_drop_max": 10000.0,
                },
                "narrow_o2_windows": [
                    {"id": "w1", "low_percent": 18.0, "high_percent": 18.8},
                    {"id": "w2", "low_percent": 18.8, "high_percent": 19.6},
                    {"id": "w3", "low_percent": 19.6, "high_percent": 20.4},
                    {"id": "w4", "low_percent": 20.4, "high_percent": 21.2},
                ],
            }
        ),
        encoding="utf-8",
    )

    written = run_ec_msw_e1r_attachment_audit(config_path, project_root=tmp_path)
    assert written == output_dir
    for name in (
        "manifest.json",
        "summary.json",
        "verdict.json",
        "frame_fidelity.json",
        "b1_parity.json",
        "narrow_o2_windows.csv",
    ):
        assert (output_dir / name).is_file()

    verdict = json.loads((output_dir / "verdict.json").read_text(encoding="utf-8"))
    assert verdict["status"] == "smoke_only"
    assert verdict["e2_allowed"] is False
    assert verdict["feature_builder"] == E1DSB_FEATURE_BUILDER

    manifest = json.loads((output_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["schema_version"] == "tv3-ec-msw-e1r-attachment-1"
    assert manifest["e1r_template_digest"] == digest
    assert manifest["feature_builder"] == E1DSB_FEATURE_BUILDER
