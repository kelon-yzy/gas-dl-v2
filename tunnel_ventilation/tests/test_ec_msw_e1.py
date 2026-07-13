from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch

from tv3.dl.cli import (
    _build_model_config,
    _parse_comma,
    _parse_waveform_stats_features,
    _resolve_ec_msw_peak_coordinate_template,
    _resolve_raw_output_prior,
    _waveform_stats_channel_count,
)
from tv3.dl.evaluation.ec_msw_e1_audit import (
    _narrow_window_rows,
    _parity_split_gate,
    _peak_error_metrics,
    _sample_frame_indices,
    _verdict,
    run_ec_msw_e1_audit,
)
from tv3.dl.models.ec_msw_e1 import (
    ECMSWE1Regressor,
    MatchedFilterPeakCoordinate,
    PositionSensitiveMultiScaleEncoder,
    PositionSensitiveStatisticsPool,
)
from tv3.dl.models.registry import MODEL_REGISTRY, build_model
from tv3.sim.generation.tunnel_ventilation import (
    TunnelVentilationBenchmarkGenerationSpec,
    generate_tunnel_ventilation_benchmark_dataset,
)


class TestPositionSensitiveStatisticsPool:
    def test_centroid_tracks_absolute_peak_position(self):
        pool = PositionSensitiveStatisticsPool()
        early = torch.zeros(1, 1, 101)
        late = torch.zeros(1, 1, 101)
        early[0, 0, 20] = 1.0
        late[0, 0, 80] = 1.0

        early_centroid = pool(early)[0, 2]
        late_centroid = pool(late)[0, 2]

        assert early_centroid.item() == pytest.approx(0.2, abs=1e-5)
        assert late_centroid.item() == pytest.approx(0.8, abs=1e-5)
        assert late_centroid > early_centroid


class TestMatchedFilterPeakCoordinate:
    def test_recovers_inserted_absolute_peak_position(self):
        template = [0.0, -1.0, 2.0, -1.0, 0.0]
        coordinate = MatchedFilterPeakCoordinate(32, template)
        waveform = torch.zeros(2, 1, 32)
        waveform[0, 0, 5:10] = torch.tensor(template)
        waveform[1, 0, 20:25] = torch.tensor(template)

        peak_index = coordinate(waveform).squeeze(1) * 31.0

        assert torch.allclose(peak_index, torch.tensor([7.0, 22.0]))

    def test_rejects_digest_mismatch(self):
        with pytest.raises(ValueError, match="digest"):
            MatchedFilterPeakCoordinate(
                32,
                [0.0, -1.0, 2.0, -1.0, 0.0],
                expected_digest="not-the-template-digest",
            )


class TestPositionSensitiveMultiScaleEncoder:
    def test_shape_and_gradient(self):
        encoder = PositionSensitiveMultiScaleEncoder(
            waveform_length=256,
            embedding_dim=12,
            stem_channels=4,
            branch_channels=4,
            kernel_sizes=(5, 9, 15),
            dilations=(1, 2, 3),
            downsample_factor=2,
            dropout=0.0,
        )
        waveform = torch.randn(2, 3, 256, requires_grad=True)
        embedding = encoder(waveform)
        embedding.square().mean().backward()

        assert embedding.shape == (2, 3, 12)
        assert waveform.grad is not None
        assert torch.isfinite(waveform.grad).all()

    def test_peak_coordinate_bypasses_learned_projection(self):
        template = [0.0, -1.0, 2.0, -1.0, 0.0]
        encoder = PositionSensitiveMultiScaleEncoder(
            waveform_length=64,
            embedding_dim=8,
            stem_channels=2,
            branch_channels=2,
            kernel_sizes=(3,),
            dilations=(1,),
            downsample_factor=2,
            dropout=0.0,
            peak_coordinate_template=template,
        ).eval()
        waveform = torch.zeros(1, 2, 64)
        waveform[0, 0, 5:10] = torch.tensor(template)
        waveform[0, 1, 40:45] = torch.tensor(template)

        with torch.inference_mode():
            embedding = encoder(waveform)

        assert embedding.shape == (1, 2, 8)
        assert torch.allclose(embedding[0, :, 0] * 63.0, torch.tensor([7.0, 42.0]))


class TestECMSWE1Regressor:
    def test_registered_and_forward_raw3(self):
        assert "ec_msw_e1" in MODEL_REGISTRY
        model = build_model(
            {
                "name": "ec_msw_e1",
                "in_channels": 73,
                "out_dim": 3,
                "slow_channels": 9,
                "ultrasonic_channels": 64,
                "fiber_mic_channels": 0,
                "waveform_embedding_dim": 8,
                "slow_embedding_dim": 4,
                "stem_channels": 4,
                "branch_channels": 4,
                "kernel_sizes": [3, 7],
                "dilations": [1, 2],
                "downsample_factor": 2,
                "head_hidden_dim": 8,
                "dropout": 0.0,
                "output_mode": "raw3",
                "raw_output_prior": [2.0, 20.0, 78.0],
            }
        )
        output = model(torch.randn(2, 5, 73))

        assert isinstance(model, ECMSWE1Regressor)
        assert output.shape == (2, 3)
        assert torch.allclose(model.head[-1].bias, torch.tensor([2.0, 20.0, 78.0]))

    def test_embedding_interfaces_and_forward_share_one_path(self):
        model = ECMSWE1Regressor(
            in_channels=73,
            slow_channels=9,
            ultrasonic_channels=64,
            waveform_embedding_dim=8,
            slow_embedding_dim=4,
            stem_channels=2,
            branch_channels=2,
            kernel_sizes=(3,),
            dilations=(1,),
            downsample_factor=2,
            head_hidden_dim=8,
            dropout=0.0,
        ).eval()
        x = torch.randn(2, 5, 73)
        slow_changed = x.clone()
        slow_changed[:, :, :9] += 100.0
        with torch.inference_mode():
            frames = model.encode_frames(x)
            sequence = model.encode_sequence(x)
            prediction = model(x)
            prediction_from_sequence = model.head(sequence)
            changed_frames = model.encode_frames(slow_changed)

        assert frames.shape == (2, 5, 8)
        assert sequence.shape == (2, 36)
        assert torch.allclose(prediction, prediction_from_sequence)
        assert torch.allclose(frames, changed_frames)

    def test_peak_coordinate_bypasses_frame_norm_and_reaches_sequence_pooling(self):
        template = [0.0, -1.0, 2.0, -1.0, 0.0]
        model = ECMSWE1Regressor(
            in_channels=41,
            slow_channels=9,
            ultrasonic_channels=32,
            waveform_embedding_dim=8,
            slow_embedding_dim=4,
            stem_channels=2,
            branch_channels=2,
            kernel_sizes=(3,),
            dilations=(1,),
            downsample_factor=2,
            head_hidden_dim=8,
            dropout=0.0,
            peak_coordinate_template=template,
        ).eval()
        x = torch.zeros(1, 3, 41)
        x[0, 0, 9 + 3 : 9 + 8] = torch.tensor(template)
        x[0, 1, 9 + 10 : 9 + 15] = torch.tensor(template)
        x[0, 2, 9 + 20 : 9 + 25] = torch.tensor(template)

        with torch.inference_mode():
            sequence = model.encode_sequence(x)

        frame_dim = 12
        assert sequence[0, 0] * 31.0 == pytest.approx(22.0)
        assert sequence[0, frame_dim] * 31.0 == pytest.approx(13.0)
        assert sequence[0, frame_dim * 2] * 31.0 == pytest.approx(22.0)

    def test_rejects_non_raw3_and_fiber_mic(self):
        with pytest.raises(ValueError, match="raw3"):
            ECMSWE1Regressor(output_mode="gas_head")
        with pytest.raises(ValueError, match="fiber_mic"):
            ECMSWE1Regressor(
                in_channels=15009,
                slow_channels=9,
                ultrasonic_channels=5000,
                fiber_mic_channels=10000,
            )

    def test_rejects_runtime_channel_mismatch(self):
        model = ECMSWE1Regressor(
            in_channels=73,
            slow_channels=9,
            ultrasonic_channels=64,
            stem_channels=2,
            branch_channels=2,
            kernel_sizes=(3,),
            dilations=(1,),
        )
        with pytest.raises(ValueError, match="Expected 73 input channels"):
            model(torch.randn(1, 2, 72))


class TestECMSWE1Config:
    def test_resolves_train_only_peak_coordinate_template(self, tmp_path: Path):
        template_path = tmp_path / "template.npy"
        np.save(
            template_path,
            np.array([0.0, -1.0, 2.0, -1.0, 0.0], dtype=np.float32),
        )
        model_config = {"peak_coordinate_template_path": str(template_path)}

        _resolve_ec_msw_peak_coordinate_template("ec_msw_e1", model_config)

        assert "peak_coordinate_template_path" not in model_config
        assert model_config["peak_coordinate_template"] == [0.0, -1.0, 2.0, -1.0, 0.0]
        assert len(model_config["peak_coordinate_template_digest"]) == 64

    def test_repair_config_keeps_failed_e1_outputs_immutable(self):
        config = json.loads(
            Path("configs/tv3_ec_msw_e1r_smoke.json").read_text(encoding="utf-8")
        )

        assert config["output_dir"] == "outputs/tv3_ec_msw/e1r_smoke_s20260704"
        assert config["model_kwargs"]["peak_coordinate_template_path"].endswith("template.npy")
        assert "aux_target_arrays" not in config

    def test_smoke_config_builds_with_waveform_stats_and_auto_prior(self):
        config_path = Path("configs/tv3_ec_msw_e1_smoke.json")
        config = json.loads(config_path.read_text(encoding="utf-8"))
        modalities = _parse_comma(config["modalities"])
        stats = _parse_waveform_stats_features(config["waveform_stats_features"])
        model_config = _build_model_config(
            config["model"],
            config["model_kwargs"],
            in_channels=5009,
            out_dim=3,
            timesteps=16,
        )
        model_config["slow_channels"] += _waveform_stats_channel_count(modalities, stats)
        _resolve_raw_output_prior(
            config["model"],
            model_config,
            np.array([[2.0, 20.0, 78.0], [4.0, 22.0, 74.0]], dtype=np.float32),
            out_dim=3,
            target_transform=None,
        )

        model = build_model(model_config)

        assert config["model"] == "ec_msw_e1"
        assert "aux_target_arrays" not in config
        assert model_config["slow_channels"] == 9
        assert model_config["raw_output_prior"] == [3.0, 21.0, 76.0]
        assert isinstance(model, ECMSWE1Regressor)


class TestECMSWE1AuditMetrics:
    def test_peak_metrics_and_sampling_are_deterministic(self):
        first = _sample_frame_indices(100, maximum=10, seed=7)
        second = _sample_frame_indices(100, maximum=10, seed=7)
        metrics = _peak_error_metrics(np.array([-0.2, 0.0, 0.1], dtype=np.float32))

        assert np.array_equal(first, second)
        assert len(np.unique(first)) == 10
        assert metrics["peak_mae_samples"] == pytest.approx(0.1)
        assert metrics["peak_bias_samples"] == pytest.approx(-1.0 / 30.0)

    def test_parity_gate_and_verdict(self):
        candidate = {
            "x_CO2": {"r2": 0.97},
            "x_O2": {"r2": 0.40},
            "x_N2": {"r2": 0.87},
        }
        reference = {
            "x_CO2": {"r2": 0.99},
            "x_O2": {"r2": 0.44},
            "x_N2": {"r2": 0.89},
        }
        gate = _parity_split_gate(
            candidate,
            reference,
            {"o2_r2_drop_max": 0.05, "co2_n2_r2_drop_max": 0.03},
        )

        assert gate["passed"] is True
        assert gate["r2_delta_vs_b1"]["x_O2"] == pytest.approx(-0.04)
        assert _verdict(True, True)["status"] == "e1_pass"
        assert _verdict(False, True)["status"] == "frame_fidelity_failed"
        assert _verdict(True, False)["status"] == "b1_parity_failed"

    def test_fixed_narrow_windows_report_p90_and_slope(self):
        y_true = np.array(
            [[1.0, 18.1, 80.9], [1.0, 18.7, 80.3], [1.0, 19.0, 80.0]],
            dtype=np.float32,
        )
        y_pred = y_true.copy()
        y_pred[:, 1] += np.array([0.1, -0.2, 0.3], dtype=np.float32)
        rows = _narrow_window_rows(
            "val",
            y_pred,
            y_true,
            [
                {"id": "w1", "low_percent": 18.0, "high_percent": 18.8},
                {"id": "w2", "low_percent": 18.8, "high_percent": 19.6},
            ],
        )

        assert rows[0]["count"] == 2
        assert rows[0]["mae_percent"] == pytest.approx(0.15, abs=1e-6)
        assert rows[0]["p90_abs_error_percent"] == pytest.approx(0.19, abs=1e-6)
        assert rows[0]["local_slope"] is not None
        assert rows[1]["count"] == 1
        assert rows[1]["local_slope"] is None


class TestECMSWE1AuditIntegration:
    def test_train_only_probe_and_parity_pipeline_writes_required_outputs(self, tmp_path: Path):
        generate_tunnel_ventilation_benchmark_dataset(
            tmp_path,
            TunnelVentilationBenchmarkGenerationSpec(
                dataset_slug="ec-msw-audit-smoke",
                sequence_count=16,
                seed=20260713,
                timesteps=16,
                storage="npz",
                optical_absorption_backend="empirical_v1",
                workers=1,
            ),
        )
        dataset_dir = tmp_path / "ec-msw-audit-smoke"
        run_dir = tmp_path / "run"
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
        audit_config = {
            "dataset_dir": str(dataset_dir),
            "training_run_dir": str(run_dir),
            "output_dir": str(tmp_path / "audit"),
            "b1_reference_metrics": str(reference_path),
            "device": "cpu",
            "batch_size": 4,
            "num_workers": 0,
            "ridge_alphas": [0.1, 1.0],
            "max_train_probe_frames": 64,
            "probe_sample_seed": 13,
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
        config_path = tmp_path / "audit.json"
        config_path.write_text(json.dumps(audit_config), encoding="utf-8")

        output_dir = run_ec_msw_e1_audit(config_path, project_root=tmp_path)

        assert (output_dir / "frame_fidelity.json").is_file()
        assert (output_dir / "b1_parity.json").is_file()
        assert (output_dir / "narrow_o2_windows.csv").is_file()
        verdict = json.loads((output_dir / "verdict.json").read_text(encoding="utf-8"))
        manifest = json.loads((output_dir / "manifest.json").read_text(encoding="utf-8"))
        assert verdict["status"] == "e1_pass"
        assert manifest["schema_version"] == "tv3-ec-msw-e1-audit-1"
        assert "ultrasonic_peak_index" in manifest["model_inputs_exclude"]
