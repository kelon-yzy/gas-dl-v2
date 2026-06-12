from __future__ import annotations

from pathlib import Path

import torch

from dl.models.base import BaseRegressor
from dl.models.cnn1d import CNN1DRegressor
from dl.models.cnn1d_tcn_fusion import CNN1DTCNFusionRegressor, GasHeadNormalize
from dl.models.lstm import LSTMRegressor
from dl.models.patchtst import PatchTSTRegressor
from dl.models.phase_window_tcn import PhaseWindowTCNRegressor
from dl.models.registry import MODEL_REGISTRY, build_model
from dl.models.tcn import CausalConv1d, TCNRegressor, tcn_channels_for_timesteps
from dl.models.transformer import TransformerRegressor
from dl.data.dataset import V4BenchmarkDataset
from sim.generation.benchmark import BenchmarkGenerationSpec, generate_benchmark_dataset


def _make_smoke_dataset(tmp_path: Path) -> Path:
    generate_benchmark_dataset(
        tmp_path,
        BenchmarkGenerationSpec(
            dataset_slug="model-smoke",
            sequence_count=8,
            seed=1,
            timesteps=16,
            storage="npz",
            optical_absorption_backend="empirical_v1",
        ),
    )
    return tmp_path / "model-smoke"


class TestModelRegistry:
    def test_registry_contains_cnn1d(self):
        assert "cnn1d" in MODEL_REGISTRY

    def test_registry_contains_tcn(self):
        assert "tcn" in MODEL_REGISTRY

    def test_registry_contains_cnn1d_tcn_fusion(self):
        assert "cnn1d_tcn_fusion" in MODEL_REGISTRY

    def test_registry_contains_phase_window_tcn(self):
        assert "phase_window_tcn" in MODEL_REGISTRY

    def test_registry_contains_sequence_models(self):
        assert {"lstm", "transformer", "patchtst"}.issubset(MODEL_REGISTRY)

    def test_build_model_from_config(self):
        model = build_model({"name": "cnn1d", "in_channels": 8, "out_dim": 4})
        assert isinstance(model, CNN1DRegressor)

    def test_build_tcn_from_config(self):
        model = build_model({"name": "tcn", "in_channels": 8, "out_dim": 4})
        assert isinstance(model, TCNRegressor)

    def test_build_cnn1d_tcn_fusion_from_config(self):
        model = build_model(
            {
                "name": "cnn1d_tcn_fusion",
                "in_channels": 14,
                "slow_channels": 2,
                "ultrasonic_channels": 5,
                "fiber_mic_channels": 7,
                "waveform_embedding_dim": 4,
                "acoustic_channels": [2, 4],
                "slow_hidden_dim": 4,
                "slow_embedding_dim": 4,
                "tcn_channels": [4],
                "shared_hidden_dims": [8, 4],
            }
        )
        assert isinstance(model, CNN1DTCNFusionRegressor)

    def test_build_phase_window_tcn_from_config(self):
        model = build_model(
            {
                "name": "phase_window_tcn",
                "in_channels": 14,
                "slow_channels": 2,
                "ultrasonic_channels": 5,
                "fiber_mic_channels": 7,
                "window_count": 3,
                "waveform_embedding_dim": 4,
                "acoustic_channels": [2, 4],
                "slow_hidden_dim": 4,
                "slow_embedding_dim": 4,
                "tcn_channels": [4],
                "shared_hidden_dims": [8, 4],
            }
        )
        assert isinstance(model, PhaseWindowTCNRegressor)

    def test_build_sequence_models_from_config(self):
        assert isinstance(build_model({"name": "lstm", "in_channels": 8, "out_dim": 4}), LSTMRegressor)
        assert isinstance(build_model({"name": "transformer", "in_channels": 8, "out_dim": 4}), TransformerRegressor)
        assert isinstance(build_model({"name": "patchtst", "in_channels": 8, "out_dim": 4}), PatchTSTRegressor)

    def test_build_model_unknown_name_raises(self):
        try:
            build_model({"name": "nonexistent"})
        except ValueError as exc:
            assert "nonexistent" in str(exc)

    def test_build_model_passes_kwargs(self):
        model = build_model({"name": "cnn1d", "in_channels": 4, "out_dim": 3, "hidden_channels": [16, 32]})
        assert model.encoder[0].in_channels == 4
        assert model.out_dim == 3

    def test_build_tcn_passes_kwargs(self):
        model = build_model({"name": "tcn", "in_channels": 4, "out_dim": 3, "channels": [16, 32]})
        assert model.encoder[0].net[0].conv.in_channels == 4
        assert model.out_dim == 3

    def test_registry_entry_is_class_or_callable(self):
        for name, entry in MODEL_REGISTRY.items():
            assert callable(entry), f"MODEL_REGISTRY[{name!r}] is not callable"


class TestCNN1DRegressor:
    def test_forward_shape_nct(self):
        model = CNN1DRegressor(in_channels=8, out_dim=4)
        x = torch.randn(2, 8, 32)  # NCT format
        out = model(x)
        assert out.shape == (2, 4)

    def test_forward_shape_different_timesteps(self):
        model = CNN1DRegressor(in_channels=8, out_dim=4)
        x = torch.randn(4, 8, 64)
        out = model(x)
        assert out.shape == (4, 4)

    def test_different_in_channels(self):
        model = CNN1DRegressor(in_channels=12, out_dim=4)
        x = torch.randn(2, 12, 32)
        out = model(x)
        assert out.shape == (2, 4)

    def test_gradient_flows(self):
        model = CNN1DRegressor(in_channels=8, out_dim=4)
        x = torch.randn(2, 8, 16, requires_grad=True)
        out = model(x)
        loss = out.sum()
        loss.backward()
        for name, param in model.named_parameters():
            assert param.grad is not None, f"{name} has no gradient"

    def test_input_format_attribute(self):
        model = CNN1DRegressor()
        assert model.input_format == "NCT"


class TestTCNRegressor:
    def test_forward_shape_nct(self):
        model = TCNRegressor(in_channels=8, out_dim=4)
        x = torch.randn(2, 8, 32)
        out = model(x)
        assert out.shape == (2, 4)

    def test_forward_shape_different_timesteps(self):
        model = TCNRegressor(in_channels=8, out_dim=4)
        x = torch.randn(4, 8, 64)
        out = model(x)
        assert out.shape == (4, 4)

    def test_different_in_channels(self):
        model = TCNRegressor(in_channels=12, out_dim=4)
        x = torch.randn(2, 12, 32)
        out = model(x)
        assert out.shape == (2, 4)

    def test_gradient_flows(self):
        model = TCNRegressor(in_channels=8, out_dim=4)
        x = torch.randn(2, 8, 16, requires_grad=True)
        out = model(x)
        loss = out.sum()
        loss.backward()
        for name, param in model.named_parameters():
            assert param.grad is not None, f"{name} has no gradient"

    def test_input_format_attribute(self):
        model = TCNRegressor()
        assert model.input_format == "NCT"

    def test_receptive_field_is_recorded(self):
        model = TCNRegressor(channels=[16, 32, 64], kernel_size=3)
        assert model.dilations == (1, 2, 4)
        assert model.receptive_field == 29

    def test_target_timesteps_expands_default_receptive_field(self):
        model = TCNRegressor(in_channels=8, out_dim=4, target_timesteps=512)
        assert len(model.dilations) == 8
        assert model.receptive_field >= 512
        assert tcn_channels_for_timesteps(1024) == [32, 64, 64, 64, 64, 64, 64, 64, 64]

    def test_target_timesteps_rejects_manual_short_receptive_field(self):
        try:
            TCNRegressor(channels=[16, 32, 64], kernel_size=3, target_timesteps=512)
        except ValueError as exc:
            assert "receptive_field=29" in str(exc)
        else:
            raise AssertionError("short manual TCN receptive field was accepted")

    def test_rejects_empty_channels(self):
        try:
            TCNRegressor(channels=[])
        except ValueError as exc:
            assert "channels must contain at least one block" in str(exc)
        else:
            raise AssertionError("empty TCN channels were accepted")

    def test_causal_conv_preserves_timesteps(self):
        conv = CausalConv1d(in_channels=3, out_channels=5, kernel_size=3, dilation=2)
        x = torch.randn(2, 3, 11)
        out = conv(x)
        assert out.shape == (2, 5, 11)


class TestCNN1DTCNFusionRegressor:
    def test_forward_shape_and_simplex_constraint(self):
        model = CNN1DTCNFusionRegressor(
            in_channels=14,
            slow_channels=2,
            ultrasonic_channels=5,
            fiber_mic_channels=7,
            waveform_embedding_dim=4,
            acoustic_channels=[2, 4],
            slow_hidden_dim=4,
            slow_embedding_dim=4,
            tcn_channels=[4],
            shared_hidden_dims=[8, 4],
        )

        out = model(torch.randn(3, 6, 14))

        assert out.shape == (3, 4)
        assert torch.all(out >= 0.0)
        assert torch.allclose(out.sum(dim=-1), torch.full((3,), 100.0), atol=1e-5)
        assert model.input_format == "NTC"
        assert model.receptive_field == 5

    def test_forward_shape_for_transformed_coordinate_head(self):
        model = CNN1DTCNFusionRegressor(
            in_channels=14,
            out_dim=3,
            slow_channels=2,
            ultrasonic_channels=5,
            fiber_mic_channels=7,
            waveform_embedding_dim=4,
            acoustic_channels=[2, 4],
            slow_hidden_dim=4,
            slow_embedding_dim=4,
            tcn_channels=[4],
            shared_hidden_dims=[8, 4],
        )

        out = model(torch.randn(3, 6, 14))

        assert out.shape == (3, 3)

    def test_gradient_flows(self):
        model = CNN1DTCNFusionRegressor(
            in_channels=14,
            slow_channels=2,
            ultrasonic_channels=5,
            fiber_mic_channels=7,
            waveform_embedding_dim=4,
            acoustic_channels=[2, 4],
            slow_hidden_dim=4,
            slow_embedding_dim=4,
            tcn_channels=[4],
            shared_hidden_dims=[8, 4],
        )
        x = torch.randn(2, 4, 14, requires_grad=True)

        loss = model(x).sum()
        loss.backward()

        for name, param in model.named_parameters():
            assert param.grad is not None, f"{name} has no gradient"

    def test_rejects_channel_mismatch(self):
        try:
            CNN1DTCNFusionRegressor(in_channels=13, slow_channels=2, ultrasonic_channels=5, fiber_mic_channels=7)
        except ValueError as exc:
            assert "does not match" in str(exc)
        else:
            raise AssertionError("channel mismatch should be rejected")

    def test_gas_head_uses_output_prior_and_keeps_sum(self):
        head = GasHeadNormalize(3, output_prior=(10.0, 70.0, 5.0, 15.0))

        out = head(torch.zeros(1, 3))

        assert torch.allclose(out.sum(dim=-1), torch.tensor([100.0]), atol=1e-5)
        assert torch.allclose(out[0], torch.tensor([10.0, 70.0, 5.0, 15.0]), atol=1e-5)


class TestPhaseWindowTCNRegressor:
    def test_forward_shape_for_raw4_head(self):
        model = PhaseWindowTCNRegressor(
            in_channels=14,
            slow_channels=2,
            ultrasonic_channels=5,
            fiber_mic_channels=7,
            window_count=3,
            waveform_embedding_dim=4,
            acoustic_channels=[2, 4],
            slow_hidden_dim=4,
            slow_embedding_dim=4,
            tcn_channels=[4],
            shared_hidden_dims=[8, 4],
        )

        out = model(torch.randn(2, 3, 6, 14))

        assert out.shape == (2, 4)
        assert model.input_format == "NTC"
        assert model.receptive_field == 5

    def test_softmax100_head_keeps_simplex_sum(self):
        model = PhaseWindowTCNRegressor(
            in_channels=14,
            slow_channels=2,
            ultrasonic_channels=5,
            fiber_mic_channels=7,
            window_count=3,
            output_mode="softmax100",
            waveform_embedding_dim=4,
            acoustic_channels=[2, 4],
            slow_hidden_dim=4,
            slow_embedding_dim=4,
            tcn_channels=[4],
            shared_hidden_dims=[8, 4],
        )

        out = model(torch.randn(2, 3, 6, 14))

        assert torch.allclose(out.sum(dim=-1), torch.full((2,), 100.0), atol=1e-5)

    def test_rejects_window_count_mismatch(self):
        model = PhaseWindowTCNRegressor(
            in_channels=14,
            slow_channels=2,
            ultrasonic_channels=5,
            fiber_mic_channels=7,
            window_count=3,
            waveform_embedding_dim=4,
            acoustic_channels=[2, 4],
            slow_hidden_dim=4,
            slow_embedding_dim=4,
            tcn_channels=[4],
            shared_hidden_dims=[8, 4],
        )

        try:
            model(torch.randn(2, 2, 6, 14))
        except ValueError as exc:
            assert "Expected 3 windows" in str(exc)
        else:
            raise AssertionError("window count mismatch should be rejected")


class TestLongSequenceRegressors:
    def test_lstm_forward_shape_ntc(self):
        model = LSTMRegressor(in_channels=8, out_dim=4, hidden_size=16)
        out = model(torch.randn(2, 32, 8))
        assert out.shape == (2, 4)

    def test_transformer_forward_shape_ntc(self):
        model = TransformerRegressor(in_channels=8, out_dim=4, d_model=16, nhead=4, num_layers=1, dim_feedforward=32)
        out = model(torch.randn(2, 32, 8))
        assert out.shape == (2, 4)

    def test_patchtst_forward_shape_ntc(self):
        model = PatchTSTRegressor(in_channels=8, out_dim=4, patch_len=8, stride=4, d_model=16, nhead=4, num_layers=1)
        out = model(torch.randn(2, 32, 8))
        assert out.shape == (2, 4)

    def test_attention_pooling_keeps_cnn_tcn_shapes(self):
        cnn = CNN1DRegressor(in_channels=8, out_dim=4, pooling="attention")
        tcn = TCNRegressor(in_channels=8, out_dim=4, pooling="last")
        x = torch.randn(2, 8, 32)
        assert cnn(x).shape == (2, 4)
        assert tcn(x).shape == (2, 4)


class TestBaseRegressor:
    def test_base_regressor_raises_not_implemented(self):
        model = BaseRegressor(out_dim=4)
        try:
            model(torch.randn(2, 8, 32))
        except NotImplementedError:
            pass
        else:
            raise AssertionError("BaseRegressor.forward should raise NotImplementedError")

    def test_base_regressor_stores_out_dim(self):
        model = BaseRegressor(out_dim=3)
        assert model.out_dim == 3


class TestEndToEndWithDataset:
    def test_dataset_to_model_forward(self, tmp_path: Path):
        dataset_dir = _make_smoke_dataset(tmp_path)
        ds = V4BenchmarkDataset(
            dataset_dir,
            split="train",
            modalities=("slow",),
            input_format="NCT",
            lazy=False,
        )
        model = CNN1DRegressor(in_channels=8, out_dim=4)
        model.eval()
        x, y = ds[0]
        x = x.unsqueeze(0)  # add batch dim
        with torch.no_grad():
            out = model(x)
        assert out.shape == (1, 4)

    def test_dataset_to_tcn_forward(self, tmp_path: Path):
        dataset_dir = _make_smoke_dataset(tmp_path)
        ds = V4BenchmarkDataset(
            dataset_dir,
            split="train",
            modalities=("slow",),
            input_format="NCT",
            lazy=False,
        )
        model = TCNRegressor(in_channels=8, out_dim=4)
        model.eval()
        x, y = ds[0]
        x = x.unsqueeze(0)
        with torch.no_grad():
            out = model(x)
        assert out.shape == (1, 4)

    def test_batch_forward(self, tmp_path: Path):
        dataset_dir = _make_smoke_dataset(tmp_path)
        ds = V4BenchmarkDataset(
            dataset_dir,
            split="train",
            modalities=("slow",),
            input_format="NCT",
            lazy=False,
        )
        model = CNN1DRegressor(in_channels=8, out_dim=4)
        model.eval()
        batch_x = torch.stack([ds[i][0] for i in range(min(4, len(ds)))])
        with torch.no_grad():
            out = model(batch_x)
        assert out.shape[0] == min(4, len(ds))
        assert out.shape[1] == 4

    def test_dataset_to_cnn1d_tcn_fusion_forward(self, tmp_path: Path):
        dataset_dir = _make_smoke_dataset(tmp_path)
        ds = V4BenchmarkDataset(
            dataset_dir,
            split="train",
            modalities=("slow", "ultrasonic", "fiber_mic"),
            input_format="NTC",
            lazy=False,
        )
        model = CNN1DTCNFusionRegressor(
            in_channels=3008,
            waveform_embedding_dim=4,
            acoustic_channels=[2, 4],
            slow_hidden_dim=4,
            slow_embedding_dim=4,
            tcn_channels=[4],
            shared_hidden_dims=[8, 4],
        )
        model.eval()
        x, _y = ds[0]
        with torch.no_grad():
            out = model(x.unsqueeze(0))
        assert out.shape == (1, 4)
        assert torch.allclose(out.sum(dim=-1), torch.tensor([100.0]), atol=1e-4)

    def test_dataset_to_phase_window_tcn_forward(self, tmp_path: Path):
        dataset_dir = _make_smoke_dataset(tmp_path)
        ds = V4BenchmarkDataset(
            dataset_dir,
            split="train",
            modalities=("slow", "ultrasonic", "fiber_mic"),
            input_format="NTC",
            phase_windows=[None, {"kind": "phase", "value": "exposure"}, {"kind": "phase", "value": "recovery"}],
            lazy=False,
        )
        model = PhaseWindowTCNRegressor(
            in_channels=3008,
            waveform_embedding_dim=4,
            acoustic_channels=[2, 4],
            slow_hidden_dim=4,
            slow_embedding_dim=4,
            tcn_channels=[4],
            shared_hidden_dims=[8, 4],
        )
        model.eval()
        x, _y = ds[0]
        with torch.no_grad():
            out = model(x.unsqueeze(0))
        assert out.shape == (1, 4)
