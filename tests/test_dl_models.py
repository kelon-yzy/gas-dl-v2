from __future__ import annotations

from pathlib import Path

import torch

from dl.models.base import BaseRegressor
from dl.models.cnn1d import CNN1DRegressor
from dl.models.registry import MODEL_REGISTRY, build_model
from dl.models.tcn import CausalConv1d, TCNRegressor
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
        ),
    )
    return tmp_path / "model-smoke"


class TestModelRegistry:
    def test_registry_contains_cnn1d(self):
        assert "cnn1d" in MODEL_REGISTRY

    def test_registry_contains_tcn(self):
        assert "tcn" in MODEL_REGISTRY

    def test_build_model_from_config(self):
        model = build_model({"name": "cnn1d", "in_channels": 8, "out_dim": 4})
        assert isinstance(model, CNN1DRegressor)

    def test_build_tcn_from_config(self):
        model = build_model({"name": "tcn", "in_channels": 8, "out_dim": 4})
        assert isinstance(model, TCNRegressor)

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

    def test_causal_conv_preserves_timesteps(self):
        conv = CausalConv1d(in_channels=3, out_channels=5, kernel_size=3, dilation=2)
        x = torch.randn(2, 3, 11)
        out = conv(x)
        assert out.shape == (2, 5, 11)


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
