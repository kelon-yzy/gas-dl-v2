"""P1 数据通路：GPU dequant/normalize 与 CPU 组装数值对齐。"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader

from tv3.dl.cli import build_parser as build_dl_cli_parser, run as run_dl_cli
from tv3.dl.data.dataset import V4BenchmarkDataset
from tv3.dl.data.waveform_preprocess import WaveformDevicePreprocessor
from tv3.dl.training.trainer import Trainer
from tv3.sim.generation.tunnel_ventilation import (
    TunnelVentilationBenchmarkGenerationSpec,
    generate_tunnel_ventilation_benchmark_dataset,
)


def _make_tv3_smoke_dataset(tmp_path: Path, slug: str = "tv3-gpu-pre-smoke", sequences: int = 12) -> Path:
    generate_tunnel_ventilation_benchmark_dataset(
        tmp_path,
        TunnelVentilationBenchmarkGenerationSpec(
            dataset_slug=slug,
            sequence_count=sequences,
            seed=20260704,
            timesteps=16,
            storage="npz",
            optical_absorption_backend="empirical_v1",
            workers=1,
        ),
    )
    return tmp_path / slug


class TestWaveformDevicePreprocessParity:
    def test_gpu_preprocess_matches_cpu_assembled_sample(self, tmp_path: Path):
        dataset_dir = _make_tv3_smoke_dataset(tmp_path)
        common = dict(
            dataset_dir=dataset_dir,
            split="train",
            modalities=("slow", "ultrasonic"),
            dequantize_waveforms=True,
            normalize_waveforms=True,
            waveform_stats_features=("log_std", "log_max_abs"),
            lazy=True,
        )
        cpu_ds = V4BenchmarkDataset(**common, waveform_preprocess="cpu")
        gpu_ds = V4BenchmarkDataset(**common, waveform_preprocess="gpu")
        cpu_x, cpu_y = cpu_ds[0]
        raw, gpu_y = gpu_ds[0]
        assert torch.allclose(cpu_y, gpu_y)
        assert isinstance(raw, dict)
        assert "x" not in raw
        assert raw["ultrasonic"].dtype == torch.int16
        preprocessor = WaveformDevicePreprocessor(
            modalities=("slow", "ultrasonic"),
            waveform_stats_features=("log_std", "log_max_abs"),
            normalize_waveforms=True,
            input_format="NTC",
        )
        batch = {key: value.unsqueeze(0) for key, value in raw.items() if key != "aux_targets"}
        assembled = preprocessor(batch).squeeze(0)
        assert assembled.shape == cpu_x.shape
        assert torch.allclose(assembled, cpu_x, atol=1e-5, rtol=1e-5)

    def test_gpu_path_rejects_window(self, tmp_path: Path):
        dataset_dir = _make_tv3_smoke_dataset(tmp_path, slug="tv3-gpu-window")
        with pytest.raises(ValueError, match="does not support window"):
            V4BenchmarkDataset(
                dataset_dir,
                split="train",
                modalities=("ultrasonic",),
                dequantize_waveforms=True,
                waveform_preprocess="gpu",
                window={"kind": "early", "value": 0.5},
            )

    def test_trainer_unpacks_raw_batch_with_input_preprocess(self, tmp_path: Path):
        dataset_dir = _make_tv3_smoke_dataset(tmp_path, slug="tv3-gpu-trainer")
        dataset = V4BenchmarkDataset(
            dataset_dir,
            split="train",
            modalities=("slow", "ultrasonic"),
            dequantize_waveforms=True,
            normalize_waveforms=True,
            waveform_stats_features=("log_std", "log_max_abs"),
            waveform_preprocess="gpu",
        )
        loader = DataLoader(dataset, batch_size=2, shuffle=False)
        batch = next(iter(loader))

        class _MeanModel(torch.nn.Module):
            def __init__(self, channels: int):
                super().__init__()
                self.proj = torch.nn.Linear(channels, 3)

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                return self.proj(x.mean(dim=1))

        model = _MeanModel(dataset.assembled_channel_count())
        trainer = Trainer(
            model=model,
            optimizer=torch.optim.AdamW(model.parameters(), lr=1e-3),
            loss_fn=torch.nn.MSELoss(),
            device="cpu",
            input_preprocess=WaveformDevicePreprocessor(
                modalities=("slow", "ultrasonic"),
                waveform_stats_features=("log_std", "log_max_abs"),
                normalize_waveforms=True,
            ),
        )
        x, y, kwargs, aux = Trainer._unpack_batch(
            batch,
            device=torch.device("cpu"),
            input_preprocess=trainer.input_preprocess,
        )
        assert x.ndim == 3
        assert x.shape[0] == 2
        assert x.shape[-1] == dataset.assembled_channel_count()
        assert y.shape[0] == 2
        assert kwargs == {}
        assert aux is None
        pred = trainer.model(x)
        assert pred.shape == (2, 3)

    def test_cli_gpu_preprocess_end_to_end(self, tmp_path: Path):
        dataset_dir = _make_tv3_smoke_dataset(tmp_path, slug="tv3-gpu-cli", sequences=16)
        output_dir = tmp_path / "runs" / "tv3-gpu-cli"
        parser = build_dl_cli_parser()
        args = parser.parse_args(
            [
                "--dataset-dir",
                str(dataset_dir),
                "--output-dir",
                str(output_dir),
                "--model",
                "cnn1d_tcn_fusion",
                "--model-kwargs",
                (
                    '{"output_mode":"raw3","slow_channels":7,"ultrasonic_channels":5000,'
                    '"fiber_mic_channels":0,"waveform_embedding_dim":8,"waveform_adc_scale":1.0,'
                    '"acoustic_channels":[4],"acoustic_kernel_size":3,"slow_hidden_dim":8,'
                    '"slow_embedding_dim":8,"tcn_channels":[4],"tcn_kernel_size":3,'
                    '"shared_hidden_dims":[8,4]}'
                ),
                "--modalities",
                "slow,ultrasonic",
                "--dequantize-waveforms",
                "--normalize-waveforms",
                "--waveform-preprocess",
                "gpu",
                "--waveform-stats-features",
                "log_std,log_max_abs",
                "--loss",
                "mse",
                "--epochs",
                "1",
                "--batch-size",
                "2",
                "--num-workers",
                "0",
                "--device",
                "cpu",
                "--eval-splits",
                "val",
            ]
        )
        payload = run_dl_cli(args)
        assert payload["waveform_preprocess"] == "gpu"
        assert payload["model_config"]["in_channels"] == 5009
        assert payload["model_config"]["slow_channels"] == 9
        assert payload["evaluations"]["val"]["metrics"] is not None
        run_config = (output_dir / "run_config.json").read_text(encoding="utf-8")
        assert '"waveform_preprocess": "gpu"' in run_config
