from __future__ import annotations

import math
from typing import Sequence

import torch
from torch import nn

from dl.models.base import BaseRegressor
from dl.models.tcn import TemporalBlock


class DeepAcousticEncoder1D(nn.Module):
    """Per-timestep waveform encoder used by the CNN1D-TCN fusion model."""

    def __init__(
        self,
        waveform_length: int,
        embedding_dim: int = 64,
        channels: Sequence[int] = (16, 32, 64, 64),
        kernel_size: int = 7,
        dropout: float = 0.15,
        waveform_int16_scale: float = 32767.0,
    ):
        super().__init__()
        if waveform_length < 1:
            raise ValueError("waveform_length must be >= 1")
        if not channels:
            raise ValueError("channels must contain at least one layer")
        if kernel_size < 2:
            raise ValueError("kernel_size must be >= 2")
        if waveform_int16_scale <= 0.0:
            raise ValueError("waveform_int16_scale must be > 0")

        self.waveform_length = waveform_length
        self.waveform_int16_scale = waveform_int16_scale

        layers: list[nn.Module] = []
        current = 1
        padding = kernel_size // 2
        for idx, hidden in enumerate(channels):
            stride = 1 if idx == len(channels) - 1 else 2
            layers.extend(
                [
                    nn.Conv1d(current, hidden, kernel_size=kernel_size, stride=stride, padding=padding, bias=False),
                    nn.BatchNorm1d(hidden),
                    nn.ReLU(),
                ]
            )
            if dropout > 0.0:
                layers.append(nn.Dropout(dropout))
            current = hidden
        self.encoder = nn.Sequential(*layers)
        self.proj = nn.Linear(current * 2 + 1, embedding_dim)

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        if waveform.ndim != 3:
            raise ValueError(f"waveform must be shaped (B, T, L), got {tuple(waveform.shape)}")
        if waveform.shape[-1] != self.waveform_length:
            raise ValueError(f"Expected waveform length {self.waveform_length}, got {waveform.shape[-1]}")

        batch_size, timesteps, waveform_length = waveform.shape
        # V4BenchmarkDataset exposes waveform values inside the fused tensor; scale arrays are not part of that input.
        flat = waveform.reshape(batch_size * timesteps, 1, waveform_length).float() / self.waveform_int16_scale
        encoded = self.encoder(flat)
        avg = nn.functional.adaptive_avg_pool1d(encoded, 1).squeeze(-1)
        mx = nn.functional.adaptive_max_pool1d(encoded, 1).squeeze(-1)
        log_amplitude = torch.log1p(flat.abs().mean(dim=-1))
        embedding = self.proj(torch.cat([avg, mx, log_amplitude], dim=-1))
        return embedding.reshape(batch_size, timesteps, -1)


class SlowFeatureEncoder(nn.Module):
    """Per-timestep slow-variable encoder: slow_channels -> hidden -> embedding."""

    def __init__(self, slow_channels: int = 8, hidden_dim: int = 32, embedding_dim: int = 64):
        super().__init__()
        if slow_channels < 1:
            raise ValueError("slow_channels must be >= 1")
        self.net = nn.Sequential(
            nn.Linear(slow_channels, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, embedding_dim),
        )

    def forward(self, slow: torch.Tensor) -> torch.Tensor:
        if slow.ndim != 3:
            raise ValueError(f"slow must be shaped (B, T, C), got {tuple(slow.shape)}")
        return self.net(slow.float())


class GasHeadNormalize(nn.Module):
    """Bounded simplex output head for [H2, CH4, CO2, N2] percentages."""

    def __init__(
        self,
        in_features: int,
        output_prior: Sequence[float] = (9.288469, 75.755157, 4.994778, 9.961745),
    ):
        super().__init__()
        if len(output_prior) != 4:
            raise ValueError("output_prior must contain four component percentages")
        if any(value <= 0.0 for value in output_prior):
            raise ValueError("output_prior values must be positive")
        self.linear = nn.Linear(in_features, 4)
        self._init_prior(output_prior)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        raw = self.linear(features)
        free_ratio = torch.softmax(raw[:, :3], dim=-1)
        free_total = 100.0 * torch.sigmoid(raw[:, 3:4])
        free = free_total * free_ratio
        n2 = 100.0 - free.sum(dim=-1, keepdim=True)
        return torch.cat([free, n2], dim=-1)

    def _init_prior(self, output_prior: Sequence[float]) -> None:
        prior = torch.tensor(output_prior, dtype=self.linear.bias.dtype)
        free = prior[:3]
        free_total = float(free.sum().item())
        if not 0.0 < free_total < 100.0:
            raise ValueError("sum of first three output_prior values must be in (0, 100)")
        free_logits = torch.log(free / free_total)
        total_prob = min(max(free_total / 100.0, 1e-6), 1.0 - 1e-6)
        total_logit = math.log(total_prob / (1.0 - total_prob))
        with torch.no_grad():
            self.linear.bias.copy_(torch.cat([free_logits, torch.tensor([total_logit], dtype=prior.dtype)]))


class CNN1DTCNFusionRegressor(BaseRegressor):
    """V4-compatible CNN1D-TCN fusion regressor with bounded simplex output.

    Expected input format is NTC with channels ordered as:
    slow[8] + ultrasonic[1000] + fiber_mic[2000].
    """

    input_format = "NTC"

    def __init__(
        self,
        in_channels: int = 3008,
        out_dim: int = 4,
        slow_channels: int = 8,
        ultrasonic_channels: int = 1000,
        fiber_mic_channels: int = 2000,
        waveform_embedding_dim: int = 64,
        acoustic_channels: Sequence[int] = (16, 32, 64, 64),
        acoustic_kernel_size: int = 7,
        acoustic_dropout: float = 0.15,
        slow_hidden_dim: int = 32,
        slow_embedding_dim: int = 64,
        tcn_channels: Sequence[int] = (64, 64, 64),
        tcn_kernel_size: int = 3,
        tcn_dropout: float = 0.25,
        shared_hidden_dims: Sequence[int] = (128, 64),
        output_prior: Sequence[float] = (9.288469, 75.755157, 4.994778, 9.961745),
    ):
        if out_dim != 4:
            raise ValueError("CNN1DTCNFusionRegressor uses bounded simplex output and requires out_dim=4")
        expected_channels = slow_channels + ultrasonic_channels + fiber_mic_channels
        if in_channels != expected_channels:
            raise ValueError(
                f"in_channels={in_channels} does not match slow+ultrasonic+fiber channels={expected_channels}"
            )
        if not tcn_channels:
            raise ValueError("tcn_channels must contain at least one block")
        if len(shared_hidden_dims) != 2:
            raise ValueError("shared_hidden_dims must contain exactly two layer widths")
        super().__init__(out_dim=out_dim)

        self.slow_channels = slow_channels
        self.ultrasonic_channels = ultrasonic_channels
        self.fiber_mic_channels = fiber_mic_channels

        self.ultrasonic_encoder = DeepAcousticEncoder1D(
            ultrasonic_channels,
            embedding_dim=waveform_embedding_dim,
            channels=acoustic_channels,
            kernel_size=acoustic_kernel_size,
            dropout=acoustic_dropout,
        )
        self.fiber_mic_encoder = DeepAcousticEncoder1D(
            fiber_mic_channels,
            embedding_dim=waveform_embedding_dim,
            channels=acoustic_channels,
            kernel_size=acoustic_kernel_size,
            dropout=acoustic_dropout,
        )
        self.slow_encoder = SlowFeatureEncoder(
            slow_channels=slow_channels,
            hidden_dim=slow_hidden_dim,
            embedding_dim=slow_embedding_dim,
        )

        fusion_channels = waveform_embedding_dim * 2 + slow_embedding_dim
        layers: list[nn.Module] = []
        current = fusion_channels
        self.dilations = tuple(2**idx for idx in range(len(tcn_channels)))
        self.receptive_field = 1 + sum(2 * (tcn_kernel_size - 1) * dilation for dilation in self.dilations)
        for hidden, dilation in zip(tcn_channels, self.dilations, strict=True):
            layers.append(TemporalBlock(current, hidden, tcn_kernel_size, dilation=dilation, dropout=tcn_dropout))
            current = hidden
        self.tcn = nn.Sequential(*layers)

        pooled_features = current * 3
        h1, h2 = shared_hidden_dims
        self.shared_head = nn.Sequential(
            nn.Linear(pooled_features, h1),
            nn.ReLU(),
            nn.Dropout(tcn_dropout),
            nn.Linear(h1, h2),
            nn.ReLU(),
            nn.Dropout(tcn_dropout),
        )
        self.gas_head = GasHeadNormalize(h2, output_prior=output_prior)

    def forward(self, x: torch.Tensor, **kwargs: object) -> torch.Tensor:
        if x.ndim != 3:
            raise ValueError(f"x must be shaped (B, T, C), got {tuple(x.shape)}")
        expected_channels = self.slow_channels + self.ultrasonic_channels + self.fiber_mic_channels
        if x.shape[-1] != expected_channels:
            raise ValueError(f"Expected {expected_channels} input channels, got {x.shape[-1]}")

        slow_end = self.slow_channels
        ultrasonic_end = slow_end + self.ultrasonic_channels
        slow = x[:, :, :slow_end]
        ultrasonic = x[:, :, slow_end:ultrasonic_end]
        fiber_mic = x[:, :, ultrasonic_end:]

        fused = torch.cat(
            [
                self.ultrasonic_encoder(ultrasonic),
                self.fiber_mic_encoder(fiber_mic),
                self.slow_encoder(slow),
            ],
            dim=-1,
        )
        feats = self.tcn(fused.transpose(1, 2))
        pooled = torch.cat([feats[:, :, -1], feats.mean(dim=-1), feats.amax(dim=-1)], dim=-1)
        return self.gas_head(self.shared_head(pooled))
