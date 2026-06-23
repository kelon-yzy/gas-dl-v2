from __future__ import annotations

from typing import Sequence

import torch
from torch import nn

from dl.models.base import BaseRegressor
from dl.models.cnn1d_tcn_fusion import (
    DeepAcousticEncoder1D,
    GasCoordinateHead,
    GasHeadNormalize,
    SlowFeatureEncoder,
)
from dl.models.tcn import TemporalBlock


class WindowedFusionEncoder(nn.Module):
    """Encode one NTC multimodal window into a fixed-size feature vector."""

    input_format = "NTC"

    def __init__(
        self,
        slow_channels: int = 8,
        ultrasonic_channels: int = 1000,
        fiber_mic_channels: int = 2000,
        waveform_embedding_dim: int = 64,
        waveform_int16_scale: float = 32767.0,
        acoustic_channels: Sequence[int] = (16, 32, 64, 64),
        acoustic_kernel_size: int = 7,
        acoustic_dropout: float = 0.15,
        slow_hidden_dim: int = 32,
        slow_embedding_dim: int = 64,
        tcn_channels: Sequence[int] = (64, 64, 64),
        tcn_kernel_size: int = 3,
        tcn_dropout: float = 0.25,
    ):
        super().__init__()
        if not tcn_channels:
            raise ValueError("tcn_channels must contain at least one block")

        self.slow_channels = slow_channels
        self.ultrasonic_channels = ultrasonic_channels
        self.fiber_mic_channels = fiber_mic_channels

        self.ultrasonic_encoder = DeepAcousticEncoder1D(
            ultrasonic_channels,
            embedding_dim=waveform_embedding_dim,
            waveform_int16_scale=waveform_int16_scale,
            channels=acoustic_channels,
            kernel_size=acoustic_kernel_size,
            dropout=acoustic_dropout,
        )
        self.fiber_mic_encoder = DeepAcousticEncoder1D(
            fiber_mic_channels,
            embedding_dim=waveform_embedding_dim,
            waveform_int16_scale=waveform_int16_scale,
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
        self.output_dim = current * 3

    @property
    def expected_channels(self) -> int:
        return self.slow_channels + self.ultrasonic_channels + self.fiber_mic_channels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3:
            raise ValueError(f"x must be shaped (B, T, C), got {tuple(x.shape)}")
        if x.shape[-1] != self.expected_channels:
            raise ValueError(f"Expected {self.expected_channels} input channels, got {x.shape[-1]}")

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
        features = self.tcn(fused.transpose(1, 2))
        return torch.cat([features[:, :, -1], features.mean(dim=-1), features.amax(dim=-1)], dim=-1)


class PhaseWindowTCNRegressor(BaseRegressor):
    """TCN regressor for true multi-window DL input shaped (B, W, T, C)."""

    input_format = "NTC"

    def __init__(
        self,
        in_channels: int = 3008,
        out_dim: int = 4,
        slow_channels: int = 8,
        ultrasonic_channels: int = 1000,
        fiber_mic_channels: int = 2000,
        window_count: int = 3,
        share_window_encoder: bool = True,
        output_mode: str = "raw4",
        waveform_embedding_dim: int = 64,
        waveform_int16_scale: float = 32767.0,
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
        if out_dim not in {3, 4}:
            raise ValueError("PhaseWindowTCNRegressor requires out_dim=4 for raw percentages or out_dim=3 for transformed targets")
        if window_count < 1:
            raise ValueError("window_count must be >= 1")
        expected_channels = slow_channels + ultrasonic_channels + fiber_mic_channels
        if in_channels != expected_channels:
            raise ValueError(
                f"in_channels={in_channels} does not match slow+ultrasonic+fiber channels={expected_channels}"
            )
        if output_mode not in {"raw4", "softmax100", "gas_head"}:
            raise ValueError("output_mode must be one of ['raw4', 'softmax100', 'gas_head']")
        if output_mode == "softmax100" and out_dim != 4:
            raise ValueError("softmax100 output_mode requires out_dim=4")
        if len(shared_hidden_dims) != 2:
            raise ValueError("shared_hidden_dims must contain exactly two layer widths")
        super().__init__(out_dim=out_dim)

        encoder_kwargs = {
            "slow_channels": slow_channels,
            "ultrasonic_channels": ultrasonic_channels,
            "fiber_mic_channels": fiber_mic_channels,
            "waveform_embedding_dim": waveform_embedding_dim,
            "waveform_int16_scale": waveform_int16_scale,
            "acoustic_channels": acoustic_channels,
            "acoustic_kernel_size": acoustic_kernel_size,
            "acoustic_dropout": acoustic_dropout,
            "slow_hidden_dim": slow_hidden_dim,
            "slow_embedding_dim": slow_embedding_dim,
            "tcn_channels": tcn_channels,
            "tcn_kernel_size": tcn_kernel_size,
            "tcn_dropout": tcn_dropout,
        }
        self.window_count = window_count
        self.share_window_encoder = share_window_encoder
        if share_window_encoder:
            self.window_encoder = WindowedFusionEncoder(**encoder_kwargs)
            encoder_output_dim = self.window_encoder.output_dim
        else:
            self.window_encoders = nn.ModuleList(WindowedFusionEncoder(**encoder_kwargs) for _ in range(window_count))
            encoder_output_dim = self.window_encoders[0].output_dim

        h1, h2 = shared_hidden_dims
        self.shared_head = nn.Sequential(
            nn.Linear(encoder_output_dim * window_count, h1),
            nn.ReLU(),
            nn.Dropout(tcn_dropout),
            nn.Linear(h1, h2),
            nn.ReLU(),
            nn.Dropout(tcn_dropout),
        )
        if output_mode == "softmax100":
            self.output_head = nn.Sequential(nn.Linear(h2, out_dim), nn.Softmax(dim=-1))
        elif output_mode == "gas_head":
            self.output_head = GasHeadNormalize(h2, output_prior=output_prior) if out_dim == 4 else GasCoordinateHead(h2, out_dim)
        else:
            self.output_head = nn.Linear(h2, out_dim)
        self.output_mode = output_mode
        self.apply(self._init_weights)

    @property
    def receptive_field(self) -> int:
        if self.share_window_encoder:
            return self.window_encoder.receptive_field
        return self.window_encoders[0].receptive_field

    def forward(self, x: torch.Tensor, **kwargs: object) -> torch.Tensor:
        if x.ndim != 4:
            raise ValueError(f"x must be shaped (B, W, T, C), got {tuple(x.shape)}")
        if x.shape[1] != self.window_count:
            raise ValueError(f"Expected {self.window_count} windows, got {x.shape[1]}")

        if self.share_window_encoder:
            encoded = [self.window_encoder(x[:, index]) for index in range(self.window_count)]
        else:
            encoded = [encoder(x[:, index]) for index, encoder in enumerate(self.window_encoders)]
        features = self.shared_head(torch.cat(encoded, dim=-1))
        out = self.output_head(features)
        if self.output_mode == "softmax100":
            return out * 100.0
        return out
