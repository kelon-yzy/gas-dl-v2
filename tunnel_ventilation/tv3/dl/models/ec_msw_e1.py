from __future__ import annotations

from typing import Sequence

import torch
from torch import nn
import torch.nn.functional as F

from tv3.dl.models.base import BaseRegressor


class PositionSensitiveStatisticsPool(nn.Module):
    """保留绝对采样位置的一阶统计，而不是只做平移不变全局池化。"""

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        if features.ndim != 3:
            raise ValueError(f"features must be shaped (N, C, L), got {tuple(features.shape)}")
        positions = torch.linspace(
            0.0,
            1.0,
            features.shape[-1],
            device=features.device,
            dtype=features.dtype,
        )
        weights = features.abs() + torch.finfo(features.dtype).eps
        centroid = (weights * positions).sum(dim=-1) / weights.sum(dim=-1)
        return torch.cat(
            [features.mean(dim=-1), features.amax(dim=-1), centroid],
            dim=-1,
        )


class PositionSensitiveMultiScaleEncoder(nn.Module):
    """共享 stem 后用多个感受野编码单帧波形，并显式输出位置统计。"""

    def __init__(
        self,
        waveform_length: int,
        embedding_dim: int = 64,
        stem_channels: int = 8,
        branch_channels: int = 16,
        kernel_sizes: Sequence[int] = (5, 15, 31),
        dilations: Sequence[int] = (1, 2, 4),
        downsample_factor: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        if waveform_length < 1:
            raise ValueError("waveform_length must be >= 1")
        if stem_channels < 1 or branch_channels < 1 or embedding_dim < 1:
            raise ValueError("encoder channel counts and embedding_dim must be >= 1")
        if len(kernel_sizes) != len(dilations) or not kernel_sizes:
            raise ValueError("kernel_sizes and dilations must have equal non-zero length")
        if any(kernel < 3 or kernel % 2 == 0 for kernel in kernel_sizes):
            raise ValueError("kernel_sizes must contain odd values >= 3")
        if any(dilation < 1 for dilation in dilations):
            raise ValueError("dilations must be >= 1")
        if downsample_factor < 1:
            raise ValueError("downsample_factor must be >= 1")

        self.waveform_length = waveform_length
        self.stem = nn.Sequential(
            nn.Conv1d(1, stem_channels, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm1d(stem_channels),
            nn.GELU(),
        )
        self.branches = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv1d(
                        stem_channels,
                        branch_channels,
                        kernel_size=kernel,
                        dilation=dilation,
                        padding=dilation * (kernel - 1) // 2,
                        bias=False,
                    ),
                    nn.BatchNorm1d(branch_channels),
                    nn.GELU(),
                )
                for kernel, dilation in zip(kernel_sizes, dilations, strict=True)
            ]
        )
        self.downsample_factor = downsample_factor
        self.pool = PositionSensitiveStatisticsPool()
        pooled_dim = len(kernel_sizes) * branch_channels * 3 + 1
        self.projection = nn.Sequential(
            nn.Linear(pooled_dim, embedding_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        if waveform.ndim != 3:
            raise ValueError(f"waveform must be shaped (B, T, L), got {tuple(waveform.shape)}")
        if waveform.shape[-1] != self.waveform_length:
            raise ValueError(f"Expected waveform length {self.waveform_length}, got {waveform.shape[-1]}")

        batch_size, timesteps, waveform_length = waveform.shape
        flat = waveform.reshape(batch_size * timesteps, 1, waveform_length).float()
        shared = self.stem(flat)
        branch_summaries = []
        for branch in self.branches:
            branch_features = branch(shared)
            if self.downsample_factor > 1:
                branch_features = F.avg_pool1d(
                    branch_features,
                    kernel_size=self.downsample_factor,
                    stride=self.downsample_factor,
                )
            branch_summaries.append(self.pool(branch_features))
        log_amplitude = torch.log1p(flat.abs().mean(dim=-1))
        embedding = self.projection(torch.cat([*branch_summaries, log_amplitude], dim=-1))
        return embedding.reshape(batch_size, timesteps, -1)


class ECMSWE1Regressor(BaseRegressor):
    """E1：位置敏感多尺度波形编码器、固定时序聚合和 raw3 小型头。"""

    input_format = "NTC"

    def __init__(
        self,
        in_channels: int = 5009,
        out_dim: int = 3,
        slow_channels: int = 9,
        ultrasonic_channels: int = 5000,
        fiber_mic_channels: int = 0,
        waveform_embedding_dim: int = 64,
        slow_embedding_dim: int = 32,
        stem_channels: int = 8,
        branch_channels: int = 16,
        kernel_sizes: Sequence[int] = (5, 15, 31),
        dilations: Sequence[int] = (1, 2, 4),
        downsample_factor: int = 4,
        dropout: float = 0.1,
        head_hidden_dim: int = 64,
        raw_output_prior: Sequence[float] | None = None,
        output_mode: str = "raw3",
    ):
        if output_mode != "raw3" or out_dim != 3:
            raise ValueError("ECMSWE1Regressor requires output_mode='raw3' and out_dim=3")
        if fiber_mic_channels != 0:
            raise ValueError("ECMSWE1Regressor E1 does not support fiber_mic")
        expected_channels = slow_channels + ultrasonic_channels
        if in_channels != expected_channels:
            raise ValueError(
                f"in_channels={in_channels} does not match slow+ultrasonic channels={expected_channels}"
            )
        if slow_channels < 1 or slow_embedding_dim < 1 or head_hidden_dim < 1:
            raise ValueError("slow and head dimensions must be >= 1")

        super().__init__(out_dim=out_dim)
        self.slow_channels = slow_channels
        self.ultrasonic_channels = ultrasonic_channels
        self.waveform_embedding_dim = waveform_embedding_dim
        self.waveform_encoder = PositionSensitiveMultiScaleEncoder(
            waveform_length=ultrasonic_channels,
            embedding_dim=waveform_embedding_dim,
            stem_channels=stem_channels,
            branch_channels=branch_channels,
            kernel_sizes=kernel_sizes,
            dilations=dilations,
            downsample_factor=downsample_factor,
            dropout=dropout,
        )
        self.slow_encoder = nn.Sequential(
            nn.Linear(slow_channels, slow_embedding_dim),
            nn.GELU(),
        )
        frame_dim = waveform_embedding_dim + slow_embedding_dim
        self.frame_norm = nn.LayerNorm(frame_dim)
        self.head = nn.Sequential(
            nn.Linear(frame_dim * 3, head_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(head_hidden_dim, out_dim),
        )
        self.apply(self._init_weights)
        if raw_output_prior is not None:
            self._init_output_prior(raw_output_prior)

    def _init_output_prior(self, prior: Sequence[float]) -> None:
        if len(prior) != 3:
            raise ValueError("raw_output_prior must contain 3 finite values")
        values = torch.tensor(prior, dtype=self.head[-1].bias.dtype)
        if not bool(torch.isfinite(values).all()):
            raise ValueError("raw_output_prior must contain 3 finite values")
        with torch.no_grad():
            self.head[-1].bias.copy_(values)

    def _split_input(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if x.ndim != 3:
            raise ValueError(f"x must be shaped (B, T, C), got {tuple(x.shape)}")
        expected_channels = self.slow_channels + self.ultrasonic_channels
        if x.shape[-1] != expected_channels:
            raise ValueError(f"Expected {expected_channels} input channels, got {x.shape[-1]}")
        return x[:, :, : self.slow_channels], x[:, :, self.slow_channels :]

    def encode_frames(self, x: torch.Tensor) -> torch.Tensor:
        """返回不含 slow 或输出头信息的逐帧 waveform embedding。"""
        _slow, waveform = self._split_input(x)
        return self.waveform_encoder(waveform)

    def encode_sequence(
        self,
        x: torch.Tensor,
        *,
        frame_embeddings: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """返回供冻结线性探针使用的固定 sequence embedding。"""
        slow, _waveform = self._split_input(x)
        if frame_embeddings is None:
            frame_embeddings = self.encode_frames(x)
        expected_shape = (x.shape[0], x.shape[1], self.waveform_embedding_dim)
        if frame_embeddings.shape != expected_shape:
            raise ValueError(
                f"frame_embeddings must be shaped {expected_shape}, got {tuple(frame_embeddings.shape)}"
            )
        frames = self.frame_norm(
            torch.cat([frame_embeddings, self.slow_encoder(slow.float())], dim=-1)
        )
        return torch.cat(
            [frames[:, -1, :], frames.mean(dim=1), frames.amax(dim=1)],
            dim=-1,
        )

    def forward(self, x: torch.Tensor, **kwargs: object) -> torch.Tensor:
        return self.head(self.encode_sequence(x))
