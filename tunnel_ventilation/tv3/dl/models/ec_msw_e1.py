from __future__ import annotations

import hashlib
from typing import Sequence

import torch
from torch import nn
import torch.nn.functional as F

from tv3.dl.models.base import BaseRegressor


class MatchedFilterPeakCoordinate(nn.Module):
    """用冻结的 train-only 模板从当前波形直接恢复绝对峰位坐标。"""

    def __init__(
        self,
        waveform_length: int,
        template: Sequence[float],
        *,
        expected_digest: str | None = None,
    ):
        super().__init__()
        values = torch.tensor(template, dtype=torch.float32)
        if values.ndim != 1 or values.numel() < 3:
            raise ValueError(
                "peak_coordinate_template must be a 1D sequence with at least 3 values"
            )
        if values.numel() > waveform_length:
            raise ValueError("peak_coordinate_template must not exceed waveform_length")
        if not bool(torch.isfinite(values).all()):
            raise ValueError("peak_coordinate_template must contain finite values")
        digest = hashlib.sha256(
            values.numpy().astype("<f4", copy=False).tobytes()
        ).hexdigest()
        if expected_digest is not None and digest != expected_digest:
            raise ValueError("peak_coordinate_template_digest does not match template values")

        centered = values - values.mean()
        norm = torch.linalg.vector_norm(centered)
        if float(norm.item()) == 0.0:
            raise ValueError("peak_coordinate_template must have non-zero centered energy")
        self.waveform_length = waveform_length
        self.template_peak_offset = int(torch.argmax(values.abs()).item())
        self.template_digest = digest
        self.register_buffer("template", (centered / norm).reshape(1, 1, -1))

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        if waveform.ndim != 3 or waveform.shape[1] != 1:
            raise ValueError(f"waveform must be shaped (N, 1, L), got {tuple(waveform.shape)}")
        if waveform.shape[-1] != self.waveform_length:
            raise ValueError(
                f"Expected waveform length {self.waveform_length}, got {waveform.shape[-1]}"
            )
        with torch.autocast(device_type=waveform.device.type, enabled=False):
            correlation = F.conv1d(waveform.float(), self.template)
        peak_index = correlation.argmax(dim=-1).float() + self.template_peak_offset
        return peak_index / float(self.waveform_length - 1)


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
        peak_coordinate_template: Sequence[float] | None = None,
        peak_coordinate_template_digest: str | None = None,
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
        self.peak_coordinate = (
            None
            if peak_coordinate_template is None
            else MatchedFilterPeakCoordinate(
                waveform_length,
                peak_coordinate_template,
                expected_digest=peak_coordinate_template_digest,
            )
        )
        learned_embedding_dim = embedding_dim - int(self.peak_coordinate is not None)
        if learned_embedding_dim < 1:
            raise ValueError(
                "embedding_dim must be >= 2 when peak_coordinate_template is configured"
            )
        self.projection = nn.Sequential(
            nn.Linear(pooled_dim, learned_embedding_dim),
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
        learned_embedding = self.projection(torch.cat([*branch_summaries, log_amplitude], dim=-1))
        embedding = (
            learned_embedding
            if self.peak_coordinate is None
            else torch.cat([self.peak_coordinate(flat), learned_embedding], dim=-1)
        )
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
        peak_coordinate_template: Sequence[float] | None = None,
        peak_coordinate_template_digest: str | None = None,
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
        self.has_peak_coordinate = peak_coordinate_template is not None
        self.waveform_encoder = PositionSensitiveMultiScaleEncoder(
            waveform_length=ultrasonic_channels,
            embedding_dim=waveform_embedding_dim,
            stem_channels=stem_channels,
            branch_channels=branch_channels,
            kernel_sizes=kernel_sizes,
            dilations=dilations,
            downsample_factor=downsample_factor,
            dropout=dropout,
            peak_coordinate_template=peak_coordinate_template,
            peak_coordinate_template_digest=peak_coordinate_template_digest,
        )
        self.slow_encoder = nn.Sequential(
            nn.Linear(slow_channels, slow_embedding_dim),
            nn.GELU(),
        )
        frame_dim = waveform_embedding_dim + slow_embedding_dim
        self.frame_norm = nn.LayerNorm(frame_dim - int(self.has_peak_coordinate))
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
        slow_embeddings = self.slow_encoder(slow.float())
        if self.has_peak_coordinate:
            peak_coordinate = frame_embeddings[:, :, :1]
            learned_frames = frame_embeddings[:, :, 1:]
            normalized = self.frame_norm(torch.cat([learned_frames, slow_embeddings], dim=-1))
            frames = torch.cat([peak_coordinate, normalized], dim=-1)
        else:
            frames = self.frame_norm(torch.cat([frame_embeddings, slow_embeddings], dim=-1))
        return torch.cat(
            [frames[:, -1, :], frames.mean(dim=1), frames.amax(dim=1)],
            dim=-1,
        )

    def forward(self, x: torch.Tensor, **kwargs: object) -> torch.Tensor:
        return self.head(self.encode_sequence(x))
