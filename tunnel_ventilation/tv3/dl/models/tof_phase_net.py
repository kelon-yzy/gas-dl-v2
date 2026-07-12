from __future__ import annotations

import math
from typing import Sequence

import torch
from torch import nn
import torch.nn.functional as F

from tv3.dl.models.base import BaseRegressor
from tv3.dl.models.tcn import TemporalBlock


class FixedQuadratureFilterBank(nn.Module):
    """固定正弦/余弦 matched filter，生成 I/Q 正交响应。

    使用 Hann 窗调制的 carrier 频率正弦/余弦核对 raw waveform 做 1D 卷积，
    输出 I 和 Q 两路响应。
    """

    def __init__(
        self,
        sample_rate_hz: float = 1_000_000.0,
        carrier_hz: float = 200_000.0,
        filter_cycles: int = 3,
    ):
        super().__init__()
        if sample_rate_hz <= 0.0 or carrier_hz <= 0.0:
            raise ValueError("sample_rate_hz and carrier_hz must be positive")
        if carrier_hz >= sample_rate_hz / 2:
            raise ValueError("carrier_hz must be less than Nyquist frequency")
        if filter_cycles < 1:
            raise ValueError("filter_cycles must be >= 1")

        samples_per_cycle = sample_rate_hz / carrier_hz
        kernel_len = max(3, int(filter_cycles * samples_per_cycle))
        if kernel_len % 2 == 0:
            kernel_len += 1

        t = torch.arange(kernel_len, dtype=torch.float32) - (kernel_len - 1) / 2
        t = t / sample_rate_hz
        omega = 2.0 * math.pi * carrier_hz

        # Hann window
        window = 0.5 * (1.0 - torch.cos(2.0 * math.pi * torch.arange(kernel_len, dtype=torch.float32) / (kernel_len - 1)))

        i_kernel = window * torch.cos(omega * t)
        q_kernel = window * torch.sin(omega * t)
        # (out=2, in=1, kernel_len)
        kernel = torch.stack([i_kernel, q_kernel], dim=0).unsqueeze(1)
        self.register_buffer("kernel", kernel)
        self.padding = kernel_len // 2
        self.register_buffer("_sample_indices", None, persistent=False)

    def forward(self, waveform: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """对 raw waveform 做正交滤波。

        Args:
            waveform: (B, T, L) raw ultrasonic samples.

        Returns:
            (i_response, q_response): 各为 (B, T, L).
        """
        if waveform.ndim != 3:
            raise ValueError(f"waveform must be shaped (B, T, L), got {tuple(waveform.shape)}")
        B, T, L = waveform.shape
        flat = waveform.reshape(B * T, 1, L).float()
        iq = F.conv1d(flat, self.kernel, padding=self.padding)
        i_resp = iq[:, 0:1, :].reshape(B, T, L)
        q_resp = iq[:, 1:2, :].reshape(B, T, L)
        return i_resp, q_resp


class EnvelopeExtractor(nn.Module):
    """从 I/Q 响应计算 envelope：sqrt(I² + Q² + eps)。"""

    def __init__(self, eps: float = 1e-8):
        super().__init__()
        self.eps = eps

    def forward(self, i_response: torch.Tensor, q_response: torch.Tensor) -> torch.Tensor:
        return torch.sqrt(i_response.square() + q_response.square() + self.eps)


class SoftArgmaxLag(nn.Module):
    """温度控制的 softargmax 在 sample 维定位连续峰值位置。

    输出归一化 index ∈ [0, 1]，可换算为 tof_s = index * (L / sample_rate_hz)
    或 tof_samples = index * L。
    """

    def __init__(self, temperature: float = 0.05):
        super().__init__()
        if temperature <= 0.0:
            raise ValueError("softargmax temperature must be > 0")
        self.temperature = temperature

    def forward(self, envelope: torch.Tensor) -> torch.Tensor:
        """对 envelope 做 softargmax。

        Args:
            envelope: (B, T, L) envelope magnitude.

        Returns:
            peak_index: (B, T) 归一化连续 index ∈ [0, 1].
        """
        B, T, L = envelope.shape
        flat = envelope.reshape(B * T, L)
        weights = F.softmax(flat / self.temperature, dim=-1)
        indices = torch.linspace(0.0, 1.0, L, device=envelope.device, dtype=envelope.dtype)
        lag = (weights * indices).sum(dim=-1)
        return lag.reshape(B, T)


class PeakShapeFeatures(nn.Module):
    """从 envelope 和 softargmax index 提取峰值形态特征。

    输出 per-timestep 特征：
    - peak_value: 峰值处的 envelope 值
    - peak_sharpness: 峰值处的负二阶差分（曲率代理）
    - local_energy: envelope 的局部总能量
    - peak_ratio: 峰值与 envelope 均值的比值
    """

    def __init__(self, local_window: int = 21):
        super().__init__()
        if local_window < 3 or local_window % 2 == 0:
            raise ValueError("local_window must be odd and >= 3")
        self.local_window = local_window

    def forward(self, envelope: torch.Tensor, peak_index_norm: torch.Tensor) -> torch.Tensor:
        """提取峰值形态特征。

        Args:
            envelope: (B, T, L).
            peak_index_norm: (B, T) 归一化 peak index ∈ [0, 1].

        Returns:
            features: (B, T, 4) — [peak_value, peak_sharpness, local_energy, peak_ratio].
        """
        B, T, L = envelope.shape
        device = envelope.device

        peak_idx = (peak_index_norm * (L - 1)).clamp(0, L - 1)
        idx_floor = peak_idx.floor().long()
        idx_ceil = (idx_floor + 1).clamp(0, L - 1)
        alpha = (peak_idx - idx_floor.float()).unsqueeze(-1)

        flat = envelope.reshape(B * T, L)
        idx_floor_flat = idx_floor.reshape(-1)
        idx_ceil_flat = idx_ceil.reshape(-1)
        batch_idx = torch.arange(B * T, device=device)

        peak_value = (
            (1.0 - alpha.reshape(-1)) * flat[batch_idx, idx_floor_flat]
            + alpha.reshape(-1) * flat[batch_idx, idx_ceil_flat]
        ).reshape(B, T, 1)

        # peak sharpness: 负二阶中心差分
        half = self.local_window // 2
        idx0 = idx_floor_flat
        idx_left = (idx0 - 1).clamp(0, L - 1)
        idx_right = (idx0 + 1).clamp(0, L - 1)
        sharpness = -(flat[batch_idx, idx_left] - 2.0 * flat[batch_idx, idx0] + flat[batch_idx, idx_right])
        sharpness = F.softplus(sharpness.reshape(B, T, 1))

        # local energy
        idx_start = (idx0 - half).clamp(0, L - 1)
        idx_end = (idx0 + half).clamp(0, L - 1)
        energy = torch.zeros(B * T, device=device)
        # 简单近似：取峰值附近窗口的总能量
        for offset in range(-half, half + 1):
            idx_cur = (idx0 + offset).clamp(0, L - 1)
            energy = energy + flat[batch_idx, idx_cur]
        local_energy = (energy / (2 * half + 1)).reshape(B, T, 1)

        # peak ratio
        mean_env = envelope.mean(dim=-1, keepdim=True)
        peak_ratio = (peak_value / (mean_env + 1e-8)).clamp(0, 100.0)

        return torch.cat([peak_value, sharpness, local_energy, peak_ratio], dim=-1)


class AcousticFrontend(nn.Module):
    """可微 TOF/Phase 前端：滤波 → envelope → softargmax → 形态特征 → 嵌入。

    将 raw ultrasonic waveform 每帧映射为固定维度的 acoustic feature vector。
    """

    def __init__(
        self,
        sample_rate_hz: float = 1_000_000.0,
        carrier_hz: float = 200_000.0,
        filter_cycles: int = 3,
        softargmax_temperature: float = 0.05,
        peak_local_window: int = 21,
        acoustic_feature_dim: int = 32,
    ):
        super().__init__()
        self.filter_bank = FixedQuadratureFilterBank(
            sample_rate_hz=sample_rate_hz,
            carrier_hz=carrier_hz,
            filter_cycles=filter_cycles,
        )
        self.envelope = EnvelopeExtractor()
        self.softargmax = SoftArgmaxLag(temperature=softargmax_temperature)
        self.peak_features = PeakShapeFeatures(local_window=peak_local_window)
        self.sample_rate_hz = sample_rate_hz
        self.waveform_length: int | None = None

        raw_feature_dim = 4 + 1  # peak shape (4) + softargmax index (1)
        self.proj = nn.Sequential(
            nn.Linear(raw_feature_dim, acoustic_feature_dim),
            nn.ReLU(),
        )

    def forward(self, waveform: torch.Tensor) -> dict[str, torch.Tensor]:
        """前向传播。

        Args:
            waveform: (B, T, L) raw ultrasonic samples.

        Returns:
            dict with:
                features: (B, T, acoustic_feature_dim) per-timestep features.
                peak_index_norm: (B, T) normalized peak index ∈ [0, 1].
                peak_index_samples: (B, T) peak index in samples.
                tof_s: (B, T) TOF in seconds.
                peak_sharpness: (B, T) peak sharpness.
        """
        L = waveform.shape[-1]
        i_resp, q_resp = self.filter_bank(waveform)
        env = self.envelope(i_resp, q_resp)
        peak_norm = self.softargmax(env)
        shape = self.peak_features(env, peak_norm)

        raw_features = torch.cat([peak_norm.unsqueeze(-1), shape], dim=-1)
        features = self.proj(raw_features)

        peak_samples = peak_norm * (L - 1)
        tof_s = peak_samples / self.sample_rate_hz

        return {
            "features": features,
            "peak_index_norm": peak_norm,
            "peak_index_samples": peak_samples,
            "tof_s": tof_s,
            "peak_sharpness": shape[..., 1],
        }


class TOFPhaseNetRegressor(BaseRegressor):
    """D2 可微 TOF-PhaseNet 回归器。

    输入: (B, T, C) NTC 格式，C = slow_channels + ultrasonic_channels.
    输出: {"prediction": (B, 3), "aux": {"tof_s": ..., "peak_index": ..., "peak_index_samples": ...}}.

    结构：AcousticFrontend + SlowEncoder → concat → LayerNorm → TCN → pooling → head.
    """

    input_format = "NTC"

    def __init__(
        self,
        in_channels: int = 5009,
        out_dim: int = 3,
        slow_channels: int = 9,
        ultrasonic_channels: int = 5000,
        fiber_mic_channels: int = 0,
        sample_rate_hz: float = 1_000_000.0,
        carrier_hz: float = 200_000.0,
        softargmax_temperature: float = 0.05,
        acoustic_feature_dim: int = 32,
        slow_embedding_dim: int = 32,
        tcn_channels: Sequence[int] = (64, 64, 64),
        tcn_kernel_size: int = 3,
        tcn_dropout: float = 0.2,
        shared_hidden_dims: Sequence[int] = (128, 64),
        raw_output_prior: Sequence[float] | None = None,
        output_mode: str = "raw3",
    ):
        if output_mode != "raw3":
            raise ValueError("TOFPhaseNetRegressor requires output_mode='raw3'")
        if out_dim != 3:
            raise ValueError("TOFPhaseNetRegressor requires out_dim=3")
        expected_channels = slow_channels + ultrasonic_channels + fiber_mic_channels
        if in_channels != expected_channels:
            raise ValueError(
                f"in_channels={in_channels} does not match slow+ultrasonic+fiber channels={expected_channels}"
            )
        if fiber_mic_channels != 0:
            raise ValueError("TOFPhaseNetRegressor does not support fiber_mic yet (plan invariance #3)")
        if not tcn_channels:
            raise ValueError("tcn_channels must contain at least one block")
        if len(shared_hidden_dims) != 2:
            raise ValueError("shared_hidden_dims must contain exactly two values")

        super().__init__()
        self.slow_channels = slow_channels
        self.ultrasonic_channels = ultrasonic_channels
        self.acoustic_feature_dim = acoustic_feature_dim
        self.slow_embedding_dim = slow_embedding_dim

        self.acoustic_frontend = AcousticFrontend(
            sample_rate_hz=sample_rate_hz,
            carrier_hz=carrier_hz,
            softargmax_temperature=softargmax_temperature,
            acoustic_feature_dim=acoustic_feature_dim,
        )

        self.slow_encoder = nn.Sequential(
            nn.Linear(slow_channels, slow_embedding_dim),
            nn.GELU(),
        )

        fusion_dim = acoustic_feature_dim + slow_embedding_dim
        self.fusion_norm = nn.LayerNorm(fusion_dim)

        tcn_input_dim = fusion_dim
        tcn_blocks: list[nn.Module] = []
        for ch in tcn_channels:
            tcn_blocks.append(
                TemporalBlock(
                    tcn_input_dim,
                    ch,
                    kernel_size=tcn_kernel_size,
                    dilation=1,
                    dropout=tcn_dropout,
                )
            )
            tcn_input_dim = ch
        self.tcn = nn.Sequential(*tcn_blocks)
        self.tcn_out_dim = tcn_channels[-1]

        pooled_dim = self.tcn_out_dim * 3  # last + mean + max
        self.shared = nn.Sequential(
            nn.Linear(pooled_dim, shared_hidden_dims[0]),
            nn.ReLU(),
            nn.Dropout(tcn_dropout),
            nn.Linear(shared_hidden_dims[0], shared_hidden_dims[1]),
            nn.ReLU(),
            nn.Dropout(tcn_dropout),
        )

        self.component_head = nn.Linear(shared_hidden_dims[1], out_dim)
        self._init_component_head(raw_output_prior)

    def _init_component_head(self, prior: Sequence[float] | None) -> None:
        if prior is not None:
            if len(prior) != 3:
                raise ValueError("raw_output_prior must contain 3 values for tv3")
            p = torch.tensor(prior, dtype=torch.float32)
            with torch.no_grad():
                self.component_head.bias.copy_(p)

    def forward(self, x: torch.Tensor, **kwargs: object) -> dict[str, torch.Tensor]:
        """前向传播。

        Args:
            x: (B, T, C) NTC 输入，C = slow_channels + ultrasonic_channels.

        Returns:
            {"prediction": (B, 3), "aux": {"tof_s": (B, T), "peak_index": (B, T), "peak_index_samples": (B, T)}}.
        """
        if x.ndim != 3:
            raise ValueError(f"TOFPhaseNetRegressor expects NTC input (B, T, C), got {tuple(x.shape)}")

        slow = x[:, :, : self.slow_channels]
        ultrasonic = x[:, :, self.slow_channels : self.slow_channels + self.ultrasonic_channels]

        frontend_out = self.acoustic_frontend(ultrasonic)
        acoustic_feat = frontend_out["features"]  # (B, T, F)

        slow_emb = self.slow_encoder(slow)  # (B, T, S)
        fused = self.fusion_norm(torch.cat([acoustic_feat, slow_emb], dim=-1))

        # TCN expects NCT format: (B, T, C) -> (B, C, T)
        tcn_in = fused.transpose(1, 2)
        tcn_out = self.tcn(tcn_in).transpose(1, 2)  # back to (B, T, C)

        last = tcn_out[:, -1, :]
        mean = tcn_out.mean(dim=1)
        mx = tcn_out.max(dim=1).values
        pooled = torch.cat([last, mean, mx], dim=-1)

        shared_feat = self.shared(pooled)

        prediction = self.component_head(shared_feat)

        return {
            "prediction": prediction,
            "aux": {
                "tof_s": frontend_out["tof_s"],
                "peak_index": frontend_out["peak_index_norm"],
                "peak_index_samples": frontend_out["peak_index_samples"],
                "peak_sharpness": frontend_out["peak_sharpness"],
            },
        }
