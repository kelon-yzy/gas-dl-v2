"""扰动注入：传感器退化/环境突变模拟（v1.2 §9.1 12 维通道布局）。

通道索引重映射（与 v1 旧 6 维布局相比）:

| 扰动类型        | 旧目标索引             | 新目标索引                | 说明 |
|----------------|---------------------|--------------------------|------|
| optical_atten  | x[..., 0]  (S_ndir) | x[..., IDX_NDIR_CO2]     | NDIR 通道乘性衰减 |
| optical_scat   | x[..., 0]  (S_ndir) | x[..., IDX_NDIR_CO2]     | NDIR 加性噪声 |
| thermal        | x[..., 1]  (S_tc)   | x[..., IDX_TCS]          | TCS 乘性扰动 |
| ultrasonic     | x[..., 2]  (S_us)   | x[..., IDX_US_SPEED]     | 超声声速估计加噪 |
| temperature    | x[..., 4]  (T)      | x[..., IDX_T_C]          | 温度阶跃偏移 |
| pressure_drift | x[..., 3]  (P)      | x[..., IDX_P_MPa]        | 压力输入通道确定性偏移 |

注意:
- ``ultrasonic`` 扰动从旧 toy S_us 改为真实 m/s 量级声速（~340-360 m/s）。
  level * scale * randn 中 scale 为每个样本自身窗口内声速均值，避免扰动幅度
  随 batch 组成变化；但相同 level 对应的绝对扰动幅度比旧版大。
- ``temperature`` 偏移量保留 80.0 K 的旧值,适配新单位 T_C 时表现为 80°C 阶跃。
- ``pressure_drift`` 是输入空间压力通道漂移，直接作用于 IDX_P_MPa；不重算
  声学/HITRAN 派生物理量。scaler-on 时扰动发生在标准化输入空间。
"""
from __future__ import annotations

import torch

from rcdw.models.single_modal import (
    IDX_NDIR_CO2,
    IDX_P_MPa,
    IDX_TCS,
    IDX_T_C,
    IDX_US_SPEED,
)


PERTURBATION_KINDS = [
    "optical_atten",
    "optical_scat",
    "thermal",
    "ultrasonic",
    "temperature",
    "pressure_drift",
]


def inject(x: torch.Tensor, kind: str, level: float) -> torch.Tensor:
    """在输入张量上注入指定类型和强度的扰动。

    Args:
        x:     (B, L, 12) 或 (N, L, 12) 滑窗输入（v1.2 新布局）
        kind:  扰动类型
        level: 扰动强度，0=无扰动，0.11=最大

    .. warning:: ``level`` 的物理含义因扰动类型而异：
       - ``optical_atten``：乘性衰减 ``*(1-level)``，分数损失
       - ``optical_scat``：加性高斯噪声 ``level*randn``，绝对尺度
       - ``thermal``：相对乘性 ``*(1+level*randn)``
       - ``ultrasonic``：逐样本量级自适应 ``level*scale*randn``，
         ``scale≈|mean(speed)|``
       - ``temperature``：绝对偏移 ``level*80``，单位°C
       - ``pressure_drift``：确定性压力输入偏移 ``level*2.0``

       同一 ``level`` 值对不同类型的物理严重度不可比；用于趋势分析可行，
       跨类型对比 level→error 曲线的斜率没有物理意义。

    Returns:
        扰动后的张量（原张量不变）
    """
    if x.shape[-1] != 12:
        raise ValueError(
            f"inject() expects 12-channel input, got last dim = {x.shape[-1]}. "
            f"v1.2 §8.2 12 维布局必需。"
        )
    x = x.clone()

    if kind == "optical_atten":
        # NDIR CO2 通道乘性衰减（光源老化 / 光路污染）
        x[..., IDX_NDIR_CO2] *= (1.0 - level)

    elif kind == "optical_scat":
        # NDIR CO2 通道加性高斯噪声（散射干扰）
        x[..., IDX_NDIR_CO2] = (
            x[..., IDX_NDIR_CO2] + level * torch.randn_like(x[..., IDX_NDIR_CO2])
        )

    elif kind == "thermal":
        # 热导通道乘性扰动（温控漂移）
        x[..., IDX_TCS] = x[..., IDX_TCS] * (
            1.0 + level * torch.randn_like(x[..., IDX_TCS])
        )

    elif kind == "ultrasonic":
        # 超声声速估计值加噪（换能器老化 / 耦合不良）
        # scale 只在序列维求均值并保留样本维，避免依赖 batch 组成。
        scale = x[..., IDX_US_SPEED].abs().mean(dim=-1, keepdim=True)
        x[..., IDX_US_SPEED] = (
            x[..., IDX_US_SPEED]
            + level * scale * torch.randn_like(x[..., IDX_US_SPEED])
        )

    elif kind == "temperature":
        # 温度阶跃偏移（T_C 通道, 索引 2）
        x[..., IDX_T_C] = x[..., IDX_T_C] + level * 80.0

    elif kind == "pressure_drift":
        # 压力输入空间确定性偏移（P_MPa 通道, 索引 3）
        x[..., IDX_P_MPa] = x[..., IDX_P_MPa] + level * 2.0

    else:
        raise ValueError(
            f"unknown perturbation kind: {kind}. valid: {PERTURBATION_KINDS}"
        )
    return x
