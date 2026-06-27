"""合成气场景的光学串扰矩阵（3×3）。

与 hydrogen_ng `sim.generation.optical_crosstalk` 的 2×2 矩阵并存。

按 docs/syngas/co_crosstalk_design.md 设计，分两步实施：
- Step 1：CO 通道仅含 CO 自身吸收（apply_full_crosstalk=False，默认）
- Step 2：扩展为完整 3×3，含 CO2↔CO 互扰

CH4↔CO 间距 873 cm⁻¹，无重叠，固定为 0。
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class SyngasOpticalCrosstalkSpec:
    """3×3 串扰系数。约定：observed_channel = true_channel + Σ ε * true_others。

    单位：无量纲（绝对吸收度的比例）。
    """

    # CH4 ↔ CO2（与 hydrogen_ng 一致，保证基线对照）
    ch4_channel_co2_response: float = 0.035
    co2_channel_ch4_response: float = 0.012
    # CO2 ↔ CO（新增）
    co_channel_co2_response: float = 0.005   # ε_32: CO2 弱热带泄漏到 CO 通道
    co2_channel_co_response: float = 0.002   # ε_23: CO R 支远翼泄漏到 CO2 通道
    # CH4 ↔ CO（间距大，固定 0）
    co_channel_ch4_response: float = 0.0
    ch4_channel_co_response: float = 0.0


DEFAULT_SYNGAS_CROSSTALK_SPEC = SyngasOpticalCrosstalkSpec()


def apply_syngas_optical_crosstalk(
    *,
    absorption_ch4: float,
    absorption_co2: float,
    absorption_co: float,
    spec: SyngasOpticalCrosstalkSpec = DEFAULT_SYNGAS_CROSSTALK_SPEC,
    enable_co_crosstalk: bool = False,
) -> dict[str, float]:
    """3×3 串扰矩阵实现。

    enable_co_crosstalk:
        False（Step 1，默认）：CO 通道无串扰，observed_co = true_co；CH4↔CO2
            行为完全保留与 hydrogen_ng 一致。
        True（Step 2）：启用 CO2↔CO 互扰。

    返回 dict 字段命名与 hydrogen_ng 兼容，并补 CO 系列。
    """
    if enable_co_crosstalk:
        ch4_observed = (
            absorption_ch4
            + spec.ch4_channel_co2_response * absorption_co2
            + spec.ch4_channel_co_response * absorption_co
        )
        co2_observed = (
            absorption_co2
            + spec.co2_channel_ch4_response * absorption_ch4
            + spec.co2_channel_co_response * absorption_co
        )
        co_observed = (
            absorption_co
            + spec.co_channel_co2_response * absorption_co2
            + spec.co_channel_ch4_response * absorption_ch4
        )
        ch4_cross_from_co2 = spec.ch4_channel_co2_response * absorption_co2
        co2_cross_from_ch4 = spec.co2_channel_ch4_response * absorption_ch4
        co_cross_from_co2 = spec.co_channel_co2_response * absorption_co2
        co2_cross_from_co = spec.co2_channel_co_response * absorption_co
    else:
        # Step 1：CO 通道纯净
        ch4_cross_from_co2 = spec.ch4_channel_co2_response * absorption_co2
        co2_cross_from_ch4 = spec.co2_channel_ch4_response * absorption_ch4
        ch4_observed = absorption_ch4 + ch4_cross_from_co2
        co2_observed = absorption_co2 + co2_cross_from_ch4
        co_observed = absorption_co
        co_cross_from_co2 = 0.0
        co2_cross_from_co = 0.0

    return {
        # CH4 / CO2（与 hydrogen_ng 字段对齐）
        "absorption_ch4_true": absorption_ch4,
        "absorption_co2_true": absorption_co2,
        "absorption_ch4_cross_from_co2": ch4_cross_from_co2,
        "absorption_co2_cross_from_ch4": co2_cross_from_ch4,
        "absorption_ch4_observed": ch4_observed,
        "absorption_co2_observed": co2_observed,
        # CO（新增）
        "absorption_co_true": absorption_co,
        "absorption_co_cross_from_co2": co_cross_from_co2,
        "absorption_co2_cross_from_co": co2_cross_from_co,
        "absorption_co_observed": co_observed,
    }
