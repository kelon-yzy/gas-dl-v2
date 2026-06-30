"""RCDW 光学交叉敏感性（仅 CO2 通道 ← H2O）。

与 HG 主线的差异：HG 有 CH4↔CO2 双向交叉；RCDW 仅 CO2 一个通道，
交叉来自 H2O。对应方案 §2.3 / §5.8。

注意：在 hitran_hapi_v1 后端下，CO2 通道对 H2O 的交叉由 spectral 子包
内部通过 ``compute_prepared_tabulated_ndir_absorbance`` 加和光谱重叠
自然产生；本模块提供 empirical fallback（当未指定 HITRAN 后端时使用）。
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class OpticalCrosstalkSpec:
    # H2O → CO2 通道交叉系数。物理上来源于 CO2 通道滤波器
    # （CWL 2347 cm⁻¹, FWHM 93 cm⁻¹）窗口内 H2O 残余吸收。
    co2_channel_h2o_response: float = 0.012


DEFAULT_OPTICAL_CROSSTALK_SPEC = OpticalCrosstalkSpec()


def apply_optical_crosstalk(
    *,
    absorption_co2: float,
    absorption_h2o: float,
    spec: OpticalCrosstalkSpec = DEFAULT_OPTICAL_CROSSTALK_SPEC,
) -> dict[str, float]:
    """Empirical 后端使用：CO2 观测吸光度 = CO2 真值 + 系数 * H2O 吸光度。"""
    co2_cross_from_h2o = spec.co2_channel_h2o_response * absorption_h2o
    return {
        "absorption_co2_true": absorption_co2,
        "absorption_h2o_true": absorption_h2o,
        "absorption_co2_cross_from_h2o": co2_cross_from_h2o,
        "absorption_co2_observed": absorption_co2 + co2_cross_from_h2o,
    }
