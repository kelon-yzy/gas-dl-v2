"""气体状态推导：H2O 摩尔分数（Magnus 公式）与单位转换。

完全等价于 HG 主线 ``src/sim/generation/gas_state.py``，但独立维护，
不 import 主线。对应方案 §5.3。
"""

from __future__ import annotations

import math


MPA_PER_ATM = 0.101325


def h2o_mole_percent_from_rh(t_c: float, p_mpa: float, h_rh: float) -> float:
    """Magnus 公式：由温度、压力、相对湿度推导 H2O 摩尔百分比。

    p_sat (kPa) = 0.61121 * exp(17.502 * T / (240.97 + T))
    H2O% = (RH/100) * (p_sat / p_amb) * 100，clamp 到 [0, 5]%。

    旧 RCDW ``synth.py`` 使用硬编码 ``rh_to_water_vol = 0.0355``；新代码
    使用 Magnus 公式由物理量推导。
    """
    p_sat_kpa = 0.61121 * math.exp(17.502 * t_c / (240.97 + t_c))
    p_amb_kpa = max(p_mpa, 1e-3) * 1000.0
    h_w_pct = (h_rh / 100.0) * (p_sat_kpa / p_amb_kpa) * 100.0
    return max(0.0, min(h_w_pct, 5.0))


def hitran_temperature_k(t_c: float) -> float:
    return round(t_c + 273.15, 3)


def hitran_pressure_atm(p_mpa: float) -> float:
    return round(p_mpa / MPA_PER_ATM, 6)
