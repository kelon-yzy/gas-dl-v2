from __future__ import annotations

import math


MPA_PER_ATM = 0.101325


def h2o_mole_percent_from_rh(t_c: float, p_mpa: float, h_rh: float) -> float:
    p_sat_kpa = 0.61121 * math.exp(17.502 * t_c / (240.97 + t_c))
    p_amb_kpa = max(p_mpa, 1e-3) * 1000.0
    h_w_pct = (h_rh / 100.0) * (p_sat_kpa / p_amb_kpa) * 100.0
    return max(0.0, min(h_w_pct, 5.0))


def hitran_temperature_k(t_c: float) -> float:
    return round(t_c + 273.15, 3)


def hitran_pressure_atm(p_mpa: float) -> float:
    return round(p_mpa / MPA_PER_ATM, 6)
