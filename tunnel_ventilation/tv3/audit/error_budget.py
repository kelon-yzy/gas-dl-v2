"""将 TOF nuisance 误差传播为等效 O₂ 误差。"""
from __future__ import annotations

import math
from collections.abc import Iterable


NORMAL_P90_Z = 1.6448536269514722


def equivalent_o2_std_percent(
    *,
    tof_per_o2_s_per_percent: float,
    tof_per_nuisance_s_per_unit: float,
    nuisance_std: float,
) -> float:
    if not math.isfinite(tof_per_o2_s_per_percent) or tof_per_o2_s_per_percent == 0.0:
        raise ValueError("tof_per_o2_s_per_percent must be finite and non-zero")
    if not math.isfinite(tof_per_nuisance_s_per_unit):
        raise ValueError("tof_per_nuisance_s_per_unit must be finite")
    if not math.isfinite(nuisance_std) or nuisance_std < 0.0:
        raise ValueError("nuisance_std must be finite and >= 0")
    return abs(tof_per_nuisance_s_per_unit * nuisance_std / tof_per_o2_s_per_percent)


def combined_p90_o2_error_percent(equivalent_stds_percent: Iterable[float]) -> float:
    variances: list[float] = []
    for value in equivalent_stds_percent:
        if not math.isfinite(value) or value < 0.0:
            raise ValueError("equivalent O2 standard deviations must be finite and >= 0")
        variances.append(value**2)
    if not variances:
        raise ValueError("at least one equivalent O2 standard deviation is required")
    return NORMAL_P90_Z * math.sqrt(sum(variances))
