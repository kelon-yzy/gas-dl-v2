"""tv3 单向 TOF 链路的局部可辨识性计算。"""
from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from itertools import product
from typing import Any

import numpy as np

from tv3.sim.generation.tunnel_ventilation.acoustic_physics import hidden_sound_speed_v2


DERIVATIVE_PARAMETERS = ("o2_percent", "co2_percent", "t_c", "path_length_m")


@dataclass(frozen=True)
class AcousticPoint:
    co2_percent: float
    o2_percent: float
    t_c: float
    path_length_m: float

    @property
    def n2_percent(self) -> float:
        return 100.0 - self.co2_percent - self.o2_percent

    def validate(self) -> None:
        values = (self.co2_percent, self.o2_percent, self.t_c, self.path_length_m)
        if not all(math.isfinite(value) for value in values):
            raise ValueError("acoustic point values must be finite")
        if self.co2_percent < 0.0 or self.o2_percent < 0.0 or self.n2_percent < 0.0:
            raise ValueError("composition must be non-negative and sum to 100 percent")
        if self.path_length_m <= 0.0:
            raise ValueError("path_length_m must be positive")


def sound_speed_m_per_s(point: AcousticPoint) -> float:
    point.validate()
    return hidden_sound_speed_v2(
        x_h2=0.0,
        x_ch4=0.0,
        x_co2=point.co2_percent,
        x_n2=point.n2_percent,
        t_c=point.t_c,
        x_o2=point.o2_percent,
    )


def observed_tof_s(point: AcousticPoint, *, fixed_delay_s: float) -> float:
    if not math.isfinite(fixed_delay_s) or fixed_delay_s < 0.0:
        raise ValueError("fixed_delay_s must be finite and >= 0")
    return point.path_length_m / sound_speed_m_per_s(point) + fixed_delay_s


def build_points(grid: Mapping[str, Sequence[float]]) -> list[AcousticPoint]:
    required = ("co2_percent", "o2_percent", "t_c", "path_length_m")
    if set(grid) != set(required):
        raise ValueError(f"grid keys must be {required}, got {tuple(grid)}")
    values = [grid[name] for name in required]
    if any(not entries for entries in values):
        raise ValueError("each grid dimension must contain at least one value")
    return [AcousticPoint(*map(float, entries)) for entries in product(*values)]


def _within_bounds(value: float, bounds: tuple[float, float]) -> bool:
    return bounds[0] <= value <= bounds[1]


def _shift_point(point: AcousticPoint, parameter: str, delta: float) -> AcousticPoint:
    if parameter == "o2_percent":
        return AcousticPoint(point.co2_percent, point.o2_percent + delta, point.t_c, point.path_length_m)
    if parameter == "co2_percent":
        return AcousticPoint(point.co2_percent + delta, point.o2_percent, point.t_c, point.path_length_m)
    if parameter == "t_c":
        return AcousticPoint(point.co2_percent, point.o2_percent, point.t_c + delta, point.path_length_m)
    if parameter == "path_length_m":
        return AcousticPoint(point.co2_percent, point.o2_percent, point.t_c, point.path_length_m + delta)
    raise ValueError(f"unsupported derivative parameter: {parameter!r}")


def _parameter_value(point: AcousticPoint, parameter: str) -> float:
    return float(getattr(point, parameter))


def _finite_difference(
    point: AcousticPoint,
    *,
    parameter: str,
    step: float,
    bounds: tuple[float, float],
    fixed_delay_s: float,
) -> tuple[float, str]:
    if not math.isfinite(step) or step <= 0.0:
        raise ValueError(f"step for {parameter} must be finite and > 0")
    current = _parameter_value(point, parameter)
    plus = current + step
    minus = current - step
    has_plus = _within_bounds(plus, bounds)
    has_minus = _within_bounds(minus, bounds)
    baseline = observed_tof_s(point, fixed_delay_s=fixed_delay_s)
    if has_plus and has_minus:
        return (
            (observed_tof_s(_shift_point(point, parameter, step), fixed_delay_s=fixed_delay_s)
            - observed_tof_s(_shift_point(point, parameter, -step), fixed_delay_s=fixed_delay_s))
            / (2.0 * step),
            "central",
        )
    if has_plus:
        return (
            (observed_tof_s(_shift_point(point, parameter, step), fixed_delay_s=fixed_delay_s) - baseline) / step,
            "forward",
        )
    if has_minus:
        return (
            (baseline - observed_tof_s(_shift_point(point, parameter, -step), fixed_delay_s=fixed_delay_s)) / step,
            "backward",
        )
    raise ValueError(f"step for {parameter} does not fit within configured bounds")


def local_tof_sensitivity(
    point: AcousticPoint,
    *,
    parameter_steps: Mapping[str, float],
    parameter_bounds: Mapping[str, tuple[float, float]],
    fixed_delay_s: float,
    max_relative_step_disagreement: float,
) -> dict[str, dict[str, float | str | bool]]:
    if not math.isfinite(max_relative_step_disagreement) or max_relative_step_disagreement < 0.0:
        raise ValueError("max_relative_step_disagreement must be finite and >= 0")
    results: dict[str, dict[str, float | str | bool]] = {}
    for parameter in DERIVATIVE_PARAMETERS:
        if parameter not in parameter_steps or parameter not in parameter_bounds:
            raise ValueError(f"missing step or bounds for {parameter}")
        step = float(parameter_steps[parameter])
        bounds = tuple(map(float, parameter_bounds[parameter]))
        derivative, scheme = _finite_difference(
            point,
            parameter=parameter,
            step=step,
            bounds=bounds,
            fixed_delay_s=fixed_delay_s,
        )
        half, _ = _finite_difference(
            point,
            parameter=parameter,
            step=step / 2.0,
            bounds=bounds,
            fixed_delay_s=fixed_delay_s,
        )
        doubled, _ = _finite_difference(
            point,
            parameter=parameter,
            step=step * 2.0,
            bounds=bounds,
            fixed_delay_s=fixed_delay_s,
        )
        denominator = max(abs(half), abs(doubled), 1e-15)
        disagreement = abs(half - doubled) / denominator
        results[parameter] = {
            "derivative_tof_s_per_unit": derivative,
            "scheme": scheme,
            "step_disagreement": disagreement,
            "stable": disagreement <= max_relative_step_disagreement,
        }
    return results


def fisher_information(
    derivatives: Mapping[str, Mapping[str, float | str | bool]],
    *,
    tof_std_s: float,
) -> dict[str, Any]:
    if not math.isfinite(tof_std_s) or tof_std_s <= 0.0:
        raise ValueError("tof_std_s must be finite and > 0")
    values = np.asarray(
        [float(derivatives[parameter]["derivative_tof_s_per_unit"]) for parameter in DERIVATIVE_PARAMETERS],
        dtype=np.float64,
    )
    if not np.isfinite(values).all():
        raise ValueError("TOF derivatives must be finite before Fisher calculation")
    covariance = np.asarray([[tof_std_s**2]], dtype=np.float64)
    if np.linalg.matrix_rank(covariance) != 1:
        raise ValueError("TOF covariance is singular")
    conditional_information = float(values[0] ** 2 / covariance[0, 0])
    if conditional_information <= 0.0:
        raise ValueError("conditional O2 Fisher information must be positive")
    joint = np.outer(values, values) / covariance[0, 0]
    rank = int(np.linalg.matrix_rank(joint))
    return {
        "conditional_o2_information": conditional_information,
        "conditional_o2_crlb_std_percent": math.sqrt(1.0 / conditional_information),
        "joint_parameters": list(DERIVATIVE_PARAMETERS),
        "joint_rank": rank,
        "joint_parameter_count": len(DERIVATIVE_PARAMETERS),
        "nuisance_marginalized_status": "unavailable_rank_deficient" if rank < len(DERIVATIVE_PARAMETERS) else "available",
        "joint_condition_number": None if rank < len(DERIVATIVE_PARAMETERS) else float(np.linalg.cond(joint)),
    }
