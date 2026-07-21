"""tv3 bidirectional TOF identifiability (F4 / v2).

Extends the v1 single-TOF audit without modifying ``identifiability.py``.
Observation model: ``t_ab = L/(c+v)+τ``, ``t_ba = L/(c-v)+τ``.
Joint Fisher uses the two-direction observation vector (optional T row).
"""
from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from itertools import product
from typing import Any

import numpy as np

from tv3.sim.generation.tunnel_ventilation.acoustic_physics import hidden_sound_speed_v2
from tv3.sim.generation.tunnel_ventilation.flow_physics import bidirectional_transit_times_s


# o2 first: conditional O2 CRLB uses index 0 (same convention as v1).
DERIVATIVE_PARAMETERS = (
    "o2_percent",
    "co2_percent",
    "t_c",
    "path_length_m",
    "v_path_m_per_s",
)


@dataclass(frozen=True)
class BidirAcousticPoint:
    co2_percent: float
    o2_percent: float
    t_c: float
    path_length_m: float
    v_path_m_per_s: float = 0.0

    @property
    def n2_percent(self) -> float:
        return 100.0 - self.co2_percent - self.o2_percent

    def validate(self) -> None:
        values = (
            self.co2_percent,
            self.o2_percent,
            self.t_c,
            self.path_length_m,
            self.v_path_m_per_s,
        )
        if not all(math.isfinite(value) for value in values):
            raise ValueError("bidir acoustic point values must be finite")
        if self.co2_percent < 0.0 or self.o2_percent < 0.0 or self.n2_percent < 0.0:
            raise ValueError("composition must be non-negative and sum to 100 percent")
        if self.path_length_m <= 0.0:
            raise ValueError("path_length_m must be positive")


def sound_speed_m_per_s(point: BidirAcousticPoint) -> float:
    point.validate()
    return hidden_sound_speed_v2(
        x_h2=0.0,
        x_ch4=0.0,
        x_co2=point.co2_percent,
        x_n2=point.n2_percent,
        t_c=point.t_c,
        x_o2=point.o2_percent,
    )


def observed_bidir_tof_s(
    point: BidirAcousticPoint,
    *,
    fixed_delay_s: float,
) -> tuple[float, float]:
    if not math.isfinite(fixed_delay_s) or fixed_delay_s < 0.0:
        raise ValueError("fixed_delay_s must be finite and >= 0")
    c = sound_speed_m_per_s(point)
    t_ab, t_ba = bidirectional_transit_times_s(
        point.path_length_m, c, point.v_path_m_per_s
    )
    return t_ab + fixed_delay_s, t_ba + fixed_delay_s


def build_bidir_points(grid: Mapping[str, Sequence[float]]) -> list[BidirAcousticPoint]:
    required = ("co2_percent", "o2_percent", "t_c", "path_length_m", "v_path_m_per_s")
    if set(grid) != set(required):
        raise ValueError(f"grid keys must be {required}, got {tuple(grid)}")
    values = [grid[name] for name in required]
    if any(not entries for entries in values):
        raise ValueError("each grid dimension must contain at least one value")
    return [BidirAcousticPoint(*map(float, entries)) for entries in product(*values)]


def _within_bounds(value: float, bounds: tuple[float, float]) -> bool:
    return bounds[0] <= value <= bounds[1]


def _shift_point(point: BidirAcousticPoint, parameter: str, delta: float) -> BidirAcousticPoint:
    if parameter == "o2_percent":
        return BidirAcousticPoint(
            point.co2_percent,
            point.o2_percent + delta,
            point.t_c,
            point.path_length_m,
            point.v_path_m_per_s,
        )
    if parameter == "co2_percent":
        return BidirAcousticPoint(
            point.co2_percent + delta,
            point.o2_percent,
            point.t_c,
            point.path_length_m,
            point.v_path_m_per_s,
        )
    if parameter == "t_c":
        return BidirAcousticPoint(
            point.co2_percent,
            point.o2_percent,
            point.t_c + delta,
            point.path_length_m,
            point.v_path_m_per_s,
        )
    if parameter == "path_length_m":
        return BidirAcousticPoint(
            point.co2_percent,
            point.o2_percent,
            point.t_c,
            point.path_length_m + delta,
            point.v_path_m_per_s,
        )
    if parameter == "v_path_m_per_s":
        return BidirAcousticPoint(
            point.co2_percent,
            point.o2_percent,
            point.t_c,
            point.path_length_m,
            point.v_path_m_per_s + delta,
        )
    raise ValueError(f"unsupported derivative parameter: {parameter!r}")


def _parameter_value(point: BidirAcousticPoint, parameter: str) -> float:
    return float(getattr(point, parameter))


def _finite_difference_pair(
    point: BidirAcousticPoint,
    *,
    parameter: str,
    step: float,
    bounds: tuple[float, float],
    fixed_delay_s: float,
) -> tuple[tuple[float, float], str]:
    if not math.isfinite(step) or step <= 0.0:
        raise ValueError(f"step for {parameter} must be finite and > 0")
    current = _parameter_value(point, parameter)
    plus = current + step
    minus = current - step
    has_plus = _within_bounds(plus, bounds)
    has_minus = _within_bounds(minus, bounds)
    base_ab, base_ba = observed_bidir_tof_s(point, fixed_delay_s=fixed_delay_s)
    if has_plus and has_minus:
        plus_ab, plus_ba = observed_bidir_tof_s(
            _shift_point(point, parameter, step), fixed_delay_s=fixed_delay_s
        )
        minus_ab, minus_ba = observed_bidir_tof_s(
            _shift_point(point, parameter, -step), fixed_delay_s=fixed_delay_s
        )
        return (
            ((plus_ab - minus_ab) / (2.0 * step), (plus_ba - minus_ba) / (2.0 * step)),
            "central",
        )
    if has_plus:
        plus_ab, plus_ba = observed_bidir_tof_s(
            _shift_point(point, parameter, step), fixed_delay_s=fixed_delay_s
        )
        return (((plus_ab - base_ab) / step, (plus_ba - base_ba) / step), "forward")
    if has_minus:
        minus_ab, minus_ba = observed_bidir_tof_s(
            _shift_point(point, parameter, -step), fixed_delay_s=fixed_delay_s
        )
        return (((base_ab - minus_ab) / step, (base_ba - minus_ba) / step), "backward")
    raise ValueError(f"step for {parameter} does not fit within configured bounds")


def local_bidir_tof_sensitivity(
    point: BidirAcousticPoint,
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
        (d_ab, d_ba), scheme = _finite_difference_pair(
            point,
            parameter=parameter,
            step=step,
            bounds=bounds,
            fixed_delay_s=fixed_delay_s,
        )
        (h_ab, h_ba), _ = _finite_difference_pair(
            point,
            parameter=parameter,
            step=step / 2.0,
            bounds=bounds,
            fixed_delay_s=fixed_delay_s,
        )
        (d2_ab, d2_ba), _ = _finite_difference_pair(
            point,
            parameter=parameter,
            step=step * 2.0,
            bounds=bounds,
            fixed_delay_s=fixed_delay_s,
        )
        denom_ab = max(abs(h_ab), abs(d2_ab), 1e-15)
        denom_ba = max(abs(h_ba), abs(d2_ba), 1e-15)
        disagreement = max(abs(h_ab - d2_ab) / denom_ab, abs(h_ba - d2_ba) / denom_ba)
        # Mid-pair TOF derivative: maps common-mode nuisances to ĉ / O2 budget.
        d_mid = 0.5 * (d_ab + d_ba)
        results[parameter] = {
            "derivative_tof_ab_s_per_unit": d_ab,
            "derivative_tof_ba_s_per_unit": d_ba,
            "derivative_tof_mid_s_per_unit": d_mid,
            "scheme": scheme,
            "step_disagreement": disagreement,
            "stable": disagreement <= max_relative_step_disagreement,
        }
    return results


def fisher_information_bidir(
    derivatives: Mapping[str, Mapping[str, float | str | bool]],
    *,
    tof_std_s: float,
    temperature_std_c: float | None = None,
) -> dict[str, Any]:
    """Multi-observation Fisher for y=[t_ab, t_ba] (+ optional T).

    Independent trigger jitter ⇒ Σ_tof = σ² I_2. Optional T row uses σ_T on the
    measured temperature channel (∂T/∂t_c=1, else 0).
    """
    if not math.isfinite(tof_std_s) or tof_std_s <= 0.0:
        raise ValueError("tof_std_s must be finite and > 0")
    n_param = len(DERIVATIVE_PARAMETERS)
    jacobian = np.zeros((2, n_param), dtype=np.float64)
    for idx, parameter in enumerate(DERIVATIVE_PARAMETERS):
        jacobian[0, idx] = float(derivatives[parameter]["derivative_tof_ab_s_per_unit"])
        jacobian[1, idx] = float(derivatives[parameter]["derivative_tof_ba_s_per_unit"])
    if not np.isfinite(jacobian).all():
        raise ValueError("TOF derivatives must be finite before Fisher calculation")
    cov_inv = np.eye(2, dtype=np.float64) / (tof_std_s**2)
    joint = jacobian.T @ cov_inv @ jacobian
    observability = "tof_ab_tof_ba"
    if temperature_std_c is not None:
        if not math.isfinite(temperature_std_c) or temperature_std_c <= 0.0:
            raise ValueError("temperature_std_c must be finite and > 0 when provided")
        t_row = np.zeros((1, n_param), dtype=np.float64)
        t_row[0, DERIVATIVE_PARAMETERS.index("t_c")] = 1.0
        joint = joint + (t_row.T @ t_row) / (temperature_std_c**2)
        observability = "tof_ab_tof_ba_plus_T"

    rank = int(np.linalg.matrix_rank(joint, tol=1e-12))
    # Conditional O2 info from mid-pair TOF (common-mode channel after reciprocal-sum).
    d_mid_o2 = float(derivatives["o2_percent"]["derivative_tof_mid_s_per_unit"])
    # Mid-pair of two independent TOFs: Var((t_ab+t_ba)/2) = σ²/2.
    mid_var = 0.5 * (tof_std_s**2)
    conditional_information = float(d_mid_o2**2 / mid_var)
    if conditional_information <= 0.0:
        raise ValueError("conditional O2 Fisher information must be positive")

    # Acoustic (c, v) subspace: columns for composition/T/L share common-mode;
    # v_path is differential-mode. Full-rank acoustic subsystem ⇒ rank >= 2.
    acoustic_full_rank = rank >= 2
    return {
        "conditional_o2_information": conditional_information,
        "conditional_o2_crlb_std_percent": math.sqrt(1.0 / conditional_information),
        "joint_parameters": list(DERIVATIVE_PARAMETERS),
        "joint_rank": rank,
        "joint_parameter_count": n_param,
        "joint_observation_model": observability,
        "acoustic_subsystem_full_rank": acoustic_full_rank,
        "nuisance_marginalized_status": (
            "available" if rank >= n_param else "unavailable_rank_deficient"
        ),
        "joint_condition_number": None if rank < n_param else float(np.linalg.cond(joint)),
    }


def midpair_tof_std_s(tof_std_s: float) -> float:
    """Std of (t_ab+t_ba)/2 under independent identical direction jitter."""
    if not math.isfinite(tof_std_s) or tof_std_s <= 0.0:
        raise ValueError("tof_std_s must be finite and > 0")
    return float(tof_std_s) / math.sqrt(2.0)


__all__ = [
    "BidirAcousticPoint",
    "DERIVATIVE_PARAMETERS",
    "build_bidir_points",
    "fisher_information_bidir",
    "local_bidir_tof_sensitivity",
    "midpair_tof_std_s",
    "observed_bidir_tof_s",
    "sound_speed_m_per_s",
]
