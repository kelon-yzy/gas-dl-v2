from __future__ import annotations

import random
from collections.abc import Iterable

from scipy.stats.qmc import LatinHypercube

from hg.sim.core.ids import make_mixture_id, make_sequence_id
from hg.sim.core.schema import COMPONENT_FIELDS


def generate_condition_rows(
    sequence_count: int,
    *,
    seed: int,
    sampling_strategy: str = "lhs",
) -> list[dict[str, str]]:
    if sequence_count <= 0:
        raise ValueError("sequence_count must be positive")
    if sampling_strategy not in {"lhs", "random"}:
        raise ValueError(f"sampling_strategy must be 'lhs' or 'random', got {sampling_strategy!r}")

    rng = random.Random(seed)
    if sampling_strategy == "lhs":
        lhs_samples = _generate_lhs_samples(sequence_count, seed=seed + 1)
        components_list = [_sample_components_lhs(u_h2, u_co2, u_n2) for u_h2, u_co2, u_n2 in lhs_samples]
    else:
        components_list = [_sample_components_random(rng) for _ in range(sequence_count)]

    rows = []
    for index, components in enumerate(components_list, start=1):
        rows.append(
            {
                "sequence_id": str(make_sequence_id(index)),
                "mixture_id": str(make_mixture_id(index)),
                **{name: _fmt(components[name], 6) for name in COMPONENT_FIELDS},
                "T_C_base": _fmt(rng.uniform(15.0, 35.0), 4),
                "P_MPa_base": _fmt(rng.uniform(0.10, 0.709), 4),
                "H_RH_base": _fmt(rng.uniform(20.0, 80.0), 4),
                "L_m_base": _fmt(rng.uniform(0.2, 0.3), 4),  # 200kHz 下长声程信号被 CH4/CO2 弛豫吸收淹没，上限压缩到 0.3m（见 Phase0 核对记录）
                "status": "synthetic_measurement",
            }
        )
    return rows


def build_label_rows(conditions: Iterable[dict[str, str]]) -> list[dict[str, str]]:
    return [{"sequence_id": row["sequence_id"], **{name: row[name] for name in COMPONENT_FIELDS}} for row in conditions]


def _generate_lhs_samples(n: int, *, seed: int) -> list[tuple[float, float, float]]:
    """Generate N 3D LHS samples in [0,1]^3, one per gas component degree of freedom."""
    sampler = LatinHypercube(d=3, seed=seed)
    unit_samples = sampler.random(n=n)
    return [(float(row[0]), float(row[1]), float(row[2])) for row in unit_samples]


def _sample_components_lhs(u_h2: float, u_co2: float, u_n2: float) -> dict[str, float]:
    """Map three LHS strata (each in [0,1]) to component percentages.

    H₂ mapping preserves the bimodal distribution: 15% low (<3%), 15% high (>25%), 70% mid.
    CO₂ and N₂ are uniformly mapped to their respective ranges.
    CH₄ is the complement; if it falls below 40%, N₂ is reduced first to make room.
    """
    x_h2 = round(_map_hydrogen_lhs(u_h2), 6)
    x_co2 = round(u_co2 * 15.0, 6)
    x_n2 = round(u_n2 * 20.0, 6)
    x_ch4 = round(100.0 - x_h2 - x_co2 - x_n2, 6)
    if x_ch4 < 40.0:
        x_n2 = round(max(0.0, min(20.0, 100.0 - x_h2 - x_co2 - 40.0)), 6)
        x_ch4 = round(100.0 - x_h2 - x_co2 - x_n2, 6)
    return {
        "x_H2": x_h2,
        "x_CH4": x_ch4,
        "x_CO2": x_co2,
        "x_N2": x_n2,
    }


def _map_hydrogen_lhs(u: float) -> float:
    """Map [0,1] LHS stratum to H₂ percentage with bimodal distribution.

    - u ∈ [0.00, 0.15) → trace-H₂: [0, 3]
    - u ∈ [0.15, 0.85] → mid-range: [0, 30]
    - u ∈ (0.85, 1.00] → high-H₂: [25, 30]
    """
    if u < 0.15:
        return (u / 0.15) * 3.0
    if u > 0.85:
        return 25.0 + ((u - 0.85) / 0.15) * 5.0
    return ((u - 0.15) / 0.70) * 30.0


def _sample_components_random(rng: random.Random) -> dict[str, float]:
    x_h2 = round(_sample_hydrogen_percent_random(rng), 6)
    x_co2 = round(rng.uniform(0.0, 15.0), 6)
    x_n2 = round(rng.uniform(0.0, 20.0), 6)
    x_ch4 = round(100.0 - x_h2 - x_co2 - x_n2, 6)
    if x_ch4 < 40.0:
        x_n2 = round(max(0.0, min(20.0, 100.0 - x_h2 - x_co2 - 40.0)), 6)
        x_ch4 = round(100.0 - x_h2 - x_co2 - x_n2, 6)
    return {
        "x_H2": x_h2,
        "x_CH4": x_ch4,
        "x_CO2": x_co2,
        "x_N2": x_n2,
    }


def _sample_hydrogen_percent_random(rng: random.Random) -> float:
    marker = rng.random()
    if marker < 0.15:
        return rng.uniform(0.0, 3.0)
    if marker > 0.85:
        return rng.uniform(25.0, 30.0)
    return rng.uniform(0.0, 30.0)


def _fmt(value: float, digits: int) -> str:
    return f"{value:.{digits}f}"
