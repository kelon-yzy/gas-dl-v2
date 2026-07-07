"""掘进通风场景的 LHS 采样。

实现见 docs/掘进通风/sampling_design.md。

采样策略：在 (x_CO2, x_O2) 二维空间做 LHS，x_N2 = 100 - x_CO2 - x_O2
被动计算。N2 在本场景是显式预测目标，写入 condition grid 和 labels。

约束（全部强制）：
- x_CO2 ∈ [0.03, 5.00] %
- x_O2 ∈ [18.00, 21.20] %
- x_N2 = 100 - x_CO2 - x_O2 ∈ [73.80, 81.97] %（由前两者范围决定，自动满足）
- x_CO2 + x_O2 + x_N2 = 100 %（严格闭包）

与 syngas 的差异：
- 自由维度 2（CO2, O2），N2 是残差但写入 labels
- 总量约束 sum=100%（严格），非 sum<100%
- N2 是预测目标，不是背景气
"""
from __future__ import annotations

import random
from collections.abc import Iterable
from dataclasses import dataclass

from scipy.stats.qmc import LatinHypercube

from tv3.sim.core.ids import make_mixture_id, make_sequence_id
from tv3.sim.core.tunnel_ventilation_schema import COMPONENT_FIELDS


# 组分区间（单位 %），见 docs/掘进通风/sampling_design.md §1.1
@dataclass(frozen=True)
class TunnelVentilationRanges:
    co2: tuple[float, float] = (0.03, 5.00)
    o2: tuple[float, float] = (18.00, 21.20)
    # N2 范围由 CO2/O2 范围间接决定：
    # min N2 = 100 - 5.00 - 21.20 = 73.80
    # max N2 = 100 - 0.03 - 18.00 = 81.97
    n2_min: float = 73.80
    n2_max: float = 81.97


TUNNEL_VENTILATION_RANGES = TunnelVentilationRanges()

# 序列基准光程采样范围（每条序列的 L_m_base，与多光程扫描档位 path_lms 独立）
# 200kHz 下长声程信号被 CH4/CO2 弛豫吸收淹没，L_m 上限 0.3m（见 Phase0 核对记录）
L_M_BASE_RANGE: tuple[float, float] = (0.2, 0.3)


def generate_tunnel_ventilation_condition_rows(
    sequence_count: int,
    *,
    seed: int,
    sampling_strategy: str = "lhs",
    ranges: TunnelVentilationRanges = TUNNEL_VENTILATION_RANGES,
) -> list[dict[str, str]]:
    """生成掘进通风 condition 行。

    与 syngas `generate_syngas_condition_rows` 接口对齐：返回 dict 列表，
    数值格式化为字符串（保留 6 位小数）。

    返回 dict 包含 COMPONENT_FIELDS（3 列目标，含 x_N2），下游物理仿真
    直接读取三组分。
    """
    if sequence_count <= 0:
        raise ValueError("sequence_count must be positive")
    if sampling_strategy not in {"lhs", "random"}:
        raise ValueError(f"sampling_strategy must be 'lhs' or 'random', got {sampling_strategy!r}")

    rng = random.Random(seed)
    if sampling_strategy == "lhs":
        components_list = _generate_lhs(sequence_count, seed=seed + 1, ranges=ranges)
    else:
        components_list = _generate_random(sequence_count, rng=rng, ranges=ranges)

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
                # 200kHz 下长声程信号被 CH4/CO2 弛豫吸收淹没，L_m 上限 0.3m
                # （见 Phase0 核对记录）。tv3 CO2 最高 5%，压力较小，沿用一致约束。
                "L_m_base": _fmt(rng.uniform(*L_M_BASE_RANGE), 4),
                "status": "synthetic_measurement",
            }
        )
    return rows


def build_tunnel_ventilation_label_rows(
    conditions: Iterable[dict[str, str]],
) -> list[dict[str, str]]:
    """labels 写入 3 列预测目标（含 x_N2）。

    与 syngas 不同，N2 在本场景是显式预测目标，进入 labels。
    """
    return [
        {"sequence_id": row["sequence_id"], **{name: row[name] for name in COMPONENT_FIELDS}}
        for row in conditions
    ]


def _generate_lhs(
    n: int,
    *,
    seed: int,
    ranges: TunnelVentilationRanges,
) -> list[dict[str, float]]:
    """2D LHS 采样：(CO2, O2) 独立采样，N2 = 100 - CO2 - O2。"""
    sampler = LatinHypercube(d=2, seed=seed)
    raw = sampler.random(n=n)

    samples: list[dict[str, float]] = []
    for row in raw:
        u_co2, u_o2 = float(row[0]), float(row[1])
        x_co2 = ranges.co2[0] + u_co2 * (ranges.co2[1] - ranges.co2[0])
        x_o2 = ranges.o2[0] + u_o2 * (ranges.o2[1] - ranges.o2[0])
        x_n2 = 100.0 - x_co2 - x_o2
        # 范围校验（CO2/O2 范围已保证 N2 ∈ [73.80, 81.97]，此处防御性 assert）
        if x_n2 < ranges.n2_min - 1e-6 or x_n2 > ranges.n2_max + 1e-6:
            raise RuntimeError(
                f"N2 out of range: x_n2={x_n2}, expected [{ranges.n2_min}, {ranges.n2_max}]"
            )
        samples.append(
            {
                "x_CO2": round(x_co2, 6),
                "x_O2": round(x_o2, 6),
                "x_N2": round(x_n2, 6),
            }
        )
    return samples


def _generate_random(
    n: int,
    *,
    rng: random.Random,
    ranges: TunnelVentilationRanges,
) -> list[dict[str, float]]:
    """纯随机采样（备用路径，用于对照）。"""
    samples: list[dict[str, float]] = []
    for _ in range(n):
        x_co2 = rng.uniform(*ranges.co2)
        x_o2 = rng.uniform(*ranges.o2)
        x_n2 = 100.0 - x_co2 - x_o2
        samples.append(
            {
                "x_CO2": round(x_co2, 6),
                "x_O2": round(x_o2, 6),
                "x_N2": round(x_n2, 6),
            }
        )
    return samples


def _fmt(value: float, digits: int) -> str:
    return f"{value:.{digits}f}"
