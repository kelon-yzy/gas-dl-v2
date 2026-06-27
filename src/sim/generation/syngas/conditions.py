"""合成气场景的 LHS 采样。

实现方案 B（煤气化技术全谱）+ 条件顺序采样。详细设计见
`docs/syngas/lhs_sampling_design.md`。

采样顺序：CO → H2 → CO2 → CH4。每一步根据已采组分动态收紧后续组分的
上下限，保证联合可行性约束在采样时直接满足。N2 = balance（被动计算）。

约束（全部强制）：
- N2 ≥ 0.2%
- H2/CO ∈ [0.1, 4.0]
- CO2/CO ∈ [0.02, 1.5]
- CO + CO2 + CH4 ∈ [35%, 75%]
"""
from __future__ import annotations

import random
from collections.abc import Iterable
from dataclasses import dataclass

from scipy.stats.qmc import LatinHypercube

from sim.core.ids import make_mixture_id, make_sequence_id
from sim.core.syngas_schema import COMPONENT_FIELDS, BACKGROUND_FIELDS


# 方案 B 边际区间（单位 %）
@dataclass(frozen=True)
class SyngasRanges:
    co: tuple[float, float] = (15.0, 65.0)
    h2: tuple[float, float] = (5.0, 55.0)
    co2: tuple[float, float] = (2.0, 30.0)
    ch4: tuple[float, float] = (0.0, 12.0)
    n2_min: float = 0.2

    # 联合约束
    h2_co_min: float = 0.1
    h2_co_max: float = 4.0
    co2_co_min: float = 0.02
    co2_co_max: float = 1.5
    carbon_min: float = 35.0
    carbon_max: float = 75.0


SYNGAS_RANGES = SyngasRanges()

# 候选样本超采率：每 n 个目标样本预生成 ceil(n * OVERSAMPLE) 个 LHS 候选，
# 应对约束冲突丢弃。文献预估丢弃率 <15%，1.2 倍足够。
_OVERSAMPLE = 1.2


def generate_syngas_condition_rows(
    sequence_count: int,
    *,
    seed: int,
    sampling_strategy: str = "lhs",
    ranges: SyngasRanges = SYNGAS_RANGES,
) -> list[dict[str, str]]:
    """生成合成气 condition 行。

    与 hydrogen_ng `generate_condition_rows` 的接口对齐：返回 dict 列表，
    数值格式化为字符串（保留 6 位小数）。

    返回 dict 中同时包含 COMPONENT_FIELDS（4 列目标）和 BACKGROUND_FIELDS
    （x_N2 计算值），下游物理仿真依赖 x_N2。
    """
    if sequence_count <= 0:
        raise ValueError("sequence_count must be positive")
    if sampling_strategy not in {"lhs", "random"}:
        raise ValueError(f"sampling_strategy must be 'lhs' or 'random', got {sampling_strategy!r}")

    rng = random.Random(seed)
    if sampling_strategy == "lhs":
        components_list = _generate_sequential_lhs(sequence_count, seed=seed + 1, ranges=ranges)
    else:
        components_list = _generate_random_rejection(sequence_count, rng=rng, ranges=ranges)

    rows = []
    for index, components in enumerate(components_list, start=1):
        rows.append(
            {
                "sequence_id": str(make_sequence_id(index)),
                "mixture_id": str(make_mixture_id(index)),
                **{name: _fmt(components[name], 6) for name in COMPONENT_FIELDS},
                **{name: _fmt(components[name], 6) for name in BACKGROUND_FIELDS},
                "T_C_base": _fmt(rng.uniform(15.0, 35.0), 4),
                "P_MPa_base": _fmt(rng.uniform(0.10, 0.709), 4),
                "H_RH_base": _fmt(rng.uniform(20.0, 80.0), 4),
                "L_m_base": _fmt(rng.uniform(0.2, 1.8), 4),
                "status": "synthetic_measurement",
            }
        )
    return rows


def build_syngas_label_rows(conditions: Iterable[dict[str, str]]) -> list[dict[str, str]]:
    """labels 仅写入 4 列预测目标，x_N2 不进入 labels。"""
    return [
        {"sequence_id": row["sequence_id"], **{name: row[name] for name in COMPONENT_FIELDS}}
        for row in conditions
    ]


def is_feasible_syngas(
    x_h2: float,
    x_co: float,
    x_co2: float,
    x_ch4: float,
    *,
    ranges: SyngasRanges = SYNGAS_RANGES,
) -> bool:
    """检查给定组分是否满足全部联合可行性约束。

    用于测试和拒绝采样。条件顺序采样不依赖此函数（约束在采样时直接编码）。
    """
    x_n2 = 100.0 - x_h2 - x_co - x_co2 - x_ch4
    if x_n2 < ranges.n2_min:
        return False
    if x_co < 1e-6:
        return False
    if not (ranges.h2_co_min <= x_h2 / x_co <= ranges.h2_co_max):
        return False
    if not (ranges.co2_co_min <= x_co2 / x_co <= ranges.co2_co_max):
        return False
    total_carbon = x_co + x_co2 + x_ch4
    if not (ranges.carbon_min <= total_carbon <= ranges.carbon_max):
        return False
    return True


def _generate_sequential_lhs(
    n: int,
    *,
    seed: int,
    ranges: SyngasRanges,
) -> list[dict[str, float]]:
    """条件顺序采样：CO → H2 → CO2 → CH4，每步动态收紧后续组分的可行区间。

    如果某步 lb > ub（约束冲突），丢弃该候选并补采。LHS 分位数在 [0,1]^4
    均匀，条件映射会拉伸或压缩各维度的区间，但 CO 边际仍然均匀，后续
    组分在各自条件可行区间内近似均匀。
    """
    n_candidates = max(n + 5, int(n * _OVERSAMPLE))
    sampler = LatinHypercube(d=4, seed=seed)
    raw = sampler.random(n=n_candidates)

    samples: list[dict[str, float]] = []
    idx = 0
    while len(samples) < n and idx < len(raw):
        u_co, u_h2, u_co2, u_ch4 = raw[idx]
        idx += 1
        sample = _sample_one_sequential(float(u_co), float(u_h2), float(u_co2), float(u_ch4), ranges)
        if sample is not None:
            samples.append(sample)

    if len(samples) < n:
        raise RuntimeError(
            f"Sequential sampling exhausted {idx} candidates but only produced "
            f"{len(samples)}/{n} feasible samples. Consider relaxing constraints "
            f"or increasing oversample rate."
        )
    return samples


def _sample_one_sequential(
    u_co: float,
    u_h2: float,
    u_co2: float,
    u_ch4: float,
    ranges: SyngasRanges,
) -> dict[str, float] | None:
    """单个条件顺序采样。返回 None 表示约束冲突。"""
    budget = 100.0 - ranges.n2_min  # 总可分配预算

    # Step 1: CO（无依赖，直接映射）
    x_co = ranges.co[0] + u_co * (ranges.co[1] - ranges.co[0])

    # Step 2: H2（依赖 CO，受 H2/CO 比值约束 + 预算约束）
    h2_ub = min(
        ranges.h2[1],
        ranges.h2_co_max * x_co,
        budget - x_co - ranges.co2[0] - ranges.ch4[0],
    )
    h2_lb = max(ranges.h2[0], ranges.h2_co_min * x_co)
    if h2_lb > h2_ub:
        return None
    x_h2 = h2_lb + u_h2 * (h2_ub - h2_lb)

    # Step 3: CO2（依赖 CO+H2，受 CO2/CO + 碳平衡上限）
    co2_ub = min(
        ranges.co2[1],
        ranges.co2_co_max * x_co,
        ranges.carbon_max - x_co - ranges.ch4[0],
        budget - x_co - x_h2 - ranges.ch4[0],
    )
    co2_lb = max(ranges.co2[0], ranges.co2_co_min * x_co)
    if co2_lb > co2_ub:
        return None
    x_co2 = co2_lb + u_co2 * (co2_ub - co2_lb)

    # Step 4: CH4（依赖 CO+H2+CO2，受预算 + 碳平衡）
    ch4_ub = min(
        ranges.ch4[1],
        ranges.carbon_max - x_co - x_co2,
        budget - x_co - x_h2 - x_co2,
    )
    ch4_lb = max(ranges.ch4[0], ranges.carbon_min - x_co - x_co2)
    if ch4_lb > ch4_ub:
        return None
    x_ch4 = ch4_lb + u_ch4 * (ch4_ub - ch4_lb)

    x_n2 = 100.0 - x_h2 - x_ch4 - x_co2 - x_co
    if x_n2 < ranges.n2_min:
        return None

    return {
        "x_H2": round(x_h2, 6),
        "x_CH4": round(x_ch4, 6),
        "x_CO2": round(x_co2, 6),
        "x_CO": round(x_co, 6),
        "x_N2": round(x_n2, 6),
    }


def _generate_random_rejection(
    n: int,
    *,
    rng: random.Random,
    ranges: SyngasRanges,
) -> list[dict[str, float]]:
    """纯随机 + 拒绝采样（备用路径，主要用于对照）。"""
    samples: list[dict[str, float]] = []
    attempts = 0
    max_attempts = n * 100  # 文献测算接受率 ~50%，100 倍上限充裕
    while len(samples) < n and attempts < max_attempts:
        attempts += 1
        x_co = rng.uniform(*ranges.co)
        x_h2 = rng.uniform(*ranges.h2)
        x_co2 = rng.uniform(*ranges.co2)
        x_ch4 = rng.uniform(*ranges.ch4)
        if not is_feasible_syngas(x_h2, x_co, x_co2, x_ch4, ranges=ranges):
            continue
        x_n2 = 100.0 - x_h2 - x_co - x_co2 - x_ch4
        samples.append(
            {
                "x_H2": round(x_h2, 6),
                "x_CH4": round(x_ch4, 6),
                "x_CO2": round(x_co2, 6),
                "x_CO": round(x_co, 6),
                "x_N2": round(x_n2, 6),
            }
        )
    if len(samples) < n:
        raise RuntimeError(
            f"Random rejection sampling produced only {len(samples)}/{n} after "
            f"{attempts} attempts. Consider switching to LHS sequential sampling."
        )
    return samples


def _fmt(value: float, digits: int) -> str:
    return f"{value:.{digits}f}"
