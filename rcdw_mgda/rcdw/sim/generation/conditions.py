"""RCDW 三组分 (O2, CO2, N2) condition rows 生成。

采样策略：LHS d=2 → simplex 映射 + N2 ≥ 55% 边界保护。
对应方案 §5.1。

与 HG 主线 ``src/sim/generation/conditions.py`` 的差异：
- HG 用 LHS d=3 + H2 双峰分布 + CH4 complement；RCDW 用 LHS d=2 + 简单 simplex 映射。
- 每个 mixture 一个 sequence（v1.x 1:1 映射）；若未来按方案 §5.2 激活其他
  PhaseSchedule，需扩展为 1:N。
"""

from __future__ import annotations

import random
from collections.abc import Iterable

from scipy.stats.qmc import LatinHypercube

from rcdw.sim.core.ids import make_mixture_id, make_sequence_id
from rcdw.sim.core.schema import COMPONENT_FIELDS


# 组分范围常量（方案 §2.1）。
_X_O2_MAX = 25.0
_X_CO2_MAX = 20.0
_X_N2_MIN = 55.0
_X_N2_MAX = 100.0  # 仅参考；实际由 100 - O2 - CO2 推出

# 环境基线范围（方案 §2.1）。
_T_C_RANGE = (15.0, 35.0)
_P_MPA_RANGE = (0.10, 0.709)
_H_RH_RANGE = (20.0, 80.0)
_L_M_RANGE = (0.2, 1.8)


def generate_condition_rows(
    sequence_count: int,
    *,
    seed: int,
    sampling_strategy: str = "lhs",
) -> list[dict[str, str]]:
    """生成 N 个 condition rows。

    返回的 dict 中所有值为字符串（与 HG 主线一致，使用 ``_fmt`` 格式化），
    下游消费时需显式 ``float()`` 转换。组分值保留 6 位小数。

    Args:
        sequence_count: 序列数；每个 sequence 对应一个 mixture（1:1）。
        seed: 主 seed；环境随机用 ``random.Random(seed)``，LHS 用 ``seed+1``。
        sampling_strategy: ``"lhs"``（默认）或 ``"random"``。

    Returns:
        长度为 ``sequence_count`` 的 dict 列表，字段为 ``CONDITION_GRID_FIELDS``。
    """
    if sequence_count <= 0:
        raise ValueError("sequence_count must be positive")
    if sampling_strategy not in {"lhs", "random"}:
        raise ValueError(
            f"sampling_strategy must be 'lhs' or 'random', got {sampling_strategy!r}"
        )

    rng = random.Random(seed)
    if sampling_strategy == "lhs":
        lhs_samples = _generate_lhs_samples(sequence_count, seed=seed + 1)
        components_list = [
            _sample_components_lhs(u_o2, u_co2) for u_o2, u_co2 in lhs_samples
        ]
    else:
        components_list = [_sample_components_random(rng) for _ in range(sequence_count)]

    rows: list[dict[str, str]] = []
    for index, components in enumerate(components_list, start=1):
        rows.append(
            {
                "sequence_id": str(make_sequence_id(index)),
                "mixture_id": str(make_mixture_id(index)),
                **{name: _fmt(components[name], 6) for name in COMPONENT_FIELDS},
                "T_C_base": _fmt(rng.uniform(*_T_C_RANGE), 4),
                "P_MPa_base": _fmt(rng.uniform(*_P_MPA_RANGE), 4),
                "H_RH_base": _fmt(rng.uniform(*_H_RH_RANGE), 4),
                "L_m_base": _fmt(rng.uniform(*_L_M_RANGE), 4),
                "status": "synthetic_measurement",
            }
        )
    return rows


def build_label_rows(conditions: Iterable[dict[str, str]]) -> list[dict[str, str]]:
    """从 condition rows 提取标签行 (sequence_id + COMPONENT_FIELDS)。"""
    return [
        {"sequence_id": row["sequence_id"], **{name: row[name] for name in COMPONENT_FIELDS}}
        for row in conditions
    ]


def _generate_lhs_samples(n: int, *, seed: int) -> list[tuple[float, float]]:
    """生成 n 个 2D LHS 样本，每个 ∈ [0,1]^2，对应 (u_O2, u_CO2)。"""
    sampler = LatinHypercube(d=2, seed=seed)
    unit_samples = sampler.random(n=n)
    return [(float(row[0]), float(row[1])) for row in unit_samples]


def _sample_components_lhs(u_o2: float, u_co2: float) -> dict[str, float]:
    """LHS 2D → simplex 映射 + N2 ≥ 55% 边界保护。

    映射规则：
    - x_O2  = u_o2  * 25.0
    - x_CO2 = u_co2 * 20.0
    - x_N2  = 100 - x_O2 - x_CO2

    若 x_N2 < 55.0，则等比例缩减 O2 与 CO2 使 x_N2 = 55.0。

    边界回退分析（方案 §5.2 细节）：max(O2)+max(CO2) = 45，恰好对应
    min(N2) = 55，理论上仅浮点精度问题会触发回退；实测回退率应近 0。
    """
    x_o2 = u_o2 * _X_O2_MAX
    x_co2 = u_co2 * _X_CO2_MAX
    x_n2 = 100.0 - x_o2 - x_co2
    if x_n2 < _X_N2_MIN:
        total_oc = x_o2 + x_co2
        if total_oc > 0.0:
            scale = (100.0 - _X_N2_MIN) / total_oc
            x_o2 *= scale
            x_co2 *= scale
        x_n2 = _X_N2_MIN
    return {
        "x_O2": round(x_o2, 6),
        "x_CO2": round(x_co2, 6),
        "x_N2": round(x_n2, 6),
    }


def _sample_components_random(rng: random.Random) -> dict[str, float]:
    """纯随机采样备选路径（不推荐用于正式 benchmark）。"""
    u_o2 = rng.random()
    u_co2 = rng.random()
    return _sample_components_lhs(u_o2, u_co2)


def _fmt(value: float, digits: int) -> str:
    return f"{value:.{digits}f}"
