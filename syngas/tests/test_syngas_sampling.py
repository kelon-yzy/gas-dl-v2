"""合成气场景的 LHS 采样 + schema 测试。

验证：
- 4D 条件顺序采样的可行性、覆盖度、接受率
- N2 残量 ≥ 阈值
- 返回 dict 含 COMPONENT_FIELDS + BACKGROUND_FIELDS
- labels 仅含 4 列预测目标
- 不影响 hydrogen_ng schema
"""
from __future__ import annotations

import numpy as np
import pytest

from sg.sim.core.syngas_schema import (
    COMPONENT_FIELDS as SG_COMPONENT_FIELDS,
    BACKGROUND_FIELDS as SG_BACKGROUND_FIELDS,
    SLOW_CHANNELS as SG_SLOW_CHANNELS,
    SLOW_DYNAMIC_CHANNELS as SG_SLOW_DYNAMIC_CHANNELS,
)
from sg.sim.core.schema import COMPONENT_FIELDS as HG_COMPONENT_FIELDS
from sg.sim.generation.syngas import (
    SYNGAS_RANGES,
    build_syngas_label_rows,
    generate_syngas_condition_rows,
    is_feasible_syngas,
)


# ---------------------------------------------------------------------------
# Schema 隔离测试
# ---------------------------------------------------------------------------


def test_syngas_schema_fields():
    """syngas COMPONENT_FIELDS 第 4 列是 x_CO，不是 x_N2。"""
    assert SG_COMPONENT_FIELDS == ("x_H2", "x_CH4", "x_CO2", "x_CO")
    assert SG_BACKGROUND_FIELDS == ("x_N2",)


def test_syngas_schema_slow_channels_includes_co():
    """syngas SLOW_CHANNELS 含 V_NDIR_CO 且为 9 通道。"""
    assert "V_NDIR_CO" in SG_SLOW_CHANNELS
    assert len(SG_SLOW_CHANNELS) == 9
    assert "V_NDIR_CO" in SG_SLOW_DYNAMIC_CHANNELS
    assert len(SG_SLOW_DYNAMIC_CHANNELS) == 4


def test_hydrogen_ng_schema_unchanged():
    """hydrogen_ng COMPONENT_FIELDS 未被影响（分支隔离前提）。"""
    assert HG_COMPONENT_FIELDS == ("x_H2", "x_CH4", "x_CO2", "x_N2")


# ---------------------------------------------------------------------------
# 采样基本契约
# ---------------------------------------------------------------------------


def test_sampling_returns_correct_count():
    rows = generate_syngas_condition_rows(50, seed=42)
    assert len(rows) == 50


def test_sampling_each_row_has_all_required_fields():
    rows = generate_syngas_condition_rows(10, seed=42)
    for row in rows:
        for field in SG_COMPONENT_FIELDS:
            assert field in row, f"missing target field {field}"
        for field in SG_BACKGROUND_FIELDS:
            assert field in row, f"missing background field {field}"
        for env in ("T_C_base", "P_MPa_base", "H_RH_base", "L_m_base"):
            assert env in row


def test_sampling_x_co_present_x_n2_in_background():
    """x_CO 在目标列，x_N2 在背景列。"""
    rows = generate_syngas_condition_rows(5, seed=42)
    for row in rows:
        # 字符串形式
        x_co = float(row["x_CO"])
        x_n2 = float(row["x_N2"])
        assert x_co > 0
        assert x_n2 > 0


def test_sampling_invalid_args():
    with pytest.raises(ValueError):
        generate_syngas_condition_rows(0, seed=42)
    with pytest.raises(ValueError):
        generate_syngas_condition_rows(10, seed=42, sampling_strategy="bogus")


# ---------------------------------------------------------------------------
# 物理可行性约束
# ---------------------------------------------------------------------------


def _samples_to_array(rows: list[dict[str, str]]) -> dict[str, np.ndarray]:
    arr = {
        name: np.array([float(r[name]) for r in rows])
        for name in (*SG_COMPONENT_FIELDS, *SG_BACKGROUND_FIELDS)
    }
    return arr


def test_sampling_satisfies_n2_floor():
    rows = generate_syngas_condition_rows(500, seed=42)
    arr = _samples_to_array(rows)
    assert (arr["x_N2"] >= SYNGAS_RANGES.n2_min - 1e-6).all()


def test_sampling_satisfies_h2_co_ratio():
    rows = generate_syngas_condition_rows(500, seed=42)
    arr = _samples_to_array(rows)
    ratio = arr["x_H2"] / arr["x_CO"]
    assert (ratio >= SYNGAS_RANGES.h2_co_min - 1e-6).all()
    assert (ratio <= SYNGAS_RANGES.h2_co_max + 1e-6).all()


def test_sampling_satisfies_co2_co_ratio():
    rows = generate_syngas_condition_rows(500, seed=42)
    arr = _samples_to_array(rows)
    ratio = arr["x_CO2"] / arr["x_CO"]
    assert (ratio >= SYNGAS_RANGES.co2_co_min - 1e-6).all()
    assert (ratio <= SYNGAS_RANGES.co2_co_max + 1e-6).all()


def test_sampling_satisfies_carbon_balance():
    rows = generate_syngas_condition_rows(500, seed=42)
    arr = _samples_to_array(rows)
    carbon = arr["x_CO"] + arr["x_CO2"] + arr["x_CH4"]
    assert (carbon >= SYNGAS_RANGES.carbon_min - 1e-6).all()
    assert (carbon <= SYNGAS_RANGES.carbon_max + 1e-6).all()


def test_sampling_mass_conservation():
    """x_H2 + x_CH4 + x_CO2 + x_CO + x_N2 = 100%，浮点误差 < 1e-4。"""
    rows = generate_syngas_condition_rows(500, seed=42)
    arr = _samples_to_array(rows)
    total = arr["x_H2"] + arr["x_CH4"] + arr["x_CO2"] + arr["x_CO"] + arr["x_N2"]
    assert np.allclose(total, 100.0, atol=1e-4)


def test_is_feasible_syngas_helper():
    """物理约束辅助函数与采样结果一致。"""
    # 工业典型工况
    assert is_feasible_syngas(x_h2=30.0, x_co=45.0, x_co2=10.0, x_ch4=2.0)
    # N2 不足
    assert not is_feasible_syngas(x_h2=40.0, x_co=50.0, x_co2=8.0, x_ch4=2.0)
    # CO = 0
    assert not is_feasible_syngas(x_h2=30.0, x_co=0.0, x_co2=10.0, x_ch4=2.0)
    # H2/CO 超界
    assert not is_feasible_syngas(x_h2=55.0, x_co=10.0, x_co2=3.0, x_ch4=0.5)
    # 总碳不足
    assert not is_feasible_syngas(x_h2=50.0, x_co=15.0, x_co2=3.0, x_ch4=1.0)


# ---------------------------------------------------------------------------
# 覆盖度（粗粒度，不强求所有边际严格均匀）
# ---------------------------------------------------------------------------


def test_sampling_co_covers_range():
    """CO 应覆盖目标区间下端和上端各 10% 的边界（条件顺序采样中 CO 边际最接近均匀）。"""
    rows = generate_syngas_condition_rows(1000, seed=42)
    arr = _samples_to_array(rows)
    lo, hi = SYNGAS_RANGES.co
    span = hi - lo
    assert arr["x_CO"].min() < lo + 0.1 * span, f"CO low end uncovered: min={arr['x_CO'].min()}"
    assert arr["x_CO"].max() > hi - 0.1 * span, f"CO high end uncovered: max={arr['x_CO'].max()}"


def test_sampling_h2_covers_range_meaningfully():
    """H2 覆盖范围至少 50%（条件采样会被约束收紧，不强求 90%）。"""
    rows = generate_syngas_condition_rows(1000, seed=42)
    arr = _samples_to_array(rows)
    span = SYNGAS_RANGES.h2[1] - SYNGAS_RANGES.h2[0]
    realized = arr["x_H2"].max() - arr["x_H2"].min()
    assert realized > 0.5 * span, f"H2 realized span too small: {realized:.2f} vs target {span:.2f}"


def test_sampling_acceptance_rate_acceptable():
    """请求 200 条样本能在 1.2 倍候选内拿满。如果失败，说明约束过严或 oversample 偏低。"""
    rows = generate_syngas_condition_rows(200, seed=42)
    assert len(rows) == 200


# ---------------------------------------------------------------------------
# Labels 行
# ---------------------------------------------------------------------------


def test_labels_only_contain_target_components():
    """labels 仅含 sequence_id + 4 列目标，x_N2 不入 labels。"""
    rows = generate_syngas_condition_rows(5, seed=42)
    labels = build_syngas_label_rows(rows)
    assert len(labels) == 5
    for label in labels:
        assert "sequence_id" in label
        for field in SG_COMPONENT_FIELDS:
            assert field in label
        assert "x_N2" not in label
        # 仅 5 个键：sequence_id + 4 目标
        assert len(label) == 5


# ---------------------------------------------------------------------------
# 随机拒绝采样路径（备用）
# ---------------------------------------------------------------------------


def test_random_rejection_sampling_also_feasible():
    rows = generate_syngas_condition_rows(50, seed=42, sampling_strategy="random")
    assert len(rows) == 50
    arr = _samples_to_array(rows)
    assert (arr["x_N2"] >= SYNGAS_RANGES.n2_min - 1e-6).all()
    ratio_h2_co = arr["x_H2"] / arr["x_CO"]
    assert (ratio_h2_co >= SYNGAS_RANGES.h2_co_min - 1e-6).all()
    assert (ratio_h2_co <= SYNGAS_RANGES.h2_co_max + 1e-6).all()
