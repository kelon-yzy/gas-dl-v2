"""测试 RCDW conditions.py 的 LHS 采样与三组分映射。

对应方案 §5.1 + §11.1。
"""
from __future__ import annotations

import pytest

from rcdw.sim.core.schema import COMPONENT_FIELDS, LEGACY_CONDITION_FIELDS
from rcdw.sim.generation.conditions import (
    build_label_rows,
    generate_condition_rows,
)


def test_condition_rows_shape():
    rows = generate_condition_rows(16, seed=42)
    assert len(rows) == 16
    expected_keys = {
        "sequence_id",
        "mixture_id",
        "T_C_base",
        "P_MPa_base",
        "H_RH_base",
        "L_m_base",
        "status",
        *COMPONENT_FIELDS,
    }
    for row in rows:
        assert set(row.keys()) >= expected_keys
        assert all(isinstance(v, str) for v in row.values())


def test_component_sum_equals_100():
    rows = generate_condition_rows(64, seed=42)
    for row in rows:
        total = sum(float(row[f]) for f in COMPONENT_FIELDS)
        assert abs(total - 100.0) < 1e-5, (
            f"组分和应为 100,实际 {total} (row={row['sequence_id']})"
        )


def test_n2_lower_bound_55_percent():
    """方案 §2.1: N2 ∈ [55, 100]%; 边界保护应保证 x_N2 >= 55。"""
    rows = generate_condition_rows(256, seed=42)
    for row in rows:
        x_n2 = float(row["x_N2"])
        assert x_n2 >= 55.0 - 1e-6, f"N2 应 >= 55%,实际 {x_n2}"
        assert x_n2 <= 100.0 + 1e-6, f"N2 应 <= 100%,实际 {x_n2}"


def test_o2_co2_ranges():
    rows = generate_condition_rows(256, seed=42)
    for row in rows:
        x_o2 = float(row["x_O2"])
        x_co2 = float(row["x_CO2"])
        # 边界回退可能等比例缩减,因此上界用 25/20 + 浮点容差
        assert -1e-6 <= x_o2 <= 25.0 + 1e-6, f"O2 越界: {x_o2}"
        assert -1e-6 <= x_co2 <= 20.0 + 1e-6, f"CO2 越界: {x_co2}"


def test_lhs_reproducibility():
    """同一 seed 应产生相同结果。"""
    rows_a = generate_condition_rows(32, seed=7)
    rows_b = generate_condition_rows(32, seed=7)
    assert rows_a == rows_b


def test_lhs_seed_sensitivity():
    """不同 seed 应产生不同结果。"""
    rows_a = generate_condition_rows(32, seed=7)
    rows_b = generate_condition_rows(32, seed=8)
    assert rows_a != rows_b


def test_ids_unique_and_prefixed():
    rows = generate_condition_rows(100, seed=42)
    seq_ids = [row["sequence_id"] for row in rows]
    mix_ids = [row["mixture_id"] for row in rows]
    assert len(set(seq_ids)) == 100
    assert len(set(mix_ids)) == 100
    assert all(s.startswith("RCDW-Q") for s in seq_ids)
    assert all(m.startswith("RCDW-M") for m in mix_ids)


def test_no_legacy_fields_emitted():
    rows = generate_condition_rows(8, seed=42)
    for row in rows:
        for legacy in LEGACY_CONDITION_FIELDS:
            assert legacy not in row, f"condition row 不应含 LEGACY 字段 {legacy}"


def test_build_label_rows():
    cond_rows = generate_condition_rows(5, seed=42)
    label_rows = build_label_rows(cond_rows)
    assert len(label_rows) == 5
    for label_row, cond_row in zip(label_rows, cond_rows, strict=True):
        assert label_row["sequence_id"] == cond_row["sequence_id"]
        for f in COMPONENT_FIELDS:
            assert label_row[f] == cond_row[f]


def test_invalid_sampling_strategy():
    with pytest.raises(ValueError, match="sampling_strategy"):
        generate_condition_rows(4, seed=0, sampling_strategy="bogus")


def test_invalid_sequence_count():
    with pytest.raises(ValueError, match="sequence_count"):
        generate_condition_rows(0, seed=0)


def test_lhs_rescaled_count_is_small():
    """方案 §13.4: O2 + CO2 max = 45 == 100 - 55,理论几乎不触发回退。

    用一个较大样本量验证回退率 < 1%（仅浮点精度引起）。
    """
    rows = generate_condition_rows(1000, seed=42)
    rescaled = 0
    for row in rows:
        x_o2 = float(row["x_O2"])
        x_co2 = float(row["x_CO2"])
        # 若发生缩减,则 O2 + CO2 严格 < u_o2*25 + u_co2*20 的上界
        # 简化判据:N2 == 55 ± 1e-5 时视为发生缩减
        if abs(float(row["x_N2"]) - 55.0) < 1e-5:
            rescaled += 1
    assert rescaled / len(rows) < 0.01, f"回退率应 < 1%,实际 {rescaled}/1000"
