"""掘进通风场景的 LHS 采样 + schema 测试。

验证：
- 2D LHS 采样在 (CO2, O2) 空间，N2 = 100 - CO2 - O2
- 组分总量严格闭包 sum=100%
- N2 范围 [73.80, 81.97] 自动满足
- labels 写入 3 列预测目标（含 x_N2）
- BACKGROUND_FIELDS 为空
- 不影响 hydrogen_ng / syngas schema
"""
from __future__ import annotations

import numpy as np
import pytest

from sim.core.schema import (
    COMPONENT_FIELDS as HG_COMPONENT_FIELDS,
    SLOW_CHANNELS as HG_SLOW_CHANNELS,
)
from sim.core.syngas_schema import (
    BACKGROUND_FIELDS as SG_BACKGROUND_FIELDS,
    COMPONENT_FIELDS as SG_COMPONENT_FIELDS,
)
from sim.core.tunnel_ventilation_schema import (
    BACKGROUND_FIELDS as TV_BACKGROUND_FIELDS,
    COMPONENT_FIELDS as TV_COMPONENT_FIELDS,
    COMPOSITION_SCHEME as TV_COMPOSITION_SCHEME,
    SCHEMA_VERSION as TV_SCHEMA_VERSION,
    SLOW_CHANNELS as TV_SLOW_CHANNELS,
    SLOW_DYNAMIC_CHANNELS as TV_SLOW_DYNAMIC_CHANNELS,
)
from sim.generation.tunnel_ventilation import (
    TUNNEL_VENTILATION_RANGES,
    TunnelVentilationRanges,
    build_tunnel_ventilation_label_rows,
    generate_tunnel_ventilation_condition_rows,
)


# ---------------------------------------------------------------------------
# Schema 隔离测试
# ---------------------------------------------------------------------------


def test_tunnel_ventilation_schema_fields():
    """tv3 COMPONENT_FIELDS 是 (x_CO2, x_O2, x_N2)，BACKGROUND_FIELDS 为空。"""
    assert TV_COMPONENT_FIELDS == ("x_CO2", "x_O2", "x_N2")
    assert TV_BACKGROUND_FIELDS == ()
    assert TV_COMPOSITION_SCHEME == "tunnel_ventilation"
    assert TV_SCHEMA_VERSION == "tunnel-ventilation-1"


def test_tunnel_ventilation_slow_channels_7_no_v_ndir_ch4():
    """tv3 SLOW_CHANNELS 是 7 通道，不含 V_NDIR_CH4（场景无 CH₄）。"""
    assert len(TV_SLOW_CHANNELS) == 7
    assert "V_NDIR_CH4" not in TV_SLOW_CHANNELS
    assert "V_NDIR_CO2" in TV_SLOW_CHANNELS
    assert "V_NDIR_CO" not in TV_SLOW_CHANNELS
    assert TV_SLOW_DYNAMIC_CHANNELS == ("V_NDIR_CO2", "V_TCS")


def test_hydrogen_ng_schema_unchanged():
    """hydrogen_ng COMPONENT_FIELDS 未被影响。"""
    assert HG_COMPONENT_FIELDS == ("x_H2", "x_CH4", "x_CO2", "x_N2")
    assert len(HG_SLOW_CHANNELS) == 8


def test_syngas_schema_unchanged():
    """syngas schema 未被影响。"""
    assert SG_COMPONENT_FIELDS == ("x_H2", "x_CH4", "x_CO2", "x_CO")
    assert SG_BACKGROUND_FIELDS == ("x_N2",)


# ---------------------------------------------------------------------------
# 采样基本契约
# ---------------------------------------------------------------------------


def test_sampling_returns_correct_count():
    rows = generate_tunnel_ventilation_condition_rows(50, seed=42)
    assert len(rows) == 50


def test_sampling_each_row_has_all_required_fields():
    rows = generate_tunnel_ventilation_condition_rows(10, seed=42)
    for row in rows:
        for field in TV_COMPONENT_FIELDS:
            assert field in row, f"missing target field {field}"
        for env in ("T_C_base", "P_MPa_base", "H_RH_base", "L_m_base"):
            assert env in row
        # 不应含 syngas/hg 字段
        assert "x_H2" not in row
        assert "x_CH4" not in row
        assert "x_CO" not in row


def test_sampling_invalid_args():
    with pytest.raises(ValueError):
        generate_tunnel_ventilation_condition_rows(0, seed=42)
    with pytest.raises(ValueError):
        generate_tunnel_ventilation_condition_rows(10, seed=42, sampling_strategy="bogus")


# ---------------------------------------------------------------------------
# 物理可行性约束
# ---------------------------------------------------------------------------


def _samples_to_array(rows: list[dict[str, str]]) -> dict[str, np.ndarray]:
    return {
        name: np.array([float(r[name]) for r in rows])
        for name in TV_COMPONENT_FIELDS
    }


def test_sampling_satisfies_co2_range():
    rows = generate_tunnel_ventilation_condition_rows(500, seed=42)
    arr = _samples_to_array(rows)
    assert (arr["x_CO2"] >= TUNNEL_VENTILATION_RANGES.co2[0] - 1e-6).all()
    assert (arr["x_CO2"] <= TUNNEL_VENTILATION_RANGES.co2[1] + 1e-6).all()


def test_sampling_satisfies_o2_range():
    rows = generate_tunnel_ventilation_condition_rows(500, seed=42)
    arr = _samples_to_array(rows)
    assert (arr["x_O2"] >= TUNNEL_VENTILATION_RANGES.o2[0] - 1e-6).all()
    assert (arr["x_O2"] <= TUNNEL_VENTILATION_RANGES.o2[1] + 1e-6).all()


def test_sampling_satisfies_n2_range():
    """N2 ∈ [73.80, 81.97]，由 CO2/O2 范围间接保证。"""
    rows = generate_tunnel_ventilation_condition_rows(500, seed=42)
    arr = _samples_to_array(rows)
    assert (arr["x_N2"] >= TUNNEL_VENTILATION_RANGES.n2_min - 1e-6).all()
    assert (arr["x_N2"] <= TUNNEL_VENTILATION_RANGES.n2_max + 1e-6).all()


def test_sampling_mass_conservation():
    """x_CO2 + x_O2 + x_N2 = 100%，浮点误差 < 1e-4。"""
    rows = generate_tunnel_ventilation_condition_rows(500, seed=42)
    arr = _samples_to_array(rows)
    total = arr["x_CO2"] + arr["x_O2"] + arr["x_N2"]
    assert np.allclose(total, 100.0, atol=1e-4)


# ---------------------------------------------------------------------------
# 覆盖度
# ---------------------------------------------------------------------------


def test_sampling_co2_covers_range():
    """CO2 应覆盖目标区间下端和上端各 10% 的边界。"""
    rows = generate_tunnel_ventilation_condition_rows(1000, seed=42)
    arr = _samples_to_array(rows)
    lo, hi = TUNNEL_VENTILATION_RANGES.co2
    span = hi - lo
    assert arr["x_CO2"].min() < lo + 0.1 * span, f"CO2 low end uncovered: min={arr['x_CO2'].min()}"
    assert arr["x_CO2"].max() > hi - 0.1 * span, f"CO2 high end uncovered: max={arr['x_CO2'].max()}"


def test_sampling_o2_covers_range():
    """O2 应覆盖目标区间下端和上端各 10% 的边界。"""
    rows = generate_tunnel_ventilation_condition_rows(1000, seed=42)
    arr = _samples_to_array(rows)
    lo, hi = TUNNEL_VENTILATION_RANGES.o2
    span = hi - lo
    assert arr["x_O2"].min() < lo + 0.1 * span, f"O2 low end uncovered: min={arr['x_O2'].min()}"
    assert arr["x_O2"].max() > hi - 0.1 * span, f"O2 high end uncovered: max={arr['x_O2'].max()}"


# ---------------------------------------------------------------------------
# Labels 行
# ---------------------------------------------------------------------------


def test_labels_contain_all_three_targets_including_n2():
    """labels 含 sequence_id + 3 列目标（含 x_N2）。"""
    rows = generate_tunnel_ventilation_condition_rows(5, seed=42)
    labels = build_tunnel_ventilation_label_rows(rows)
    assert len(labels) == 5
    for label in labels:
        assert "sequence_id" in label
        for field in TV_COMPONENT_FIELDS:
            assert field in label
        assert "x_N2" in label  # N2 在本场景是目标
        # 4 个键：sequence_id + 3 目标
        assert len(label) == 4


# ---------------------------------------------------------------------------
# 可复现性
# ---------------------------------------------------------------------------


def test_sampling_reproducible_with_same_seed():
    rows_a = generate_tunnel_ventilation_condition_rows(20, seed=42)
    rows_b = generate_tunnel_ventilation_condition_rows(20, seed=42)
    assert rows_a == rows_b


def test_sampling_different_with_different_seed():
    rows_a = generate_tunnel_ventilation_condition_rows(20, seed=42)
    rows_b = generate_tunnel_ventilation_condition_rows(20, seed=43)
    assert rows_a != rows_b


# ---------------------------------------------------------------------------
# 随机采样路径（备用）
# ---------------------------------------------------------------------------


def test_random_sampling_also_feasible():
    rows = generate_tunnel_ventilation_condition_rows(50, seed=42, sampling_strategy="random")
    assert len(rows) == 50
    arr = _samples_to_array(rows)
    assert (arr["x_CO2"] >= TUNNEL_VENTILATION_RANGES.co2[0] - 1e-6).all()
    assert (arr["x_O2"] >= TUNNEL_VENTILATION_RANGES.o2[0] - 1e-6).all()
    total = arr["x_CO2"] + arr["x_O2"] + arr["x_N2"]
    assert np.allclose(total, 100.0, atol=1e-4)


# ---------------------------------------------------------------------------
# 自定义 ranges
# ---------------------------------------------------------------------------


def test_custom_ranges_respected():
    """可传入自定义 ranges（例如收窄 CO2 上限）。"""
    custom = TunnelVentilationRanges(co2=(0.04, 2.0), o2=(19.0, 21.0))
    rows = generate_tunnel_ventilation_condition_rows(
        50, seed=42, ranges=custom
    )
    arr = _samples_to_array(rows)
    assert (arr["x_CO2"] >= 0.04 - 1e-6).all()
    assert (arr["x_CO2"] <= 2.0 + 1e-6).all()
    assert (arr["x_O2"] >= 19.0 - 1e-6).all()
    assert (arr["x_O2"] <= 21.0 + 1e-6).all()
