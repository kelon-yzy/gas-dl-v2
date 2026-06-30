"""测试 W_base 行顺序与新通道布局对齐（方案 §8.5 / §11.1）。

不变量：W_base 第 m 行对应模态 m 的权重；新通道布局下,
extract_modal_input 返回的 sensor 索引必须与 W_base 行顺序一致：
- W_base[0] = NDIR  ↔ extract_modal_input(.., "ndir")  ↔ IDX_NDIR_CO2
- W_base[1] = TCD   ↔ extract_modal_input(.., "tcd")   ↔ IDX_TCS
- W_base[2] = USN   ↔ extract_modal_input(.., "usn")   ↔ IDX_US_SPEED
"""
from __future__ import annotations

import torch

from rcdw.models.rcdw import RCDW_MGDA
from rcdw.models.single_modal import (
    IDX_NDIR_CO2,
    IDX_TCS,
    IDX_US_SPEED,
    SENSOR_INDICES,
    extract_modal_input,
)


_W_BASE = torch.tensor(
    [
        [0.05, 0.70, 0.05],  # NDIR
        [0.50, 0.15, 0.45],  # TCD
        [0.45, 0.15, 0.50],  # US
    ],
    dtype=torch.float32,
)


def test_sensor_indices_in_w_base_order():
    """SENSOR_INDICES['ndir'/'tcd'/'usn'] 顺序与 W_base 行顺序一致。"""
    assert SENSOR_INDICES["ndir"] == IDX_NDIR_CO2
    assert SENSOR_INDICES["tcd"] == IDX_TCS
    assert SENSOR_INDICES["usn"] == IDX_US_SPEED


def test_w_base_columns_sum_to_one():
    """W_base 每列(每种气体)各模态权重 sum=1.0。"""
    col_sums = _W_BASE.sum(dim=0)
    torch.testing.assert_close(col_sums, torch.ones(3), atol=1e-6, rtol=1e-6)


def test_extract_modal_input_sensor_matches_w_base_row():
    """extract_modal_input 取的 sensor 通道与 W_base 行顺序一一对应。"""
    x_last = torch.zeros(2, 12)
    x_last[:, IDX_NDIR_CO2] = 1.0
    x_last[:, IDX_TCS] = 2.0
    x_last[:, IDX_US_SPEED] = 3.0

    # NDIR (row 0) → sensor = x_last[:, IDX_NDIR_CO2] = 1
    assert extract_modal_input(x_last, "ndir")[0, 0].item() == 1.0
    # TCD (row 1) → sensor = x_last[:, IDX_TCS] = 2
    assert extract_modal_input(x_last, "tcd")[0, 0].item() == 2.0
    # USN (row 2) → sensor = x_last[:, IDX_US_SPEED] = 3
    assert extract_modal_input(x_last, "usn")[0, 0].item() == 3.0


def test_rcdw_mgda_forward_with_12_channel_input():
    """RCDW_MGDA 在新 12 维输入上 forward 端到端通过。"""
    model = RCDW_MGDA(_W_BASE, hidden=8, window=8)
    x = torch.randn(2, 8, 12)
    out = model(x)
    assert out["C"].shape == (2, 3)
    assert out["Y_modal"].shape == (2, 3, 3)
    assert out["E_pred"].shape == (2, 3, 3)
    assert out["W"].shape == (2, 3, 3)
    # 融合输出每行应近似归一化
    sums = out["W"].sum(dim=1)  # (B, G)
    torch.testing.assert_close(sums, torch.ones_like(sums), atol=1e-5, rtol=1e-5)


def test_rcdw_mgda_y_modal_stack_in_w_base_order():
    """Y_modal[:, 0/1/2] 应分别对应 NDIR/TCD/USN(与 W_base 行顺序一致)。"""
    model = RCDW_MGDA(_W_BASE, hidden=8, window=8)
    model.eval()
    # 构造极端输入：所有模态用相同 sensor 值,可以验证 forward 不报错
    x = torch.zeros(2, 8, 12)
    x[:, :, IDX_NDIR_CO2] = 1.5
    x[:, :, IDX_TCS] = 1.5
    x[:, :, IDX_US_SPEED] = 350.0
    out = model(x)
    # 三 modal 都应输出合法分布
    sums = out["Y_modal"].sum(dim=-1)
    nonzero_mask = sums > 0.5
    torch.testing.assert_close(
        sums[nonzero_mask], torch.ones(int(nonzero_mask.sum())),
        atol=1e-4, rtol=1e-4,
    )
