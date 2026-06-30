"""测试单模态网络（v1.2 §8.3 12 维通道布局）。"""
from __future__ import annotations

import pytest
import torch

from rcdw.models.single_modal import (
    ENV_INDICES,
    IDX_NDIR_CO2,
    IDX_TCS,
    IDX_T_C,
    IDX_US_SPEED,
    INPUT_CHANNELS,
    SENSOR_INDICES,
    NDIRNet,
    TCDNet,
    USNet,
    extract_modal_input,
)


def test_input_channels_constant():
    """v1.2 §8.2: 输入维度从旧 6 升级为 12。"""
    assert INPUT_CHANNELS == 12


def test_sensor_indices_match_w_base_row_order():
    """v1.2 §8.5: NDIR/TCD/USN 行顺序与传感器索引一致。"""
    assert SENSOR_INDICES["ndir"] == IDX_NDIR_CO2 == 0
    assert SENSOR_INDICES["tcd"] == IDX_TCS == 1
    assert SENSOR_INDICES["usn"] == IDX_US_SPEED == 8


def test_env_indices_order():
    """v1.2 §8.2: 环境顺序 [T_C, P_MPa, H_RH] 对应索引 [2, 3, 4]。"""
    assert ENV_INDICES == [2, 3, 4]
    assert IDX_T_C == 2


def test_output_shape():
    net = NDIRNet(in_dim=4, hidden=32)
    x = torch.randn(8, 4)
    y = net(x)
    assert y.shape == (8, 3)


def test_output_sum_one():
    torch.manual_seed(0)
    net = TCDNet(in_dim=4, hidden=32)
    x = torch.randn(16, 4)
    y = net(x)
    sums = y.sum(dim=-1)
    nonzero_mask = sums > 0.5
    torch.testing.assert_close(
        sums[nonzero_mask], torch.ones(int(nonzero_mask.sum())),
        atol=1e-4, rtol=1e-4,
    )


def test_output_non_negative():
    net = USNet(in_dim=4, hidden=32)
    x = torch.randn(16, 4)
    y = net(x)
    assert (y >= 0).all()


def test_extract_modal_input_ndir():
    """NDIR 取 IDX_NDIR_CO2 (=0) + ENV_INDICES (=[2,3,4])。"""
    x_last = torch.randn(4, 12)
    inp = extract_modal_input(x_last, "ndir")
    assert inp.shape == (4, 4)
    torch.testing.assert_close(inp[:, 0], x_last[:, IDX_NDIR_CO2])
    torch.testing.assert_close(inp[:, 1], x_last[:, IDX_T_C])
    torch.testing.assert_close(inp[:, 2], x_last[:, 3])  # P_MPa
    torch.testing.assert_close(inp[:, 3], x_last[:, 4])  # H_RH


def test_extract_modal_input_tcd():
    x_last = torch.randn(4, 12)
    inp = extract_modal_input(x_last, "tcd")
    assert inp.shape == (4, 4)
    torch.testing.assert_close(inp[:, 0], x_last[:, IDX_TCS])


def test_extract_modal_input_usn():
    """v1.2: USN 主信号改为 IDX_US_SPEED (=8) 而非旧 idx 2。"""
    x_last = torch.randn(4, 12)
    inp = extract_modal_input(x_last, "usn")
    assert inp.shape == (4, 4)
    torch.testing.assert_close(inp[:, 0], x_last[:, IDX_US_SPEED])


def test_extract_modal_input_unknown_modality():
    x_last = torch.randn(4, 12)
    with pytest.raises(ValueError, match="unknown modality"):
        extract_modal_input(x_last, "bogus")
