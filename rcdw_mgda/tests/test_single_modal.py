"""测试单模态网络。"""
import torch
import pytest
from rcdw.models.single_modal import NDIRNet, TCDNet, USNet, extract_modal_input


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
    # 当 raw 经 clamp(min=0) 后全为 0 时 sum=0，否则 L1 归一化保证 sum=1
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


def test_extract_modal_input():
    x_last = torch.randn(4, 6)  # [S_ndir, S_tc, S_us, P, T, RH]
    inp = extract_modal_input(x_last, "ndir")
    assert inp.shape == (4, 4)
    torch.testing.assert_close(inp[:, 0], x_last[:, 0])  # S_ndir
    torch.testing.assert_close(inp[:, 1], x_last[:, 3])  # P
    torch.testing.assert_close(inp[:, 2], x_last[:, 4])  # T
    torch.testing.assert_close(inp[:, 3], x_last[:, 5])  # RH
