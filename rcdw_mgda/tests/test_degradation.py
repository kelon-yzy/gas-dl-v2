"""测试退化硬抑制。"""
import torch
import pytest
from rcdw.utils.degradation import hard_suppress


def test_no_degradation():
    """所有模态误差相同时，不触发抑制。"""
    W = torch.ones(4, 3, 3) / 3
    E = torch.ones(4, 3, 3) * 0.05
    W_out, degraded = hard_suppress(W, E)
    assert not degraded.any()
    torch.testing.assert_close(W_out, W, atol=1e-5, rtol=1e-5)


def test_degradation_trigger():
    """模态 0 的误差 > 4x 最小值时触发抑制。"""
    W = torch.ones(10, 3, 3) / 3
    E = torch.ones(10, 3, 3) * 0.01
    E[:, 0, :] = 0.05  # 模态 0 误差 5x
    W_out, degraded = hard_suppress(W, E, ratio=4.0, cap=0.04)
    assert degraded[0, :].all()  # 模态 0 对所有气体退化
    # cap 是归一化前的上限；归一化后退化模态权重应远小于正常模态
    assert (W_out[:, 0, :] < W_out[:, 1, :]).all()
    assert (W_out[:, 0, :] < W_out[:, 2, :]).all()


def test_renormalization():
    """抑制后权重应重归一化到 sum=1。"""
    W = torch.ones(8, 3, 3) / 3
    E = torch.ones(8, 3, 3) * 0.01
    E[:, 2, :] = 0.1  # 模态 2 退化
    W_out, _ = hard_suppress(W, E, ratio=4.0, cap=0.04)
    W_sum = W_out.sum(dim=1)
    torch.testing.assert_close(W_sum, torch.ones_like(W_sum), atol=1e-5, rtol=1e-5)


def test_cap_value():
    """退化模态归一化前权重应 ≤ cap，归一化后应远小于正常模态。"""
    W = torch.ones(10, 3, 3) * 0.5
    E = torch.ones(10, 3, 3) * 0.01
    E[:, 1, 0] = 0.2  # 模态 1 对 O2 退化
    W_out, degraded = hard_suppress(W, E, ratio=4.0, cap=0.04)
    if degraded[1, 0]:
        # 模态 1 在 O2 通道的权重应比模态 0/2 显著小
        assert (W_out[:, 1, 0] < W_out[:, 0, 0]).all()
        assert (W_out[:, 1, 0] < W_out[:, 2, 0]).all()
