"""测试 RCDWFusion 数值对齐、维度、可微性。"""
import torch
import numpy as np
import pytest
from rcdw.models.rcdw import RCDWFusion


@pytest.fixture
def W_base():
    return torch.tensor([
        [0.05, 0.70, 0.05],
        [0.50, 0.15, 0.45],
        [0.45, 0.15, 0.50],
    ], dtype=torch.float32)


@pytest.fixture
def fusion(W_base):
    return RCDWFusion(W_base)


def test_output_shape(fusion):
    Y = torch.rand(8, 3, 3)
    E = torch.rand(8, 3, 3).abs() * 0.1
    C, W = fusion(Y, E)
    assert C.shape == (8, 3)
    assert W.shape == (8, 3, 3)


def test_weight_sum_one(fusion):
    Y = torch.rand(16, 3, 3)
    E = torch.rand(16, 3, 3).abs() * 0.1
    _, W = fusion(Y, E)
    W_sum = W.sum(dim=1)  # (B, G)
    torch.testing.assert_close(W_sum, torch.ones_like(W_sum), atol=1e-5, rtol=1e-5)


def test_differentiable(fusion):
    Y = torch.rand(4, 3, 3, requires_grad=True)
    E = (torch.rand(4, 3, 3).abs() * 0.1 + 0.01).requires_grad_(True)
    C, W = fusion(Y, E)
    loss = C.sum()
    loss.backward()
    assert Y.grad is not None
    assert E.grad is not None


def test_alpha_boundary(W_base):
    fusion = RCDWFusion(W_base, alpha_min=0.0, alpha_max=1.0, tau_a=0.05)
    # 所有模态误差相同 → dE=0 → alpha=alpha_min=0 → W=W_base
    Y = torch.rand(4, 3, 3)
    E = torch.ones(4, 3, 3) * 0.05
    _, W = fusion(Y, E)
    for b in range(4):
        for g in range(3):
            torch.testing.assert_close(
                W[b, :, g], W_base[:, g], atol=1e-4, rtol=1e-4
            )


def test_shift_clamp(W_base):
    fusion = RCDWFusion(W_base, s_min=0.01, s_max=0.01, tau_s=0.05)
    Y = torch.rand(4, 3, 3)
    E = torch.rand(4, 3, 3).abs()
    _, W = fusion(Y, E)
    diff = (W - W_base.unsqueeze(0)).abs()
    # clamp 后差异应 <= s_max + 归一化微调
    assert diff.max().item() < 0.05


def test_zero_error_uses_baseline(W_base):
    # E=0 → dE=0 → alpha=alpha_min；只有 alpha_min=0 时 W 才严格回退到 W_base
    fusion = RCDWFusion(W_base, alpha_min=0.0, alpha_max=1.0, tau_a=0.05)
    Y = torch.rand(4, 3, 3)
    E = torch.zeros(4, 3, 3)
    _, W = fusion(Y, E)
    for b in range(4):
        torch.testing.assert_close(W[b], W_base, atol=1e-4, rtol=1e-4)


def test_numerical_alignment(W_base):
    """与手算结果对齐（简化版 numerical_check）。"""
    torch.manual_seed(123)
    B = 2
    Y = torch.rand(B, 3, 3)
    E = torch.rand(B, 3, 3).abs() * 0.1

    fusion = RCDWFusion(W_base)
    C, W = fusion(Y, E)

    # 手算 Wmix
    logits = -8.0 * E
    Wmix = torch.softmax(logits, dim=1)

    dE = E.max(dim=1).values - E.min(dim=1).values
    alpha = 0.1 + 0.8 * dE / (dE + 0.05)
    alpha = alpha.unsqueeze(1)

    W_expected = (1 - alpha) * W_base + alpha * Wmix
    shift = 0.05 + 0.35 * dE.unsqueeze(1) / (dE.unsqueeze(1) + 0.05)
    W_expected = W_expected.clamp(W_base - shift, W_base + shift)
    W_expected = W_expected / W_expected.sum(dim=1, keepdim=True)

    torch.testing.assert_close(W, W_expected, atol=1e-5, rtol=1e-5)


def test_w_base_shape_assertion():
    with pytest.raises(AssertionError):
        RCDWFusion(torch.rand(3, 4))  # 错误 shape


def test_rcdw_mgda_fusion_kwargs_passthrough():
    """配置中的 fusion 超参必须真实传到 RCDWFusion，不能被静默忽略。"""
    from rcdw.models.rcdw import RCDW_MGDA

    W_base = torch.tensor([
        [0.05, 0.70, 0.05],
        [0.50, 0.15, 0.45],
        [0.45, 0.15, 0.50],
    ], dtype=torch.float32)

    fusion_kwargs = {
        "beta": 2.0,
        "alpha_min": 0.2,
        "alpha_max": 0.7,
        "tau_a": 0.1,
        "s_min": 0.02,
        "s_max": 0.30,
        "tau_s": 0.08,
    }
    model = RCDW_MGDA(W_base, hidden=32, fusion_kwargs=fusion_kwargs)

    assert model.fuse.beta == 2.0
    assert model.fuse.a_min == 0.2
    assert model.fuse.a_max == 0.7
    assert model.fuse.tau_a == 0.1
    assert model.fuse.s_min == 0.02
    assert model.fuse.s_max == 0.30
    assert model.fuse.tau_s == 0.08


def test_rcdw_mgda_default_fusion_kwargs():
    """fusion_kwargs=None 时使用 RCDWFusion 默认值。"""
    from rcdw.models.rcdw import RCDW_MGDA

    W_base = torch.tensor([
        [0.05, 0.70, 0.05],
        [0.50, 0.15, 0.45],
        [0.45, 0.15, 0.50],
    ], dtype=torch.float32)

    model = RCDW_MGDA(W_base, hidden=32)
    assert model.fuse.beta == 8.0
    assert model.fuse.tau_a == 0.05
